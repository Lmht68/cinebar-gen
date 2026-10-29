"""Behavior tests for stable directory barcode processing."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cinebar import (
    CinebarError,
    CropMode,
    DirectoryProcessor,
    FailureCategory,
    GenerationSettings,
    ProcessingResult,
)
from cinebar.crop import CropSelection
from cinebar.progress import ProgressReporter

VIDEO_SUFFIXES = (
    ".avi",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".ts",
    ".webm",
    ".wmv",
)


def _mark_hidden(path: Path) -> None:
    if os.name == "nt":
        command = ["attrib", "+H", str(path)]
    elif shutil.which("chflags") is not None:
        command = ["chflags", "hidden", str(path)]
    else:
        pytest.skip("Host does not expose a native hidden-entry mechanism.")

    completed_process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed_process.returncode != 0:
        pytest.skip("Host refused to mark the test entry hidden.")


def _create_symlink_or_skip(link_path: Path, target_path: Path) -> None:
    try:
        link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())
    except OSError:
        pytest.skip("Host refused to create the test symlink.")


class _WritingBarcodeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, int, int | None, bool, CropSelection]] = []

    def process(
        self,
        media_path: Path,
        output_path: Path,
        height: int = 300,
        workers: int | None = None,
        overwrite: bool = False,
        crop: CropSelection = CropMode.AUTO,
        progress_reporter: ProgressReporter | None = None,
    ) -> ProcessingResult:
        self.calls.append((media_path, output_path, height, workers, overwrite, crop))
        output_path.write_bytes(media_path.name.encode())
        return ProcessingResult(
            output_path=output_path,
            decoded_frame_count=1,
            output_width=1,
            output_height=height,
        )


class _ReentrancyDetectingBarcodeGenerator(_WritingBarcodeGenerator):
    def __init__(self) -> None:
        super().__init__()
        self._is_processing = False
        self.completed_sources: list[Path] = []

    def process(
        self,
        media_path: Path,
        output_path: Path,
        height: int = 300,
        workers: int | None = None,
        overwrite: bool = False,
        crop: CropSelection = CropMode.AUTO,
        progress_reporter: ProgressReporter | None = None,
    ) -> ProcessingResult:
        prior_sources = [call[0] for call in self.calls]
        if self._is_processing or prior_sources != self.completed_sources:
            raise AssertionError("A prior barcode generator call has not returned.")
        self._is_processing = True
        try:
            return super().process(
                media_path,
                output_path,
                height,
                workers,
                overwrite,
                crop,
                progress_reporter,
            )
        finally:
            self._is_processing = False
            self.completed_sources.append(media_path)


class _FailingBarcodeGenerator(_WritingBarcodeGenerator):
    def process(
        self,
        media_path: Path,
        output_path: Path,
        height: int = 300,
        workers: int | None = None,
        overwrite: bool = False,
        crop: CropSelection = CropMode.AUTO,
        progress_reporter: ProgressReporter | None = None,
    ) -> ProcessingResult:
        self.calls.append((media_path, output_path, height, workers, overwrite, crop))
        raise CinebarError(FailureCategory.NO_VIDEO_STREAM)


def test_directory_processor_generates_nested_barcodes_beside_sources(
    tmp_path: Path,
) -> None:
    """Generate ordered beside-input barcodes from nested supported sources."""
    root = tmp_path / "library"
    first_source = root / "Alpha" / "First.mkv"
    second_source = root / "Beta" / "Second.MP4"
    first_source.parent.mkdir(parents=True)
    second_source.parent.mkdir(parents=True)
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    barcode_generator = _WritingBarcodeGenerator()

    results = DirectoryProcessor(barcode_generator).process_batch(
        root,
        GenerationSettings(height=3, workers=1, crop=CropMode.NONE),
    )

    expected_outputs = (
        first_source.with_name("First.cinebar.png"),
        second_source.with_name("Second.cinebar.png"),
    )
    assert tuple(result.output_path for result in results) == expected_outputs
    assert [call[:2] for call in barcode_generator.calls] == [
        (first_source, expected_outputs[0]),
        (second_source, expected_outputs[1]),
    ]
    assert [call[2:] for call in barcode_generator.calls] == [
        (3, 1, False, CropMode.NONE),
        (3, 1, False, CropMode.NONE),
    ]

    assert [output_path.read_bytes() for output_path in expected_outputs] == [
        b"First.mkv",
        b"Second.MP4",
    ]


def test_directory_processor_rejects_missing_root_before_generation(
    tmp_path: Path,
) -> None:
    """Classify a missing batch root without entering the barcode generator."""
    barcode_generator = _WritingBarcodeGenerator()

    with pytest.raises(CinebarError) as raised_error:
        DirectoryProcessor(barcode_generator).process_batch(tmp_path / "missing")

    assert raised_error.value.category is FailureCategory.ROOT_NOT_FOUND
    assert barcode_generator.calls == []


def test_directory_processor_rejects_file_root_before_generation(
    tmp_path: Path,
) -> None:
    """Classify a regular-file batch root without entering the generator."""
    root = tmp_path / "not-a-directory.mkv"
    root.write_bytes(b"source")
    barcode_generator = _WritingBarcodeGenerator()

    with pytest.raises(CinebarError) as raised_error:
        DirectoryProcessor(barcode_generator).process_batch(root)

    assert raised_error.value.category is FailureCategory.ROOT_UNREADABLE
    assert barcode_generator.calls == []


def test_directory_processor_rejects_scan_failure_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify initial-tree scan failures without entering the generator."""
    root = tmp_path / "library"
    root.mkdir()
    (root / "movie.mkv").write_bytes(b"source")
    barcode_generator = _WritingBarcodeGenerator()

    def fail_scandir(path: object) -> object:
        raise PermissionError

    monkeypatch.setattr(os, "scandir", fail_scandir)

    with pytest.raises(CinebarError) as raised_error:
        DirectoryProcessor(barcode_generator).process_batch(root)

    assert raised_error.value.category is FailureCategory.ROOT_UNREADABLE
    assert barcode_generator.calls == []


def test_directory_processor_accepts_all_case_insensitive_video_suffixes(
    tmp_path: Path,
) -> None:
    """Generate barcodes for every supported lowercase and uppercase suffix."""
    root = tmp_path / "library"
    sources_directory = root / "formats"
    sources_directory.mkdir(parents=True)
    source_paths = []
    for index, suffix in enumerate(VIDEO_SUFFIXES):
        for name_suffix in (suffix, suffix.upper()):
            source = sources_directory / f"movie-{index}{name_suffix}"
            source.write_bytes(b"source")
            source_paths.append(source)
    barcode_generator = _WritingBarcodeGenerator()

    results = DirectoryProcessor(barcode_generator).process_batch(root)

    expected_outputs = {
        source.with_name(f"{source.stem}.cinebar.png") for source in source_paths
    }
    assert {result.output_path for result in results} == expected_outputs
    assert {call[0] for call in barcode_generator.calls} == set(source_paths)
    assert all(output_path.exists() for output_path in expected_outputs)


def test_directory_processor_returns_empty_tuple_for_empty_tree(tmp_path: Path) -> None:
    """Return no results when a valid tree has no supported source media."""
    root = tmp_path / "empty-library"
    root.mkdir()

    results = DirectoryProcessor(_WritingBarcodeGenerator()).process_batch(root)

    assert results == ()


def test_directory_processor_skips_dot_entries_and_non_video_artifacts(
    tmp_path: Path,
) -> None:
    """Ignore hidden-by-name, generated, temporary, and non-video entries."""
    root = tmp_path / "library"
    root.mkdir()
    source = root / "movie.mkv"
    source.write_bytes(b"source")
    (root / ".dot-file.mkv").write_bytes(b"ignored")
    dot_directory = root / ".dot-directory"
    dot_directory.mkdir()
    (dot_directory / "nested.mkv").write_bytes(b"ignored")
    (root / "movie.cinebar.png").write_bytes(b"generated")
    (root / ".movie.cinebar.png.token.tmp").write_bytes(b"temporary")
    (root / "notes.txt").write_text("ignored")
    barcode_generator = _WritingBarcodeGenerator()

    DirectoryProcessor(barcode_generator).process_batch(root)

    assert [call[0] for call in barcode_generator.calls] == [source]


def test_directory_processor_skips_host_hidden_files_and_directories(
    tmp_path: Path,
) -> None:
    """Ignore native-hidden media and complete native-hidden subtrees."""
    root = tmp_path / "library"
    root.mkdir()
    visible_source = root / "visible.mkv"
    hidden_file = root / "hidden-file.mkv"
    hidden_directory = root / "hidden-directory"
    visible_source.write_bytes(b"visible")
    hidden_file.write_bytes(b"hidden")
    hidden_directory.mkdir()
    (hidden_directory / "hidden-nested.mkv").write_bytes(b"hidden")
    _mark_hidden(hidden_file)
    _mark_hidden(hidden_directory)
    barcode_generator = _WritingBarcodeGenerator()

    DirectoryProcessor(barcode_generator).process_batch(root)

    assert [call[0] for call in barcode_generator.calls] == [visible_source]


def test_directory_processor_skips_file_and_directory_symlinks(
    tmp_path: Path,
) -> None:
    """Ignore linked media and linked directory trees without following targets."""
    root = tmp_path / "library"
    root.mkdir()
    visible_source = root / "visible.mkv"
    visible_source.write_bytes(b"visible")
    linked_file_target = tmp_path / "linked-target.mkv"
    linked_file_target.write_bytes(b"linked")
    linked_directory_target = tmp_path / "linked-directory-target"
    linked_directory_target.mkdir()
    (linked_directory_target / "nested.mkv").write_bytes(b"linked")
    _create_symlink_or_skip(root / "linked-file.mkv", linked_file_target)
    _create_symlink_or_skip(root / "linked-directory", linked_directory_target)
    barcode_generator = _WritingBarcodeGenerator()

    DirectoryProcessor(barcode_generator).process_batch(root)

    assert [call[0] for call in barcode_generator.calls] == [visible_source]


def test_directory_processor_preserves_source_stems_in_beside_input_names(
    tmp_path: Path,
) -> None:
    """Keep spaces, capitalization, dots, and years in derived output names."""
    root = tmp_path / "library"
    root.mkdir()
    first_source = root / "Movie.One (2024).MKV"
    second_source = root / "mixed Case.part.1999.mp4"
    second_source.write_bytes(b"second")
    first_source.write_bytes(b"first")

    results = DirectoryProcessor(_WritingBarcodeGenerator()).process_batch(root)

    assert tuple(result.output_path.name for result in results) == (
        "mixed Case.part.1999.cinebar.png",
        "Movie.One (2024).cinebar.png",
    )


def test_directory_processor_rejects_output_collisions_before_generation(
    tmp_path: Path,
) -> None:
    """Abort the whole batch before either source maps to one output path."""
    root = tmp_path / "library"
    root.mkdir()
    (root / "Movie.mkv").write_bytes(b"mkv")
    (root / "Movie.mp4").write_bytes(b"mp4")
    barcode_generator = _WritingBarcodeGenerator()

    with pytest.raises(CinebarError) as raised_error:
        DirectoryProcessor(barcode_generator).process_batch(
            root,
            GenerationSettings(overwrite=True),
        )

    assert raised_error.value.category is FailureCategory.OUTPUT_COLLISION
    assert barcode_generator.calls == []


def test_directory_processor_rejects_symlink_root_before_generation(
    tmp_path: Path,
) -> None:
    """Classify a linked root without following it or entering the generator."""
    root_target = tmp_path / "root-target"
    root_target.mkdir()
    (root_target / "movie.mkv").write_bytes(b"source")
    root_link = tmp_path / "root-link"
    _create_symlink_or_skip(root_link, root_target)
    barcode_generator = _WritingBarcodeGenerator()

    with pytest.raises(CinebarError) as raised_error:
        DirectoryProcessor(barcode_generator).process_batch(root_link)

    assert raised_error.value.category is FailureCategory.ROOT_UNREADABLE
    assert barcode_generator.calls == []


def test_directory_processor_processes_one_movie_at_a_time(tmp_path: Path) -> None:
    """Start each later generator call only after the preceding call returns."""
    root = tmp_path / "library"
    root.mkdir()
    first_source = root / "Alpha.mkv"
    second_source = root / "Beta.mp4"
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    barcode_generator = _ReentrancyDetectingBarcodeGenerator()

    results = DirectoryProcessor(barcode_generator).process_batch(root)

    assert tuple(result.output_path for result in results) == (
        first_source.with_name("Alpha.cinebar.png"),
        second_source.with_name("Beta.cinebar.png"),
    )
    assert barcode_generator.completed_sources == [first_source, second_source]


def test_directory_processor_stops_after_first_generator_failure(
    tmp_path: Path,
) -> None:
    """Propagate the first generator failure without starting later media."""
    root = tmp_path / "library"
    root.mkdir()
    first_source = root / "Alpha.mkv"
    second_source = root / "Beta.mp4"
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    barcode_generator = _FailingBarcodeGenerator()

    with pytest.raises(CinebarError) as raised_error:
        DirectoryProcessor(barcode_generator).process_batch(root)

    assert raised_error.value.category is FailureCategory.NO_VIDEO_STREAM
    assert [call[0] for call in barcode_generator.calls] == [first_source]
