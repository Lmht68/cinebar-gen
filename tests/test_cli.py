"""End-to-end tests for the installed cinebar command."""

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import png
import pytest

RGB_COLORS = [(40, 50, 60), (0, 0, 0), (200, 80, 20), (9, 150, 230)]
PRESENTATION_TIMESTAMPS = [0, 1, 4, 10]


def _run_cinebar(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the installed Cinebar command with captured text output."""
    cinebar_executable = shutil.which("cinebar")
    assert cinebar_executable is not None, "Installed cinebar executable was not found."
    return subprocess.run(
        [cinebar_executable, *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_installed_failure(
    completed_process: subprocess.CompletedProcess[str],
    category: str,
    message: str,
    sensitive_media_path: Path,
) -> None:
    """Require one safe, actionable installed-command failure."""
    assert completed_process.returncode != 0
    assert completed_process.stderr == f"ERROR [{category}] {message}\n"
    assert str(sensitive_media_path) not in completed_process.stderr


def test_installed_command_processes_all_variable_timestamp_frames(
    tmp_path: Path,
) -> None:
    """Invoke cinebar and preserve all lossless fixture colors in order."""
    media_path = tmp_path / "fixture.mkv"
    output_path = tmp_path / "barcode.png"
    _write_lossless_fixture(media_path)

    with av.open(str(media_path)) as fixture:
        selected_stream = fixture.streams.video[0]
        assert selected_stream.frames == 0

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--height",
            "3",
            "--workers",
            "1",
            "--crop",
            "none",
        ]
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert _read_barcode(output_path) == (4, 3, RGB_COLORS)


def test_installed_command_applies_explicit_crop(tmp_path: Path) -> None:
    """Apply one exact-black explicit picture region to every lossless frame."""
    media_path = tmp_path / "explicit-crop-fixture.mkv"
    output_path = tmp_path / "explicit-crop-barcode.png"
    _write_explicit_crop_fixture(media_path)

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--height",
            "3",
            "--workers",
            "1",
            "--crop",
            "1,1,3,3",
        ]
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert _read_barcode(output_path) == (2, 3, [(60, 60, 60), (75, 30, 15)])


@pytest.mark.parametrize(
    "crop_arguments",
    [[], ["--crop", "auto"]],
    ids=["default", "explicit-auto"],
)
def test_installed_command_applies_automatic_crop(
    tmp_path: Path,
    crop_arguments: list[str],
) -> None:
    """Crop stable encoded borders for omitted and explicit automatic selection."""
    media_path = tmp_path / "automatic-crop-fixture.mkv"
    output_path = tmp_path / "automatic-crop-barcode.png"
    _write_automatic_crop_fixture(media_path)

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--height",
            "3",
            "--workers",
            "1",
            *crop_arguments,
        ]
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert _read_barcode(output_path) == (
        12,
        3,
        [
            (60, 30, 15),
            (75, 60, 45),
            (15, 75, 135),
            (191, 96, 48),
            (90, 150, 60),
            (33, 66, 99),
            (3, 6, 9),
            (120, 90, 30),
            (165, 15, 105),
            (24, 48, 72),
            (67, 112, 157),
            (180, 150, 120),
        ],
    )


def _write_lossless_fixture(media_path: Path) -> None:
    """Write a Matroska fixture with lossless bgr0 frames and uneven PTS values."""
    with av.open(str(media_path), mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width = 2
        stream.height = 2
        stream.pix_fmt = "bgr0"

        for presentation_timestamp, color in zip(
            PRESENTATION_TIMESTAMPS, RGB_COLORS, strict=True
        ):
            frame = av.VideoFrame.from_ndarray(
                np.full((2, 2, 3), color, dtype=np.uint8), format="rgb24"
            )
            frame.pts = presentation_timestamp
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def _write_explicit_crop_fixture(media_path: Path) -> None:
    """Write lossless frames with a black border and black picture content."""
    picture_colors = [(80, 80, 80), (100, 40, 20)]
    with av.open(str(media_path), mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width = 4
        stream.height = 4
        stream.pix_fmt = "bgr0"

        for presentation_timestamp, picture_color in enumerate(picture_colors):
            rgb_frame = np.zeros((4, 4, 3), dtype=np.uint8)
            rgb_frame[1:3, 1:3] = picture_color
            rgb_frame[1, 1] = (0, 0, 0)
            frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            frame.pts = presentation_timestamp
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def _write_automatic_crop_fixture(media_path: Path) -> None:
    """Write twelve lossless frames with stable encoded borders."""
    picture_colors = [
        (80, 40, 20),
        (100, 80, 60),
        (20, 100, 180),
        (255, 128, 64),
        (120, 200, 80),
        (44, 88, 132),
        (4, 8, 12),
        (160, 120, 40),
        (220, 20, 140),
        (32, 64, 96),
        (90, 150, 210),
        (240, 200, 160),
    ]
    with av.open(str(media_path), mode="w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=24)
        stream.width = 4
        stream.height = 4
        stream.pix_fmt = "bgr0"

        for presentation_timestamp, picture_color in enumerate(picture_colors):
            rgb_frame = np.zeros((4, 4, 3), dtype=np.uint8)
            rgb_frame[1:3, 1:3] = picture_color
            rgb_frame[1, 1] = (0, 0, 0)
            frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            frame.pts = presentation_timestamp
            frame.time_base = Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def _read_barcode(barcode_path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    """Read dimensions and first-row RGB columns from a lossless barcode PNG."""
    with barcode_path.open("rb") as barcode_file:
        width, height, rows, _ = png.Reader(file=barcode_file).read()
        pixel_rows = [list(row) for row in rows]

    first_row = pixel_rows[0]
    colors = [
        (first_row[index], first_row[index + 1], first_row[index + 2])
        for index in range(0, len(first_row), 3)
    ]
    assert pixel_rows == [first_row] * height
    return width, height, colors


@pytest.mark.parametrize(
    "crop_selection",
    ["1,1,1,2", "1,1,2", "1,one,2,3"],
)
def test_installed_command_reports_invalid_explicit_crop_without_output(
    tmp_path: Path,
    crop_selection: str,
) -> None:
    """Report empty and malformed crop input without creating a barcode."""
    media_path = tmp_path / "sensitive-source-name.mp4"
    output_path = tmp_path / "barcode.png"

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--crop",
            crop_selection,
        ]
    )

    _assert_installed_failure(
        completed_process,
        "invalid_crop",
        (
            "Explicit crop must use four non-negative integers as "
            "LEFT,TOP,RIGHT,BOTTOM, define a non-empty rectangle, and fit every "
            "decoded frame."
        ),
        media_path,
    )
    assert not output_path.exists()


def test_installed_command_reports_missing_input_without_source_path(
    tmp_path: Path,
) -> None:
    """Report a missing source through the safe installed-command failure boundary."""
    media_path = tmp_path / "sensitive-source-name.mp4"
    output_path = tmp_path / "barcode.png"

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--workers",
            "1",
            "--crop",
            "none",
        ]
    )

    _assert_installed_failure(
        completed_process,
        "input_not_found",
        "Input media was not found. Verify the source path and try again.",
        media_path,
    )
    assert not output_path.exists()


def test_installed_command_reports_non_regular_input_without_source_path(
    tmp_path: Path,
) -> None:
    """Report a directory source through the safe installed-command boundary."""
    media_path = tmp_path / "sensitive-source-directory"
    output_path = tmp_path / "barcode.png"
    media_path.mkdir()

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--workers",
            "1",
            "--crop",
            "none",
        ]
    )

    _assert_installed_failure(
        completed_process,
        "input_unreadable",
        (
            "Input media is not a readable regular file. Check its permissions and "
            "file type."
        ),
        media_path,
    )
    assert not output_path.exists()


def test_installed_command_reports_media_without_video_stream(
    tmp_path: Path,
) -> None:
    """Report readable non-video media without publishing an output."""
    media_path = tmp_path / "sensitive-source.txt"
    output_path = tmp_path / "barcode.png"
    media_path.write_text("not a video", encoding="utf-8")

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--workers",
            "1",
            "--crop",
            "none",
        ]
    )

    _assert_installed_failure(
        completed_process,
        "no_video_stream",
        "Input media has no usable video stream. Choose a supported video file.",
        media_path,
    )
    assert not output_path.exists()


def test_installed_command_reports_malformed_height_without_source_path(
    tmp_path: Path,
) -> None:
    """Route malformed heights through the safe logging boundary."""
    media_path = tmp_path / "sensitive-source-name.mp4"
    output_path = tmp_path / "barcode.png"

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--height",
            "not-an-integer",
        ]
    )

    _assert_installed_failure(
        completed_process,
        "invalid_height",
        "Height must be a positive integer.",
        media_path,
    )
    assert not output_path.exists()


def test_installed_command_reports_non_positive_height_without_source_path(
    tmp_path: Path,
) -> None:
    """Route non-positive heights through the safe logging boundary."""
    media_path = tmp_path / "sensitive-source-name.mp4"
    output_path = tmp_path / "barcode.png"

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--height",
            "0",
        ]
    )

    _assert_installed_failure(
        completed_process,
        "invalid_height",
        "Height must be a positive integer.",
        media_path,
    )
    assert not output_path.exists()


def test_installed_command_routes_parser_errors_through_logging(
    tmp_path: Path,
) -> None:
    """Report parser failures without echoing sensitive media paths."""
    media_path = tmp_path / "sensitive-source-name.mp4"

    completed_process = _run_cinebar([str(media_path)])

    _assert_installed_failure(
        completed_process,
        "invalid_argument",
        "Invalid command-line arguments. Run 'cinebar --help' for usage.",
        media_path,
    )


def test_installed_command_preserves_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    """Refuse to replace a final barcode unless callers request it."""
    media_path = tmp_path / "sensitive-source.mkv"
    output_path = tmp_path / "barcode.png"
    _write_lossless_fixture(media_path)
    output_path.write_bytes(b"previous barcode")

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--workers",
            "1",
            "--crop",
            "none",
        ]
    )

    _assert_installed_failure(
        completed_process,
        "output_exists",
        "Output already exists. Use --overwrite to replace it intentionally.",
        media_path,
    )
    assert output_path.read_bytes() == b"previous barcode"


def test_installed_command_replaces_existing_output_with_overwrite(
    tmp_path: Path,
) -> None:
    """Replace an existing output only when the installed command is explicit."""
    media_path = tmp_path / "fixture.mkv"
    output_path = tmp_path / "barcode.png"
    _write_lossless_fixture(media_path)
    output_path.write_bytes(b"previous barcode")

    completed_process = _run_cinebar(
        [
            str(media_path),
            "--output",
            str(output_path),
            "--height",
            "3",
            "--workers",
            "1",
            "--crop",
            "none",
            "--overwrite",
        ]
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert _read_barcode(output_path) == (4, 3, RGB_COLORS)
