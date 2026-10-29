"""Behavior tests for complete barcode processing."""

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast

import cv2
import numpy as np
import png
import pytest

from cinebar import (
    CinebarError,
    CropMode,
    CropRectangle,
    FailureCategory,
    PhaseProgress,
    ProcessingCompleted,
    ProgressPhase,
    ProgressState,
    ProgressUnit,
)
from cinebar.extract import FrameExtractionProgress
from cinebar.processor import SingleFileProcessor
from cinebar.progress import ProgressTracker


@pytest.mark.parametrize(
    ("category", "message"),
    [
        (
            FailureCategory.INVALID_ARGUMENT,
            "Invalid command-line arguments. Run 'cinebar --help' for usage.",
        ),
        (FailureCategory.INVALID_HEIGHT, "Height must be a positive integer."),
        (
            FailureCategory.INVALID_CROP,
            (
                "Explicit crop must use four non-negative integers as "
                "LEFT,TOP,RIGHT,BOTTOM, define a non-empty rectangle, and fit every "
                "decoded frame."
            ),
        ),
        (
            FailureCategory.NON_BLACK_CROP_BOUNDARY,
            (
                "Every applicable crop boundary must have at least 99% of pixels with "
                "every RGB channel at 16 or below in every frame. Adjust --crop or "
                "use --crop none."
            ),
        ),
        (
            FailureCategory.INPUT_NOT_FOUND,
            "Input media was not found. Verify the source path and try again.",
        ),
        (
            FailureCategory.INPUT_UNREADABLE,
            (
                "Input media is not a readable regular file. Check its permissions "
                "and file type."
            ),
        ),
        (
            FailureCategory.ROOT_NOT_FOUND,
            "Batch root was not found. Verify the directory path and try again.",
        ),
        (
            FailureCategory.ROOT_UNREADABLE,
            (
                "Batch root is not a readable directory. Check its permissions and "
                "file type."
            ),
        ),
        (
            FailureCategory.NO_VIDEO_STREAM,
            "Input media has no usable video stream. Choose a supported video file.",
        ),
        (
            FailureCategory.ZERO_FRAMES,
            (
                "The video stream decoded zero frames. Verify that the media is not "
                "empty or damaged."
            ),
        ),
        (
            FailureCategory.DECODE_FAILED,
            (
                "Video decoding failed before completion. Check that the media is not "
                "damaged; when using multiple workers, retry with --workers 1."
            ),
        ),
        (
            FailureCategory.CORRUPT_FRAME,
            (
                "The decoder returned a corrupt video frame. Verify or transcode the "
                "source media."
            ),
        ),
        (
            FailureCategory.OUTPUT_EXISTS,
            "Output already exists. Use --overwrite to replace it intentionally.",
        ),
        (
            FailureCategory.OUTPUT_WRITE_FAILED,
            (
                "The barcode could not be written. Check destination permissions and "
                "available space."
            ),
        ),
        (
            FailureCategory.OUTPUT_COLLISION,
            "Distinct input media files map to the same barcode output.",
        ),
        (
            FailureCategory.READINESS_TIMEOUT,
            "Input media did not become ready before the batch readiness timeout.",
        ),
        (
            FailureCategory.INTERRUPTED,
            "Processing was interrupted; no new barcode was published.",
        ),
    ],
)
def test_cinebar_error_is_picklable_and_path_safe(
    category: FailureCategory, message: str
) -> None:
    """Expose one stable picklable failure interface with safe fixed messages."""
    error = CinebarError(category)

    restored_error = pickle.loads(pickle.dumps(error))

    assert error.args == (category,)
    assert error.category is category
    assert str(error) == message
    assert restored_error.category is category
    assert str(restored_error) == message


def test_cinebar_error_preserves_safe_detail_through_pickling() -> None:
    """Retain explicitly safe details in error text and serialized arguments."""
    safe_detail = (
        "Destination: C:\\barcodes\\Movie.png\n"
        "Conflicting inputs:\n"
        "- C:\\media\\Movie.mkv"
    )
    error = CinebarError(FailureCategory.OUTPUT_COLLISION, safe_detail)

    restored_error = pickle.loads(pickle.dumps(error))

    assert error.args == (FailureCategory.OUTPUT_COLLISION, safe_detail)
    assert str(error) == (
        f"Distinct input media files map to the same barcode output.\n{safe_detail}"
    )
    assert restored_error.args == (FailureCategory.OUTPUT_COLLISION, safe_detail)
    assert restored_error.category is FailureCategory.OUTPUT_COLLISION
    assert str(restored_error) == str(error)


class _SequentialCapture:
    """Minimal opened capture returning predetermined sequential read results."""

    def __init__(
        self,
        read_results: list[tuple[bool, object | None] | BaseException],
    ) -> None:
        """Store sequential reads and initialize their current index."""
        self._read_results = tuple(read_results)
        self._next_read_index = 0

    def isOpened(self) -> bool:
        """Report that the synthetic capture opened."""
        return True

    def read(self) -> tuple[bool, object | None]:
        """Return the next result or the normal end-of-stream sentinel."""
        if self._next_read_index >= len(self._read_results):
            return False, None

        read_result = self._read_results[self._next_read_index]
        self._next_read_index += 1
        if isinstance(read_result, BaseException):
            raise read_result
        return read_result

    def release(self) -> None:
        """Release the synthetic capture."""


def test_process_defaults_to_one_automatic_crop_selection(
    tmp_path: Path,
) -> None:
    """Resolve automatic crop once and forward it to one complete extraction."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    selected_crop = CropRectangle(left=1, top=2, right=7, bottom=8)
    selector_calls: list[Path] = []
    extractor_calls: list[tuple[Path, int | None, CropRectangle | None]] = []

    def select_crop(
        media: Path, *, progress_callback: Callable[[int], None] | None
    ) -> CropRectangle:
        selector_calls.append(media)
        return selected_crop

    def extract_colors(
        media: Path,
        worker_count: int | None,
        crop: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        extractor_calls.append((media, worker_count, crop))
        return bytes((1, 2, 3))

    SingleFileProcessor(
        frame_color_extractor=extract_colors,
        automatic_crop_selector=select_crop,
    ).process(media_path, output_path, height=1, workers=3)

    assert selector_calls == [media_path]
    assert extractor_calls == [(media_path, 3, selected_crop)]


@pytest.mark.parametrize(
    ("requested_crop", "expected_crop"),
    [
        (CropMode.NONE, None),
        (
            CropRectangle(left=1, top=2, right=7, bottom=8),
            CropRectangle(left=1, top=2, right=7, bottom=8),
        ),
    ],
)
def test_process_skips_sampling_for_disabled_or_explicit_crop(
    tmp_path: Path,
    requested_crop: CropMode | CropRectangle,
    expected_crop: CropRectangle | None,
) -> None:
    """Pass disabled and explicit crop selections directly to complete extraction."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    selector_calls: list[Path] = []
    extractor_calls: list[CropRectangle | None] = []

    def select_crop(
        media: Path, *, progress_callback: Callable[[int], None] | None
    ) -> None:
        selector_calls.append(media)
        return None

    def extract_colors(
        _: Path,
        __: int | None,
        crop: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        extractor_calls.append(crop)
        return bytes((1, 2, 3))

    SingleFileProcessor(
        frame_color_extractor=extract_colors,
        automatic_crop_selector=select_crop,
    ).process(media_path, output_path, height=1, crop=requested_crop)

    assert selector_calls == []
    assert extractor_calls == [expected_crop]


def test_process_rejects_later_automatic_crop_boundary_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abort publication when later decoded content reaches an automatic boundary."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    first_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    later_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    later_frame[3, 1] = (17, 0, 0)
    capture = _SequentialCapture([(True, first_frame), (True, later_frame)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(
            automatic_crop_selector=lambda _, *, progress_callback: CropRectangle(
                1, 1, 3, 3
            )
        ).process(media_path, output_path, workers=1)

    assert raised_error.value.category is FailureCategory.NON_BLACK_CROP_BOUNDARY
    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_classifies_zero_decoded_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject an opened source whose sequential decoder yields no frames."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    capture = _SequentialCapture([])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor().process(
            media_path,
            output_path,
            workers=1,
            crop=CropMode.NONE,
        )

    assert raised_error.value.category is FailureCategory.ZERO_FRAMES
    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_classifies_decoder_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject an OpenCV decoder exception without publishing a barcode."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    capture = _SequentialCapture([cv2.error("decoder failure")])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor().process(
            media_path,
            output_path,
            workers=1,
            crop=CropMode.NONE,
        )

    assert raised_error.value.category is FailureCategory.DECODE_FAILED
    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_rejects_malformed_decoded_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a non-uint8 decoded frame without publishing a barcode."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    capture = _SequentialCapture([(True, np.ones((1, 1, 3), dtype=np.int16))])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor().process(
            media_path,
            output_path,
            workers=1,
            crop=CropMode.NONE,
        )

    assert raised_error.value.category is FailureCategory.CORRUPT_FRAME
    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_applies_letterbox_crop_with_black_picture_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish the exact mean of a valid letterboxed picture region."""
    media_path = tmp_path / "letterbox.mkv"
    output_path = tmp_path / "letterbox.png"
    media_path.write_bytes(b"placeholder")
    bgr_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr_frame[1:3] = (40, 80, 120)
    bgr_frame[1, 1] = (0, 0, 0)
    capture = _SequentialCapture([(True, bgr_frame)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    result = SingleFileProcessor().process(
        media_path,
        output_path,
        height=2,
        workers=1,
        crop=CropRectangle(left=0, top=1, right=4, bottom=3),
    )

    with output_path.open("rb") as barcode_file:
        width, height, rows, _ = png.Reader(file=barcode_file).read()
        actual_rows = [list(row) for row in rows]

    assert result.decoded_frame_count == 1
    assert width == 1
    assert height == 2
    assert actual_rows == [[105, 70, 35], [105, 70, 35]]


def test_process_applies_pillarbox_crop_with_black_picture_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publish the exact mean of a valid pillarboxed picture region."""
    media_path = tmp_path / "pillarbox.mkv"
    output_path = tmp_path / "pillarbox.png"
    media_path.write_bytes(b"placeholder")
    bgr_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr_frame[:, 1:3] = (80, 40, 20)
    bgr_frame[0, 1] = (0, 0, 0)
    capture = _SequentialCapture([(True, bgr_frame)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    result = SingleFileProcessor().process(
        media_path,
        output_path,
        height=2,
        workers=1,
        crop=CropRectangle(left=1, top=0, right=3, bottom=4),
    )

    with output_path.open("rb") as barcode_file:
        width, height, rows, _ = png.Reader(file=barcode_file).read()
        actual_rows = [list(row) for row in rows]

    assert result.decoded_frame_count == 1
    assert width == 1
    assert height == 2
    assert actual_rows == [[17, 35, 70], [17, 35, 70]]


def test_process_accepts_crop_when_later_frame_gains_a_black_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep one rectangle when a later wider frame begins excluding its right side."""
    media_path = tmp_path / "changing-dimensions.mkv"
    output_path = tmp_path / "changing-dimensions.png"
    media_path.write_bytes(b"placeholder")
    first_frame = np.zeros((2, 3, 3), dtype=np.uint8)
    first_frame[:, 1:3] = (40, 20, 10)
    later_frame = np.zeros((2, 4, 3), dtype=np.uint8)
    later_frame[:, 1:3] = (80, 40, 20)
    capture = _SequentialCapture([(True, first_frame), (True, later_frame)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    SingleFileProcessor().process(
        media_path,
        output_path,
        height=1,
        workers=1,
        crop=CropRectangle(left=1, top=0, right=3, bottom=2),
    )

    with output_path.open("rb") as barcode_file:
        width, height, rows, _ = png.Reader(file=barcode_file).read()
        actual_rows = [list(row) for row in rows]

    assert width == 2
    assert height == 1
    assert actual_rows == [[10, 20, 40, 20, 40, 80]]


def test_process_rejects_crop_out_of_bounds_on_later_frame_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a rectangle that a later decoded frame can no longer contain."""
    media_path = tmp_path / "smaller-frame.mkv"
    output_path = tmp_path / "smaller-frame.png"
    media_path.write_bytes(b"placeholder")
    capture = _SequentialCapture(
        [
            (True, np.zeros((3, 3, 3), dtype=np.uint8)),
            (True, np.zeros((2, 3, 3), dtype=np.uint8)),
        ]
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor().process(
            media_path,
            output_path,
            workers=1,
            crop=CropRectangle(left=1, top=1, right=3, bottom=3),
        )

    assert raised_error.value.category is FailureCategory.INVALID_CROP
    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_rejects_crop_boundary_below_near_black_coverage_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject an applicable crop boundary below near-black coverage."""
    media_path = tmp_path / "near-black-boundary.mkv"
    output_path = tmp_path / "near-black-boundary.png"
    media_path.write_bytes(b"placeholder")
    later_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    later_frame[3, 1] = (17, 0, 0)
    capture = _SequentialCapture(
        [(True, np.zeros((3, 3, 3), dtype=np.uint8)), (True, later_frame)]
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor().process(
            media_path,
            output_path,
            workers=1,
            crop=CropRectangle(left=1, top=1, right=3, bottom=3),
        )

    assert raised_error.value.category is FailureCategory.NON_BLACK_CROP_BOUNDARY
    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


@pytest.mark.parametrize("invalid_crop", [object(), "none"])
def test_process_rejects_runtime_crop_value_before_media_access(
    tmp_path: Path,
    invalid_crop: object,
) -> None:
    """Reject a non-selection crop before reading the source or creating output."""
    color_extractor_called = False

    def extract_colors(
        _: Path,
        __: int | None,
        ___: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        nonlocal color_extractor_called
        color_extractor_called = True
        return bytes((1, 2, 3))

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(extract_colors).process(
            tmp_path / "missing-source.mkv",
            tmp_path / "barcode.png",
            crop=cast(CropMode, invalid_crop),
        )

    assert raised_error.value.category is FailureCategory.INVALID_CROP
    assert color_extractor_called is False


@pytest.mark.parametrize("height", [0, -1])
def test_process_rejects_non_positive_height_before_media_access(
    tmp_path: Path, height: int
) -> None:
    """Reject invalid heights before accessing the source or calling the extractor."""
    color_extractor_called = False

    def extract_colors(
        _: Path,
        __: int | None,
        ___: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        nonlocal color_extractor_called
        color_extractor_called = True
        return b""

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(extract_colors).process(
            tmp_path / "missing-source.mkv",
            tmp_path / "barcode.png",
            height=height,
        )

    assert raised_error.value.category is FailureCategory.INVALID_HEIGHT
    assert color_extractor_called is False


@pytest.mark.parametrize("workers", [0, -1])
def test_process_rejects_non_positive_worker_count_before_media_access(
    tmp_path: Path, workers: int
) -> None:
    """Reject an explicit invalid worker count before accessing the source."""
    color_extractor_called = False

    def extract_colors(
        _: Path,
        __: int | None,
        ___: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        nonlocal color_extractor_called
        color_extractor_called = True
        return b""

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(extract_colors).process(
            tmp_path / "missing-source.mkv",
            tmp_path / "barcode.png",
            workers=workers,
        )

    assert raised_error.value.category is FailureCategory.INVALID_ARGUMENT
    assert color_extractor_called is False


def test_process_refuses_existing_output_before_decoding(tmp_path: Path) -> None:
    """Preserve an existing output unless callers intentionally request replacement."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    output_path.write_bytes(b"existing barcode")
    color_extractor_called = False

    def extract_colors(
        _: Path,
        __: int | None,
        ___: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        nonlocal color_extractor_called
        color_extractor_called = True
        return bytes((1, 2, 3))

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(extract_colors).process(media_path, output_path)

    assert raised_error.value.category is FailureCategory.OUTPUT_EXISTS
    assert output_path.read_bytes() == b"existing barcode"
    assert color_extractor_called is False


def test_process_rejects_missing_input_before_decoding(tmp_path: Path) -> None:
    """Classify a missing source without calling the decoder or publishing output."""
    output_path = tmp_path / "barcode.png"
    color_extractor_called = False

    def extract_colors(
        _: Path,
        __: int | None,
        ___: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        nonlocal color_extractor_called
        color_extractor_called = True
        return bytes((1, 2, 3))

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(extract_colors).process(
            tmp_path / "missing-source.mkv", output_path
        )

    assert raised_error.value.category is FailureCategory.INPUT_NOT_FOUND
    assert color_extractor_called is False
    assert not output_path.exists()


def test_process_rejects_non_regular_input_before_decoding(tmp_path: Path) -> None:
    """Classify a directory source without calling the decoder or publishing output."""
    output_path = tmp_path / "barcode.png"
    color_extractor_called = False

    def extract_colors(
        _: Path,
        __: int | None,
        ___: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        nonlocal color_extractor_called
        color_extractor_called = True
        return bytes((1, 2, 3))

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(extract_colors).process(tmp_path, output_path)

    assert raised_error.value.category is FailureCategory.INPUT_UNREADABLE
    assert color_extractor_called is False
    assert not output_path.exists()


def test_process_writes_ordered_colors_with_black_frame(tmp_path: Path) -> None:
    """Write exact lossless barcode rows from complete ordered frame colors."""
    compact_colors = bytes(
        (
            10,
            20,
            30,
            25,
            35,
            45,
            1,
            2,
            3,
            7,
            80,
            200,
            0,
            0,
            0,
        )
    )

    def extract_colors(
        _: Path,
        __: int | None,
        ___: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        return compact_colors

    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    result = SingleFileProcessor(extract_colors).process(
        media_path,
        output_path,
        height=2,
        workers=2,
        crop=CropMode.NONE,
    )

    with output_path.open("rb") as barcode_file:
        width, height, rows, _ = png.Reader(file=barcode_file).read()
        actual_rows = [list(row) for row in rows]

    expected_row = [
        10,
        20,
        30,
        25,
        35,
        45,
        1,
        2,
        3,
        7,
        80,
        200,
        0,
        0,
        0,
    ]
    assert result.output_path == output_path
    assert result.decoded_frame_count == 5
    assert result.output_width == 5
    assert result.output_height == 2
    assert width == 5
    assert height == 2
    assert actual_rows == [expected_row, expected_row]


def test_process_defaults_to_300_pixel_height(tmp_path: Path) -> None:
    """Publish a 300-pixel-tall PNG when height is omitted."""
    compact_colors = bytes((0, 0, 0))
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "default-height.png"
    media_path.write_bytes(b"placeholder")

    result = SingleFileProcessor(
        lambda _path, _workers, _crop, *, progress_callback: compact_colors
    ).process(
        media_path,
        output_path,
        crop=CropMode.NONE,
    )

    with output_path.open("rb") as barcode_file:
        width, height, rows, _ = png.Reader(file=barcode_file).read()
        actual_rows = [list(row) for row in rows]

    assert result.output_height == 300
    assert width == 1
    assert height == 300
    assert actual_rows == [[0, 0, 0]] * 300


def test_process_rejects_empty_decode_before_creating_output(tmp_path: Path) -> None:
    """Reject an empty frame source without publishing a barcode file."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(
            lambda _path, _workers, _crop, *, progress_callback: b""
        ).process(
            media_path,
            output_path,
            crop=CropMode.NONE,
        )

    assert raised_error.value.category is FailureCategory.ZERO_FRAMES
    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_rejects_incomplete_frame_colors_before_creating_output(
    tmp_path: Path,
) -> None:
    """Reject an incomplete extracted RGB sequence without publishing a barcode."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(
            lambda _path, _workers, _crop, *, progress_callback: bytes((1, 2))
        ).process(
            media_path,
            output_path,
            crop=CropMode.NONE,
        )

    assert raised_error.value.category is FailureCategory.CORRUPT_FRAME
    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_overwrite_replaces_output_with_complete_png(tmp_path: Path) -> None:
    """Replace a prior output only when callers explicitly request it."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    output_path.write_bytes(b"previous barcode")
    compact_colors = bytes((10, 20, 30, 40, 50, 60))

    SingleFileProcessor(
        lambda _path, _workers, _crop, *, progress_callback: compact_colors
    ).process(
        media_path,
        output_path,
        height=2,
        overwrite=True,
        crop=CropMode.NONE,
    )

    with output_path.open("rb") as barcode_file:
        width, height, rows, _ = png.Reader(file=barcode_file).read()
        actual_rows = [list(row) for row in rows]

    assert width == 2
    assert height == 2
    assert actual_rows == [[10, 20, 30, 40, 50, 60]] * 2
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_encodes_open_destination_temporary_before_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the existing final output untouched while encoding a temporary PNG."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    output_path.write_bytes(b"previous barcode")
    original_write = png.Writer.write
    observed_temporary_paths: list[Path] = []

    def observe_write(writer: object, output_file: BinaryIO, rows: object) -> None:
        temporary_path = Path(output_file.name)
        observed_temporary_paths.append(temporary_path)
        assert output_file.closed is False
        assert temporary_path.parent == output_path.parent
        assert temporary_path != output_path
        assert output_path.read_bytes() == b"previous barcode"
        original_write(writer, output_file, rows)

    monkeypatch.setattr(png.Writer, "write", observe_write)

    SingleFileProcessor(
        lambda _path, _workers, _crop, *, progress_callback: bytes((1, 2, 3))
    ).process(
        media_path,
        output_path,
        overwrite=True,
        crop=CropMode.NONE,
    )

    assert len(observed_temporary_paths) == 1
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_removes_temporary_after_partial_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve the final output when encoding writes only a partial PNG."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    output_path.write_bytes(b"previous barcode")

    def raise_partial_write(_: object, output_file: BinaryIO, __: object) -> None:
        output_file.write(b"partial PNG")
        raise OSError("write failure")

    monkeypatch.setattr(png.Writer, "write", raise_partial_write)

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(
            lambda _path, _workers, _crop, *, progress_callback: bytes((1, 2, 3))
        ).process(
            media_path,
            output_path,
            overwrite=True,
            crop=CropMode.NONE,
        )

    assert raised_error.value.category is FailureCategory.OUTPUT_WRITE_FAILED
    assert output_path.read_bytes() == b"previous barcode"
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_removes_temporary_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve the final output when encoding is interrupted."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    output_path.write_bytes(b"previous barcode")

    def raise_interruption(_: object, __: BinaryIO, ___: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(png.Writer, "write", raise_interruption)

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(
            lambda _path, _workers, _crop, *, progress_callback: bytes((1, 2, 3))
        ).process(
            media_path,
            output_path,
            overwrite=True,
            crop=CropMode.NONE,
        )

    assert raised_error.value.category is FailureCategory.INTERRUPTED
    assert output_path.read_bytes() == b"previous barcode"
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_refuses_output_created_while_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave a competing publisher's final output untouched before replacement."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    original_write = png.Writer.write

    def create_competing_output(
        writer: object, output_file: BinaryIO, rows: object
    ) -> None:
        original_write(writer, output_file, rows)
        output_path.write_bytes(b"competing barcode")

    monkeypatch.setattr(png.Writer, "write", create_competing_output)

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(
            lambda _path, _workers, _crop, *, progress_callback: bytes((1, 2, 3))
        ).process(
            media_path,
            output_path,
            crop=CropMode.NONE,
        )

    assert raised_error.value.category is FailureCategory.OUTPUT_EXISTS
    assert output_path.read_bytes() == b"competing barcode"
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_process_classifies_interruption_during_decoding(tmp_path: Path) -> None:
    """Classify decoding interruption before any temporary output is created."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")

    def interrupt_decoding(
        _: Path,
        __: int | None,
        ___: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        raise KeyboardInterrupt

    with pytest.raises(CinebarError) as raised_error:
        SingleFileProcessor(interrupt_decoding).process(
            media_path,
            output_path,
            crop=CropMode.NONE,
        )

    assert raised_error.value.category is FailureCategory.INTERRUPTED
    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_progress_tracker_emits_throttled_phase_facts_with_rate_and_eta() -> None:
    """Emit truthful phase snapshots and an unconditional completion."""
    now = 0.0
    events: list[PhaseProgress] = []
    tracker = ProgressTracker(events.append, clock=lambda: now)

    tracker.start(
        ProgressPhase.CROP_ANALYSIS,
        ProgressUnit.SAMPLES,
        total_units=12,
    )
    now = 0.05
    tracker.update(3)
    now = 1.0
    tracker.update(6, fraction_complete=0.5, estimate_eta=True)
    now = 2.0
    tracker.complete(12)

    assert events == [
        PhaseProgress(
            phase=ProgressPhase.CROP_ANALYSIS,
            state=ProgressState.STARTED,
            completed_units=0,
            total_units=12,
            unit=ProgressUnit.SAMPLES,
            fraction_complete=0.0,
            is_estimate=False,
            elapsed_seconds=0.0,
            rate_per_second=None,
            eta_seconds=None,
            worker_count=None,
        ),
        PhaseProgress(
            phase=ProgressPhase.CROP_ANALYSIS,
            state=ProgressState.RUNNING,
            completed_units=6,
            total_units=12,
            unit=ProgressUnit.SAMPLES,
            fraction_complete=0.5,
            is_estimate=False,
            elapsed_seconds=1.0,
            rate_per_second=6.0,
            eta_seconds=1.0,
            worker_count=None,
        ),
        PhaseProgress(
            phase=ProgressPhase.CROP_ANALYSIS,
            state=ProgressState.COMPLETED,
            completed_units=12,
            total_units=12,
            unit=ProgressUnit.SAMPLES,
            fraction_complete=1.0,
            is_estimate=False,
            elapsed_seconds=2.0,
            rate_per_second=6.0,
            eta_seconds=None,
            worker_count=None,
        ),
    ]


def test_process_reports_complete_progress_and_success_summary(tmp_path: Path) -> None:
    """Publish progress facts without coupling extraction to terminal rendering."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    selected_crop = CropRectangle(left=1, top=2, right=7, bottom=8)
    now = 0.0
    events: list[PhaseProgress | ProcessingCompleted] = []

    def clock() -> float:
        nonlocal now
        now += 0.1
        return now

    def select_crop(
        _: Path,
        *,
        progress_callback: Callable[[int], None] | None,
    ) -> CropRectangle:
        assert progress_callback is not None
        for sample_count in range(1, 13):
            progress_callback(sample_count)
        return selected_crop

    def extract_colors(
        _: Path,
        __: int | None,
        ___: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        assert progress_callback is not None
        progress_callback(
            FrameExtractionProgress(
                decoded_frame_count=0,
                expected_frame_count=None,
                worker_count=1,
                presentation_seconds=None,
                duration_seconds=4.0,
            )
        )
        progress_callback(
            FrameExtractionProgress(
                decoded_frame_count=1,
                expected_frame_count=None,
                worker_count=1,
                presentation_seconds=1.0,
                duration_seconds=4.0,
            )
        )
        progress_callback(
            FrameExtractionProgress(
                decoded_frame_count=3,
                expected_frame_count=None,
                worker_count=1,
                presentation_seconds=3.0,
                duration_seconds=4.0,
            )
        )
        return bytes((1, 2, 3, 4, 5, 6, 7, 8, 9))

    result = SingleFileProcessor(
        frame_color_extractor=extract_colors,
        automatic_crop_selector=select_crop,
        clock=clock,
    ).process(
        media_path,
        output_path,
        height=2,
        workers=1,
        progress_reporter=events.append,
    )

    phase_events = [event for event in events if isinstance(event, PhaseProgress)]
    completed_phases = [
        event.phase for event in phase_events if event.state is ProgressState.COMPLETED
    ]
    successful_completion = events[-1]

    assert result.output_width == 3
    assert completed_phases == [
        ProgressPhase.CROP_ANALYSIS,
        ProgressPhase.FRAME_DECODING,
        ProgressPhase.PNG_ENCODING,
        ProgressPhase.OUTPUT_PUBLICATION,
    ]
    completed_events = {
        event.phase: event
        for event in phase_events
        if event.state is ProgressState.COMPLETED
    }
    assert completed_events[ProgressPhase.CROP_ANALYSIS].completed_units == 12
    assert completed_events[ProgressPhase.CROP_ANALYSIS].total_units == 12
    assert completed_events[ProgressPhase.FRAME_DECODING].completed_units == 3
    assert completed_events[ProgressPhase.PNG_ENCODING].completed_units == 2
    assert completed_events[ProgressPhase.OUTPUT_PUBLICATION].completed_units == 1
    assert any(
        event.phase is ProgressPhase.FRAME_DECODING
        and event.eta_seconds is not None
        and event.is_estimate
        for event in phase_events
    )
    assert isinstance(successful_completion, ProcessingCompleted)
    assert successful_completion.result == result
    assert successful_completion.elapsed_seconds > 0
    assert successful_completion.average_frames_per_second is not None
    assert successful_completion.worker_count == 1
    assert successful_completion.accepted_crop == selected_crop


def test_process_omits_decode_eta_without_timestamp_progress(tmp_path: Path) -> None:
    """Keep sequential decoding indeterminate when timestamp facts are absent."""
    media_path = tmp_path / "source.mkv"
    output_path = tmp_path / "barcode.png"
    media_path.write_bytes(b"placeholder")
    now = 0.0
    events: list[PhaseProgress | ProcessingCompleted] = []

    def clock() -> float:
        nonlocal now
        now += 0.1
        return now

    def extract_colors(
        _: Path,
        __: int | None,
        ___: CropRectangle | None,
        *,
        progress_callback: Callable[[FrameExtractionProgress], None] | None,
    ) -> bytes:
        assert progress_callback is not None
        progress_callback(
            FrameExtractionProgress(
                decoded_frame_count=0,
                expected_frame_count=None,
                worker_count=1,
                presentation_seconds=None,
                duration_seconds=None,
            )
        )
        progress_callback(
            FrameExtractionProgress(
                decoded_frame_count=1,
                expected_frame_count=None,
                worker_count=1,
                presentation_seconds=None,
                duration_seconds=None,
            )
        )
        return bytes((1, 2, 3))

    SingleFileProcessor(extract_colors, clock=clock).process(
        media_path,
        output_path,
        height=1,
        crop=CropMode.NONE,
        progress_reporter=events.append,
    )

    assert all(
        event.eta_seconds is None
        for event in events
        if isinstance(event, PhaseProgress)
        and event.phase is ProgressPhase.FRAME_DECODING
    )


def test_phase_progress_rejects_mismatched_phase_units() -> None:
    """Prevent renderers from receiving semantically impossible phase facts."""
    with pytest.raises(ValueError, match="unit must match phase"):
        PhaseProgress(
            phase=ProgressPhase.CROP_ANALYSIS,
            state=ProgressState.STARTED,
            completed_units=0,
            total_units=12,
            unit=ProgressUnit.FRAMES,
            fraction_complete=0.0,
            is_estimate=False,
            elapsed_seconds=0.0,
            rate_per_second=None,
            eta_seconds=None,
            worker_count=None,
        )
