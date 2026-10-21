"""Tests for parallel frame-color extraction helpers."""

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

import cinebar.extract as extract_module
from cinebar import CinebarError, FailureCategory
from cinebar.crop import CropRectangle
from cinebar.extract import (
    ExtractedFrameRange,
    FrameRange,
    combine_extracted_frame_ranges,
    extract_frame_colors,
    extract_frame_range,
    opencv_mean_frame_color,
    partition_frame_ranges,
)


def test_opencv_mean_frame_color_truncates_and_converts_bgr_to_rgb() -> None:
    """Return deterministic RGB bytes from OpenCV's fractional BGR means."""
    bgr_frame = np.array([[[30, 20, 10], [31, 21, 11]]], dtype=np.uint8)

    frame_color = opencv_mean_frame_color(bgr_frame)

    assert frame_color == bytes((10, 20, 30))
    assert cv2.mean(bgr_frame)[:3] == (30.5, 20.5, 10.5)


def test_partition_frame_ranges_covers_each_frame_index_once() -> None:
    """Create ordered contiguous ranges without empty worker tasks."""
    frame_ranges = partition_frame_ranges(frame_count=10, workers=3)

    assert frame_ranges == [
        FrameRange(start=0, end=3),
        FrameRange(start=4, end=6),
        FrameRange(start=7, end=9),
    ]


class FakeVideoCapture:
    """Minimal seekable OpenCV capture adapter for extraction tests."""

    def __init__(self, frames: list[NDArray[np.uint8]]) -> None:
        """Store indexed BGR frames."""
        self.frames = frames
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        """Report that the synthetic capture opened."""
        return True

    def set(self, property_id: int, value: float) -> bool:
        """Seek to one synthetic frame index."""
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        return True

    def get(self, property_id: int) -> float:
        """Return capture metadata or the next synthetic frame index."""
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        return float(self.position)

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        """Read the next synthetic frame or report end of input."""
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self) -> None:
        """Record capture release."""
        self.released = True


def test_extract_frame_range_returns_colors_and_boundary_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extract only the assigned range and inspect its adjacent frame."""
    frames = [
        np.full((1, 1, 3), (index + 2, index + 1, index), dtype=np.uint8)
        for index in range(4)
    ]
    capture = FakeVideoCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    extracted_range = extract_frame_range(
        "movie.mp4",
        FrameRange(start=1, end=2),
        frame_count=4,
    )

    assert extracted_range.colors == bytes((1, 2, 3, 2, 3, 4))
    assert extracted_range.first_frame_digest == sha256(frames[1]).digest()
    assert extracted_range.following_frame_digest == sha256(frames[3]).digest()
    assert capture.released is True


def test_combine_extracted_frame_ranges_preserves_order_and_coverage() -> None:
    """Join ordered colors only when adjacent raw-frame evidence agrees."""
    first_range = FrameRange(start=0, end=1)
    second_range = FrameRange(start=2, end=3)
    extracted_ranges = [
        ExtractedFrameRange(
            frame_range=first_range,
            colors=bytes((1, 2, 3, 4, 5, 6)),
            first_frame_digest=b"first",
            following_frame_digest=b"boundary",
        ),
        ExtractedFrameRange(
            frame_range=second_range,
            colors=bytes((7, 8, 9, 10, 11, 12)),
            first_frame_digest=b"boundary",
            following_frame_digest=None,
        ),
    ]

    colors = combine_extracted_frame_ranges(
        extracted_ranges,
        expected_ranges=[first_range, second_range],
        frame_count=4,
    )

    assert colors == bytes(range(1, 13))


class FakePool:
    """Run process-pool tasks inline while retaining the pool interface."""

    def __init__(self, processes: int) -> None:
        """Record the requested process count."""
        self.processes = processes

    def __enter__(self) -> Self:
        """Enter the inline pool context."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave the inline pool context."""

    def starmap(
        self,
        function: Callable[
            [str, FrameRange, int, CropRectangle | None],
            ExtractedFrameRange,
        ],
        tasks: list[tuple[str, FrameRange, int, CropRectangle | None]],
    ) -> list[ExtractedFrameRange]:
        """Execute submitted extraction tasks in order."""
        return [function(*task) for task in tasks]


def test_extract_frame_colors_validates_and_combines_worker_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Produce one ordered RGB color for every reported source frame."""
    frames = [
        np.full((1, 1, 3), (index + 2, index + 1, index), dtype=np.uint8)
        for index in range(4)
    ]
    captures: list[FakeVideoCapture] = []

    def create_capture(_: str) -> FakeVideoCapture:
        capture = FakeVideoCapture(frames)
        captures.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", create_capture)
    monkeypatch.setattr(extract_module, "Pool", FakePool)

    colors = extract_frame_colors(Path("movie.mp4"), workers=2)

    assert colors == bytes((0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5))
    assert len(captures) == 3
    assert all(capture.released for capture in captures)


def test_extract_frame_colors_applies_crop_to_every_worker_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reduce each assigned frame's fixed picture region in compact RGB order."""
    rectangle = CropRectangle(left=1, top=1, right=3, bottom=3)
    frames = []
    for picture_bgr in [
        (36, 24, 12),
        (60, 40, 20),
        (84, 56, 28),
        (108, 72, 36),
    ]:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[1:3, 1:3] = picture_bgr
        frame[1, 1] = (0, 0, 0)
        frames.append(frame)

    captures: list[FakeVideoCapture] = []

    def create_capture(_: str) -> FakeVideoCapture:
        capture = FakeVideoCapture(frames)
        captures.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", create_capture)
    monkeypatch.setattr(extract_module, "Pool", FakePool)

    colors = extract_frame_colors(Path("movie.mp4"), workers=2, crop=rectangle)

    assert colors == bytes((9, 18, 27, 15, 30, 45, 21, 42, 63, 27, 54, 81))
    assert len(captures) == 3
    assert all(capture.released for capture in captures)


def test_combine_extracted_frame_ranges_rejects_boundary_mismatch() -> None:
    """Fail rather than publish colors from ranges that sought to different frames."""
    first_range = FrameRange(start=0, end=0)
    second_range = FrameRange(start=1, end=1)
    extracted_ranges = [
        ExtractedFrameRange(
            frame_range=first_range,
            colors=bytes((1, 2, 3)),
            first_frame_digest=b"first",
            following_frame_digest=b"expected-boundary",
        ),
        ExtractedFrameRange(
            frame_range=second_range,
            colors=bytes((4, 5, 6)),
            first_frame_digest=b"different-frame",
            following_frame_digest=None,
        ),
    ]

    with pytest.raises(CinebarError) as raised_error:
        combine_extracted_frame_ranges(
            extracted_ranges,
            expected_ranges=[first_range, second_range],
            frame_count=2,
        )

    assert raised_error.value.category is FailureCategory.DECODE_FAILED
