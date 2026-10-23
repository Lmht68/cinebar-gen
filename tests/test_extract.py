"""Tests for parallel frame-color extraction helpers."""

import random
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from queue import Empty
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
    FrameExtractionProgress,
    FrameRange,
    combine_extracted_frame_ranges,
    extract_frame_colors,
    extract_frame_range,
    opencv_mean_frame_color,
    partition_frame_ranges,
    sample_automatic_crop,
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

    def __init__(
        self,
        frames: list[NDArray[np.uint8]],
        reported_frame_count: float | None = None,
        fps: float = 30.0,
        presentation_milliseconds: list[float] | None = None,
    ) -> None:
        """Store indexed frames, metadata, and capture-observation history.

        Args:
            frames: Ordered BGR frames returned by the synthetic source.
            reported_frame_count: Metadata count exposed to the extractor.
            fps: Metadata frame rate exposed to the extractor.
            presentation_milliseconds: Post-read presentation positions by frame.
        """
        self.frames = frames
        self.reported_frame_count = (
            float(len(frames)) if reported_frame_count is None else reported_frame_count
        )
        self.fps = fps
        self.presentation_milliseconds = (
            [1_000.0 * (index + 1) / fps for index in range(len(frames))]
            if presentation_milliseconds is None
            else presentation_milliseconds
        )
        self.position = 0
        self.seek_history: list[int] = []
        self.read_history: list[int] = []
        self.released = False

    def isOpened(self) -> bool:
        """Report that the synthetic capture opened."""
        return True

    def set(self, property_id: int, value: float) -> bool:
        """Seek to one synthetic frame index."""
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        self.seek_history.append(self.position)
        return True

    def get(self, property_id: int) -> float:
        """Return capture metadata or the next synthetic frame index."""
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.reported_frame_count
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_POS_MSEC:
            if self.position == 0:
                return 0.0
            return self.presentation_milliseconds[self.position - 1]
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        return float(self.position)

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        """Read the next synthetic frame or report end of input."""
        self.read_history.append(self.position)
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self) -> None:
        """Record capture release."""
        self.released = True


def test_sample_automatic_crop_reads_twelve_selected_frame_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seek and decode only twelve unique randomized positions in ascending order."""
    frames = []
    for index in range(16):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[1:3] = (index + 20, index + 30, index + 40)
        frames.append(frame)
    capture = FakeVideoCapture(frames)
    sampled_unit_counts: list[int] = []
    selected_positions = [12, 1, 15, 3, 8, 5, 13, 2, 10, 7, 14, 4]

    def choose_positions(population: range, k: int) -> list[int]:
        assert population == range(16)
        assert k == 12
        return selected_positions

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(random, "sample", choose_positions)

    crop = sample_automatic_crop(
        Path("movie.mp4"),
        progress_callback=sampled_unit_counts.append,
    )

    assert crop == CropRectangle(left=0, top=1, right=6, bottom=3)
    assert sampled_unit_counts == list(range(1, 13))
    assert capture.seek_history == sorted(selected_positions)
    assert capture.read_history == sorted(selected_positions)
    assert capture.released is True


@pytest.mark.parametrize(
    "reported_frame_count",
    [11.0, 0.0, -1.0, float("nan"), float("inf"), 12.5],
)
def test_sample_automatic_crop_rejects_unusable_frame_count_metadata(
    monkeypatch: pytest.MonkeyPatch,
    reported_frame_count: float,
) -> None:
    """Return no crop without reading when sampling metadata is unusable."""
    capture = FakeVideoCapture(
        [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(12)],
        reported_frame_count=reported_frame_count,
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    crop = sample_automatic_crop(Path("movie.mp4"))

    assert crop is None
    assert capture.seek_history == []
    assert capture.read_history == []
    assert capture.released is True


def test_sample_automatic_crop_rejects_an_unusable_sampled_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return no crop and release the probe after a corrupt sampled frame."""
    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(12)]
    frames[5] = np.zeros((0, 6, 3), dtype=np.uint8)
    capture = FakeVideoCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(
        random,
        "sample",
        lambda _population, k: list(range(k)),
    )

    crop = sample_automatic_crop(Path("movie.mp4"))

    assert crop is None
    assert capture.read_history == [0, 1, 2, 3, 4, 5]
    assert capture.released is True


def test_sample_automatic_crop_rejects_inconsistent_sample_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return no crop after all samples expose inconsistent decoded dimensions."""
    frames = [np.zeros((4, 6, 3), dtype=np.uint8) for _ in range(12)]
    frames[-1] = np.zeros((5, 6, 3), dtype=np.uint8)
    capture = FakeVideoCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(
        random,
        "sample",
        lambda _population, k: list(range(k)),
    )

    crop = sample_automatic_crop(Path("movie.mp4"))

    assert crop is None
    assert capture.read_history == list(range(12))
    assert capture.released is True


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
        crop=None,
        worker_slot=0,
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


class FakeQueue:
    """In-memory queue exposing the progress transport methods under test."""

    def __init__(self) -> None:
        """Create an empty synthetic worker-progress queue."""
        self._items: list[tuple[int, int]] = []

    def put(self, item: tuple[int, int]) -> None:
        """Append one worker-slot decoded-frame count."""
        self._items.append(item)

    def get_nowait(self) -> tuple[int, int]:
        """Return one pending count without blocking."""
        if not self._items:
            raise Empty
        return self._items.pop(0)

    def close(self) -> None:
        """Match the multiprocessing queue shutdown contract."""

    def join_thread(self) -> None:
        """Match the multiprocessing queue shutdown contract."""


class FakeAsyncResult:
    """Return inline pool results through the asynchronous-result interface."""

    def __init__(self, extracted_ranges: list[ExtractedFrameRange]) -> None:
        """Store completed inline worker results."""
        self._extracted_ranges = extracted_ranges

    def ready(self) -> bool:
        """Report that inline execution has completed."""
        return True

    def wait(self, timeout: float | None = None) -> None:
        """Match the asynchronous-result wait contract."""

    def get(self) -> list[ExtractedFrameRange]:
        """Return submitted worker results in task order."""
        return self._extracted_ranges


class FakePool:
    """Run process-pool tasks inline while retaining the pool interface."""

    def __init__(
        self,
        processes: int,
        initializer: Callable[..., None] | None = None,
        initargs: tuple[object, ...] = (),
    ) -> None:
        """Record the requested process count and initialize synthetic workers."""
        self.processes = processes
        if initializer is not None:
            initializer(*initargs)

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

    def starmap_async(
        self,
        function: Callable[
            [str, FrameRange, int, CropRectangle | None, int],
            ExtractedFrameRange,
        ],
        tasks: list[tuple[str, FrameRange, int, CropRectangle | None, int]],
    ) -> FakeAsyncResult:
        """Execute submitted extraction tasks in order."""
        return FakeAsyncResult([function(*task) for task in tasks])


def test_extract_frame_colors_validates_and_combines_worker_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Produce one ordered RGB color for every reported source frame."""
    frames = [
        np.full((1, 1, 3), (index + 2, index + 1, index), dtype=np.uint8)
        for index in range(4)
    ]
    captures: list[FakeVideoCapture] = []
    progress_facts: list[FrameExtractionProgress] = []

    def create_capture(_: str) -> FakeVideoCapture:
        capture = FakeVideoCapture(frames)
        captures.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", create_capture)
    monkeypatch.setattr(extract_module, "Pool", FakePool)
    monkeypatch.setattr(extract_module, "Queue", FakeQueue)

    colors = extract_frame_colors(
        Path("movie.mp4"),
        workers=2,
        progress_callback=progress_facts.append,
    )

    assert colors == bytes((0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5))
    assert progress_facts == [
        FrameExtractionProgress(0, 4, 2, None, None),
        FrameExtractionProgress(4, 4, 2, None, None),
        FrameExtractionProgress(4, 4, 2, None, None),
    ]
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


def test_extract_frame_colors_reports_sequential_timestamp_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Suppress later timestamp fractions after a regressing presentation position."""
    frames = [
        np.full((1, 1, 3), (index, index, index), dtype=np.uint8) for index in range(3)
    ]
    capture = FakeVideoCapture(
        frames,
        reported_frame_count=4.0,
        fps=4.0,
        presentation_milliseconds=[250.0, 500.0, 250.0],
    )
    progress_facts: list[FrameExtractionProgress] = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    colors = extract_frame_colors(
        Path("movie.mp4"),
        workers=1,
        progress_callback=progress_facts.append,
    )

    assert colors == bytes((0, 0, 0, 1, 1, 1, 2, 2, 2))
    assert progress_facts == [
        FrameExtractionProgress(0, None, 1, None, 1.0),
        FrameExtractionProgress(1, None, 1, 0.25, 1.0),
        FrameExtractionProgress(2, None, 1, 0.5, 1.0),
        FrameExtractionProgress(3, None, 1, None, 1.0),
        FrameExtractionProgress(3, None, 1, None, 1.0),
    ]


@pytest.mark.parametrize(
    ("fps", "presentation_milliseconds"),
    [
        (0.0, [250.0, 500.0, 750.0]),
        (4.0, [float("nan"), 500.0, 750.0]),
        (4.0, [1_250.0, 500.0, 750.0]),
    ],
    ids=["missing-duration", "non-finite-timestamp", "out-of-range-timestamp"],
)
def test_extract_frame_colors_suppresses_untrustworthy_timestamp_progress(
    monkeypatch: pytest.MonkeyPatch,
    fps: float,
    presentation_milliseconds: list[float],
) -> None:
    """Never recover a sequential fraction after timestamp prerequisites fail."""
    frames = [np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(3)]
    capture = FakeVideoCapture(
        frames,
        reported_frame_count=4.0,
        fps=fps,
        presentation_milliseconds=presentation_milliseconds,
    )
    progress_facts: list[FrameExtractionProgress] = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)

    extract_frame_colors(
        Path("movie.mp4"),
        workers=1,
        progress_callback=progress_facts.append,
    )

    assert all(fact.presentation_seconds is None for fact in progress_facts)
