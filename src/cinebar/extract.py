"""Extract ordered frame colors from OpenCV-decoded video frames."""

import random
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from multiprocessing import Pool, Queue
from os import cpu_count
from pathlib import Path
from queue import Empty
from typing import Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .crop import (
    AUTOMATIC_CROP_SAMPLE_COUNT,
    CropRectangle,
    apply_crop,
    select_sampled_crop,
)
from .errors import CinebarError, FailureCategory
from .progress import UnitProgressCallback


@dataclass(frozen=True, slots=True)
class FrameExtractionProgress:
    """One immutable fact from complete frame-color extraction.

    Args:
        decoded_frame_count: Frames successfully decoded so far.
        expected_frame_count: Reported parallel total, when available.
        worker_count: Actual active decoder process count.
        presentation_seconds: Current sequential presentation position, if valid.
        duration_seconds: Sequential media-duration estimate, if valid.
    """

    decoded_frame_count: int
    expected_frame_count: int | None
    worker_count: int
    presentation_seconds: float | None
    duration_seconds: float | None


type ExtractionProgressCallback = Callable[[FrameExtractionProgress], None]
MAX_AUTO_WORKERS = 16


class _WorkerProgressQueue(Protocol):
    """Queue methods shared by multiprocessing and test progress transports."""

    def put(self, item: tuple[int, int]) -> None:
        """Publish one worker slot and decoded-frame count."""
        ...

    def get_nowait(self) -> tuple[int, int]:
        """Return one pending worker count without blocking."""
        ...

    def close(self) -> None:
        """Close the parent-side queue transport."""
        ...

    def join_thread(self) -> None:
        """Join the queue feeder thread after closing."""
        ...


_WORKER_PROGRESS_QUEUE: _WorkerProgressQueue | None = None


@dataclass(frozen=True, slots=True)
class FrameRange:
    """Inclusive decoded-frame indexes assigned to one worker.

    Args:
        start: First zero-based frame index.
        end: Last zero-based frame index.
    """

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ExtractedFrameRange:
    """Compact colors and boundary evidence returned by one worker.

    Args:
        frame_range: Inclusive source indexes represented by ``colors``.
        colors: Three packed RGB bytes per assigned frame.
        first_frame_digest: Digest of the first assigned raw BGR frame.
        following_frame_digest: Digest of the first frame after the assigned range.
    """

    frame_range: FrameRange
    colors: bytes
    first_frame_digest: bytes
    following_frame_digest: bytes | None


def partition_frame_ranges(frame_count: int, workers: int) -> list[FrameRange]:
    """Partition reported frame indexes into ordered non-empty ranges.

    Args:
        frame_count: Positive reported frame count.
        workers: Positive requested worker count.

    Returns:
        Contiguous ranges covering every index exactly once.

    Raises:
        ValueError: If ``frame_count`` or ``workers`` is not positive.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")

    active_workers = min(workers, frame_count)
    base_length, longer_range_count = divmod(frame_count, active_workers)
    frame_ranges: list[FrameRange] = []
    start = 0
    for worker_index in range(active_workers):
        length = base_length + (worker_index < longer_range_count)
        end = start + length - 1
        frame_ranges.append(FrameRange(start=start, end=end))
        start = end + 1
    return frame_ranges


def sample_automatic_crop(
    video_path: Path,
    progress_callback: UnitProgressCallback | None = None,
) -> CropRectangle | None:
    """Select a crop from twelve random decoded frame positions when possible.

    Args:
        video_path: Local media file to sample.
        progress_callback: Recipient of each successfully decoded probe count.

    Returns:
        One fixed picture-region rectangle, or ``None`` when sampling cannot
        provide unambiguous crop evidence.
    """
    try:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                return None
            frame_count = _usable_frame_count(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count is None or frame_count < AUTOMATIC_CROP_SAMPLE_COUNT:
                return None

            sampled_positions = sorted(
                random.sample(
                    range(frame_count),
                    k=AUTOMATIC_CROP_SAMPLE_COUNT,
                )
            )
            sampled_frames: list[NDArray[np.uint8]] = []
            for sampled_position in sampled_positions:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(sampled_position)):
                    return None
                _require_next_frame_position(capture, sampled_position)
                success, decoded_frame = capture.read()
                if not success:
                    return None
                sampled_frames.append(_validate_decoded_frame(decoded_frame))
                if progress_callback is not None:
                    progress_callback(len(sampled_frames))
                _require_next_frame_position(capture, sampled_position + 1)
            return select_sampled_crop(sampled_frames)
        finally:
            capture.release()
    except (CinebarError, cv2.error):
        return None


def extract_frame_colors(
    video_path: Path,
    workers: int | None = None,
    crop: CropRectangle | None = None,
    *,
    progress_callback: ExtractionProgressCallback | None = None,
) -> bytes:
    """Extract one ordered RGB frame color for every OpenCV-decoded frame.

    Args:
        video_path: Local media file to decode.
        workers: Process count, or ``None`` for a bounded automatic count.
        crop: Optional fixed picture region to validate and select per frame.
        progress_callback: Recipient of extraction facts from the parent process.

    Returns:
        Three packed RGB bytes per decoded frame in source index order.

    Raises:
        CinebarError: If decoding, crop validation, or parallel range validation
            fails.
        ValueError: If ``workers`` is not positive.
    """
    resolved_workers = (
        min(MAX_AUTO_WORKERS, cpu_count() or 1) if workers is None else workers
    )
    if resolved_workers <= 0:
        raise ValueError("workers must be positive.")
    if resolved_workers == 1:
        return _extract_frame_colors_sequentially(
            video_path,
            crop,
            progress_callback=progress_callback,
        )

    video_path_text = str(video_path)
    try:
        metadata_capture = cv2.VideoCapture(video_path_text)
        try:
            if not metadata_capture.isOpened():
                raise CinebarError(FailureCategory.NO_VIDEO_STREAM)
            reported_frame_count = metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            metadata_capture.release()
    except cv2.error:
        raise CinebarError(FailureCategory.DECODE_FAILED) from None

    frame_count = _usable_frame_count(reported_frame_count)
    if frame_count is None:
        raise CinebarError(FailureCategory.DECODE_FAILED)
    frame_ranges = partition_frame_ranges(frame_count, resolved_workers)
    active_worker_count = len(frame_ranges)
    if progress_callback is not None:
        progress_callback(
            FrameExtractionProgress(
                decoded_frame_count=0,
                expected_frame_count=frame_count,
                worker_count=active_worker_count,
                presentation_seconds=None,
                duration_seconds=None,
            )
        )

    progress_queue: _WorkerProgressQueue | None = (
        Queue() if progress_callback is not None else None
    )
    tasks = [
        (video_path_text, frame_range, frame_count, crop, worker_slot)
        for worker_slot, frame_range in enumerate(frame_ranges)
    ]
    try:
        with Pool(
            processes=active_worker_count,
            initializer=(
                _initialize_worker_progress if progress_queue is not None else None
            ),
            initargs=(progress_queue,) if progress_queue is not None else (),
        ) as pool:
            asynchronous_result = pool.starmap_async(extract_frame_range, tasks)
            latest_worker_counts: dict[int, int] = {}
            reported_decoded_frame_count = 0
            while not asynchronous_result.ready():
                reported_decoded_frame_count = _drain_worker_progress(
                    progress_queue,
                    latest_worker_counts,
                    reported_decoded_frame_count,
                    frame_count,
                    active_worker_count,
                    progress_callback,
                )
                asynchronous_result.wait(timeout=0.1)
            _drain_worker_progress(
                progress_queue,
                latest_worker_counts,
                reported_decoded_frame_count,
                frame_count,
                active_worker_count,
                progress_callback,
            )
            extracted_ranges: list[ExtractedFrameRange] = asynchronous_result.get()
    except cv2.error:
        raise CinebarError(FailureCategory.DECODE_FAILED) from None
    finally:
        if progress_queue is not None:
            progress_queue.close()
            progress_queue.join_thread()

    compact_colors = combine_extracted_frame_ranges(
        extracted_ranges,
        expected_ranges=frame_ranges,
        frame_count=frame_count,
    )
    if progress_callback is not None:
        progress_callback(
            FrameExtractionProgress(
                decoded_frame_count=len(compact_colors) // 3,
                expected_frame_count=len(compact_colors) // 3,
                worker_count=active_worker_count,
                presentation_seconds=None,
                duration_seconds=None,
            )
        )
    return compact_colors


def _extract_frame_colors_sequentially(
    video_path: Path,
    crop: CropRectangle | None,
    *,
    progress_callback: ExtractionProgressCallback | None,
) -> bytes:
    """Extract complete ordered colors without frame-count metadata or seeking."""
    colors = bytearray()
    duration_seconds: float | None = None
    presentation_progress_is_valid = False
    last_presentation_seconds: float | None = None
    try:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise CinebarError(FailureCategory.NO_VIDEO_STREAM)
            if progress_callback is not None:
                duration_seconds = _media_duration_seconds(capture)
                presentation_progress_is_valid = duration_seconds is not None
                progress_callback(
                    FrameExtractionProgress(
                        decoded_frame_count=0,
                        expected_frame_count=None,
                        worker_count=1,
                        presentation_seconds=None,
                        duration_seconds=duration_seconds,
                    )
                )
            while True:
                success, decoded_frame = capture.read()
                if not success:
                    break
                bgr_frame = _validate_decoded_frame(decoded_frame)
                picture_region = (
                    bgr_frame if crop is None else apply_crop(bgr_frame, crop)
                )
                colors.extend(opencv_mean_frame_color(picture_region))
                if progress_callback is None:
                    continue

                presentation_seconds: float | None = None
                if presentation_progress_is_valid:
                    candidate_presentation_seconds = _presentation_seconds(
                        capture.get(cv2.CAP_PROP_POS_MSEC)
                    )
                    if (
                        candidate_presentation_seconds is None
                        or duration_seconds is None
                        or candidate_presentation_seconds > duration_seconds
                        or (
                            last_presentation_seconds is not None
                            and candidate_presentation_seconds
                            < last_presentation_seconds
                        )
                    ):
                        presentation_progress_is_valid = False
                    else:
                        presentation_seconds = candidate_presentation_seconds
                        last_presentation_seconds = candidate_presentation_seconds
                progress_callback(
                    FrameExtractionProgress(
                        decoded_frame_count=len(colors) // 3,
                        expected_frame_count=None,
                        worker_count=1,
                        presentation_seconds=presentation_seconds,
                        duration_seconds=duration_seconds,
                    )
                )
        finally:
            capture.release()
    except cv2.error:
        raise CinebarError(FailureCategory.DECODE_FAILED) from None

    if not colors:
        raise CinebarError(FailureCategory.ZERO_FRAMES)
    if progress_callback is not None:
        progress_callback(
            FrameExtractionProgress(
                decoded_frame_count=len(colors) // 3,
                expected_frame_count=None,
                worker_count=1,
                presentation_seconds=(
                    last_presentation_seconds
                    if presentation_progress_is_valid
                    else None
                ),
                duration_seconds=duration_seconds,
            )
        )
    return bytes(colors)


def _media_duration_seconds(capture: cv2.VideoCapture) -> float | None:
    """Return a finite positive duration estimate from OpenCV metadata."""
    frame_count = _usable_frame_count(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = capture.get(cv2.CAP_PROP_FPS)
    if (
        frame_count is None
        or isinstance(frames_per_second, bool)
        or not isinstance(frames_per_second, (int, float))
        or not isfinite(frames_per_second)
        or frames_per_second <= 0
    ):
        return None
    return frame_count / frames_per_second


def _presentation_seconds(presentation_milliseconds: object) -> float | None:
    """Return one finite positive presentation position in seconds."""
    if (
        isinstance(presentation_milliseconds, bool)
        or not isinstance(presentation_milliseconds, (int, float))
        or not isfinite(presentation_milliseconds)
        or presentation_milliseconds <= 0
    ):
        return None
    return presentation_milliseconds / 1_000.0


def _initialize_worker_progress(progress_queue: _WorkerProgressQueue) -> None:
    """Configure one pool worker to publish decoded-frame counts."""
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue


def _publish_worker_progress(worker_slot: int, decoded_frame_count: int) -> None:
    """Publish one worker's latest decoded-frame count when progress is enabled."""
    if _WORKER_PROGRESS_QUEUE is not None:
        _WORKER_PROGRESS_QUEUE.put((worker_slot, decoded_frame_count))


def _drain_worker_progress(
    progress_queue: _WorkerProgressQueue | None,
    latest_worker_counts: dict[int, int],
    reported_decoded_frame_count: int,
    expected_frame_count: int,
    worker_count: int,
    progress_callback: ExtractionProgressCallback | None,
) -> int:
    """Aggregate queued worker counts and emit one strictly advancing parent fact."""
    if progress_queue is None or progress_callback is None:
        return reported_decoded_frame_count

    while True:
        try:
            worker_slot, decoded_frame_count = progress_queue.get_nowait()
        except Empty:
            break
        if decoded_frame_count > latest_worker_counts.get(worker_slot, 0):
            latest_worker_counts[worker_slot] = decoded_frame_count

    aggregated_decoded_frame_count = sum(latest_worker_counts.values())
    if aggregated_decoded_frame_count <= reported_decoded_frame_count:
        return reported_decoded_frame_count
    progress_callback(
        FrameExtractionProgress(
            decoded_frame_count=aggregated_decoded_frame_count,
            expected_frame_count=expected_frame_count,
            worker_count=worker_count,
            presentation_seconds=None,
            duration_seconds=None,
        )
    )
    return aggregated_decoded_frame_count


def extract_frame_range(
    video_path: str,
    frame_range: FrameRange,
    frame_count: int,
    crop: CropRectangle | None,
    worker_slot: int,
) -> ExtractedFrameRange:
    """Extract one validated inclusive OpenCV frame range.

    Args:
        video_path: Local media path.
        frame_range: Inclusive frame indexes assigned to this worker.
        frame_count: Positive reported frame count for the complete stream.
        crop: Optional fixed picture region to validate and select per frame.
        worker_slot: Zero-based active worker identifier for progress aggregation.

    Returns:
        Ordered compact RGB colors plus range-boundary evidence.

    Raises:
        CinebarError: If the decoder cannot open, seek, read, or crop the range.
    """
    if (
        frame_range.start < 0
        or frame_range.end < frame_range.start
        or frame_range.end >= frame_count
    ):
        raise CinebarError(FailureCategory.DECODE_FAILED)

    try:
        capture = cv2.VideoCapture(video_path)
        try:
            if not capture.isOpened():
                raise CinebarError(FailureCategory.DECODE_FAILED)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_range.start)):
                raise CinebarError(FailureCategory.DECODE_FAILED)
            _require_next_frame_position(capture, frame_range.start)

            range_length = frame_range.end - frame_range.start + 1
            colors = bytearray(range_length * 3)
            first_frame_digest = b""
            color_offset = 0
            for expected_index in range(frame_range.start, frame_range.end + 1):
                success, decoded_frame = capture.read()
                if not success:
                    raise CinebarError(FailureCategory.DECODE_FAILED)
                bgr_frame = _validate_decoded_frame(decoded_frame)
                _require_next_frame_position(capture, expected_index + 1)
                if expected_index == frame_range.start:
                    first_frame_digest = sha256(bgr_frame).digest()
                picture_region = (
                    bgr_frame if crop is None else apply_crop(bgr_frame, crop)
                )
                colors[color_offset : color_offset + 3] = opencv_mean_frame_color(
                    picture_region
                )
                color_offset += 3
                decoded_frame_count = expected_index - frame_range.start + 1
                if decoded_frame_count % 32 == 0:
                    _publish_worker_progress(worker_slot, decoded_frame_count)

            following_frame_digest: bytes | None = None
            if frame_range.end + 1 < frame_count:
                success, decoded_frame = capture.read()
                if not success:
                    raise CinebarError(FailureCategory.DECODE_FAILED)
                following_frame = _validate_decoded_frame(decoded_frame)
                _require_next_frame_position(capture, frame_range.end + 2)
                following_frame_digest = sha256(following_frame).digest()
            else:
                success, decoded_frame = capture.read()
                if success:
                    _validate_decoded_frame(decoded_frame)
                    raise CinebarError(FailureCategory.DECODE_FAILED)
            _publish_worker_progress(worker_slot, range_length)

            return ExtractedFrameRange(
                frame_range=frame_range,
                colors=bytes(colors),
                first_frame_digest=first_frame_digest,
                following_frame_digest=following_frame_digest,
            )
        finally:
            capture.release()
    except cv2.error:
        raise CinebarError(FailureCategory.DECODE_FAILED) from None


def combine_extracted_frame_ranges(
    extracted_ranges: list[ExtractedFrameRange],
    expected_ranges: list[FrameRange],
    frame_count: int,
) -> bytes:
    """Validate and combine ordered worker results.

    Args:
        extracted_ranges: Worker results in requested range order.
        expected_ranges: Contiguous ranges submitted to workers.
        frame_count: Expected complete decoded-frame count.

    Returns:
        Three packed RGB bytes for every expected frame.

    Raises:
        CinebarError: If coverage, ordering, or boundary evidence differs.
    """
    if len(extracted_ranges) != len(expected_ranges):
        raise CinebarError(FailureCategory.DECODE_FAILED)
    if (
        not expected_ranges
        or expected_ranges[0].start != 0
        or expected_ranges[-1].end != frame_count - 1
    ):
        raise CinebarError(FailureCategory.DECODE_FAILED)

    compact_colors = bytearray(frame_count * 3)
    color_offset = 0
    previous_result: ExtractedFrameRange | None = None
    for expected_range, extracted_range in zip(
        expected_ranges, extracted_ranges, strict=True
    ):
        if extracted_range.frame_range != expected_range:
            raise CinebarError(FailureCategory.DECODE_FAILED)
        expected_color_bytes = (expected_range.end - expected_range.start + 1) * 3
        if len(extracted_range.colors) != expected_color_bytes:
            raise CinebarError(FailureCategory.DECODE_FAILED)
        if expected_range.start > 0:
            if previous_result is None:
                raise CinebarError(FailureCategory.DECODE_FAILED)
            if previous_result.frame_range.end + 1 != expected_range.start:
                raise CinebarError(FailureCategory.DECODE_FAILED)
            if (
                previous_result.following_frame_digest
                != extracted_range.first_frame_digest
            ):
                raise CinebarError(FailureCategory.DECODE_FAILED)

        next_offset = color_offset + expected_color_bytes
        compact_colors[color_offset:next_offset] = extracted_range.colors
        color_offset = next_offset
        previous_result = extracted_range

    if previous_result is None or previous_result.following_frame_digest is not None:
        raise CinebarError(FailureCategory.DECODE_FAILED)
    return bytes(compact_colors)


def _validate_decoded_frame(decoded_frame: object) -> NDArray[np.uint8]:
    """Require OpenCV to return a contiguous non-empty 8-bit BGR frame."""
    if (
        not isinstance(decoded_frame, np.ndarray)
        or decoded_frame.dtype != np.uint8
        or decoded_frame.ndim != 3
        or decoded_frame.size == 0
        or decoded_frame.shape[2] != 3
        or not decoded_frame.flags.c_contiguous
    ):
        raise CinebarError(FailureCategory.CORRUPT_FRAME)
    return cast(NDArray[np.uint8], decoded_frame)


def _usable_frame_count(reported_frame_count: float) -> int | None:
    """Return one finite, positive integer OpenCV frame count when available."""
    if (
        not isfinite(reported_frame_count)
        or reported_frame_count <= 0
        or not reported_frame_count.is_integer()
    ):
        return None
    return int(reported_frame_count)


def _require_next_frame_position(
    capture: cv2.VideoCapture,
    expected_index: int,
) -> None:
    """Require OpenCV to report the expected next frame index."""
    actual_index = capture.get(cv2.CAP_PROP_POS_FRAMES)
    if actual_index != float(expected_index):
        raise CinebarError(FailureCategory.DECODE_FAILED)


def opencv_mean_frame_color(bgr_frame: NDArray[np.uint8]) -> bytes:
    """Compute one truncated RGB frame color from OpenCV BGR pixels.

    Args:
        bgr_frame: Packed 8-bit BGR pixels decoded by OpenCV.

    Returns:
        Three bytes containing the truncated arithmetic means in RGB order.
    """
    blue_mean, green_mean, red_mean, _ = cv2.mean(bgr_frame)
    return bytes((int(red_mean), int(green_mean), int(blue_mean)))
