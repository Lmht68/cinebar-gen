"""Extract ordered frame colors from OpenCV-decoded video frames."""

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

MAX_AUTO_WORKERS = 16


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


class FrameExtractionError(RuntimeError):
    """Report unsafe or incomplete frame extraction."""


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


def extract_frame_colors(video_path: Path, workers: int | None = None) -> bytes:
    """Extract one ordered RGB frame color for every OpenCV-decoded frame.

    Args:
        video_path: Local media file to decode.
        workers: Process count, or ``None`` for a bounded automatic count.

    Returns:
        Three packed RGB bytes per decoded frame in source index order.

    Raises:
        FrameExtractionError: If decoding or parallel range validation fails.
        ValueError: If ``workers`` is not positive.
    """
    resolved_workers = (
        min(MAX_AUTO_WORKERS, cpu_count() or 1) if workers is None else workers
    )
    if resolved_workers <= 0:
        raise ValueError("workers must be positive.")
    if resolved_workers == 1:
        return _extract_frame_colors_sequentially(video_path)

    video_path_text = str(video_path)
    metadata_capture = cv2.VideoCapture(video_path_text)
    try:
        if not metadata_capture.isOpened():
            raise FrameExtractionError(f"Unable to open video: {video_path}")
        reported_frame_count = metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        metadata_capture.release()

    if (
        not isfinite(reported_frame_count)
        or reported_frame_count <= 0
        or not reported_frame_count.is_integer()
    ):
        raise FrameExtractionError(
            "Parallel processing requires a positive integer OpenCV frame count. "
            "Choose --workers 1 to decode sequentially."
        )

    frame_count = int(reported_frame_count)
    frame_ranges = partition_frame_ranges(frame_count, resolved_workers)
    tasks = [
        (video_path_text, frame_range, frame_count) for frame_range in frame_ranges
    ]
    with Pool(processes=len(frame_ranges)) as pool:
        extracted_ranges: list[ExtractedFrameRange] = pool.starmap(
            extract_frame_range,
            tasks,
        )
    return combine_extracted_frame_ranges(
        extracted_ranges,
        expected_ranges=frame_ranges,
        frame_count=frame_count,
    )


def _extract_frame_colors_sequentially(video_path: Path) -> bytes:
    """Extract complete ordered colors without frame-count metadata or seeking."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise FrameExtractionError(f"Unable to open video: {video_path}")
        colors = bytearray()
        while True:
            success, bgr_frame = capture.read()
            if not success:
                break
            if bgr_frame is None:
                raise FrameExtractionError("Decoder returned an empty video frame.")
            colors.extend(opencv_mean_frame_color(cast(NDArray[np.uint8], bgr_frame)))
    finally:
        capture.release()

    if not colors:
        raise FrameExtractionError("No video frames were decoded.")
    return bytes(colors)


def extract_frame_range(
    video_path: str,
    frame_range: FrameRange,
    frame_count: int,
) -> ExtractedFrameRange:
    """Extract one validated inclusive OpenCV frame range.

    Args:
        video_path: Local media path.
        frame_range: Inclusive frame indexes assigned to this worker.
        frame_count: Positive reported frame count for the complete stream.

    Returns:
        Ordered compact RGB colors plus range-boundary evidence.

    Raises:
        FrameExtractionError: If the decoder cannot open, seek, or read the range.
        ValueError: If the assigned range is outside ``frame_count``.
    """
    if (
        frame_range.start < 0
        or frame_range.end < frame_range.start
        or frame_range.end >= frame_count
    ):
        raise ValueError("frame_range must be within the reported frame count.")

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise FrameExtractionError(f"Unable to open video: {video_path}")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_range.start)):
            raise FrameExtractionError(f"Unable to seek to frame {frame_range.start}.")
        _require_next_frame_position(capture, frame_range.start)

        range_length = frame_range.end - frame_range.start + 1
        colors = bytearray(range_length * 3)
        first_frame_digest = b""
        color_offset = 0
        for expected_index in range(frame_range.start, frame_range.end + 1):
            success, bgr_frame = capture.read()
            if not success or bgr_frame is None:
                raise FrameExtractionError(
                    f"Decoder stopped before frame {expected_index}."
                )
            bgr_frame = cast(NDArray[np.uint8], bgr_frame)
            _require_next_frame_position(capture, expected_index + 1)
            if expected_index == frame_range.start:
                first_frame_digest = sha256(bgr_frame).digest()
            colors[color_offset : color_offset + 3] = opencv_mean_frame_color(bgr_frame)
            color_offset += 3

        following_frame_digest: bytes | None = None
        if frame_range.end + 1 < frame_count:
            success, following_frame = capture.read()
            if not success or following_frame is None:
                raise FrameExtractionError(
                    f"Decoder stopped before overlap frame {frame_range.end + 1}."
                )
            _require_next_frame_position(capture, frame_range.end + 2)
            following_frame_digest = sha256(following_frame).digest()
        else:
            success, _ = capture.read()
            if success:
                raise FrameExtractionError(
                    "Reported frame count excludes decodable trailing frames."
                )

        return ExtractedFrameRange(
            frame_range=frame_range,
            colors=bytes(colors),
            first_frame_digest=first_frame_digest,
            following_frame_digest=following_frame_digest,
        )
    finally:
        capture.release()


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
        FrameExtractionError: If coverage, ordering, or boundary evidence differs.
    """
    if len(extracted_ranges) != len(expected_ranges):
        raise FrameExtractionError(
            "Worker result count does not match submitted ranges."
        )
    if (
        not expected_ranges
        or expected_ranges[0].start != 0
        or expected_ranges[-1].end != frame_count - 1
    ):
        raise FrameExtractionError("Submitted ranges do not cover the reported video.")

    compact_colors = bytearray(frame_count * 3)
    color_offset = 0
    previous_result: ExtractedFrameRange | None = None
    for expected_range, extracted_range in zip(
        expected_ranges, extracted_ranges, strict=True
    ):
        if extracted_range.frame_range != expected_range:
            raise FrameExtractionError(
                "Worker result order does not match frame ranges."
            )
        expected_color_bytes = (expected_range.end - expected_range.start + 1) * 3
        if len(extracted_range.colors) != expected_color_bytes:
            raise FrameExtractionError(
                f"Frame range {expected_range.start}-{expected_range.end} "
                "returned an incomplete color sequence."
            )
        if expected_range.start > 0:
            if previous_result is None:
                raise FrameExtractionError(
                    "Worker results omit a preceding contiguous frame range."
                )
            if previous_result.frame_range.end + 1 != expected_range.start:
                raise FrameExtractionError("Submitted frame ranges are not contiguous.")
            if (
                previous_result.following_frame_digest
                != extracted_range.first_frame_digest
            ):
                raise FrameExtractionError(
                    f"Decoder seek validation failed at frame {expected_range.start}."
                )

        next_offset = color_offset + expected_color_bytes
        compact_colors[color_offset:next_offset] = extracted_range.colors
        color_offset = next_offset
        previous_result = extracted_range

    if previous_result is None or previous_result.following_frame_digest is not None:
        raise FrameExtractionError(
            "Final worker did not validate the reported video end."
        )
    return bytes(compact_colors)


def _require_next_frame_position(
    capture: cv2.VideoCapture,
    expected_index: int,
) -> None:
    """Require OpenCV to report the expected next frame index."""
    actual_index = capture.get(cv2.CAP_PROP_POS_FRAMES)
    if actual_index != float(expected_index):
        raise FrameExtractionError(
            f"Decoder reported frame position {actual_index}, "
            f"expected {expected_index}."
        )


def opencv_mean_frame_color(bgr_frame: NDArray[np.uint8]) -> bytes:
    """Compute one truncated RGB frame color from OpenCV BGR pixels.

    Args:
        bgr_frame: Packed 8-bit BGR pixels decoded by OpenCV.

    Returns:
        Three bytes containing the truncated arithmetic means in RGB order.
    """
    blue_mean, green_mean, red_mean, _ = cv2.mean(bgr_frame)
    return bytes((int(red_mean), int(green_mean), int(blue_mean)))
