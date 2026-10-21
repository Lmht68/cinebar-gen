"""Select and validate fixed decoded-frame picture-region crops."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import ceil

import numpy as np
from numpy.typing import NDArray

from .errors import CinebarError, FailureCategory

AUTOMATIC_CROP_SAMPLE_COUNT = 12
NEAR_BLACK_CHANNEL_MAX = 16
NEAR_BLACK_MIN_COVERAGE = 0.99


class CropMode(StrEnum):
    """Select automatic or disabled picture-region cropping."""

    AUTO = "auto"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class CropRectangle:
    """One fixed inclusive-left/top and exclusive-right/bottom picture region.

    Args:
        left: Inclusive zero-based horizontal coordinate.
        top: Inclusive zero-based vertical coordinate.
        right: Exclusive horizontal coordinate.
        bottom: Exclusive vertical coordinate.

    Raises:
        CinebarError: If a coordinate is malformed, negative, or defines an empty
            rectangle.
    """

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        """Reject malformed, negative, empty, and inverted coordinates."""
        coordinates = (self.left, self.top, self.right, self.bottom)
        if any(
            isinstance(coordinate, bool) or not isinstance(coordinate, int)
            for coordinate in coordinates
        ):
            raise CinebarError(FailureCategory.INVALID_CROP)
        if self.left < 0 or self.top < 0:
            raise CinebarError(FailureCategory.INVALID_CROP)
        if self.left >= self.right or self.top >= self.bottom:
            raise CinebarError(FailureCategory.INVALID_CROP)


type CropSelection = CropMode | CropRectangle


def select_sampled_crop(
    sampled_frames: Sequence[NDArray[np.uint8]],
) -> CropRectangle | None:
    """Select one fixed crop from twelve sampled decoded frames.

    Each accepted edge has a unanimous near-black run length and an exact-black
    crop boundary in every sample.

    Args:
        sampled_frames: Twelve validated packed 8-bit BGR frames.

    Returns:
        One fixed picture-region rectangle, or ``None`` when the sampled border
        evidence is insufficient, contradictory, or ambiguous.
    """
    if len(sampled_frames) != AUTOMATIC_CROP_SAMPLE_COUNT:
        return None

    frame_height, frame_width = sampled_frames[0].shape[:2]
    if any(frame.shape[:2] != (frame_height, frame_width) for frame in sampled_frames):
        return None

    top_candidates: list[int | None] = []
    bottom_candidates: list[int | None] = []
    left_candidates: list[int | None] = []
    right_candidates: list[int | None] = []
    minimum_black_pixels_per_row = ceil(frame_width * NEAR_BLACK_MIN_COVERAGE)
    minimum_black_pixels_per_column = ceil(frame_height * NEAR_BLACK_MIN_COVERAGE)
    for sampled_frame in sampled_frames:
        near_black_pixels = np.all(
            sampled_frame <= NEAR_BLACK_CHANNEL_MAX,
            axis=2,
        )
        qualifying_rows = (
            np.count_nonzero(near_black_pixels, axis=1) >= minimum_black_pixels_per_row
        )
        qualifying_columns = (
            np.count_nonzero(near_black_pixels, axis=0)
            >= minimum_black_pixels_per_column
        )
        top_run = _leading_qualifying_run_length(qualifying_rows)
        bottom_run = _trailing_qualifying_run_length(qualifying_rows)
        left_run = _leading_qualifying_run_length(qualifying_columns)
        right_run = _trailing_qualifying_run_length(qualifying_columns)
        top_candidates.append(top_run if top_run < frame_height else None)
        bottom_candidates.append(
            frame_height - bottom_run if bottom_run < frame_height else None
        )
        left_candidates.append(left_run if left_run < frame_width else None)
        right_candidates.append(
            frame_width - right_run if right_run < frame_width else None
        )

    top = _accept_sampled_edge(
        sampled_frames,
        top_candidates,
        uncropped_coordinate=0,
        axis=0,
        boundary_offset=-1,
    )
    bottom = _accept_sampled_edge(
        sampled_frames,
        bottom_candidates,
        uncropped_coordinate=frame_height,
        axis=0,
        boundary_offset=0,
    )
    left = _accept_sampled_edge(
        sampled_frames,
        left_candidates,
        uncropped_coordinate=0,
        axis=1,
        boundary_offset=-1,
    )
    right = _accept_sampled_edge(
        sampled_frames,
        right_candidates,
        uncropped_coordinate=frame_width,
        axis=1,
        boundary_offset=0,
    )

    resolved_top = 0 if top is None else top
    resolved_bottom = frame_height if bottom is None else bottom
    resolved_left = 0 if left is None else left
    resolved_right = frame_width if right is None else right
    if (
        (top is None and bottom is None and left is None and right is None)
        or resolved_top >= resolved_bottom
        or resolved_left >= resolved_right
    ):
        return None
    return CropRectangle(
        left=resolved_left,
        top=resolved_top,
        right=resolved_right,
        bottom=resolved_bottom,
    )


def _leading_qualifying_run_length(qualifying_lines: NDArray[np.bool_]) -> int:
    run_length = 0
    for line_qualifies in qualifying_lines:
        if not line_qualifies:
            return run_length
        run_length += 1
    return run_length


def _trailing_qualifying_run_length(qualifying_lines: NDArray[np.bool_]) -> int:
    run_length = 0
    for line_index in range(len(qualifying_lines) - 1, -1, -1):
        if not qualifying_lines[line_index]:
            return run_length
        run_length += 1
    return run_length


def _accept_sampled_edge(
    sampled_frames: Sequence[NDArray[np.uint8]],
    candidates: Sequence[int | None],
    *,
    uncropped_coordinate: int,
    axis: int,
    boundary_offset: int,
) -> int | None:
    candidate: int | None = None
    for other_candidate in candidates:
        if other_candidate is None:
            continue
        if candidate is None:
            candidate = other_candidate
        elif other_candidate != candidate:
            return None

    if candidate is None or candidate == uncropped_coordinate:
        return None

    boundary_coordinate = candidate + boundary_offset
    for sampled_frame in sampled_frames:
        boundary = (
            sampled_frame[boundary_coordinate]
            if axis == 0
            else sampled_frame[:, boundary_coordinate]
        )
        if np.any(boundary):
            return None
    return candidate


def apply_crop(
    bgr_frame: NDArray[np.uint8],
    rectangle: CropRectangle,
) -> NDArray[np.uint8]:
    """Validate crop boundaries and select the fixed picture region without copying.

    Args:
        bgr_frame: Validated packed 8-bit BGR frame decoded by OpenCV.
        rectangle: Fixed inclusive-left/top and exclusive-right/bottom bounds.

    Returns:
        A view of the accepted picture region.

    Raises:
        CinebarError: If the rectangle does not fit or a boundary is non-black.
    """
    frame_height, frame_width = bgr_frame.shape[:2]
    if rectangle.right > frame_width or rectangle.bottom > frame_height:
        raise CinebarError(FailureCategory.INVALID_CROP)

    if rectangle.top > 0 and np.any(bgr_frame[rectangle.top - 1]):
        raise CinebarError(FailureCategory.NON_BLACK_CROP_BOUNDARY)
    if rectangle.bottom < frame_height and np.any(bgr_frame[rectangle.bottom]):
        raise CinebarError(FailureCategory.NON_BLACK_CROP_BOUNDARY)
    if rectangle.left > 0 and np.any(bgr_frame[:, rectangle.left - 1]):
        raise CinebarError(FailureCategory.NON_BLACK_CROP_BOUNDARY)
    if rectangle.right < frame_width and np.any(bgr_frame[:, rectangle.right]):
        raise CinebarError(FailureCategory.NON_BLACK_CROP_BOUNDARY)

    return bgr_frame[rectangle.top : rectangle.bottom, rectangle.left : rectangle.right]
