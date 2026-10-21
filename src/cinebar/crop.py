"""Validate explicit picture-region crop rectangles."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import CinebarError, FailureCategory


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


def apply_explicit_crop(
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
