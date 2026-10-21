"""Tests for explicit picture-region crop validation."""

from typing import cast

import numpy as np
import pytest

from cinebar import CinebarError, FailureCategory
from cinebar.crop import CropRectangle, apply_explicit_crop


@pytest.mark.parametrize(
    ("left", "top", "right", "bottom"),
    [
        (1, 1, 1, 2),
        (2, 1, 1, 2),
        (-1, 0, 1, 1),
        (True, 0, 1, 1),
        ("0", 0, 1, 1),
    ],
)
def test_crop_rectangle_rejects_invalid_coordinates(
    left: object,
    top: object,
    right: object,
    bottom: object,
) -> None:
    """Reject empty, inverted, negative, boolean, and non-integer rectangles."""
    with pytest.raises(CinebarError) as raised_error:
        CropRectangle(
            left=cast(int, left),
            top=cast(int, top),
            right=cast(int, right),
            bottom=cast(int, bottom),
        )

    assert raised_error.value.category is FailureCategory.INVALID_CROP


def test_apply_explicit_crop_returns_picture_region_view() -> None:
    """Select every picture pixel, including genuine black picture content."""
    bgr_frame = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [30, 20, 10], [0, 0, 0]],
            [[0, 0, 0], [60, 50, 40], [90, 80, 70], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )

    picture_region = apply_explicit_crop(
        bgr_frame,
        CropRectangle(left=1, top=1, right=3, bottom=3),
    )

    assert np.shares_memory(picture_region, bgr_frame)
    assert picture_region.tolist() == [
        [[0, 0, 0], [30, 20, 10]],
        [[60, 50, 40], [90, 80, 70]],
    ]


@pytest.mark.parametrize(
    "non_black_position",
    [(0, 1), (3, 1), (1, 0), (1, 3)],
)
def test_apply_explicit_crop_rejects_non_black_boundary(
    non_black_position: tuple[int, int],
) -> None:
    """Reject a non-black pixel on any applicable last-excluded boundary."""
    bgr_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr_frame[non_black_position] = (1, 0, 0)

    with pytest.raises(CinebarError) as raised_error:
        apply_explicit_crop(
            bgr_frame,
            CropRectangle(left=1, top=1, right=3, bottom=3),
        )

    assert raised_error.value.category is FailureCategory.NON_BLACK_CROP_BOUNDARY
