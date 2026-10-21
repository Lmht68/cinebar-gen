"""Tests for explicit picture-region crop validation."""

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from cinebar import CinebarError, CropMode, FailureCategory
from cinebar.crop import CropRectangle, apply_crop, select_sampled_crop


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


def test_crop_mode_exposes_automatic_and_no_crop_selections() -> None:
    """Expose the caller-visible automatic and explicit no-crop selections."""
    assert CropMode.AUTO.value == "auto"
    assert CropMode.NONE.value == "none"


def _letterboxed_frame(picture_color: tuple[int, int, int]) -> NDArray[np.uint8]:
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    frame[1:4] = picture_color
    frame[2, 3] = (0, 0, 0)
    return frame


def test_select_sampled_crop_accepts_a_stable_letterbox() -> None:
    """Select a fixed picture region from twelve stable encoded borders."""
    sampled_frames = [
        _letterboxed_frame((index + 20, index + 30, index + 40)) for index in range(12)
    ]

    crop = select_sampled_crop(sampled_frames)

    assert crop == CropRectangle(left=0, top=1, right=8, bottom=4)


def _pillarboxed_frame(picture_color: tuple[int, int, int]) -> NDArray[np.uint8]:
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    frame[:, 2:7] = picture_color
    frame[3, 3] = (0, 0, 0)
    return frame


def test_select_sampled_crop_accepts_a_stable_pillarbox() -> None:
    """Select a fixed picture region from twelve stable encoded side borders."""
    sampled_frames = [
        _pillarboxed_frame((index + 20, index + 30, index + 40)) for index in range(12)
    ]

    crop = select_sampled_crop(sampled_frames)

    assert crop == CropRectangle(left=2, top=0, right=7, bottom=6)


def _fully_bordered_frame() -> NDArray[np.uint8]:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[1:99, 1:99] = (30, 40, 50)
    return frame


@pytest.mark.parametrize(
    ("edge", "expected_crop"),
    [
        ("top", CropRectangle(left=1, top=0, right=99, bottom=99)),
        ("bottom", CropRectangle(left=1, top=1, right=99, bottom=100)),
        ("left", CropRectangle(left=0, top=1, right=99, bottom=99)),
        ("right", CropRectangle(left=1, top=1, right=100, bottom=99)),
    ],
)
def test_select_sampled_crop_leaves_each_disagreeing_edge_uncropped(
    edge: str,
    expected_crop: CropRectangle,
) -> None:
    """Keep stable sides while leaving one sampled disagreement uncropped."""
    sampled_frames = [_fully_bordered_frame() for _ in range(12)]
    differing_frame = sampled_frames[-1]
    if edge == "top":
        differing_frame[0, 50:52] = (30, 40, 50)
    elif edge == "bottom":
        differing_frame[-1, 50:52] = (30, 40, 50)
    elif edge == "left":
        differing_frame[50:52, 0] = (30, 40, 50)
    else:
        differing_frame[50:52, -1] = (30, 40, 50)

    crop = select_sampled_crop(sampled_frames)

    assert crop == expected_crop


def test_select_sampled_crop_rejects_ambiguous_or_unbordered_samples() -> None:
    """Return no crop for fully black frames or picture content at every edge."""
    fully_black_frames = [np.zeros((6, 8, 3), dtype=np.uint8) for _ in range(12)]
    unbordered_frames = [
        np.full((6, 8, 3), (30, 40, 50), dtype=np.uint8) for _ in range(12)
    ]

    assert select_sampled_crop(fully_black_frames) is None
    assert select_sampled_crop(unbordered_frames) is None


def test_select_sampled_crop_rejects_varying_encoded_border_width() -> None:
    """Return no crop when the only encoded border has inconsistent width."""
    sampled_frames = [
        np.full((6, 8, 3), (30, 40, 50), dtype=np.uint8) for _ in range(12)
    ]
    for sampled_frame in sampled_frames[:-1]:
        sampled_frame[:1] = (0, 0, 0)
    sampled_frames[-1][:2] = (0, 0, 0)

    assert select_sampled_crop(sampled_frames) is None


def test_select_sampled_crop_rejects_inconsistent_sample_dimensions() -> None:
    """Return no crop when sampled frames do not share one frame size."""
    sampled_frames = [_letterboxed_frame((30, 40, 50)) for _ in range(11)]
    sampled_frames.append(np.zeros((7, 8, 3), dtype=np.uint8))

    assert select_sampled_crop(sampled_frames) is None


def test_select_sampled_crop_requires_exactly_twelve_samples() -> None:
    """Return no crop unless all twelve configured samples are present."""
    sampled_frames = [_letterboxed_frame((30, 40, 50)) for _ in range(11)]

    assert select_sampled_crop(sampled_frames) is None


def test_select_sampled_crop_requires_an_exact_black_candidate_boundary() -> None:
    """Reject a near-black but non-exact-black last-excluded candidate line."""
    sampled_frames = [
        np.full((6, 8, 3), (30, 40, 50), dtype=np.uint8) for _ in range(12)
    ]
    for sampled_frame in sampled_frames:
        sampled_frame[0] = (16, 16, 16)

    assert select_sampled_crop(sampled_frames) is None


def test_apply_crop_returns_picture_region_view() -> None:
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

    picture_region = apply_crop(
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
def test_apply_crop_rejects_non_black_boundary(
    non_black_position: tuple[int, int],
) -> None:
    """Reject a non-black pixel on any applicable last-excluded boundary."""
    bgr_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr_frame[non_black_position] = (1, 0, 0)

    with pytest.raises(CinebarError) as raised_error:
        apply_crop(
            bgr_frame,
            CropRectangle(left=1, top=1, right=3, bottom=3),
        )

    assert raised_error.value.category is FailureCategory.NON_BLACK_CROP_BOUNDARY
