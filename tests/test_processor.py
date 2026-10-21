"""Behavior tests for no-crop barcode processing."""

from pathlib import Path

import png
import pytest

from cinebar.processor import SingleFileProcessor


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

    def extract_colors(_: Path, __: int | None) -> bytes:
        return compact_colors

    output_path = tmp_path / "barcode.png"
    result = SingleFileProcessor(extract_colors).process(
        Path("source.mkv"), output_path, height=2, workers=2
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
    output_path = tmp_path / "default-height.png"

    result = SingleFileProcessor(lambda _path, _workers: compact_colors).process(
        Path("source.mkv"), output_path
    )

    with output_path.open("rb") as barcode_file:
        width, height, rows, _ = png.Reader(file=barcode_file).read()
        actual_rows = [list(row) for row in rows]

    assert result.output_height == 300
    assert width == 1
    assert height == 300
    assert actual_rows == [[0, 0, 0]] * 300


@pytest.mark.parametrize("height", [0, -1])
def test_process_rejects_non_positive_height_before_decoding(
    tmp_path: Path, height: int
) -> None:
    """Reject invalid dimensions before the frame-source factory is called."""
    color_extractor_called = False

    def extract_colors(_: Path, __: int | None) -> bytes:
        nonlocal color_extractor_called
        color_extractor_called = True
        return b""

    with pytest.raises(ValueError, match="height must be positive"):
        SingleFileProcessor(extract_colors).process(
            Path("source.mkv"), tmp_path / "barcode.png", height=height
        )

    assert color_extractor_called is False


def test_process_rejects_empty_decode_before_creating_output(tmp_path: Path) -> None:
    """Reject an empty frame source without publishing a barcode file."""
    output_path = tmp_path / "barcode.png"

    with pytest.raises(ValueError, match="No video frames were decoded"):
        SingleFileProcessor(lambda _path, _workers: b"").process(
            Path("source.mkv"), output_path
        )

    assert not output_path.exists()
