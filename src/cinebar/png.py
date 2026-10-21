"""Stream compact frame colors into lossless barcode PNGs."""

from collections.abc import Iterator
from typing import BinaryIO, Protocol, cast

import png


class _ReadableInto(Protocol):
    """Read binary data directly into a caller-provided buffer."""

    def readinto(self, buffer: bytearray) -> int | None:
        """Read data into ``buffer`` and return the byte count."""
        ...


def write_barcode_png(
    output_file: BinaryIO,
    color_spool: BinaryIO,
    frame_count: int,
    height: int,
) -> None:
    """Write a lossless barcode PNG from compact ordered frame colors.

    Args:
        output_file: Open binary stream that receives the complete PNG.
        color_spool: Seekable binary stream containing three RGB bytes per frame.
        frame_count: Number of frame colors in ``color_spool`` and PNG width.
        height: Requested PNG height in pixels.

    Raises:
        ValueError: If dimensions are non-positive or the spool is incomplete.
        OSError: If the output stream cannot be written.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if height <= 0:
        raise ValueError("height must be positive.")

    writer = png.Writer(
        width=frame_count,
        height=height,
        greyscale=False,
        alpha=False,
        bitdepth=8,
    )
    writer.write(output_file, _barcode_rows(color_spool, frame_count, height))


def _barcode_rows(
    color_spool: BinaryIO, frame_count: int, height: int
) -> Iterator[bytearray]:
    """Yield each barcode row from one reusable compact color buffer."""
    row = bytearray(frame_count * 3)
    color_spool.seek(0)
    if cast(_ReadableInto, color_spool).readinto(row) != len(row):
        raise ValueError("Frame color spool does not contain every decoded frame.")
    for _ in range(height):
        yield row
