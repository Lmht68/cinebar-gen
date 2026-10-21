"""Application-owned values used during barcode processing."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Details of a successfully published barcode PNG.

    Args:
        output_path: Final path of the published barcode PNG.
        decoded_frame_count: Number of validated decoded frame colors.
        output_width: PNG width in frame strips.
        output_height: PNG height in pixels.

    Raises:
        ValueError: If dimensions do not describe a non-empty barcode.
    """

    output_path: Path
    decoded_frame_count: int
    output_width: int
    output_height: int

    def __post_init__(self) -> None:
        """Validate the completed barcode dimensions."""
        if self.decoded_frame_count <= 0:
            raise ValueError("decoded_frame_count must be positive.")
        if self.output_width != self.decoded_frame_count:
            raise ValueError("output_width must equal decoded_frame_count.")
        if self.output_height <= 0:
            raise ValueError("output_height must be positive.")
