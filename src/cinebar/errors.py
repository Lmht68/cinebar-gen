"""Stable public failure categories for Cinebar processing."""

from enum import StrEnum


class FailureCategory(StrEnum):
    """Classify user-actionable Cinebar processing failures."""

    INVALID_ARGUMENT = "invalid_argument"
    INVALID_HEIGHT = "invalid_height"
    INPUT_NOT_FOUND = "input_not_found"
    INPUT_UNREADABLE = "input_unreadable"
    NO_VIDEO_STREAM = "no_video_stream"
    ZERO_FRAMES = "zero_frames"
    DECODE_FAILED = "decode_failed"
    CORRUPT_FRAME = "corrupt_frame"
    OUTPUT_EXISTS = "output_exists"
    OUTPUT_WRITE_FAILED = "output_write_failed"
    INTERRUPTED = "interrupted"


_FAILURE_MESSAGES: dict[FailureCategory, str] = {
    FailureCategory.INVALID_ARGUMENT: (
        "Invalid command-line arguments. Run 'cinebar --help' for usage."
    ),
    FailureCategory.INVALID_HEIGHT: "Height must be a positive integer.",
    FailureCategory.INPUT_NOT_FOUND: (
        "Input media was not found. Verify the source path and try again."
    ),
    FailureCategory.INPUT_UNREADABLE: (
        "Input media is not a readable regular file. Check its permissions and "
        "file type."
    ),
    FailureCategory.NO_VIDEO_STREAM: (
        "Input media has no usable video stream. Choose a supported video file."
    ),
    FailureCategory.ZERO_FRAMES: (
        "The video stream decoded zero frames. Verify that the media is not empty or "
        "damaged."
    ),
    FailureCategory.DECODE_FAILED: (
        "Video decoding failed before completion. Check that the media is not damaged; "
        "when using multiple workers, retry with --workers 1."
    ),
    FailureCategory.CORRUPT_FRAME: (
        "The decoder returned a corrupt video frame. Verify or transcode the "
        "source media."
    ),
    FailureCategory.OUTPUT_EXISTS: (
        "Output already exists. Use --overwrite to replace it intentionally."
    ),
    FailureCategory.OUTPUT_WRITE_FAILED: (
        "The barcode could not be written. Check destination permissions and available "
        "space."
    ),
    FailureCategory.INTERRUPTED: (
        "Processing was interrupted; no new barcode was published."
    ),
}


class CinebarError(Exception):
    """Report one stable, user-actionable Cinebar failure.

    Args:
        category: Stable category describing the failure.
    """

    category: FailureCategory

    def __init__(self, category: FailureCategory) -> None:
        """Create a failure with a fixed safe message.

        Args:
            category: Stable category describing the failure.
        """
        self.category = category
        super().__init__(category)

    def __str__(self) -> str:
        """Return the path-safe actionable failure message."""
        return _FAILURE_MESSAGES[self.category]
