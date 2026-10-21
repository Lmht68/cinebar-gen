"""Command-line interface for complete no-crop barcode generation."""

import argparse
import logging
from collections.abc import Sequence
from multiprocessing import freeze_support
from pathlib import Path

from .extract import FrameExtractionError
from .processor import SingleFileProcessor


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one no-crop barcode from a local media file.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Zero after the barcode PNG is successfully published.
    """
    freeze_support()
    parser = argparse.ArgumentParser(
        description="Generate a complete lossless no-crop media barcode."
    )
    parser.add_argument("media_path", type=Path, help="Local media file to decode.")
    parser.add_argument("--output", required=True, type=Path, help="Final PNG path.")
    parser.add_argument(
        "--height",
        type=_positive_integer,
        default=300,
        help="Barcode height in pixels. Defaults to 300.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_worker_count,
        default=None,
        help="Worker processes. Defaults to at most 16; use 1 for sequential decode.",
    )
    parser.add_argument(
        "--crop",
        choices=("none",),
        default="none",
        help="Crop mode. Only 'none' is currently supported.",
    )
    arguments = parser.parse_args(argv)

    try:
        SingleFileProcessor().process(
            media_path=arguments.media_path,
            output_path=arguments.output,
            height=arguments.height,
            workers=arguments.workers,
        )
    except (FrameExtractionError, OSError, ValueError) as error:
        logging.getLogger(__name__).error("%s", error)
        return 1
    return 0


def _positive_integer(value: str) -> int:
    """Parse one strictly positive command-line integer.

    Args:
        value: Raw command-line argument.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("height must be an integer.") from error
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("height must be positive.")
    return parsed_value


def _positive_worker_count(value: str) -> int:
    """Parse one strictly positive worker count.

    Args:
        value: Raw command-line argument.

    Returns:
        Parsed positive worker count.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("workers must be an integer.") from error
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("workers must be positive.")
    return parsed_value
