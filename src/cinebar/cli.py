"""Command-line interface for complete no-crop barcode generation."""

import argparse
import logging
from collections.abc import Sequence
from multiprocessing import freeze_support
from pathlib import Path
from typing import NoReturn

from .errors import CinebarError, FailureCategory
from .processor import SingleFileProcessor

LOGGER = logging.getLogger(__name__)


class _CinebarArgumentParser(argparse.ArgumentParser):
    """Translate argument-parser failures into Cinebar's public error model."""

    def error(self, message: str) -> NoReturn:
        """Raise an argument failure without printing parser-controlled output.

        Args:
            message: Parser-generated detail that must not reach users.
        """
        raise CinebarError(FailureCategory.INVALID_ARGUMENT)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one no-crop barcode from a local media file.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Zero after publication, one after a failure, or 130 after interruption.
    """
    freeze_support()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _CinebarArgumentParser(
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing final PNG after complete encoding.",
    )

    try:
        arguments = parser.parse_args(argv)
        SingleFileProcessor().process(
            media_path=arguments.media_path,
            output_path=arguments.output,
            height=arguments.height,
            workers=arguments.workers,
            overwrite=arguments.overwrite,
        )
    except CinebarError as error:
        LOGGER.error("[%s] %s", error.category.value, error)
        return 130 if error.category is FailureCategory.INTERRUPTED else 1
    except KeyboardInterrupt:
        interruption = CinebarError(FailureCategory.INTERRUPTED)
        LOGGER.error("[%s] %s", interruption.category.value, interruption)
        return 130
    return 0


def _positive_integer(value: str) -> int:
    """Parse one strictly positive command-line integer.

    Args:
        value: Raw command-line argument.

    Returns:
        Parsed positive integer.

    Raises:
        CinebarError: If the value is not a positive integer.
    """
    try:
        parsed_value = int(value)
    except ValueError:
        raise CinebarError(FailureCategory.INVALID_HEIGHT) from None
    if parsed_value <= 0:
        raise CinebarError(FailureCategory.INVALID_HEIGHT)
    return parsed_value


def _positive_worker_count(value: str) -> int:
    """Parse one strictly positive worker count.

    Args:
        value: Raw command-line argument.

    Returns:
        Parsed positive worker count.

    Raises:
        CinebarError: If the value is not a positive integer.
    """
    try:
        parsed_value = int(value)
    except ValueError:
        raise CinebarError(FailureCategory.INVALID_ARGUMENT) from None
    if parsed_value <= 0:
        raise CinebarError(FailureCategory.INVALID_ARGUMENT)
    return parsed_value
