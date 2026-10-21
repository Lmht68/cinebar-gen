"""Generate lossless complete-film color barcodes."""

from .errors import CinebarError, FailureCategory
from .models import ProcessingResult
from .processor import SingleFileProcessor

__all__ = [
    "CinebarError",
    "FailureCategory",
    "ProcessingResult",
    "SingleFileProcessor",
]
