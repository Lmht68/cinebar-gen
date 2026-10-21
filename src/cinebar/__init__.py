"""Generate lossless complete-film color barcodes."""

from .crop import CropRectangle
from .errors import CinebarError, FailureCategory
from .models import ProcessingResult
from .processor import SingleFileProcessor

__all__ = [
    "CinebarError",
    "CropRectangle",
    "FailureCategory",
    "ProcessingResult",
    "SingleFileProcessor",
]
