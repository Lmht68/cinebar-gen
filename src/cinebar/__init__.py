"""Generate lossless complete-film color barcodes."""

from .crop import CropMode, CropRectangle, CropSelection
from .errors import CinebarError, FailureCategory
from .models import ProcessingResult
from .processor import SingleFileProcessor

__all__ = [
    "CinebarError",
    "CropMode",
    "CropRectangle",
    "CropSelection",
    "FailureCategory",
    "ProcessingResult",
    "SingleFileProcessor",
]
