"""Generate lossless complete-film color barcodes."""

from .models import ProcessingResult
from .processor import SingleFileProcessor

__all__ = ["ProcessingResult", "SingleFileProcessor"]
