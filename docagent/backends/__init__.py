"""OCR backends.

Importing this package registers every backend under its name; the modules are
imported for their side effect and re-exported for direct use.
"""

from .base import OCRBackend, OCRResult, available_backends, get_backend, register
from . import glm_ocr as glm_ocr  # noqa: F401  (registers "glm_ocr")
from . import trocr as trocr  # noqa: F401  (registers "trocr")

__all__ = [
    "OCRBackend",
    "OCRResult",
    "available_backends",
    "get_backend",
    "register",
    "glm_ocr",
    "trocr",
]
