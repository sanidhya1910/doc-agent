"""The contract every OCR backend implements, plus the backend registry."""

from __future__ import annotations

from typing import Callable, Protocol

from PIL import Image

from ..state import Block

__all__ = ["OCRResult", "OCRBackend", "register", "get_backend", "available_backends"]


class OCRResult:
    """What a backend returns for one page.

    ``markdown`` is the reading-order text (GLM-OCR produces real Markdown with
    tables and LaTeX; TrOCR produces plain lines).  ``blocks`` carry the
    coordinates that the searchable-PDF exporter needs.
    """

    __slots__ = ("markdown", "blocks", "confidence", "warnings")

    def __init__(
        self,
        markdown: str = "",
        blocks: list[Block] | None = None,
        confidence: float = 1.0,
        warnings: list[str] | None = None,
    ) -> None:
        self.markdown = markdown
        self.blocks = blocks or []
        self.confidence = confidence
        self.warnings = warnings or []


class OCRBackend(Protocol):
    """Read one page image into text plus positioned blocks."""

    name: str

    def read(self, image: Image.Image, **kwargs: object) -> OCRResult:
        ...


_REGISTRY: dict[str, Callable[[], OCRBackend]] = {}


def register(name: str) -> Callable[[Callable[[], OCRBackend]], Callable[[], OCRBackend]]:
    """Register a zero-arg factory under ``name``.

    Factories rather than instances so that importing this package never
    triggers a model download.
    """

    def decorator(factory: Callable[[], OCRBackend]) -> Callable[[], OCRBackend]:
        _REGISTRY[name] = factory
        return factory

    return decorator


def get_backend(name: str) -> OCRBackend:
    if name not in _REGISTRY:
        raise KeyError("unknown OCR backend %r (have: %s)" % (name, ", ".join(sorted(_REGISTRY))))
    return _REGISTRY[name]()


def available_backends() -> list[str]:
    return sorted(_REGISTRY)
