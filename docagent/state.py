"""The blackboard that the agent tools read from and write to.

A ``Document`` is built once by :mod:`docagent.ingest`, filled in by the OCR
backends, then consumed by the exporters.  Every tool in :mod:`docagent.tools`
takes a ``Document`` and mutates it in place, so the planner never has to
thread state through its tool calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from PIL import Image

__all__ = [
    "DocKind",
    "Box",
    "Block",
    "Table",
    "Field",
    "Page",
    "Document",
    "Artifact",
]


class DocKind(str, Enum):
    """What a page looks like, as judged by :func:`docagent.tools.classify_pages`."""

    INVOICE = "invoice"
    RECEIPT = "receipt"
    FORM = "form"
    TABLE = "table"
    PROSE = "prose"
    HANDWRITING = "handwriting"
    ID_DOCUMENT = "id_document"
    MIXED = "mixed"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, raw: str) -> "DocKind":
        """Best-effort coercion of free-form model output to a member."""
        token = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
        # Longest first so "id_document" is not shadowed by a shorter member.
        for member in sorted(cls, key=lambda m: -len(m.value)):
            if member.value in token:
                return member
        return cls.UNKNOWN

    @property
    def is_tabular(self) -> bool:
        return self in {DocKind.TABLE, DocKind.INVOICE, DocKind.RECEIPT}

    @property
    def is_form_like(self) -> bool:
        return self in {DocKind.FORM, DocKind.INVOICE, DocKind.RECEIPT, DocKind.ID_DOCUMENT}


@dataclass
class Box:
    """Pixel bounding box in the coordinate space of the page image."""

    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def scaled(self, fx: float, fy: float) -> "Box":
        return Box(int(self.x * fx), int(self.y * fy), int(self.w * fx), int(self.h * fy))


@dataclass
class Block:
    """One recognised region of text, with the coordinates it came from.

    Coordinates are what make the searchable-PDF export possible, so every
    backend supplies them even when the model itself is coordinate-free
    (TrOCR gets them from the line segmenter).
    """

    text: str
    box: Box | None = None
    kind: str = "text"  # text | heading | table | formula | list
    confidence: float = 1.0


@dataclass
class Table:
    """A detected table, kept in a library-agnostic row/column form."""

    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    page_no: int = 0
    caption: str = ""
    confidence: float = 1.0

    @property
    def shape(self) -> tuple[int, int]:
        width = len(self.header) or (len(self.rows[0]) if self.rows else 0)
        return (len(self.rows), width)

    def to_records(self) -> list[dict[str, str]]:
        """Row dicts keyed by header, for JSON export."""
        if not self.header:
            return [{"col_%d" % i: v for i, v in enumerate(row)} for row in self.rows]
        out: list[dict[str, str]] = []
        for row in self.rows:
            padded = list(row) + [""] * (len(self.header) - len(row))
            out.append(dict(zip(self.header, padded)))
        return out


@dataclass
class Field:
    """A key-value pair pulled out of a form-like document."""

    key: str
    value: str
    page_no: int = 0
    confidence: float = 1.0


@dataclass
class Page:
    """A single page: its image, how it was read, and what came out."""

    index: int  # 0-based
    image: Image.Image | None = None
    source: str = ""  # originating filename
    kind: DocKind = DocKind.UNKNOWN
    kind_confidence: float = 0.0
    markdown: str = ""
    blocks: list[Block] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    fields: list[Field] = field(default_factory=list)
    #: True when text came from a PDF/Office text layer and no OCR is needed.
    #: This is the single largest GPU saving in the pipeline.
    has_text_layer: bool = False
    backend: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def page_no(self) -> int:
        """1-based page number, which is what users and the planner see."""
        return self.index + 1

    @property
    def needs_ocr(self) -> bool:
        return not self.has_text_layer and not self.markdown

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size if self.image is not None else (0, 0)

    def text(self) -> str:
        if self.markdown:
            return self.markdown
        return "\n".join(b.text for b in self.blocks)

    def preview(self, limit: int = 200) -> str:
        """Short excerpt for the planner prompt; never the full page."""
        flat = " ".join(self.text().split())
        return flat[:limit] + ("..." if len(flat) > limit else "")

    def thumbnail(self, width: int = 320):
        """Downscaled RGB copy for the PDF report, or ``None`` if imageless."""
        if self.image is None:
            return None
        img = self.image.copy()
        img.thumbnail((width, width * 4))
        return img.convert("RGB")


@dataclass
class Artifact:
    """A file the agent produced, plus enough metadata to describe it."""

    path: str
    kind: str  # xlsx | csv | docx | pdf | json | markdown | txt | zip
    label: str = ""
    detail: str = ""


@dataclass
class Document:
    """Everything known about one agent run."""

    pages: list[Page] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    instruction: str = ""
    summary: str = ""
    artifacts: list[Artifact] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    workdir: str = ""
    #: Per-run pre-processing switches, honoured by the OCR tools. Kept on the
    #: document rather than in globals so concurrent requests cannot affect
    #: each other.
    preprocess: dict[str, bool] = field(default_factory=dict)
    #: Set to "glm_ocr" or "trocr" to override backend selection for the whole
    #: run; empty means pick per page from the classification.
    force_backend: str = ""

    # -- convenience views ------------------------------------------------

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def page(self, page_no: int) -> Page:
        """Look up by 1-based page number, the numbering the planner uses."""
        if not 1 <= page_no <= len(self.pages):
            raise IndexError("page %d out of range (1..%d)" % (page_no, len(self.pages)))
        return self.pages[page_no - 1]

    def resolve_pages(self, spec: Any) -> list["Page"]:
        """Turn a loose page spec into pages.

        Accepts ``None``/``"all"`` for everything, a single 1-based number, or
        an iterable of numbers.  The planner is not reliable about which of
        those it emits, so all three are tolerated.
        """
        if spec is None or spec == "all" or spec == []:
            return list(self.pages)
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            return [self.page(int(spec))]
        if isinstance(spec, str):
            nums = [int(tok) for tok in spec.replace(",", " ").split() if tok.strip("-").isdigit()]
            return [self.page(n) for n in nums] if nums else list(self.pages)
        out: list[Page] = []
        for item in spec:
            out.append(self.page(int(item)))
        return out

    def pages_needing_ocr(self) -> list["Page"]:
        return [p for p in self.pages if p.needs_ocr]

    def all_tables(self) -> list[Table]:
        return [t for p in self.pages for t in p.tables]

    def all_fields(self) -> list[Field]:
        return [f for p in self.pages for f in p.fields]

    def full_text(self) -> str:
        return "\n\n".join(p.text() for p in self.pages if p.text().strip())

    def dominant_kind(self) -> DocKind:
        """The kind that best describes the document as a whole.

        Weighted by confidence so one low-confidence outlier page cannot
        redirect the whole run.
        """
        if not self.pages:
            return DocKind.UNKNOWN
        scores: dict[DocKind, float] = {}
        for p in self.pages:
            if p.kind is DocKind.UNKNOWN:
                continue
            scores[p.kind] = scores.get(p.kind, 0.0) + max(p.kind_confidence, 0.1)
        if not scores:
            return DocKind.UNKNOWN
        best, best_score = max(scores.items(), key=lambda kv: kv[1])
        total = sum(scores.values())
        # No clear winner across a multi-kind document -> mixed.
        if len(scores) > 1 and best_score / total < 0.6:
            return DocKind.MIXED
        return best

    def min_confidence(self) -> float:
        vals = [p.kind_confidence for p in self.pages if p.kind is not DocKind.UNKNOWN]
        return min(vals) if vals else 0.0

    # -- bookkeeping ------------------------------------------------------

    def log(self, tool: str, args: dict[str, Any], result: str, gpu: bool = False) -> None:
        self.trace.append({"tool": tool, "args": args, "result": result, "gpu": gpu})

    def add_artifact(self, path: str, kind: str, label: str = "", detail: str = "") -> Artifact:
        art = Artifact(path=path, kind=kind, label=label or kind.upper(), detail=detail)
        self.artifacts.append(art)
        return art

    def brief(self, preview_chars: int = 200) -> str:
        """Compact description of the blackboard for the planner prompt.

        Deliberately excludes full page text: the planner decides *what to do*,
        it does not need to read the document.
        """
        lines = [
            "Document: %d page(s) from %s" % (self.n_pages, ", ".join(self.sources) or "upload"),
            "Overall type: %s" % self.dominant_kind().value,
        ]
        for p in self.pages:
            lines.append(
                "  p%d: %s (conf %.2f) text_layer=%s tables=%d fields=%d"
                % (p.page_no, p.kind.value, p.kind_confidence, p.has_text_layer,
                   len(p.tables), len(p.fields))
            )
            excerpt = p.preview(preview_chars)
            if excerpt:
                lines.append("     excerpt: %s" % excerpt)
        if self.artifacts:
            lines.append("Artifacts so far: " + ", ".join(a.kind for a in self.artifacts))
        return "\n".join(lines)
