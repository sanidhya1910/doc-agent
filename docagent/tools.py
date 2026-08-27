"""The agent tool layer.

Every tool takes the :class:`~docagent.state.Document` blackboard plus keyword
arguments and returns a short string observation for the planner.  Tools are
declared with a JSON schema so the same registry drives the planner, the
deterministic router, and the MCP surface.

Tools marked ``gpu=False`` are pure CPU and cost no ZeroGPU quota.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from .export import writers
from .models import ModelsUnavailable, model_run_count
from .preprocess import has_table_rules
from .state import Document, DocKind, Field, Page
from .tables import strip_tables, tables_from_markdown

log = logging.getLogger("docagent.tools")

__all__ = ["Tool", "TOOLS", "tool_schemas", "call_tool", "perceive", "summarize_text"]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    gpu: bool = False


TOOLS: dict[str, Tool] = {}


def _register(name: str, description: str, parameters: dict, gpu: bool = False) -> Callable:
    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        TOOLS[name] = Tool(name=name, description=description, parameters=parameters,
                           fn=fn, gpu=gpu)
        return fn

    return decorator


_PAGES_PARAM = {
    "type": "array",
    "items": {"type": "integer"},
    "description": "1-based page numbers. Omit for every page.",
}


def tool_schemas(exclude: set[str] | None = None) -> list[dict]:
    """OpenAI/Hermes-style function schemas for the planner."""
    exclude = exclude or set()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": {
                    "type": "object",
                    "properties": t.parameters,
                    "required": [],
                },
            },
        }
        for t in TOOLS.values()
        if t.name not in exclude
    ]


def call_tool(doc: Document, name: str, arguments: dict | None = None) -> str:
    """Run a tool by name, recording it on the trace.

    Errors are returned as observations rather than raised: a planner that
    passes a bad argument should be able to see what went wrong and retry.
    """
    arguments = dict(arguments or {})
    tool = TOOLS.get(name)
    if tool is None:
        return "error: no such tool %r. Available: %s" % (name, ", ".join(sorted(TOOLS)))

    before = model_run_count()
    try:
        result = tool.fn(doc, **arguments)
    except TypeError as exc:
        result = "error: bad arguments for %s (%s)" % (name, exc)
    except ModelsUnavailable as exc:
        result = "error: model unavailable for %s (%s)" % (name, exc)
    except Exception as exc:  # noqa: BLE001 - the planner needs to see anything
        log.warning("tool %s failed: %s", name, exc)
        result = "error: %s failed (%s)" % (name, exc)
    # Record GPU work that actually happened, not work the tool might have
    # done: a tool that fell back to CPU must not be counted against quota.
    doc.log(name, arguments, result, gpu=model_run_count() > before)
    return result


# ---------------------------------------------------------------------------
# inspection
# ---------------------------------------------------------------------------

@_register(
    "list_pages",
    "Describe every page: its type, confidence, whether it already has a text "
    "layer, and how many tables and fields were found.",
    {},
)
def list_pages(doc: Document) -> str:
    if not doc.pages:
        return "no pages loaded"
    return doc.brief(preview_chars=120)


# ---------------------------------------------------------------------------
# perception (GPU)
# ---------------------------------------------------------------------------

def _classify_page(page: Page) -> None:
    """Work out what a page is, from its text.

    Deliberately CPU-only. GLM-OCR has no classification mode -- it only
    answers its three parsing prompts -- and a page whose text is already on
    the blackboard can be judged with keyword heuristics for free, which is
    both more predictable and costs no GPU quota.
    """
    # An Office file whose tables were parsed natively is tabular by
    # construction; no need to infer it from the prose around them.
    if page.tables and not page.kind.is_form_like:
        page.kind, page.kind_confidence = DocKind.TABLE, 0.85
        return
    page.kind, page.kind_confidence = _classify_from_text(page.text())


_INVOICE_HINTS = re.compile(
    r"\b(invoice|bill to|amount due|subtotal|tax|vat|purchase order|po number)\b", re.I
)
_RECEIPT_HINTS = re.compile(r"\b(receipt|change due|cashier|thank you for shopping)\b", re.I)
_FORM_HINTS = re.compile(r"(\b\w[\w ]{2,24}:\s*\S)")

#: A number, with optional currency symbol, thousands separators or percent.
_NUMERIC = re.compile(r"^[$€£¥]?[-+(]?\d[\d,. ]*\)?%?$")
#: Two or more runs of whitespace wide enough to be column padding.
_COLUMN_GAPS = re.compile(r"\S {2,}\S")


def _looks_columnar(text: str, min_rows: int = 3) -> bool:
    """True when consecutive lines read like table rows.

    Needed because ``Text Recognition:`` transcribes a table as ordinary prose
    with single spaces -- the ruling lines and column padding are gone, so the
    only remaining signal is that several consecutive lines share a token count
    and carry mostly numbers. Without this a real ruled table reads as prose
    and never gets its table-recognition pass.
    """
    best = run = 0
    expected: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        tokens = line.split()
        row_like = (
            len(tokens) >= 3
            and sum(1 for t in tokens if _NUMERIC.match(t)) >= 2
        ) or len(_COLUMN_GAPS.findall(raw)) >= 2
        if row_like and (expected is None or len(tokens) == expected):
            run += 1
            expected = len(tokens)
        else:
            best = max(best, run)
            run = 1 if row_like else 0
            expected = len(tokens) if row_like else None
    return max(best, run) >= min_rows


def _classify_from_text(text: str) -> tuple[DocKind, float]:
    """Cheap CPU classification used for text-layer pages and as a fallback.

    Confidence is deliberately modest -- these are keyword heuristics, and the
    manifest should say so rather than overstating certainty.
    """
    if not text.strip():
        return DocKind.UNKNOWN, 0.0
    if tables_from_markdown(text):
        if _INVOICE_HINTS.search(text):
            return DocKind.INVOICE, 0.75
        if _RECEIPT_HINTS.search(text):
            return DocKind.RECEIPT, 0.75
        return DocKind.TABLE, 0.72
    if _INVOICE_HINTS.search(text):
        return DocKind.INVOICE, 0.68
    if _RECEIPT_HINTS.search(text):
        return DocKind.RECEIPT, 0.68
    if _looks_columnar(text):
        # Column-aligned rows with no invoice wording: a plain data table.
        # Lower confidence than a parsed table, since this is inferred shape.
        return DocKind.TABLE, 0.66
    if len(_FORM_HINTS.findall(text)) >= 4:
        return DocKind.FORM, 0.65
    return DocKind.PROSE, 0.70


@_register(
    "classify_pages",
    "Work out what each page is (invoice, receipt, form, table, prose, "
    "handwriting, id_document) from its recognised text. Cheap: no GPU needed. "
    "Run read_pages first so there is text to judge.",
    {"pages": _PAGES_PARAM},
)
def classify_pages(doc: Document, pages: Any = None) -> str:
    targets = doc.resolve_pages(pages)
    for page in targets:
        if page.kind is DocKind.UNKNOWN:
            _classify_page(page)
    return "; ".join(
        "p%d=%s(%.2f)" % (p.page_no, p.kind.value, p.kind_confidence) for p in targets
    ) or "nothing to classify"


@_register(
    "read_pages",
    "Run OCR and put the text on the blackboard. Backend 'auto' picks GLM-OCR "
    "for print and TrOCR for handwriting. Pages with a text layer are skipped "
    "because their text is already available.",
    {
        "pages": _PAGES_PARAM,
        "backend": {
            "type": "string",
            "enum": ["auto", "glm_ocr", "trocr"],
            "description": "Which recogniser to use. Default 'auto'.",
        },
    },
    gpu=True,
)
def read_pages(doc: Document, pages: Any = None, backend: str = "auto") -> str:
    from .backends import get_backend
    from .preprocess import prepare

    targets = [p for p in doc.resolve_pages(pages) if p.image is not None]
    if not targets:
        return "no image pages to read"

    if backend == "auto" and doc.force_backend:
        backend = doc.force_backend

    done, skipped, retried = 0, 0, 0
    for page in targets:
        if page.has_text_layer:
            skipped += 1
            continue
        chosen = backend
        if chosen == "auto":
            chosen = "trocr" if page.kind is DocKind.HANDWRITING else "glm_ocr"
        opts = doc.preprocess
        try:
            image = prepare(
                page.image,
                do_deskew=opts.get("deskew", True),
                do_denoise=opts.get("denoise", True),
                do_rescale=opts.get("rescale", True),
            )
            result = get_backend(chosen).read(image)
        except ModelsUnavailable as exc:
            page.warnings.append("OCR unavailable: %s" % exc)
            continue

        # A page that yields almost nothing while clearly carrying lines of ink
        # is the signature of script the primary reader could not handle. Only
        # then is a second, more expensive pass worth the quota.
        if backend == "auto" and chosen == "glm_ocr" and _looks_unread(result.markdown, image):
            try:
                fallback = get_backend("trocr").read(image)
                if len(fallback.markdown.strip()) > len(result.markdown.strip()):
                    result, chosen = fallback, "trocr"
                    retried += 1
                    page.warnings.append("re-read with TrOCR after a low-yield first pass")
            except ModelsUnavailable:
                pass

        page.markdown = result.markdown
        page.blocks = result.blocks
        page.backend = chosen
        page.warnings.extend(result.warnings)
        if result.confidence and page.kind_confidence == 0.0:
            page.kind_confidence = result.confidence
        done += 1

    parts = ["read %d page(s)" % done]
    if skipped:
        parts.append("skipped %d with an existing text layer" % skipped)
    if retried:
        parts.append("%d re-read with TrOCR" % retried)
    return "; ".join(parts)


#: Below this, a page has effectively not been read.
_MIN_CHARS_PER_LINE = 3


def _looks_unread(text: str, image: Any) -> bool:
    """True when a page has visible text lines but produced almost no text."""
    if len(text.strip()) >= 40:
        return False
    try:
        from .backends.trocr import segment_lines

        lines = len(segment_lines(image))
    except Exception:  # noqa: BLE001
        return False
    return lines >= 3 and len(text.strip()) < lines * _MIN_CHARS_PER_LINE


@_register(
    "extract_fields",
    "Pull key-value pairs out of form-like pages (invoice number, total, dates, "
    "names). Optionally restrict to specific keys.",
    {
        "pages": _PAGES_PARAM,
        "keys": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Field names to look for. Omit to let the model choose.",
        },
    },
    gpu=True,
)
def extract_fields(doc: Document, pages: Any = None, keys: Any = None) -> str:
    from .backends import get_backend

    schema = [str(k) for k in keys] if isinstance(keys, (list, tuple)) and keys else None
    targets = doc.resolve_pages(pages)
    total = 0
    for page in targets:
        if page.fields:
            total += len(page.fields)
            continue
        pairs: dict[str, str] = {}
        if page.image is not None and not page.has_text_layer:
            try:
                pairs = get_backend("glm_ocr").extract_fields(page.image, schema)
            except ModelsUnavailable as exc:
                page.warnings.append("field extraction unavailable: %s" % exc)
        if not pairs:
            pairs = _fields_from_text(page.text(), schema)
        page.fields = [
            Field(key=k, value=v, page_no=page.page_no, confidence=0.8 if page.image else 0.6)
            for k, v in pairs.items()
            if str(v).strip()
        ]
        total += len(page.fields)
    return "extracted %d field(s) across %d page(s)" % (total, len(targets))


_KV_LINE = re.compile(r"^\s*([A-Za-z][\w &/().-]{1,40}?)\s*[:：]\s*(.+?)\s*$")


def _fields_from_text(text: str, schema: list[str] | None) -> dict[str, str]:
    """Regex key-value fallback for text-layer pages and model outages."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = _KV_LINE.match(line)
        if not match:
            continue
        key, value = match.group(1).strip(), match.group(2).strip()
        if not value or value.startswith("|"):
            continue
        found.setdefault(key, value)
    if schema:
        lowered = {k.lower(): v for k, v in found.items()}
        return {k: lowered.get(k.lower(), "") for k in schema}
    return found


# ---------------------------------------------------------------------------
# structure (CPU)
# ---------------------------------------------------------------------------

@_register(
    "extract_tables",
    "Pull tables out of the pages into rows and columns, ready for a "
    "spreadsheet. Free for text that already contains table markup; runs a "
    "table-recognition pass on image pages that look tabular.",
    {"pages": _PAGES_PARAM},
    gpu=True,
)
def extract_tables(doc: Document, pages: Any = None) -> str:
    targets = doc.resolve_pages(pages)
    total = 0
    for page in targets:
        if not page.tables:
            page.tables = tables_from_markdown(page.text(), page_no=page.page_no)
        if not page.tables:
            _recognise_tables_visually(doc, page)
        for table in page.tables:
            # Office loaders build tables before page indices are assigned.
            if not table.page_no:
                table.page_no = page.page_no
        total += len(page.tables)
    if not total:
        return "no tables found"
    shapes = ", ".join(
        "p%d %dx%d" % (t.page_no, t.shape[0], t.shape[1]) for t in doc.all_tables()
    )
    return "found %d table(s): %s" % (total, shapes)


def _recognise_tables_visually(doc: Document, page: Page) -> None:
    """Second pass over a tabular-looking page image, for real structure.

    ``Text Recognition:`` transcribes a table as space-aligned plain text, so a
    page can read perfectly and still yield nothing exportable. This asks the
    model for the table markup instead. Gated on the page looking tabular so an
    ordinary page of prose never pays for it.
    """
    if page.image is None or page.has_text_layer:
        return
    if not (
        page.kind.is_tabular
        or page.kind is DocKind.UNKNOWN
        or has_table_rules(page.image)
    ):
        return
    try:
        from .backends import get_backend

        backend = get_backend("glm_ocr")
        markup = backend.read_tables(page.image)
    except ModelsUnavailable as exc:
        log.info("table recognition unavailable (%s)", exc)
        return
    except Exception as exc:  # noqa: BLE001
        page.warnings.append("table recognition failed: %s" % exc)
        return

    found = tables_from_markdown(markup, page_no=page.page_no)
    if found:
        page.tables = found
        # The structure pass is stronger evidence than the text heuristic that
        # ran before it, so let it correct the page type.
        if not page.kind.is_tabular:
            page.kind, page.kind_confidence = DocKind.TABLE, 0.8
    elif page.kind.is_tabular:
        page.warnings.append("looked tabular but no table structure was recovered")


# ---------------------------------------------------------------------------
# summarisation (GPU, with a CPU fallback)
# ---------------------------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the to was were
    will with this these those there their they you your we our""".split()
)

_LENGTH_SENTENCES = {"short": 2, "medium": 4, "long": 8}


def _extractive_summary(text: str, sentences: int) -> str:
    """Word-frequency sentence ranking. No model, no dependency, no quota.

    Used when the brain model is unavailable so the summary feature never
    simply disappears.
    """
    clean = " ".join(strip_tables(text).split())
    if not clean:
        return ""
    all_parts = [s.strip() for s in _SENTENCE.split(clean) if s.strip()]
    # Prefer whole sentences over fragments like "Page 3.", but never let the
    # filter empty out a document that is simply written in short sentences.
    parts = [s for s in all_parts if len(s.split()) >= 4]
    if len(parts) < sentences:
        parts = all_parts
    if len(parts) <= sentences:
        return " ".join(parts)

    freq: dict[str, int] = {}
    for word in re.findall(r"[a-z']+", clean.lower()):
        if word in _STOPWORDS or len(word) < 3:
            continue
        freq[word] = freq.get(word, 0) + 1
    if not freq:
        return " ".join(parts[:sentences])

    scored = []
    for i, sentence in enumerate(parts):
        words = re.findall(r"[a-z']+", sentence.lower())
        if not words:
            continue
        score = sum(freq.get(w, 0) for w in words) / len(words)
        scored.append((score, i, sentence))
    scored.sort(reverse=True)
    chosen = sorted(scored[:sentences], key=lambda item: item[1])
    return " ".join(s for _, _, s in chosen)


def summarize_text(text: str, length: str = "medium") -> tuple[str, bool]:
    """Summarise, preferring the brain model. Returns ``(summary, used_model)``."""
    n_sentences = _LENGTH_SENTENCES.get(length, 4)
    clean = strip_tables(text).strip()
    if not clean:
        return "", False

    # Keep the prompt bounded: the planner budget matters more than covering
    # every last page of a long document.
    excerpt = clean[:12000]
    prompt = (
        "Summarise the following document in %d sentences. Be factual and "
        "specific, name concrete figures and dates where present, and do not "
        "add anything that is not in the text.\n\n%s" % (n_sentences, excerpt)
    )
    try:
        from .llm import chat

        summary, _ = chat(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_new_tokens=90 * n_sentences,
        )
        if summary.strip():
            return summary.strip(), True
    except ModelsUnavailable as exc:
        log.info("summariser unavailable (%s); using extractive fallback", exc)
    except Exception as exc:  # noqa: BLE001
        log.warning("summariser failed (%s); using extractive fallback", exc)
    return _extractive_summary(clean, n_sentences), False


@_register(
    "summarize",
    "Write a summary of the document onto the blackboard, for the summary PDF "
    "and report.",
    {
        "length": {
            "type": "string",
            "enum": ["short", "medium", "long"],
            "description": "Roughly 2, 4 or 8 sentences. Default 'medium'.",
        }
    },
    gpu=True,
)
def summarize(doc: Document, length: str = "medium") -> str:
    summary, used_model = summarize_text(doc.full_text(), length)
    if not summary:
        return "nothing to summarise"
    doc.summary = summary
    if not used_model:
        doc.warnings.append("summary is extractive (the summarisation model was unavailable)")
    return "summary written (%d chars, %s)" % (
        len(summary), "model" if used_model else "extractive"
    )


# ---------------------------------------------------------------------------
# export (CPU)
# ---------------------------------------------------------------------------

def _export(doc: Document, kind: str, label: str, fn: Callable[[], Any], detail: str = "") -> str:
    produced = fn()
    paths = produced if isinstance(produced, list) else [produced]
    if not paths:
        return "nothing to write for %s" % kind
    for path in paths:
        doc.add_artifact(path, kind, label, detail)
    import os

    return "wrote %s" % ", ".join(os.path.basename(p) for p in paths)


@_register("write_xlsx", "Write every extracted table to an Excel workbook, one sheet per "
                         "table, plus Fields and Manifest sheets.", {})
def write_xlsx(doc: Document) -> str:
    if not doc.all_tables():
        return "no tables to write; call extract_tables first"
    return _export(doc, "xlsx", "Tables (Excel)",
                   lambda: writers.write_xlsx(doc, doc.workdir),
                   "%d table(s)" % len(doc.all_tables()))


@_register("write_csv", "Write each extracted table to its own CSV file.", {})
def write_csv(doc: Document) -> str:
    if not doc.all_tables():
        return "no tables to write; call extract_tables first"
    return _export(doc, "csv", "Tables (CSV)", lambda: writers.write_csv(doc, doc.workdir))


@_register("write_docx", "Write a Word document preserving headings, lists and tables.", {})
def write_docx(doc: Document) -> str:
    return _export(doc, "docx", "Word document", lambda: writers.write_docx(doc, doc.workdir))


@_register("write_pdf_report", "Write a PDF report with the summary, key fields, tables and "
                               "per-page provenance.", {})
def write_pdf_report(doc: Document) -> str:
    return _export(doc, "pdf", "Summary report",
                   lambda: writers.write_pdf_report(doc, doc.workdir))


@_register("write_searchable_pdf", "Write a PDF of the original pages with an invisible, "
                                   "selectable text layer.", {})
def write_searchable_pdf(doc: Document) -> str:
    if not any(p.image is not None for p in doc.pages):
        return "no page images available for a searchable PDF"
    return _export(doc, "pdf", "Searchable PDF",
                   lambda: writers.write_searchable_pdf(doc, doc.workdir))


@_register("write_json", "Write the whole blackboard as structured JSON.", {})
def write_json(doc: Document) -> str:
    return _export(doc, "json", "Structured JSON", lambda: writers.write_json(doc, doc.workdir))


@_register("write_markdown", "Write the full recognised text as Markdown.", {})
def write_markdown(doc: Document) -> str:
    return _export(doc, "markdown", "Full text (Markdown)",
                   lambda: writers.write_markdown(doc, doc.workdir))


@_register(
    "finish",
    "Call this when the request has been satisfied, with a one-sentence reply "
    "for the user.",
    {"message": {"type": "string", "description": "What you did, in one sentence."}},
)
def finish(doc: Document, message: str = "") -> str:
    return message or "done"


# ---------------------------------------------------------------------------
# the perception pre-pass
# ---------------------------------------------------------------------------

def perceive(doc: Document) -> None:
    """Read, then classify, then recover structure.

    Reading comes first because classification is done from the text, on the
    CPU: judging a page by what it says is both more reliable than a prompt
    GLM-OCR does not support, and free. Pages that arrive with a text layer
    skip the read entirely and so never reach the GPU at all.
    """
    if doc.pages_needing_ocr():
        call_tool(doc, "read_pages", {})
    call_tool(doc, "classify_pages", {})
    call_tool(doc, "extract_tables", {})
