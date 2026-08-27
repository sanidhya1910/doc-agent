"""MCP tool surface.

Gradio derives MCP tool schemas from the type hints and docstrings of the
functions registered with ``gr.api``, so these are deliberately thin wrappers
with plain-typed arguments and explicit docs.  The internal tools in
:mod:`docagent.tools` take a live ``Document`` and are not shaped for that, so
they are not exposed directly.

Registered by :func:`register` from ``app.py``; every function here also works
as a normal Python API.
"""

from __future__ import annotations

import json
import tempfile
from typing import Any

import gradio as gr

from docagent import load, process
from docagent.tools import call_tool, perceive, summarize_text

__all__ = [
    "register",
    "process_document",
    "ocr_document",
    "classify_document",
    "extract_tables",
    "extract_fields",
    "summarize_document",
    "document_to_xlsx",
]


def _workdir() -> str:
    return tempfile.mkdtemp(prefix="docagent_mcp_")


def _artifact_list(doc: Any) -> list[dict]:
    return [
        {"kind": a.kind, "label": a.label, "path": a.path, "detail": a.detail}
        for a in doc.artifacts
    ]


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

def process_document(file_path: str, instruction: str = "") -> str:
    """Run the full document agent on a file and return a JSON result.

    The agent works out what the document is, reads it with the right model and
    produces the structured artifacts that suit it. Use this when you want the
    agent to decide; use the narrower tools when you already know what you want.

    Args:
        file_path: Path to an image, PDF, DOCX, PPTX, XLSX or ZIP file.
        instruction: Optional plain-English request, e.g. "put the line items
            in a spreadsheet and summarise the rest".

    Returns:
        JSON with the reply, the detected document type, per-page confidence,
        any warnings, and the paths of every artifact written.
    """
    doc = process(file_path, instruction, workdir=_workdir())
    reply = next(
        (t["result"] for t in reversed(doc.trace) if t["tool"] == "reply"), ""
    )
    return json.dumps(
        {
            "reply": reply,
            "document_type": doc.dominant_kind().value,
            "page_count": doc.n_pages,
            "tables": len(doc.all_tables()),
            "fields": len(doc.all_fields()),
            "summary": doc.summary,
            "warnings": doc.warnings,
            "artifacts": _artifact_list(doc),
        },
        indent=2,
        ensure_ascii=False,
    )


def ocr_document(file_path: str, backend: str = "auto") -> str:
    """Read a document and return its full text as Markdown.

    Tables come back as Markdown pipe tables and formulas as LaTeX, in reading
    order. Pages that already carry a text layer are used as-is without OCR.

    Args:
        file_path: Path to an image, PDF, DOCX, PPTX, XLSX or ZIP file.
        backend: "auto" (default), "glm_ocr" for print, or "trocr" for
            handwriting.

    Returns:
        The document text as Markdown, with a comment marking each page.
    """
    doc = load(file_path, workdir=_workdir())
    call_tool(doc, "read_pages", {"backend": backend})
    call_tool(doc, "classify_pages", {})
    return "\n\n".join(
        "<!-- page %d -->\n%s" % (p.page_no, p.text().strip())
        for p in doc.pages
        if p.text().strip()
    )


def classify_document(file_path: str) -> str:
    """Identify what a document is, without extracting tables or fields.

    Page types are inferred from the document's text, so an image or a scanned
    PDF is read first. Files that already carry a text layer are classified
    without any OCR at all.

    Args:
        file_path: Path to an image, PDF, DOCX, PPTX, XLSX or ZIP file.

    Returns:
        JSON with the overall type and a per-page type and confidence. Types
        are: invoice, receipt, form, table, prose, handwriting, id_document,
        mixed, unknown.
    """
    doc = load(file_path, workdir=_workdir())
    if doc.pages_needing_ocr():
        call_tool(doc, "read_pages", {})
    call_tool(doc, "classify_pages", {})
    return json.dumps(
        {
            "document_type": doc.dominant_kind().value,
            "pages": [
                {
                    "page": p.page_no,
                    "type": p.kind.value,
                    "confidence": round(p.kind_confidence, 3),
                    "has_text_layer": p.has_text_layer,
                }
                for p in doc.pages
            ],
        },
        indent=2,
    )


def extract_tables(file_path: str) -> str:
    """Extract every table from a document as JSON rows.

    Args:
        file_path: Path to an image, PDF, DOCX, PPTX, XLSX or ZIP file.

    Returns:
        JSON list of tables, each with its page number, caption, header and
        rows, plus records keyed by column name.
    """
    doc = load(file_path, workdir=_workdir())
    perceive(doc)
    return json.dumps(
        [
            {
                "page": t.page_no,
                "caption": t.caption,
                "header": t.header,
                "rows": t.rows,
                "records": t.to_records(),
            }
            for t in doc.all_tables()
        ],
        indent=2,
        ensure_ascii=False,
    )


def extract_fields(file_path: str, keys: str = "") -> str:
    """Extract key-value pairs from a form, invoice, receipt or ID document.

    Args:
        file_path: Path to an image, PDF, DOCX, PPTX, XLSX or ZIP file.
        keys: Optional comma-separated field names to look for, e.g.
            "invoice number, date, total". Leave empty to let the model choose.

    Returns:
        JSON list of fields, each with its key, value, page and confidence.
    """
    doc = load(file_path, workdir=_workdir())
    perceive(doc)
    wanted = [k.strip() for k in keys.split(",") if k.strip()]
    call_tool(doc, "extract_fields", {"keys": wanted} if wanted else {})
    return json.dumps(
        [
            {
                "key": f.key,
                "value": f.value,
                "page": f.page_no,
                "confidence": round(f.confidence, 3),
            }
            for f in doc.all_fields()
        ],
        indent=2,
        ensure_ascii=False,
    )


def summarize_document(file_path: str, length: str = "medium") -> str:
    """Summarise a document in plain prose.

    Args:
        file_path: Path to an image, PDF, DOCX, PPTX, XLSX or ZIP file.
        length: "short" (about 2 sentences), "medium" (4) or "long" (8).

    Returns:
        The summary text.
    """
    doc = load(file_path, workdir=_workdir())
    perceive(doc)
    summary, _ = summarize_text(doc.full_text(), length)
    return summary or "The document contained no text to summarise."


def document_to_xlsx(file_path: str) -> str:
    """Extract a document into an Excel workbook and return the file.

    One sheet per detected table, plus a Fields sheet for any key-value pairs
    and a Manifest sheet recording how each page was read.

    Args:
        file_path: Path to an image, PDF, DOCX, PPTX, XLSX or ZIP file.

    Returns:
        Path to the written .xlsx file.
    """
    work = _workdir()
    doc = load(file_path, workdir=work)
    doc.workdir = work
    perceive(doc)
    if doc.dominant_kind().is_form_like:
        call_tool(doc, "extract_fields", {})
    call_tool(doc, "write_xlsx", {})
    for artifact in doc.artifacts:
        if artifact.kind == "xlsx":
            return artifact.path
    # Still return a workbook so the caller gets a file rather than an error;
    # it will contain only the Manifest sheet.
    from docagent.export import writers

    return writers.write_xlsx(doc, work)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

#: (function, api_name, one-line description) for every exposed MCP tool.
#: The MCP tool name is taken from the function's ``__name__``, not from
#: ``api_name``, so the functions above are named exactly as they should appear
#: to an MCP client.
TOOLS = [
    (process_document, "process_document",
     "Run the full document agent: detect the document type and produce the "
     "structured output that suits it."),
    (ocr_document, "ocr_document",
     "Read a document and return its text as Markdown, tables included."),
    (classify_document, "classify_document",
     "Identify what a document is, per page, with confidence."),
    (extract_tables, "extract_tables",
     "Extract every table from a document as JSON rows."),
    (extract_fields, "extract_fields",
     "Extract key-value pairs from a form, invoice, receipt or ID document."),
    (summarize_document, "summarize_document",
     "Summarise a document in plain prose."),
    (document_to_xlsx, "document_to_xlsx",
     "Extract a document into an Excel workbook and return the file."),
]


def register() -> None:
    """Register every tool as a Gradio API endpoint.

    Must be called inside a ``gr.Blocks`` context. ``demo.launch(mcp_server=True)``
    then publishes these as MCP tools at ``/gradio_api/mcp/sse``.
    """
    for fn, api_name, description in TOOLS:
        gr.api(fn, api_name=api_name, api_description=description)
