"""The confidence-reporting bundle.

Chosen policy: when the agent is not fully confident it never blocks and never
fails -- it emits every plausible artifact plus a manifest saying what it was
unsure about.  A visitor who does not stay to answer a question still leaves
with usable output.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path

from ..state import Document
from . import writers

log = logging.getLogger("docagent.export.bundle")

__all__ = ["build_manifest", "write_manifest", "build_bundle", "LOW_CONFIDENCE"]

#: Pages below this are called out in the manifest and the chat reply.
LOW_CONFIDENCE = 0.70


def build_manifest(doc: Document) -> dict:
    """Per-page provenance and an explicit list of what to double-check."""
    low = [p for p in doc.pages if 0.0 < p.kind_confidence < LOW_CONFIDENCE]
    unread = [p for p in doc.pages if not p.text().strip()]

    concerns: list[str] = []
    for p in low:
        concerns.append(
            "page %d read as %s with confidence %.2f -- verify before relying on it"
            % (p.page_no, p.kind.value, p.kind_confidence)
        )
    for p in unread:
        concerns.append("page %d produced no text" % p.page_no)
    concerns.extend(doc.warnings)

    return {
        "sources": doc.sources,
        "instruction": doc.instruction,
        "page_count": doc.n_pages,
        "overall_type": doc.dominant_kind().value,
        "min_page_confidence": round(doc.min_confidence(), 4),
        "tables_found": len(doc.all_tables()),
        "fields_found": len(doc.all_fields()),
        "needs_review": bool(concerns),
        "concerns": concerns,
        "pages": [
            {
                "page": p.page_no,
                "source": p.source,
                "type": p.kind.value,
                "confidence": round(p.kind_confidence, 4),
                "backend": p.backend or "none",
                "has_text_layer": p.has_text_layer,
                "chars": len(p.text()),
                "tables": len(p.tables),
                "fields": len(p.fields),
                "warnings": p.warnings,
            }
            for p in doc.pages
        ],
        "artifacts": [
            {"file": os.path.basename(a.path), "kind": a.kind, "label": a.label,
             "detail": a.detail}
            for a in doc.artifacts
        ],
        "trace": doc.trace,
    }


def write_manifest(doc: Document, directory: str | os.PathLike,
                   name: str = "manifest.json") -> str:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / name
    target.write_text(
        json.dumps(build_manifest(doc), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return str(target)


def build_bundle(
    doc: Document,
    directory: str | os.PathLike,
    name: str = "output.zip",
    *,
    include_searchable_pdf: bool = True,
) -> str:
    """Write every artifact that makes sense for this document, then zip it.

    Writers are attempted independently: a failure in one format is recorded as
    a warning and the rest of the bundle is still produced.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = {os.path.abspath(a.path) for a in doc.artifacts}

    def attempt(kind: str, label: str, fn, detail: str = "") -> None:
        try:
            produced = fn()
        except Exception as exc:
            doc.warnings.append("could not write %s: %s" % (kind, exc))
            log.warning("writer %s failed: %s", kind, exc)
            return
        for p in produced if isinstance(produced, list) else [produced]:
            if os.path.abspath(p) not in existing:
                doc.add_artifact(p, kind, label, detail)
                existing.add(os.path.abspath(p))

    attempt("markdown", "Full text (Markdown)", lambda: writers.write_markdown(doc, directory))
    attempt("json", "Structured JSON", lambda: writers.write_json(doc, directory))

    if doc.all_tables():
        attempt("xlsx", "Tables (Excel)", lambda: writers.write_xlsx(doc, directory),
                "%d table(s)" % len(doc.all_tables()))
        attempt("csv", "Tables (CSV)", lambda: writers.write_csv(doc, directory))

    if doc.all_fields():
        attempt(
            "json", "Key fields",
            lambda: writers.write_fields_json(doc, directory),
            "%d field(s)" % len(doc.all_fields()),
        )

    attempt("docx", "Word document", lambda: writers.write_docx(doc, directory))
    attempt("pdf", "Summary report", lambda: writers.write_pdf_report(doc, directory))
    if include_searchable_pdf and any(p.image is not None for p in doc.pages):
        attempt("pdf", "Searchable PDF",
                lambda: writers.write_searchable_pdf(doc, directory))

    manifest = write_manifest(doc, directory)

    zip_path = directory / name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for artifact in doc.artifacts:
            if os.path.exists(artifact.path):
                zf.write(artifact.path, os.path.basename(artifact.path))
        zf.write(manifest, os.path.basename(manifest))

    doc.add_artifact(str(zip_path), "zip", "Everything (ZIP)",
                     "%d file(s)" % (len(doc.artifacts) + 1))
    return str(zip_path)
