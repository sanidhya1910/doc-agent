"""The deterministic fast path.

A ZeroGPU visitor gets five minutes of GPU per day, so spending planner tokens
on a document whose handling is obvious is waste.  When the page types point
clearly at one output and the user has not asked for anything specific, the
router picks the plan directly and the planner LLM never runs.

:func:`route` returns ``None`` when the document is genuinely ambiguous, which
is the signal for :mod:`docagent.planner` to take over.
"""

from __future__ import annotations

import logging

from .state import Document, DocKind
from .tools import call_tool

log = logging.getLogger("docagent.router")

__all__ = ["route", "plan_for"]


def plan_for(doc: Document) -> list[tuple[str, dict]] | None:
    """The tool sequence for an unambiguous document, or ``None``.

    The plan covers only what is needed *beyond* perception, which has already
    read the pages, classified them and pulled out any tables. So a plain table
    needs nothing further, while an invoice still needs its header fields.

    Plans produce *data* only -- fields, a summary. They never call the
    ``write_*`` tools: the bundle written at the end of every run already emits
    each artifact the data warrants, and doing that work here would put
    pure-CPU file writing inside the GPU session.

    Deliberately conservative: anything mixed or unknown goes to the planner
    rather than being guessed at here.
    """
    if not doc.pages:
        return None

    kind = doc.dominant_kind()
    if kind in (DocKind.UNKNOWN, DocKind.MIXED):
        return None

    # An invoice or receipt is a table *and* a form: perception got the line
    # items, but the header fields are usually the point.
    if kind.is_form_like:
        return [("extract_fields", {})]

    if kind.is_tabular:
        return []  # perception already produced the tables

    if kind in (DocKind.PROSE, DocKind.HANDWRITING):
        return [("summarize", {"length": "medium"})]

    return None


def route(doc: Document) -> str | None:
    """Run the fast path if one applies. Returns a reply, or ``None`` to defer.

    A free-text instruction always defers: interpreting it is exactly what the
    planner is for.
    """
    if doc.instruction.strip():
        return None

    plan = plan_for(doc)
    if plan is None:
        log.info("no deterministic plan for %s; deferring to the planner",
                 doc.dominant_kind().value)
        return None

    for name, args in plan:
        call_tool(doc, name, args)

    kind = doc.dominant_kind()
    if kind.is_tabular:
        n = len(doc.all_tables())
        if not n:
            return ("Read %d page(s) as %s, but found no table structure to export."
                    % (doc.n_pages, kind.value))
        extra = (" and %d field(s)" % len(doc.all_fields())) if doc.all_fields() else ""
        return ("Read %d page(s) as %s and pulled %d table(s)%s into a spreadsheet."
                % (doc.n_pages, kind.value, n, extra))
    if kind.is_form_like:
        return "Read %d page(s) as %s and extracted %d field(s)." % (
            doc.n_pages, kind.value, len(doc.all_fields())
        )
    return "Read %d page(s) as %s and wrote a summary report." % (doc.n_pages, kind.value)
