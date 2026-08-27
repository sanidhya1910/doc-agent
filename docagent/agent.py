"""Top-level orchestration: one document in, artifacts and a reply out.

The whole run -- perception, routing or planning, and export -- happens inside
a *single* ``@spaces.GPU`` session.  One session per tool would multiply queue
waits and burn the visitor quota on scheduling rather than work.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any, Iterator, Sequence

from .export import build_bundle
from .ingest import load
from .models import estimate_duration, gpu_task
from .state import Document
from .tools import perceive

log = logging.getLogger("docagent.agent")

__all__ = ["run", "run_stream", "process"]


def _finalise(doc: Document, reply: str) -> str:
    """Always write the bundle, then add anything the user should check.

    This is the chosen low-confidence policy: never block, never fail, always
    hand back artifacts plus an explicit note about what was uncertain.
    """
    build_bundle(doc, doc.workdir)

    from .export.bundle import LOW_CONFIDENCE

    concerns = [
        "p%d read as %s (confidence %.2f)" % (p.page_no, p.kind.value, p.kind_confidence)
        for p in doc.pages
        if 0.0 < p.kind_confidence < LOW_CONFIDENCE
    ]
    parts = [reply.strip()] if reply.strip() else []
    if concerns:
        parts.append("Worth checking: " + "; ".join(concerns) + ".")
    if doc.warnings:
        parts.append("Notes: " + "; ".join(doc.warnings) + ".")
    return " ".join(parts)


def _agent_events(doc: Document) -> Iterator[dict]:
    """Perceive, then either route deterministically or hand off to the planner."""
    from .planner import iter_plan
    from .router import route

    started = time.time()
    perceive(doc)
    yield {"type": "step", "tool": "perceive", "args": {},
           "observation": doc.brief(preview_chars=80)}

    reply = route(doc)
    if reply is not None:
        yield {"type": "step", "tool": "router", "args": {},
               "observation": "deterministic plan (no planner tokens used)"}
        yield {"type": "final", "message": reply}
    else:
        yield from iter_plan(doc)

    log.info("agent run finished in %.1fs", time.time() - started)


@gpu_task(duration=estimate_duration)
def _gpu_phase(doc: Document) -> list[dict]:
    """Everything that may touch a model, and nothing else.

    Returns a list rather than a generator because a ZeroGPU function has to
    complete inside its session; the caller replays the events afterwards.

    Export deliberately happens *outside* this function. Writing a workbook, a
    Word file and two PDFs takes seconds of pure CPU, and inside the session
    every one of those seconds would be billed against the visitor's GPU quota.
    """
    return list(_agent_events(doc))


def _reply_from(events: list[dict]) -> str:
    return next(
        (e.get("message", "") for e in reversed(events) if e.get("type") == "final"), ""
    )


def run(doc: Document) -> str:
    """Process an already-ingested document. Returns the user-facing reply."""
    return _finalise(doc, _reply_from(_gpu_phase(doc)))


def run_stream(doc: Document) -> list[dict]:
    """Same as :func:`run`, but returns the full event list for the UI trace."""
    events = _gpu_phase(doc)
    events.append({"type": "result", "message": _finalise(doc, _reply_from(events))})
    return events


def process(
    sources: Any | Sequence[Any],
    instruction: str = "",
    *,
    dpi: int = 200,
    workdir: str | os.PathLike | None = None,
) -> Document:
    """Ingest and process in one call, returning the finished Document.

    Artifacts are on ``doc.artifacts`` and the user-facing sentence is the last
    ``reply`` entry of ``doc.trace``. Callers that only want that sentence can
    call :func:`load` and :func:`run` directly instead.
    """
    work = workdir or tempfile.mkdtemp(prefix="docagent_")
    doc = load(sources, instruction=instruction, dpi=dpi, workdir=work)
    doc.workdir = str(work)
    if not doc.pages:
        doc.warnings.append("no readable pages were found in the input")
        return doc
    reply = run(doc)
    doc.log("reply", {}, reply)
    return doc
