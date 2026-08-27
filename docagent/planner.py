"""The planner: a step-capped tool-calling loop over the brain model.

The planner is only reached when :mod:`docagent.router` cannot settle the
document deterministically -- a mixed document, or a free-text instruction.
It sees the blackboard *summary*, never the full text: its job is to decide
what to do, not to read the document.
"""

from __future__ import annotations

import logging
from typing import Iterator

from .models import ModelsUnavailable
from .state import Document
from .tools import TOOLS, call_tool, tool_schemas

log = logging.getLogger("docagent.planner")

__all__ = ["plan_and_execute", "MAX_STEPS"]

MAX_STEPS = 8

SYSTEM_PROMPT = """You are a document-processing agent. You are given a document \
that has already been read, and a request from the user. Decide which tools to \
call to satisfy the request, then call `finish`.

Rules:
- Call one tool at a time and read the observation before deciding the next one.
- The text has already been recognised. Only call `read_pages` if a page is \
reported as having no text.
- `extract_tables` must run before `write_xlsx` or `write_csv`.
- `summarize` must run before `write_pdf_report` if the user wants a summary.
- Prefer the fewest tools that satisfy the request. Every tool marked GPU costs \
the user quota.
- When you are done, call `finish` with one sentence describing what you produced.

Reply with exactly one tool call and nothing else:
<tool_call>
<function=tool_name>
<parameter=argument_name>value</parameter>
</function>
</tool_call>"""


def _tool_catalogue() -> str:
    """Human-readable tool list, for templates that cannot take ``tools=``."""
    lines = []
    for tool in TOOLS.values():
        params = ", ".join(tool.parameters) or "no arguments"
        lines.append(
            "- %s(%s)%s: %s" % (tool.name, params, " [GPU]" if tool.gpu else "", tool.description)
        )
    return "\n".join(lines)


def _initial_messages(doc: Document) -> list[dict]:
    request = doc.instruction.strip() or (
        "No specific request was given. Produce the structured output that best "
        "suits this document."
    )
    user = "Available tools:\n%s\n\nDocument state:\n%s\n\nUser request: %s" % (
        _tool_catalogue(),
        doc.brief(),
        request,
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]


def plan_and_execute(doc: Document, max_steps: int = MAX_STEPS) -> str:
    """Run the loop to completion and return the user-facing reply."""
    reply = ""
    for event in iter_plan(doc, max_steps=max_steps):
        if event.get("type") == "final":
            reply = event.get("message", "")
    return reply


def iter_plan(doc: Document, max_steps: int = MAX_STEPS) -> Iterator[dict]:
    """Stream the loop, yielding an event per step so the UI can show a trace.

    Events: ``{"type": "step", "tool", "args", "observation"}`` and a closing
    ``{"type": "final", "message"}``.
    """
    from .llm import chat_with_tools

    schemas = tool_schemas()
    messages = _initial_messages(doc)

    try:
        for step in range(max_steps):
            text, calls = chat_with_tools(messages, schemas, max_new_tokens=384)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

            if not calls:
                # No parseable call. If the model wrote prose on its first turn
                # it is probably answering rather than planning; accept it.
                if step == 0 and text.strip():
                    yield {"type": "final", "message": text.strip()}
                    return
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text":
                                 "That was not a tool call. Reply with exactly one "
                                 "<tool_call><function=NAME>...</function></tool_call> "
                                 "block and nothing else."}],
                })
                continue

            for call in calls:
                name, args = call["name"], call["arguments"]
                if name == "finish":
                    message = str(args.get("message") or "").strip()
                    doc.log("finish", args, message or "done")
                    yield {"type": "final", "message": message or _fallback_reply(doc)}
                    return

                observation = call_tool(doc, name, args)
                yield {"type": "step", "tool": name, "args": args, "observation": observation}
                messages.append({
                    "role": "user",
                    "content": [{"type": "text",
                                 "text": "Observation from %s: %s" % (name, observation)}],
                })

        doc.warnings.append("planner hit the %d-step limit" % max_steps)
        yield {"type": "final", "message": _fallback_reply(doc)}
    except ModelsUnavailable as exc:
        log.info("planner unavailable (%s); falling back to the deterministic plan", exc)
        yield from _fallback_plan(doc, str(exc))


def _fallback_plan(doc: Document, reason: str) -> Iterator[dict]:
    """Run the router plan when the planner model cannot be used at all."""
    from .router import plan_for

    doc.warnings.append("planner model unavailable (%s); used the default plan" % reason)
    # Perception already read, classified and pulled tables, so the only thing
    # left worth doing for an otherwise-unplannable document is a summary.
    plan = plan_for(doc)
    if plan is None:
        plan = [("summarize", {"length": "medium"})]
    for name, args in plan:
        observation = call_tool(doc, name, args)
        yield {"type": "step", "tool": name, "args": args, "observation": observation}
    yield {"type": "final", "message": _fallback_reply(doc)}


def _fallback_reply(doc: Document) -> str:
    """Describe what actually got produced, when the model did not say."""
    bits = ["Processed %d page(s)" % doc.n_pages]
    if doc.all_tables():
        bits.append("%d table(s)" % len(doc.all_tables()))
    if doc.all_fields():
        bits.append("%d field(s)" % len(doc.all_fields()))
    if doc.summary:
        bits.append("a summary")
    return ", ".join(bits) + "."
