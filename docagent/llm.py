"""Thin wrapper around the planner/summariser model.

Kept separate from :mod:`docagent.planner` so that summarisation can use the
same resident model without importing the agent loop, and so the tool-calling
details live in one place.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .models import HUB, device, note_model_run

log = logging.getLogger("docagent.llm")

__all__ = ["chat", "chat_with_tools", "parse_tool_calls"]


def _apply_template(processor: Any, messages: list[dict], tools: list[dict] | None) -> Any:
    """Render messages, passing tools through when the template supports them.

    Not every chat template accepts ``tools=``; when it does not, the caller
    falls back to describing the tools in the system prompt instead.
    """
    kwargs: dict[str, Any] = dict(
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    if tools:
        try:
            return processor.apply_chat_template(messages, tools=tools, **kwargs), True
        except (TypeError, ValueError) as exc:
            log.info("chat template does not accept tools (%s); using prompt fallback", exc)
    return processor.apply_chat_template(messages, **kwargs), False


def chat(
    messages: list[dict],
    *,
    max_new_tokens: int = 512,
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    stop: list[str] | None = None,
) -> tuple[str, bool]:
    """One turn of the brain. Returns ``(text, tools_were_templated)``."""
    import torch

    processor, model = HUB.brain()
    note_model_run()
    inputs, templated = _apply_template(processor, messages, tools)
    inputs = inputs.to(device())
    inputs.pop("token_type_ids", None)
    prompt_len = inputs["input_ids"].shape[1]
    tokenizer = getattr(processor, "tokenizer", processor)

    kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
    )
    if stop:
        # Not every version supports stop_strings; losing it only costs tokens.
        kwargs["stop_strings"] = list(stop)
        kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        try:
            generated = model.generate(**inputs, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("stop_strings", None)
            kwargs.pop("tokenizer", None)
            generated = model.generate(**inputs, **kwargs)

    text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    return text.strip(), templated


#: Qwen3/Qwen3.5 style: <function=name><parameter=key>value</parameter></function>
_XML_FUNCTION = re.compile(
    r"<function\s*=\s*([\w.-]+)\s*>(.*?)</function\s*>", re.DOTALL | re.IGNORECASE
)
_XML_PARAMETER = re.compile(
    r"<parameter\s*=\s*([\w.-]+)\s*>(.*?)</parameter\s*>", re.DOTALL | re.IGNORECASE
)
#: GLM style: <tool_call>name <arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
_ARG_PAIR = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL | re.IGNORECASE
)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL | re.IGNORECASE)


def _coerce(value: str) -> Any:
    """JSON-decode an argument value, else keep it as a trimmed string."""
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _from_json(blob: str) -> list[dict]:
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    for item in parsed if isinstance(parsed, list) else [parsed]:
        if not isinstance(item, dict):
            continue
        # Accept both {"name": ..., "arguments": ...} and the OpenAI-style
        # {"function": {"name": ..., "arguments": ...}} nesting.
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments", fn.get("parameters", {}))
        if isinstance(args, str):
            args = _coerce(args)
        out.append({"name": str(name), "arguments": args if isinstance(args, dict) else {}})
    return out


def parse_tool_calls(text: str) -> list[dict]:
    """Pull tool calls out of a model turn.

    Model families disagree on the wire format, and the same loop has to work
    across whichever brain model is configured. All of these are handled:

    * ``<function=name><parameter=k>v</parameter></function>`` -- Qwen3/Qwen3.5,
      the default brain. Verified against the real model, which never emits JSON.
    * ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` -- Hermes and
      Qwen2.5.
    * ``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``
      -- the GLM convention.
    * a fenced or bare JSON object, which smaller models fall back to.

    Anything unparseable yields an empty list, and the caller turns that into an
    observation so the model can correct itself.
    """
    calls: list[dict] = []

    # 1. Qwen3.5 XML functions, which may or may not sit inside <tool_call>.
    for name, body in _XML_FUNCTION.findall(text):
        args = {key: _coerce(value) for key, value in _XML_PARAMETER.findall(body)}
        calls.append({"name": name.strip(), "arguments": args})
    if calls:
        return calls

    # 2. <tool_call> blocks: JSON inside, or the GLM arg_key/arg_value pairs.
    #    The closing tag is optional because generation can stop short.
    for block in _TOOL_CALL_BLOCK.findall(text):
        found = _from_json(block.strip())
        if found:
            calls.extend(found)
            continue
        pairs = _ARG_PAIR.findall(block)
        if pairs:
            name = block.split("<arg_key>", 1)[0].strip().splitlines()[0].strip()
            if name:
                calls.append(
                    {"name": name, "arguments": {k.strip(): _coerce(v) for k, v in pairs}}
                )
    if calls:
        return calls

    # 3. A fenced or bare JSON object.
    for blob in re.findall(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL):
        calls.extend(_from_json(blob))
    if calls:
        return calls
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return _from_json(match.group(0)) if match else []


#: Once a tool call is closed the turn is over. Without this the model runs on
#: to the token limit inventing further turns, which wastes GPU seconds and
#: buries the call that mattered.
_STOP_STRINGS = ["</tool_call>", "</function>"]


def chat_with_tools(
    messages: list[dict], tools: list[dict], *, max_new_tokens: int = 512
) -> tuple[str, list[dict]]:
    """One planner turn: returns the raw text and any tool calls parsed from it."""
    text, _ = chat(
        messages, max_new_tokens=max_new_tokens, tools=tools, stop=_STOP_STRINGS
    )
    return text, parse_tool_calls(text)
