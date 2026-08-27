"""GLM-OCR backend: the primary perception model.

GLM-OCR is *prompt-limited*: it is not a general instruction-following VLM.
Its model card defines two scenarios, and it was verified against both here:

* document parsing, via exactly three prompts -- ``Text Recognition:`` (plain
  text in reading order), ``Table Recognition:`` (a real HTML ``<table>``) and
  ``Formula Recognition:`` (LaTeX);
* information extraction, which needs a literal JSON skeleton to fill in.

Two consequences shape the pipeline. Reading a table with ``Text Recognition:``
gives space-aligned plain text with no structure to export, so tabular pages
get a second ``Table Recognition:`` pass. And classification is not a mode this
model has, so page types are inferred from the recognised text on the CPU --
which is both more reliable here and free.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from typing import Any

from PIL import Image

from ..models import HUB, device, note_model_run
from ..state import Block
from .base import OCRResult, register

log = logging.getLogger("docagent.backends.glm_ocr")

MAX_NEW_TOKENS = int(os.environ.get("DOCAGENT_MAX_NEW_TOKENS", "4096"))

# GLM-OCR is prompt-limited: the model card defines exactly three document
# parsing prompts, and it does not follow free-form instructions. Anything
# else (classification, for instance) has to be done elsewhere.
READ_PROMPT = "Text Recognition:"
TABLE_PROMPT = "Table Recognition:"
FORMULA_PROMPT = "Formula Recognition:"

#: Fields are pulled by the model's second supported mode, information
#: extraction, which requires a literal JSON skeleton rather than an English
#: instruction. Verified: an English prompt returns prose, this returns JSON.
DEFAULT_FIELD_KEYS = [
    "document_type",
    "document_number",
    "date",
    "issued_by",
    "issued_to",
    "total_amount",
]


def _messages(image_ref: Any, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_ref},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _mean_token_confidence(scores: Any) -> float:
    """Mean top-1 softmax probability across generated tokens.

    A genuine, if rough, signal -- unlike asking the model to rate itself.
    Used to flag low-confidence pages in the manifest rather than to gate
    anything, so approximate is fine.
    """
    try:
        import torch

        if not scores:
            return 1.0
        probs = [torch.softmax(step.float(), dim=-1).max().item() for step in scores]
        return float(sum(probs) / len(probs))
    except Exception:  # pragma: no cover - never worth failing a run over
        return 1.0


def _generate(image: Image.Image, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    """Run one GLM-OCR turn, returning (text, confidence)."""
    import torch

    processor, model = HUB.glm_ocr()
    note_model_run()

    # Processors differ on whether they accept a PIL object inline; falling
    # back to a temp file keeps this working across transformers releases.
    tmp_path = None
    try:
        try:
            inputs = processor.apply_chat_template(
                _messages(image, prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            image.convert("RGB").save(tmp_path)
            inputs = processor.apply_chat_template(
                _messages(tmp_path, prompt),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        inputs = inputs.to(device())
        inputs.pop("token_type_ids", None)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        text = processor.decode(out.sequences[0][prompt_len:], skip_special_tokens=True)
        return text.strip(), _mean_token_confidence(getattr(out, "scores", None))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _json_key(name: str) -> str:
    """Make a field name safe to embed in the JSON skeleton."""
    return re.sub(r'[^\w .:/-]', "", str(name)).strip() or "field"


def _strip_fence(text: str) -> str:
    """Drop a wrapping code fence if the model added one."""
    fence = re.match(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", text, re.DOTALL)
    return fence.group(1) if fence else text


class GLMOCRBackend:
    """Markdown-producing document reader."""

    name = "glm_ocr"

    def read(self, image: Image.Image, **kwargs: Any) -> OCRResult:
        max_new_tokens = int(kwargs.get("max_new_tokens", MAX_NEW_TOKENS))
        text, confidence = _generate(image, READ_PROMPT, max_new_tokens)
        text = _strip_fence(text)

        # GLM-OCR is coordinate-free in this mode, so blocks carry text and
        # reading order but no boxes.  The searchable-PDF writer detects the
        # missing boxes and lays text out by line instead; positions are then
        # approximate while the text stays selectable and searchable.
        blocks = [
            Block(text=line, kind="text", confidence=confidence)
            for line in text.splitlines()
            if line.strip()
        ]
        return OCRResult(markdown=text, blocks=blocks, confidence=confidence)

    def read_tables(self, image: Image.Image, **kwargs: Any) -> str:
        """Recognise table structure, returning HTML.

        ``Text Recognition:`` transcribes a table as space-aligned plain text,
        which carries no structure to export.  This mode returns a real
        ``<table>``, which :mod:`docagent.tables` parses into rows.
        """
        max_new_tokens = int(kwargs.get("max_new_tokens", MAX_NEW_TOKENS))
        text, _ = _generate(image, TABLE_PROMPT, max_new_tokens)
        return _strip_fence(text)

    def read_formulas(self, image: Image.Image) -> str:
        """Recognise formulas, returning LaTeX."""
        text, _ = _generate(image, FORMULA_PROMPT, 1024)
        return _strip_fence(text)

    def extract_fields(self, image: Image.Image, schema: list[str] | None = None) -> dict[str, str]:
        """Pull key-value pairs by asking for a filled-in JSON skeleton.

        The model requires a literal JSON template here; an English request
        for "the key information" comes back as prose.
        """
        keys = [k for k in (schema or DEFAULT_FIELD_KEYS) if str(k).strip()]
        skeleton = "{\n" + ",\n".join('  "%s": ""' % _json_key(k) for k in keys) + "\n}"
        prompt = "Output the information in the image in the following JSON format:\n" + skeleton

        text, _ = _generate(image, prompt, 1024)
        raw = _strip_fence(text)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Models sometimes prepend a sentence; salvage the first object.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                log.warning("field extraction returned no JSON object")
                return {}
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                log.warning("field extraction returned malformed JSON")
                return {}
        if not isinstance(parsed, dict):
            return {}
        return {
            str(k): ("" if v is None else str(v))
            for k, v in parsed.items()
            if not isinstance(v, (dict, list))
        }


@register("glm_ocr")
def _factory() -> GLMOCRBackend:
    return GLMOCRBackend()
