"""TrOCR backend, kept for handwriting.

This is the original project pipeline -- segment a page into text lines, run
TrOCR on each -- with its three real weaknesses fixed:

* the morphology kernel was hardcoded to ``(30, 1)``, which only suits one
  font size; it is now derived from measured glyph height;
* lines that morphology merged were never recovered; a horizontal projection
  profile now splits them;
* recognition ran one line at a time in a Python loop; it is now batched.

Line boxes are kept, so TrOCR pages give the searchable-PDF exporter real
coordinates.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..models import HUB, device, note_model_run
from ..preprocess import binarize, estimate_text_height, to_gray
from ..state import Block, Box
from .base import OCRResult, register

log = logging.getLogger("docagent.backends.trocr")

BATCH_SIZE = int(os.environ.get("DOCAGENT_TROCR_BATCH", "8"))
MAX_NEW_TOKENS = 128


def _split_tall_box(
    binary: np.ndarray, box: tuple[int, int, int, int], text_h: int
) -> list[tuple[int, int, int, int]]:
    """Split a box that swallowed several lines, using row ink density.

    Rows with no ink are line gaps.  A box is only split where a gap is at
    least a third of a glyph tall, which avoids cutting through descenders.
    """
    x, y, w, h = box
    strip = binary[y : y + h, x : x + w]
    row_ink = (strip > 0).sum(axis=1)
    threshold = max(1, int(0.02 * w))
    min_gap = max(2, text_h // 3)

    segments: list[tuple[int, int]] = []
    start = None
    gap = 0
    for i, ink in enumerate(row_ink):
        if ink > threshold:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                segments.append((start, i - gap + 1))
                start = None
                gap = 0
    if start is not None:
        segments.append((start, len(row_ink)))

    if len(segments) <= 1:
        return [box]
    out = []
    for top, bottom in segments:
        if bottom - top >= max(4, text_h // 2):
            out.append((x, y + top, w, bottom - top))
    return out or [box]


def segment_lines(image: Image.Image) -> list[Box]:
    """Find text-line boxes, top-to-bottom then left-to-right."""
    gray = to_gray(image)
    binary = binarize(gray)
    text_h = estimate_text_height(gray)

    # Wide enough to bridge inter-word gaps, narrow enough not to bridge
    # columns.  Scales with the text rather than assuming a fixed size.
    kernel_w = max(9, int(text_h * 1.5))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Drop specks and the page border that thresholding sometimes finds.
        if h < max(4, text_h * 0.4) or w < max(6, text_h * 0.5):
            continue
        if w >= gray.shape[1] * 0.99 and h >= gray.shape[0] * 0.99:
            continue
        raw.append((x, y, w, h))

    split: list[tuple[int, int, int, int]] = []
    for box in raw:
        if box[3] > text_h * 2.2:
            split.extend(_split_tall_box(binary, box, text_h))
        else:
            split.append(box)

    # Reading order: group boxes whose vertical spans overlap into a row, then
    # order within the row left-to-right. A plain sort by y interleaves
    # side-by-side columns.
    split.sort(key=lambda b: b[1])
    ordered: list[tuple[int, int, int, int]] = []
    row: list[tuple[int, int, int, int]] = []
    row_bottom = -1
    for box in split:
        if row and box[1] > row_bottom - text_h * 0.4:
            row.sort(key=lambda b: b[0])
            ordered.extend(row)
            row = []
            row_bottom = -1
        row.append(box)
        row_bottom = max(row_bottom, box[1] + box[3])
    if row:
        row.sort(key=lambda b: b[0])
        ordered.extend(row)

    return [Box(*b) for b in ordered]


class TrOCRBackend:
    """Line-segmented handwriting recogniser."""

    name = "trocr"

    def read(self, image: Image.Image, **kwargs: Any) -> OCRResult:
        import torch

        boxes = segment_lines(image)
        if not boxes:
            return OCRResult(
                markdown="", blocks=[], confidence=0.0,
                warnings=["no text lines detected"],
            )

        processor, model = HUB.trocr()
        note_model_run()
        gray = image.convert("L")
        crops = [
            gray.crop((b.x, b.y, b.x + b.w, b.y + b.h)).convert("RGB") for b in boxes
        ]

        batch_size = max(1, int(kwargs.get("batch_size", BATCH_SIZE)))
        texts: list[str] = []
        for start in range(0, len(crops), batch_size):
            chunk = crops[start : start + batch_size]
            pixel_values = processor(images=chunk, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device=device(), dtype=model.dtype)
            with torch.no_grad():
                generated = model.generate(pixel_values, max_new_tokens=MAX_NEW_TOKENS)
            texts.extend(processor.batch_decode(generated, skip_special_tokens=True))

        blocks = [
            Block(text=t.strip(), box=b, kind="text")
            for t, b in zip(texts, boxes)
            if t.strip()
        ]
        markdown = "\n".join(b.text for b in blocks)
        # TrOCR has no calibrated score; report a coverage proxy so a page
        # where most lines came back empty is visibly flagged.
        confidence = len(blocks) / len(boxes) if boxes else 0.0
        return OCRResult(markdown=markdown, blocks=blocks, confidence=confidence)


@register("trocr")
def _factory() -> TrOCRBackend:
    return TrOCRBackend()
