"""CPU image clean-up applied before any model sees a page.

Everything here is deterministic OpenCV/PIL work and costs no GPU quota, so it
is always worth doing: a deskewed, adequately-sized page measurably improves
both OCR backends.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

__all__ = [
    "prepare",
    "deskew",
    "estimate_skew",
    "to_gray",
    "binarize",
    "estimate_text_height",
    "has_table_rules",
]

#: Below this height most recognisers start dropping characters.
MIN_EDGE = 640
#: Above this the models downscale internally anyway and we only waste memory.
MAX_EDGE = 2400


def to_gray(image: Image.Image) -> np.ndarray:
    """Grayscale ``uint8`` array from any PIL mode."""
    return np.array(image.convert("L"))


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold, inverted so text is white on black.

    Inverted is what the morphology in the line segmenter expects, and what
    ``cv2.findContours`` treats as foreground.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def estimate_skew(gray: np.ndarray, max_angle: float = 15.0) -> float:
    """Skew in degrees, positive meaning the page leans counter-clockwise.

    Uses the minimum-area rectangle around all foreground pixels, which is
    cheap and robust for scans.  Returns 0.0 when the estimate is implausible
    (a nearly-blank page, or a rotation large enough to be a real 90-degree
    orientation problem rather than skew).
    """
    binary = binarize(gray)
    # np.where yields (row, col); minAreaRect wants (x, y), so swap the axes.
    # Feeding it (y, x) transposes the page and reports the wrong angle.
    coords = np.column_stack(np.where(binary > 0))[:, ::-1]
    if coords.shape[0] < 50:
        return 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    # OpenCV reports the angle in (0, 90]; map it to a small signed rotation.
    if angle > 45:
        angle -= 90
    if abs(angle) > max_angle:
        return 0.0
    return float(angle)


def deskew(image: Image.Image, max_angle: float = 15.0) -> Image.Image:
    """Rotate a page upright. Returns the input unchanged for tiny angles."""
    angle = estimate_skew(to_gray(image), max_angle=max_angle)
    if abs(angle) < 0.25:
        return image
    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")


def estimate_text_height(gray: np.ndarray) -> int:
    """Median height of connected components that look like glyphs.

    Drives the adaptive morphology kernel in the TrOCR line segmenter, which
    the original implementation hardcoded to ``(30, 1)`` -- a width that only
    suits one particular font size.
    """
    binary = binarize(gray)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return 20
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    # Drop specks and full-page frames; what remains is mostly glyphs.
    keep = heights[(areas > 8) & (heights > 4) & (heights < gray.shape[0] * 0.25)]
    if keep.size == 0:
        return 20
    return int(np.median(keep))


def has_table_rules(image: Image.Image, min_lines: int = 3) -> bool:
    """Detect the ruling lines of a table, using morphology alone.

    A text-independent signal, which matters because the reader sometimes
    returns a table as a single run-on line -- leaving nothing columnar to spot
    in the text. Long horizontal and vertical runs of ink are what a ruled
    table has and a page of prose does not, and finding them costs no GPU.
    """
    gray = to_gray(image)
    height, width = gray.shape
    if height < 40 or width < 40:
        return False
    binary = binarize(gray)

    def count_runs(kernel_size: tuple[int, int], axis: int) -> int:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        found = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(found, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        span = width if axis == 0 else height
        return sum(
            1
            for c in contours
            if (cv2.boundingRect(c)[2] if axis == 0 else cv2.boundingRect(c)[3]) > span * 0.3
        )

    horizontal = count_runs((max(width // 8, 12), 1), axis=0)
    if horizontal >= min_lines:
        return True
    # A borderless table often still has column separators, so accept a
    # combination of a couple of rules in each direction.
    vertical = count_runs((1, max(height // 8, 12)), axis=1)
    return horizontal >= 2 and vertical >= 2


def _rescale(image: Image.Image) -> Image.Image:
    """Bring the long edge into a range both backends handle well."""
    w, h = image.size
    longest = max(w, h)
    if longest == 0:
        return image
    if longest < MIN_EDGE:
        factor = MIN_EDGE / longest
    elif longest > MAX_EDGE:
        factor = MAX_EDGE / longest
    else:
        return image
    return image.resize((max(int(w * factor), 1), max(int(h * factor), 1)), Image.LANCZOS)


def _denoise(image: Image.Image) -> Image.Image:
    """Light bilateral filter: removes scan grain without softening strokes."""
    arr = np.array(image.convert("RGB"))
    return Image.fromarray(cv2.bilateralFilter(arr, d=5, sigmaColor=45, sigmaSpace=45))


def prepare(
    image: Image.Image,
    *,
    do_deskew: bool = True,
    do_denoise: bool = True,
    do_rescale: bool = True,
) -> Image.Image:
    """Full pre-processing chain, each step individually skippable.

    Always returns RGB, because both backends expect three channels.
    """
    out = image
    if out.mode not in ("RGB", "L"):
        out = out.convert("RGB")
    if do_deskew:
        out = deskew(out)
    if do_denoise:
        out = _denoise(out)
    if do_rescale:
        out = _rescale(out)
    return out.convert("RGB")
