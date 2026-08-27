"""Line segmentation and the pieces of the backends that need no weights.

Segmentation is the part of the original TrOCR pipeline that was rewritten, so
it is worth testing on its own: it runs entirely on OpenCV and needs no model.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from docagent.backends import available_backends, get_backend
from docagent.backends.trocr import segment_lines
from docagent.llm import parse_tool_calls
from docagent.preprocess import deskew, estimate_skew, estimate_text_height, prepare, to_gray
from docagent.tools import _extractive_summary, _fields_from_text


def _page(lines: list[str], size=(700, 400), step: int = 46, font_scale: int = 1) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((30, 30 + i * step), line, fill="black")
    return image


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

def test_segments_one_box_per_line():
    image = _page(["first line here", "second line here", "third line here"])
    boxes = segment_lines(image)
    assert len(boxes) == 3


def test_boxes_come_back_in_reading_order():
    image = _page(["alpha", "bravo", "charlie"])
    boxes = segment_lines(image)
    assert [b.y for b in boxes] == sorted(b.y for b in boxes)


def test_tightly_spaced_lines_are_still_separated():
    """The original code merged these; the projection-profile split recovers them."""
    image = _page(["line one", "line two", "line three", "line four"], step=20,
                  size=(700, 220))
    assert len(segment_lines(image)) >= 3


def test_blank_page_yields_no_boxes():
    assert segment_lines(Image.new("RGB", (400, 200), "white")) == []


def test_boxes_stay_inside_the_page():
    image = _page(["alpha", "bravo"])
    width, height = image.size
    for box in segment_lines(image):
        assert 0 <= box.x and 0 <= box.y
        assert box.x + box.w <= width
        assert box.y + box.h <= height


def test_segmentation_adapts_to_text_size():
    """A fixed 30x1 kernel could not do this; the adaptive one can."""
    small = _page(["tiny text line one", "tiny text line two"], step=18, size=(700, 120))
    large = Image.new("RGB", (900, 400), "white")
    draw = ImageDraw.Draw(large)
    for i, line in enumerate(["BIG ONE", "BIG TWO"]):
        draw.text((30, 40 + i * 150), line, fill="black", font_size=60)
    assert len(segment_lines(small)) == 2
    assert len(segment_lines(large)) == 2


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def test_estimate_text_height_tracks_font_size():
    small = to_gray(_page(["small text"], size=(600, 120)))
    large = Image.new("RGB", (900, 300), "white")
    ImageDraw.Draw(large).text((30, 40), "LARGE", fill="black", font_size=72)
    assert estimate_text_height(to_gray(large)) > estimate_text_height(small)


def test_deskew_corrects_a_rotated_page():
    image = _page(["the quick brown fox", "jumps over the lazy dog"])
    rotated = image.rotate(-4, expand=True, fillcolor="white")
    before = abs(estimate_skew(to_gray(rotated)))
    after = abs(estimate_skew(to_gray(deskew(rotated))))
    assert after <= before


def test_deskew_leaves_a_straight_page_alone():
    image = _page(["already straight"])
    assert deskew(image).size == image.size


def test_prepare_always_returns_rgb():
    grayscale = Image.new("L", (800, 600), 255)
    assert prepare(grayscale).mode == "RGB"


def test_prepare_upscales_tiny_pages():
    tiny = Image.new("RGB", (120, 80), "white")
    assert max(prepare(tiny, do_deskew=False, do_denoise=False).size) >= 640


def test_prepare_downscales_huge_pages():
    huge = Image.new("RGB", (6000, 4000), "white")
    assert max(prepare(huge, do_deskew=False, do_denoise=False).size) <= 2400


def test_prepare_steps_are_individually_skippable():
    image = _page(["content"])
    out = prepare(image, do_deskew=False, do_denoise=False, do_rescale=False)
    assert out.size == image.size


def test_ruling_lines_detected_on_a_ruled_table(fixture_path):
    """Text-independent table signal, for pages the reader returns as one line."""
    from docagent.preprocess import has_table_rules

    assert has_table_rules(Image.open(fixture_path("ruled_table.png")))
    assert has_table_rules(Image.open(fixture_path("table.png")))


def test_ruling_lines_not_detected_on_prose(fixture_path):
    from docagent.preprocess import has_table_rules

    assert not has_table_rules(Image.open(fixture_path("prose.png")))
    assert not has_table_rules(Image.open(fixture_path("invoice.png")))


@pytest.mark.parametrize("text,expected", [
    # How the reader actually transcribes a table: single-spaced, no ruling.
    ("Item Qty Price\nWidget 2 9.99\nGadget 1 24.50\nBracket 10 0.40", True),
    # A PDF text layer keeps its column padding.
    ("Region      2025      2026\nNorth         41        52\nSouth   28   27", True),
    ("The review covers the period ending March. Revenue rose to 4.2 million.", False),
    # Prose that merely mentions numbers must not be mistaken for a table.
    ("In 2026 we grew. The board approved 14 hires and costs were 3.1 million.", False),
])
def test_columnar_detection(text, expected):
    """Needed because the reader flattens tables to single-spaced prose."""
    from docagent.tools import _looks_columnar

    assert _looks_columnar(text) is expected


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_both_backends_are_registered():
    assert set(available_backends()) == {"glm_ocr", "trocr"}


def test_unknown_backend_raises_a_helpful_error():
    with pytest.raises(KeyError, match="unknown OCR backend"):
        get_backend("tesseract")


# ---------------------------------------------------------------------------
# planner output parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    '<tool_call>{"name": "extract_tables", "arguments": {"pages": [1]}}</tool_call>',
    '```json\n{"name": "extract_tables", "arguments": {"pages": [1]}}\n```',
    '{"name": "extract_tables", "arguments": {"pages": [1]}}',
    '{"function": {"name": "extract_tables", "arguments": "{\\"pages\\": [1]}"}}',
])
def test_tool_calls_parse_from_every_shape_models_emit(raw):
    (call,) = parse_tool_calls(raw)
    assert call["name"] == "extract_tables"
    assert call["arguments"] == {"pages": [1]}


def test_unparseable_output_yields_no_calls():
    assert parse_tool_calls("I think we should extract the tables now.") == []


def test_multiple_tool_calls_are_all_returned():
    raw = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
           '<tool_call>{"name": "b", "arguments": {}}</tool_call>')
    assert [c["name"] for c in parse_tool_calls(raw)] == ["a", "b"]


def test_multiple_xml_tool_calls_are_all_returned():
    raw = ("<function=a><parameter=x>1</parameter></function>"
           "<function=b><parameter=x>2</parameter></function>")
    assert [c["name"] for c in parse_tool_calls(raw)] == ["a", "b"]


# ---------------------------------------------------------------------------
# CPU fallbacks
# ---------------------------------------------------------------------------

def test_extractive_summary_selects_source_sentences():
    text = (
        "Revenue rose to four million this quarter. Revenue growth was driven by "
        "the new offices. The cat sat on the mat. Revenue is expected to keep rising."
    )
    summary = _extractive_summary(text, 2)
    assert summary
    assert "cat sat on the mat" not in summary


def test_extractive_summary_keeps_source_order():
    text = "Alpha one here. Beta two here. Gamma three here. Delta four here."
    summary = _extractive_summary(text, 4)
    assert summary.index("Alpha") < summary.index("Delta")


def test_extractive_summary_of_empty_text():
    assert _extractive_summary("", 3) == ""


def test_regex_field_extraction():
    text = "Invoice number: 2026-114\nDate: 2026-03-12\nTotal: 48.48\nnot a field line"
    fields = _fields_from_text(text, None)
    assert fields["Invoice number"] == "2026-114"
    assert fields["Total"] == "48.48"


def test_regex_field_extraction_honours_a_schema():
    text = "Invoice number: 2026-114\nDate: 2026-03-12"
    fields = _fields_from_text(text, ["invoice number", "missing key"])
    assert fields["invoice number"] == "2026-114"
    assert fields["missing key"] == ""
