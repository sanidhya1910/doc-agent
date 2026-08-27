"""Table recovery from the Markdown/HTML that the reader emits."""

from __future__ import annotations

from docagent.state import Table
from docagent.tables import (
    strip_tables,
    table_to_markdown,
    tables_from_markdown,
    to_dataframe,
)

PIPE = """Intro prose.

**Line items**

| Item | Qty | Price |
| ---- | --: | ----: |
| Widget | 2 | 9.99 |
| Gadget | 1 | 24.50 |

Closing prose.
"""


def test_pipe_table_header_and_rows():
    (table,) = tables_from_markdown(PIPE, page_no=3)
    assert table.header == ["Item", "Qty", "Price"]
    assert table.rows == [["Widget", "2", "9.99"], ["Gadget", "1", "24.50"]]
    assert table.shape == (2, 3)
    assert table.page_no == 3


def test_caption_comes_from_the_line_above_blank_lines():
    (table,) = tables_from_markdown(PIPE)
    assert table.caption == "Line items"


def test_escaped_pipe_is_not_a_delimiter():
    text = "| A | B |\n| --- | --- |\n| left \\| right | 2 |\n"
    (table,) = tables_from_markdown(text)
    assert table.rows == [["left | right", "2"]]


def test_prose_containing_a_pipe_is_not_a_table():
    """Without a separator row it is prose, however many pipes it has."""
    assert tables_from_markdown("this | that | the other\nand | more | pipes") == []


def test_ragged_rows_are_normalised_to_header_width():
    text = "| A | B | C |\n| - | - | - |\n| 1 | 2 |\n| 1 | 2 | 3 | 4 |\n"
    (table,) = tables_from_markdown(text)
    assert table.rows == [["1", "2", ""], ["1", "2", "3"]]


def test_html_tables_are_parsed():
    text = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    (table,) = tables_from_markdown(text)
    assert table.header == ["A", "B"]
    assert table.rows == [["1", "2"]]


def test_multiple_tables_in_one_page():
    text = PIPE + "\n| X | Y |\n| - | - |\n| 1 | 2 |\n"
    assert len(tables_from_markdown(text)) == 2


def test_records_keyed_by_header():
    (table,) = tables_from_markdown(PIPE)
    assert table.to_records()[0] == {"Item": "Widget", "Qty": "2", "Price": "9.99"}


def test_strip_tables_leaves_prose_only():
    stripped = strip_tables(PIPE)
    assert "Intro prose." in stripped
    assert "Closing prose." in stripped
    assert "Widget" not in stripped


def test_strip_tables_removes_html_tables():
    text = "before <table><tr><td>x</td></tr></table> after"
    assert "<table" not in strip_tables(text)


def test_table_renders_back_to_markdown_and_round_trips():
    original = Table(header=["Item", "Qty"], rows=[["Widget", "2"], ["Gadget", "1"]],
                     page_no=1, caption="Line items")
    (parsed,) = tables_from_markdown(table_to_markdown(original))
    assert parsed.header == original.header
    assert parsed.rows == original.rows
    assert parsed.caption == "Line items"


def test_rendered_table_escapes_pipes_so_it_round_trips():
    original = Table(header=["A", "B"], rows=[["left | right", "2"]])
    (parsed,) = tables_from_markdown(table_to_markdown(original))
    assert parsed.rows == [["left | right", "2"]]


def test_rendered_table_pads_short_rows():
    original = Table(header=["A", "B", "C"], rows=[["1"]])
    (parsed,) = tables_from_markdown(table_to_markdown(original))
    assert parsed.rows == [["1", "", ""]]


def test_rendered_table_without_a_header_gets_placeholder_columns():
    """Synthetic columns mean no original row is consumed as the header."""
    original = Table(rows=[["1", "2"], ["3", "4"]])
    rendered = table_to_markdown(original)
    assert "col 1" in rendered and "col 2" in rendered
    (parsed,) = tables_from_markdown(rendered)
    assert parsed.header == ["col 1", "col 2"]
    assert parsed.rows == original.rows


def test_rendered_tables_are_removed_by_strip_tables():
    """The Word and summary exports rely on this to avoid duplication."""
    rendered = table_to_markdown(Table(header=["A"], rows=[["1"]]))
    assert not tables_from_markdown(strip_tables(rendered))


def test_dataframe_disambiguates_duplicate_columns():
    text = "| A | A | |\n| - | - | - |\n| 1 | 2 | 3 |\n"
    (table,) = tables_from_markdown(text)
    frame = to_dataframe(table)
    assert len(set(frame.columns)) == 3
