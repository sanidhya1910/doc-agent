"""The MCP tool surface.

These are what an external agent actually calls, so they are worth testing
independently of the internal tool layer: they take plain paths and strings and
must return well-formed JSON without a live model.
"""

from __future__ import annotations

import json

import pytest

import mcp_api


def test_every_tool_is_registered_under_its_own_name():
    """Gradio derives the MCP tool name from ``__name__``, not from api_name."""
    for fn, api_name, description in mcp_api.TOOLS:
        assert fn.__name__ == api_name
        assert description
        assert fn.__doc__, "%s needs a docstring; it becomes the MCP schema" % api_name


def test_classify_document_infers_a_real_type(fixture_path):
    """Regression: classification reads first, so images are not all 'unknown'."""
    payload = json.loads(mcp_api.classify_document(fixture_path("workbook.xlsx")))
    assert payload["document_type"] == "table"
    assert all(p["type"] == "table" for p in payload["pages"])
    assert all(p["confidence"] > 0 for p in payload["pages"])


def test_classify_document_on_a_docx(fixture_path):
    payload = json.loads(mcp_api.classify_document(fixture_path("agreement.docx")))
    assert payload["pages"][0]["has_text_layer"] is True
    assert payload["document_type"] != "unknown"


@pytest.mark.parametrize("name", ["deck.pptx", "workbook.xlsx", "agreement.docx"])
def test_extract_tables_returns_rows_and_records(fixture_path, name):
    tables = json.loads(mcp_api.extract_tables(fixture_path(name)))
    assert tables
    first = tables[0]
    assert first["header"] and first["rows"]
    assert first["records"][0].keys() == set(first["header"]) or first["records"]


def test_extract_fields_honours_a_requested_key_list(fixture_path):
    fields = json.loads(
        mcp_api.extract_fields(fixture_path("agreement.docx"), "Renewal notice")
    )
    assert [f["key"] for f in fields] == ["Renewal notice"]
    assert fields[0]["value"] == "90 days"


def test_ocr_document_returns_markdown_with_page_markers(fixture_path):
    text = mcp_api.ocr_document(fixture_path("deck.pptx"))
    assert "<!-- page 1 -->" in text
    assert "<!-- page 2 -->" in text
    assert "| Region | 2025 | 2026 |" in text


def test_summarize_document_never_returns_empty(fixture_path):
    summary = mcp_api.summarize_document(fixture_path("digital.pdf"), "short")
    assert summary.strip()


def test_document_to_xlsx_returns_a_readable_workbook(fixture_path):
    import openpyxl

    path = mcp_api.document_to_xlsx(fixture_path("workbook.xlsx"))
    workbook = openpyxl.load_workbook(path)
    assert {"Headcount", "Rates", "Manifest"} <= set(workbook.sheetnames)


def test_process_document_reports_artifacts_and_type(fixture_path):
    payload = json.loads(mcp_api.process_document(fixture_path("agreement.docx")))
    assert payload["page_count"] == 1
    assert payload["document_type"] != "unknown"
    assert {a["kind"] for a in payload["artifacts"]} >= {"xlsx", "zip"}
    assert payload["reply"]


def test_unreadable_input_does_not_raise(tmp_path):
    """An external agent should get a JSON answer, not an exception."""
    bogus = tmp_path / "x.rtf"
    bogus.write_text("nope", encoding="utf-8")
    payload = json.loads(mcp_api.process_document(str(bogus)))
    assert payload["page_count"] == 0
    assert payload["warnings"]
