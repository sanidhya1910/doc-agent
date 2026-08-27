"""Every supported input shape reaches the blackboard correctly."""

from __future__ import annotations

import zipfile

import pytest

from docagent import load


@pytest.mark.parametrize(
    "name,expected_pages,expect_text_layer",
    [
        ("invoice.png", 1, False),
        ("table.png", 1, False),
        ("photo.heic", 1, False),
        ("digital.pdf", 2, True),
        ("scanned.pdf", 3, False),
        ("multipage.tiff", 3, False),
        ("agreement.docx", 1, True),
        ("deck.pptx", 2, True),       # one page per slide
        ("workbook.xlsx", 2, True),   # one page per sheet
        ("batch.zip", 2, False),
    ],
)
def test_page_counts_and_text_layer(fixture_path, workdir, name, expected_pages,
                                    expect_text_layer):
    doc = load(fixture_path(name), workdir=workdir)
    assert doc.n_pages == expected_pages
    assert not doc.warnings
    assert all(p.has_text_layer == expect_text_layer for p in doc.pages)


def test_digital_pdf_needs_no_ocr(fixture_path, workdir):
    """The quota guarantee starts here: a text layer means no OCR at all."""
    doc = load(fixture_path("digital.pdf"), workdir=workdir)
    assert doc.pages_needing_ocr() == []
    assert "quarterly review" in doc.full_text().lower()


def test_scanned_pdf_needs_ocr(fixture_path, workdir):
    doc = load(fixture_path("scanned.pdf"), workdir=workdir)
    assert len(doc.pages_needing_ocr()) == 3
    assert all(p.image is not None for p in doc.pages)


def test_docx_tables_parsed_natively(fixture_path, workdir):
    doc = load(fixture_path("agreement.docx"), workdir=workdir)
    tables = doc.all_tables()
    assert len(tables) == 1
    assert tables[0].header == ["Term", "Value", "Unit"]
    assert ["Volume", "12000", "units"] in tables[0].rows


def test_pptx_one_page_per_slide_with_tables(fixture_path, workdir):
    doc = load(fixture_path("deck.pptx"), workdir=workdir)
    assert "Revenue up eleven percent" in doc.pages[0].text()
    (table,) = doc.pages[1].tables
    assert table.header == ["Region", "2025", "2026"]
    assert ["North", "41", "52"] in table.rows


def test_xlsx_one_page_per_sheet_named_after_it(fixture_path, workdir):
    doc = load(fixture_path("workbook.xlsx"), workdir=workdir)
    assert [p.source for p in doc.pages] == [
        "workbook.xlsx#Headcount",
        "workbook.xlsx#Rates",
    ]
    assert [t.caption for t in doc.all_tables()] == ["Headcount", "Rates"]
    assert doc.all_tables()[1].rows == [
        ["Volume", "12000", "units"],
        ["Rate", "0.4", "each"],
    ]


def test_heic_decodes_at_full_resolution(fixture_path, workdir):
    """Phone photos arrive as HEIC; it must not silently degrade."""
    doc = load(fixture_path("photo.heic"), workdir=workdir)
    (page,) = doc.pages
    assert page.image is not None
    assert page.image.size == (1100, 780)
    assert page.needs_ocr


@pytest.mark.parametrize("name", ["deck.pptx", "workbook.xlsx", "agreement.docx"])
def test_office_tables_appear_in_the_page_text(fixture_path, workdir, name):
    """Otherwise a table-only slide or sheet exports an empty document.md."""
    from docagent.tables import strip_tables, tables_from_markdown

    doc = load(fixture_path(name), workdir=workdir)
    for page in doc.pages:
        if not page.tables:
            continue
        assert page.text().strip(), "page %d has tables but no text" % page.page_no
        # The rendered tables must round-trip back to the same structure...
        assert len(tables_from_markdown(page.text())) == len(page.tables)
        # ...and strip cleanly, so Word/summary exports do not duplicate them.
        assert not tables_from_markdown(strip_tables(page.text()))


def test_pages_are_indexed_consecutively(fixture_path, workdir):
    doc = load([fixture_path("invoice.png"), fixture_path("digital.pdf")], workdir=workdir)
    assert [p.page_no for p in doc.pages] == [1, 2, 3]
    assert doc.sources == ["invoice.png", "digital.pdf"]


def test_multiple_files_combine(fixture_path, workdir):
    doc = load([fixture_path("invoice.png"), fixture_path("table.png")], workdir=workdir)
    assert doc.n_pages == 2


def test_unreadable_input_warns_without_raising(workdir, tmp_path):
    bogus = tmp_path / "notes.rtf"
    bogus.write_text("nope", encoding="utf-8")
    doc = load(str(bogus), workdir=workdir)
    assert doc.n_pages == 0
    assert any("could not read" in w for w in doc.warnings)


def test_missing_file_warns_without_raising(workdir):
    doc = load("definitely-not-here.png", workdir=workdir)
    assert doc.n_pages == 0
    assert any("missing file" in w for w in doc.warnings)


def test_one_bad_file_does_not_lose_the_batch(fixture_path, workdir, tmp_path):
    bogus = tmp_path / "broken.rtf"
    bogus.write_text("nope", encoding="utf-8")
    doc = load([fixture_path("invoice.png"), str(bogus)], workdir=workdir)
    assert doc.n_pages == 1
    assert doc.warnings


def test_zip_cannot_escape_the_workdir(workdir, tmp_path):
    """A path-traversal member must land inside the workdir, by basename."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escaped.txt", "should not escape")
    doc = load(str(archive), workdir=workdir)
    assert not (tmp_path.parent / "escaped.txt").exists()
    # The member is still read, just from a safe location inside the workdir.
    assert doc.n_pages <= 1
