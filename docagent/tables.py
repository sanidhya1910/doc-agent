"""Recover tables from the Markdown/HTML that GLM-OCR emits.

Pure CPU string work, so extracting tables costs no GPU quota: the model has
already done the structural analysis and written it down as a pipe table or an
HTML ``<table>``.  All this does is parse it back into rows.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from .state import Table

__all__ = [
    "tables_from_markdown",
    "table_to_markdown",
    "strip_tables",
    "to_dataframe",
]

#: A Markdown separator row: | --- | :--: | ---: |. One dash is enough --
#: strict Markdown wants three, but models routinely emit "| - | - |".
_SEPARATOR = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")


def _split_row(line: str) -> list[str]:
    """Split one pipe-table row, tolerating missing outer pipes."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    # An escaped pipe inside a cell is not a delimiter.
    parts = re.split(r"(?<!\\)\|", stripped)
    return [p.replace(r"\|", "|").strip() for p in parts]


def _looks_like_row(line: str) -> bool:
    return "|" in line and line.strip() != ""


class _TableHTMLParser(HTMLParser):
    """Minimal <table> reader for the HTML some pages come back as."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag == "table":
            self._rows = []
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell = []
        elif tag == "br" and self._in_cell:
            self._cell.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._row.append(" ".join("".join(self._cell).split()))
            self._in_cell = False
        elif tag == "tr":
            if self._row:
                self._rows.append(self._row)
            self._row = []
        elif tag == "table":
            if self._rows:
                self.tables.append(self._rows)
            self._rows = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell.append(data)


def _tables_from_html(text: str, page_no: int) -> list[Table]:
    if "<table" not in text.lower():
        return []
    parser = _TableHTMLParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        return []
    out: list[Table] = []
    for rows in parser.tables:
        rows = [r for r in rows if any(c.strip() for c in r)]
        if len(rows) < 2:
            continue
        header = rows[0]
        # Models emit short rows for things like a trailing "Subtotal" line;
        # pad them so every row lines up with the header in the spreadsheet.
        body = [
            (list(r) + [""] * (len(header) - len(r)))[: len(header)] for r in rows[1:]
        ]
        out.append(Table(header=header, rows=body, page_no=page_no))
    return out


def tables_from_markdown(text: str, page_no: int = 0) -> list[Table]:
    """Every pipe table and HTML table found in ``text``.

    A pipe table needs a header, a separator row and at least one body row --
    without the separator a line of prose containing a pipe would be misread as
    a table.
    """
    tables: list[Table] = list(_tables_from_html(text, page_no))

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not _looks_like_row(lines[i]):
            i += 1
            continue
        if i + 1 >= len(lines) or not _SEPARATOR.match(lines[i + 1]):
            i += 1
            continue

        header = _split_row(lines[i])
        rows: list[list[str]] = []
        j = i + 2
        while j < len(lines) and _looks_like_row(lines[j]) and not _SEPARATOR.match(lines[j]):
            cells = _split_row(lines[j])
            # Normalise ragged rows against the header width.
            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))
            elif len(cells) > len(header):
                cells = cells[: len(header)]
            if any(c.strip() for c in cells):
                rows.append(cells)
            j += 1

        if rows:
            # A bold or italic line just above often names the table; skip
            # over any blank lines separating it from the table itself.
            caption = ""
            k = i - 1
            while k >= 0 and not lines[k].strip():
                k -= 1
            if k >= 0:
                above = lines[k].strip()
                if not _looks_like_row(above) and len(above) < 120:
                    caption = above.strip("*_# ")
            tables.append(Table(header=header, rows=rows, page_no=page_no, caption=caption))
        i = max(j, i + 1)

    return tables


def table_to_markdown(table: Table) -> str:
    """Render a table back to a Markdown pipe table.

    Office loaders parse tables into structure directly, which leaves the
    page's text without them -- so a spreadsheet-only slide would export an
    empty ``document.md``. Rendering them back into the text keeps every export
    complete. Round-tripping is safe: :func:`strip_tables` removes them again
    for the Word and summary paths, so nothing is duplicated.
    """
    header = table.header or ["col %d" % (i + 1) for i in range(table.shape[1])]
    width = len(header)

    def row(cells: list[str]) -> str:
        padded = (list(cells) + [""] * width)[:width]
        return "| " + " | ".join(c.replace("|", r"\|").strip() for c in padded) + " |"

    lines = []
    if table.caption:
        lines.append("**%s**" % table.caption)
        lines.append("")
    lines.append(row(header))
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend(row(r) for r in table.rows)
    return "\n".join(lines)


def strip_tables(text: str) -> str:
    """The same text with pipe-table blocks removed.

    Used for summarisation, where a wall of table rows crowds out the prose
    the summary should actually be about.
    """
    text = re.sub(r"<table\b.*?</table>", "", text, flags=re.DOTALL | re.IGNORECASE)
    lines = text.splitlines()
    keep: list[str] = []
    i = 0
    while i < len(lines):
        if (
            _looks_like_row(lines[i])
            and i + 1 < len(lines)
            and _SEPARATOR.match(lines[i + 1])
        ):
            i += 2
            while i < len(lines) and _looks_like_row(lines[i]):
                i += 1
            continue
        keep.append(lines[i])
        i += 1
    return "\n".join(keep)


def to_dataframe(table: Table):
    """pandas DataFrame view of a table, for callers that want one."""
    import pandas as pd

    if table.header:
        width = len(table.header)
        rows = [list(r) + [""] * (width - len(r)) for r in table.rows]
        return pd.DataFrame(rows, columns=_unique(table.header))
    return pd.DataFrame(table.rows)


def _unique(names: list[str]) -> list[str]:
    """Disambiguate duplicate or blank column names for pandas/openpyxl."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for idx, name in enumerate(names):
        base = name.strip() or ("col_%d" % idx)
        if base in seen:
            seen[base] += 1
            base = "%s_%d" % (base, seen[base])
        else:
            seen[base] = 0
        out.append(base)
    return out
