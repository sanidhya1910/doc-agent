"""An agentic document-processing pipeline.

Drop in anything -- image, PDF, Office file, ZIP -- and the agent works out
what it is, reads it with the right model, and produces the structured output
that suits it: a spreadsheet for tabular data, JSON for a form, a summary PDF
for prose.

Typical use::

    from docagent import process
    doc = process("invoice.pdf", "pull the line items into a spreadsheet")
    for artifact in doc.artifacts:
        print(artifact.kind, artifact.path)
"""

from .agent import process, run, run_stream
from .ingest import load, supported_extensions
from .state import Artifact, Block, Box, DocKind, Document, Field, Page, Table

__version__ = "0.2.0"

__all__ = [
    "process",
    "run",
    "run_stream",
    "load",
    "supported_extensions",
    "Artifact",
    "Block",
    "Box",
    "DocKind",
    "Document",
    "Field",
    "Page",
    "Table",
    "__version__",
]
