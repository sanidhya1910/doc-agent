---
title: Doc Agent
emoji: 📄
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.14.0
python_version: "3.12"
hardware: zero-gpu
app_file: app.py
pinned: false
license: apache-2.0
short_description: Any document in; spreadsheet, JSON or summary out
tags:
  - ocr
  - document-parsing
  - agent
  - mcp-server
---

# Doc Agent

**Drop in any document. Get back the structured output that actually suits it.**

Not "here is the text, good luck" — a spreadsheet when the page is a table,
JSON when it is a form, a summary PDF when it is prose. You can say what you
want in plain English, or say nothing and let the agent decide.

It is also an [MCP](#use-it-from-another-agent) server, so Claude, Cursor or any
MCP client can call it as a tool.

---

## What you get back

| You give it | It gives you |
| --- | --- |
| An invoice or receipt | Line items in Excel and CSV, header fields as JSON |
| A ruled table, a spreadsheet scan | One sheet per table, headers frozen |
| A form or an ID document | Key-value pairs as JSON |
| A report, an article, a letter | An abstractive summary as a PDF report |
| A handwritten note | Text via TrOCR, with line coordinates |
| **Anything at all** | Markdown, a searchable PDF, a Word document, and a manifest |

Every run also writes `output.zip` with all of the above plus `manifest.json`,
recording how each page was read, how confident it was, and anything worth a
second look.

### It never leaves you empty-handed

When the agent is unsure it does not block to ask a question and it does not
fail. It writes everything that plausibly applies, then tells you what it was
unsure about — in its reply and in the manifest:

```
Read 3 page(s) and pulled 2 table(s) into a spreadsheet.
Worth checking: p1 read as invoice (confidence 0.68).
```

A visitor who closes the tab still leaves with usable files.

### What it accepts

Images (PNG, JPEG, WebP, BMP, GIF), multi-frame TIFF, **HEIC** straight off a
phone, PDF (scanned or digital), DOCX, PPTX, XLSX, TXT, Markdown, CSV, and ZIP
archives of any of those. Multi-page and multi-file throughout.

---

## How it works

```
ingest ─► read ─► classify ─► tables ─┬─► router  (deterministic, no LLM) ─┐
         (GPU)     (CPU)     (GPU if  │                                    ├─► export
                             tabular) └─► planner (Qwen3.5 tool loop)     ─┘   (CPU)
```

1. **Ingest** builds a page list. PDFs and Office files that already carry a
   text layer are marked as such and **never reach the GPU**.
2. **Read** runs OCR, but only on pages that need it.
3. **Classify** infers each page type *from the recognised text*, on the CPU.
   Doing it after reading rather than before makes it both free and more
   reliable — see [the note on GLM-OCR](#a-note-on-glm-ocr).
4. **Recover tables.** Text that already contains table markup is parsed for
   free. A page image that looks tabular gets a second, structure-aware model
   pass; a page of prose never pays for one.
5. **Route or plan.** If the page types point clearly at one output and you
   asked for nothing specific, a deterministic router picks the tool sequence
   and **the planner LLM never runs**. Mixed documents and free-text requests go
   to the planner, which calls tools one at a time and reads each observation.
6. **Export** writes the artifacts.

### The agent's tools

The planner drives a real tool loop. Anything marked GPU is the only thing that
can cost you quota:

| Tool | GPU | Purpose |
| --- | :-: | --- |
| `list_pages` | | Page count, types, confidence, text-layer status |
| `read_pages` | ● | OCR, choosing GLM-OCR or TrOCR per page |
| `classify_pages` | | Infer page types from the text |
| `extract_tables` | ◐ | Parse table markup; structure pass on tabular images |
| `extract_fields` | ● | Key-value pairs from forms and invoices |
| `summarize` | ● | Abstractive summary |
| `write_*` | | xlsx, csv, docx, pdf report, searchable pdf, json, markdown |
| `finish` | | End the loop with a one-sentence reply |

You can watch it work — the app shows the full trace, including which steps
actually touched the GPU.

---

## Models

| Role | Model | Params | License |
| --- | --- | --- | --- |
| Reading | [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) | 1.3B | MIT |
| Planning, summarising | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B | Apache-2.0 |
| Handwriting | [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten) | 333M | MIT |

GLM-OCR ranks first on OmniDocBench V1.5 at a size that fits comfortably in a
ZeroGPU slice alongside the other two (~12GB of 48GB).

### A note on GLM-OCR

GLM-OCR is **prompt-limited** rather than a general instruction-following VLM.
Its card defines three document-parsing prompts — `Text Recognition:`,
`Table Recognition:`, `Formula Recognition:` — plus an information-extraction
mode that needs a literal JSON skeleton to fill in. Two things follow, both
verified against the model rather than assumed:

- Reading a table with `Text Recognition:` returns space-aligned plain text with
  no structure to export, so tabular pages get a second `Table Recognition:`
  pass, which returns a real HTML `<table>`.
- The model has no classification mode, so page types are inferred from the
  recognised text on the CPU instead. That is why reading comes before
  classifying in the pipeline above.

---

## Being careful with your GPU quota

ZeroGPU gives each visitor about **5 minutes of GPU per day** on a free account
(2 minutes signed out, 40 on PRO). That budget shaped the design:

- **Digital PDFs and Office files cost zero GPU.** Their text layer is used
  as-is — this is the single largest saving.
- **The router handles unambiguous documents with no planner tokens at all.**
- **One GPU session per run**, sized by the number of pages actually needing
  OCR, not one session per tool — which would burn the budget on queue waits.
- **Export happens outside the session.** Writing a workbook, a Word file and
  two PDFs is seconds of pure CPU that would otherwise be billed as GPU time.

The "GPU" column in the trace reports work that *actually happened*, not work a
tool might have done — a step that fell back to CPU is not counted against you.

---

## Use it from another agent

The Space is an MCP server. Point a client at:

```
https://sanidhya1910-doc-agent.hf.space/gradio_api/mcp/
```

The trailing slash matters (the bare path 307-redirects). That is Gradio 6's
streamable-HTTP transport; Gradio 5 served MCP over SSE at
`/gradio_api/mcp/sse`.

| Tool | What it does |
| --- | --- |
| `process_document` | The agentic one — give it a file and an optional instruction, it decides the rest |
| `ocr_document` | Full text as Markdown, tables included |
| `classify_document` | What the document is, per page, with confidence |
| `extract_tables` | Every table as JSON rows and records |
| `extract_fields` | Key-value pairs, optionally constrained to keys you name |
| `summarize_document` | Plain-prose summary |
| `document_to_xlsx` | Returns a populated workbook |

---

## Use it as a library

```python
from docagent import process

doc = process("invoice.pdf", "pull the line items into a spreadsheet")

print(doc.dominant_kind().value)        # "invoice"
for table in doc.all_tables():
    print(table.header, table.rows)
for artifact in doc.artifacts:
    print(artifact.kind, artifact.path)
```

Lower-level pieces are available too:

```python
from docagent import load
from docagent.tools import perceive, call_tool

doc = load(["scan1.png", "scan2.png"])
perceive(doc)                                  # read, classify, find tables
call_tool(doc, "extract_fields", {"keys": ["total", "date"]})
```

---

## Running it yourself

```bash
pip install -r requirements.txt
python app.py
```

Without a GPU everything still runs — `@spaces.GPU` is a no-op off-Space and the
models fall back to CPU. Slow, but the whole pipeline is exercisable.

```bash
python -m pytest tests -q
```

The suite runs with `DOCAGENT_DISABLE_MODELS=1`, so it needs **no weights and no
network**: 135 tests covering ingest, table recovery, routing, GPU accounting,
every exporter and the MCP surface. Fixtures are synthesised locally by
`tests/make_fixtures.py` (generated automatically on first run).

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCAGENT_OCR_MODEL` | `zai-org/GLM-OCR` | Reading model |
| `DOCAGENT_BRAIN` | `Qwen/Qwen3.5-4B` | Planner; `Qwen/Qwen3.5-2B` halves latency and quota |
| `DOCAGENT_TROCR_MODEL` | `microsoft/trocr-base-handwritten` | Handwriting model |
| `DOCAGENT_DEVICE` | auto | Force `cpu` or `cuda` |
| `DOCAGENT_DISABLE_MODELS` | unset | `1` runs the CPU-only paths |
| `DOCAGENT_MAX_NEW_TOKENS` | `4096` | Cap per page read |
| `DOCAGENT_TROCR_BATCH` | `8` | TrOCR line batch size |
| `DOCAGENT_LOG_LEVEL` | `INFO` | Logging verbosity |

### Deploying to ZeroGPU

Free accounts in good standing (verified email, older than 30 days) can host two
ZeroGPU Spaces. Select **ZeroGPU** in the Space settings — the `hardware:` key in
this README is only a hint. Nothing in the code needs changing.

To avoid re-downloading ~12GB on every restart, attach a storage bucket at
`/data` and point `HF_HOME` and `HF_HUB_CACHE` at it.

Pushes to `main` on GitHub sync to the Space automatically via
`.github/workflows/sync-to-hub.yml`, which needs an `HF_TOKEN` repository secret.

---

## Where it came from

This began as a line-splitting TrOCR demo: dilate, find contours, run TrOCR on
each line, return a flat string. That model is still here as the handwriting
backend, with its three real weaknesses fixed — the morphology kernel now scales
with measured glyph height instead of being hardcoded to 30px, merged lines are
recovered with a projection profile, and recognition is batched instead of
looping in Python.

## Known limitations

- GLM-OCR runs coordinate-free, so the searchable PDF positions text by line on
  those pages. Fully searchable and copy-pasteable, but character positions are
  approximate. TrOCR pages and PDF text layers get real coordinates.
- Page classification is keyword- and layout-heuristic rather than a model
  judgement. Its confidence is reported conservatively and surfaces in the
  manifest; a page typed wrongly still gets the full bundle.
- Handwriting routes to TrOCR when a page is already known to be handwritten, or
  when GLM-OCR returns almost nothing for a page that visibly has text lines.
  You can also force a backend in the UI.
- Summaries fall back to extractive sentence ranking if the planner model is
  unavailable, and say so in the manifest.
- Fetching documents from a URL is deliberately unsupported: server-side
  fetching of arbitrary user-supplied URLs is an SSRF surface.

## License

Apache-2.0. The models carry their own licenses — MIT for GLM-OCR and TrOCR,
Apache-2.0 for Qwen3.5.
