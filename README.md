---
title: Multi Line OCR Document Agent
emoji: 📚
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.14.0
python_version: "3.12"
hardware: zero-gpu
app_file: app.py
pinned: false
license: apache-2.0
short_description: An agent that reads any document and returns structured output
tags:
  - ocr
  - document-parsing
  - agent
  - mcp-server
---

# Document Agent

Drop in a scan, a photo, a PDF, an Office file or a ZIP of pages, and say what
you want out of it — or say nothing and let the agent decide. It works out what
the document is, reads it with the right model, and writes the structured output
that suits it.

This started life as a line-splitting TrOCR demo. TrOCR is still here, as the
handwriting backend.

## What it does

| You give it | It gives you |
| --- | --- |
| An invoice or receipt | Line items in Excel/CSV, header fields as JSON |
| A ruled table, a spreadsheet scan | One sheet per table, headers frozen |
| A form or an ID document | Key-value pairs as JSON |
| A report or an article | An abstractive summary as a PDF report |
| A handwritten note | Text via TrOCR, with line coordinates |
| Anything at all | Markdown, a searchable PDF, a Word document, and a manifest |

Every run also produces `output.zip` containing all of the above plus
`manifest.json`, which records how each page was read, the confidence, and
anything worth double-checking. The agent never blocks to ask a question and
never fails outright — if it is unsure, it says so in the manifest and in its
reply, and you still get the artifacts.

### Input formats

PNG, JPEG, WebP, BMP, GIF, multi-frame TIFF, HEIC/HEIF, PDF (scanned or
digital), DOCX, PPTX, XLSX/XLSM, TXT, Markdown, CSV, and ZIP archives of any of
those.

## How the agent works

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
   reliable — see the note on GLM-OCR below.
4. **Recover tables.** Text that already contains table markup is parsed for
   free. A page image that looks tabular gets a second, structure-aware model
   pass; a page of prose never pays for one.
5. **Route or plan.** If the page types point clearly at one output and you did
   not ask for anything specific, a deterministic router picks the tool sequence
   and the planner LLM never runs. Mixed documents and free-text requests go to
   the planner, which calls tools one at a time and reads each observation.
6. **Export** writes the artifacts.

Steps 2–5 happen inside a **single** `@spaces.GPU` session whose duration scales
with the number of pages actually needing OCR. One session per tool would
multiply queue waits and burn quota on scheduling rather than work. Export is
deliberately left *outside* that session: writing a workbook, a Word file and
two PDFs is seconds of pure CPU that would otherwise be billed as GPU time.

## Models

| Role | Model | Params | License |
| --- | --- | --- | --- |
| Reading | [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) | 1.3B | MIT |
| Planning, summarising | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | 4B | Apache-2.0 |
| Handwriting | [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten) | 333M | MIT |

GLM-OCR ranks first on OmniDocBench V1.5 at a size that fits comfortably in a
ZeroGPU slice alongside the other two (~12GB of 48GB).

It is worth knowing that GLM-OCR is **prompt-limited** rather than a general
instruction-following VLM. Its card defines three document-parsing prompts —
`Text Recognition:`, `Table Recognition:`, `Formula Recognition:` — plus an
information-extraction mode that requires a literal JSON skeleton to fill in.
Two things follow, both verified against the model here:

- Reading a table with `Text Recognition:` returns space-aligned plain text with
  no structure to export, so tabular pages get a second `Table Recognition:`
  pass, which returns a real HTML `<table>`.
- The model has no classification mode, so page types are inferred from the
  recognised text on the CPU instead. That is why reading comes before
  classifying in the pipeline above.

## GPU quota

ZeroGPU gives each visitor roughly **5 minutes of GPU per day** on a free
account, 2 minutes signed out, and 40 minutes on PRO. This shapes the design:

- Digital PDFs and Office files cost **zero** GPU — their text layer is used as-is.
- The deterministic router handles unambiguous documents with **no planner tokens**.
- GPU reservation scales with page count rather than being fixed.

The "GPU" column in the app's agent trace reports work that *actually* happened,
not work a tool might have done.

## Running it locally

```bash
pip install -r requirements.txt
python app.py
```

Without a GPU everything still runs — `@spaces.GPU` is a no-op off-Space and the
models fall back to CPU. It is slow, but the whole pipeline is exercisable. With
`DOCAGENT_DISABLE_MODELS=1` the models are skipped entirely and the CPU paths
(ingest, table parsing, extractive summary, every exporter) run on their own.

```bash
python tests/make_fixtures.py   # generate the fixture matrix
python -m pytest tests -q
```

## Using it as an MCP server

The Space is also an MCP server. Point a client at:

```
https://<your-space-host>/gradio_api/mcp/
```

That is the streamable-HTTP transport used by Gradio 6; the trailing slash
matters, as the bare path 307-redirects. Gradio 5 served MCP over SSE at
`/gradio_api/mcp/sse` instead.

Tools: `process_document`, `ocr_document`, `classify_document`,
`extract_tables`, `extract_fields`, `summarize_document`, `document_to_xlsx`.

`process_document` is the agentic one — give it a file and an optional
plain-English instruction and it decides the rest. The others are deterministic
if you already know what you want.

## As a Python library

```python
from docagent import process

doc = process("invoice.pdf", "pull the line items into a spreadsheet")
print(doc.dominant_kind().value)
for artifact in doc.artifacts:
    print(artifact.kind, artifact.path)
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCAGENT_OCR_MODEL` | `zai-org/GLM-OCR` | Reading model |
| `DOCAGENT_BRAIN` | `Qwen/Qwen3.5-4B` | Planner; set to `Qwen/Qwen3.5-2B` to halve latency and quota |
| `DOCAGENT_TROCR_MODEL` | `microsoft/trocr-base-handwritten` | Handwriting model |
| `DOCAGENT_DEVICE` | auto | Force `cpu` or `cuda` |
| `DOCAGENT_DISABLE_MODELS` | unset | Set to `1` to run CPU-only paths |
| `DOCAGENT_MAX_NEW_TOKENS` | `4096` | Cap per page read |
| `DOCAGENT_TROCR_BATCH` | `8` | TrOCR line batch size |
| `DOCAGENT_LOG_LEVEL` | `INFO` | Logging verbosity |

### Deploying to ZeroGPU

Free accounts in good standing (verified email, older than 30 days) can host two
ZeroGPU Spaces. Select **ZeroGPU** in Space settings; nothing in the code needs
changing. To avoid re-downloading ~12GB on every restart, attach a storage
bucket at `/data` and point `HF_HOME` and `HF_HUB_CACHE` at it.

## Known limitations

- GLM-OCR runs coordinate-free, so the searchable PDF positions text by line for
  those pages. It is fully searchable and copy-pasteable, but character
  positions are approximate. TrOCR pages and PDF text layers get real
  coordinates.
- Page classification is keyword- and layout-heuristic rather than a model
  judgement, so its confidence is reported conservatively and shows up in the
  manifest. A page whose type is guessed wrong still gets the full bundle.
- Handwriting is routed to TrOCR when a page is already known to be handwritten,
  or when GLM-OCR returns almost nothing for a page that visibly has text lines.
  You can also force a backend in the UI.
- Summaries fall back to extractive sentence ranking when the planner model is
  unavailable, and say so in the manifest.
- Fetching documents from a URL is deliberately not supported: server-side
  fetching of arbitrary user-supplied URLs is an SSRF surface.
