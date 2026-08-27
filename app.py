"""Multi-line OCR document agent -- Gradio app and MCP server.

Drop in any document and the agent decides what it is, how to read it, and
which structured artifacts to produce. The same tools are published over MCP
so external agents can call this Space directly.

On ZeroGPU the models are built at import time, which is what the platform
wants: a CUDA emulation layer is active outside ``@spaces.GPU`` functions and
placements made at startup are far cheaper than transfers inside a GPU call.
"""

from __future__ import annotations

import logging
import os
import tempfile

import gradio as gr

import mcp_api
from docagent import __version__, load, run_stream, supported_extensions
from docagent.export import build_bundle
from docagent.models import BRAIN_MODEL_ID, OCR_MODEL_ID, TROCR_MODEL_ID, is_zerogpu, preload

logging.basicConfig(
    level=os.environ.get("DOCAGENT_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("app")

# ZeroGPU requires module-level construction. Off-Space we stay lazy so local
# runs and tests do not pull ~12GB of weights.
MODEL_STATUS: dict[str, bool] = {}
if is_zerogpu():
    MODEL_STATUS = preload(ocr=True, brain=True, trocr=False)
    log.info("preloaded models: %s", MODEL_STATUS)

ACCEPTED = supported_extensions()

INTRO = """\
Drop in a scan, a photo, a PDF, an Office file or a ZIP of pages, and say what
you want out of it -- or say nothing and let the agent decide. It works out
what the document is, reads it with the right model, and writes the structured
output that suits it: a spreadsheet for tabular data, JSON for a form, a
summary PDF for prose. You always get the full bundle plus a manifest saying
what it was unsure about.
"""

EXAMPLES = [
    "Pull the line items into a spreadsheet.",
    "Summarise this and give me a PDF.",
    "Extract the invoice number, date and total as JSON.",
    "Just give me the plain text.",
]


def _format_trace(doc) -> str:
    """Render the tool trace as Markdown.

    Read from ``doc.trace`` rather than the streamed events because that is
    where the *actual* GPU usage is recorded -- a tool that fell back to CPU
    must not be shown as having spent quota.
    """
    if not doc.trace:
        return "_No tool calls recorded._"
    lines = ["| step | tool | GPU | result |", "| ---: | --- | :-: | --- |"]
    for i, entry in enumerate(doc.trace, start=1):
        result = " ".join(str(entry.get("result", "")).split())
        if len(result) > 160:
            result = result[:160] + "..."
        lines.append(
            "| %d | `%s` | %s | %s |"
            % (i, entry.get("tool", "?"), "yes" if entry.get("gpu") else "", result)
        )
    gpu_steps = sum(1 for e in doc.trace if e.get("gpu"))
    lines.append("")
    lines.append("_%d of %d steps used the GPU._" % (gpu_steps, len(doc.trace)))
    return "\n".join(lines)


def _artifact_rows(doc) -> str:
    if not doc.artifacts:
        return "_No artifacts written._"
    lines = ["| file | what it is |", "| --- | --- |"]
    for a in doc.artifacts:
        detail = (" - " + a.detail) if a.detail else ""
        lines.append("| `%s` | %s%s |" % (os.path.basename(a.path), a.label, detail))
    return "\n".join(lines)


def handle(files, instruction, backend, dpi, deskew):
    """Main event handler. Yields progressively so the UI stays responsive."""
    if not files:
        yield (
            "Add at least one file first.",
            "_Nothing to do._",
            "_No artifacts._",
            None,
            None,
        )
        return

    yield ("Reading input...", "_Working..._", "_No artifacts yet._", None, None)

    work = tempfile.mkdtemp(prefix="docagent_ui_")
    doc = load(files, instruction=instruction or "", dpi=int(dpi), workdir=work)
    doc.workdir = work

    if not doc.pages:
        note = "Could not read any pages. " + "; ".join(doc.warnings)
        yield (note, "_Nothing to do._", "_No artifacts._", None, None)
        return

    gallery = [(p.image, "p%d - %s" % (p.page_no, p.source)) for p in doc.pages if p.image]

    yield (
        "Processing %d page(s)..." % doc.n_pages,
        "_Working..._",
        "_No artifacts yet._",
        gallery or None,
        None,
    )

    doc.preprocess = {"deskew": bool(deskew), "denoise": bool(deskew)}
    doc.force_backend = "" if backend == "auto" else backend

    try:
        events = run_stream(doc)
        reply = next(
            (e.get("message", "") for e in reversed(events) if e.get("type") == "result"), ""
        )
    except Exception as exc:  # noqa: BLE001
        # The whole design is never to send someone away empty-handed, so even
        # an unexpected failure still writes whatever reached the blackboard.
        log.exception("agent run failed")
        doc.warnings.append("the run failed partway through: %s" % exc)
        try:
            build_bundle(doc, doc.workdir)
        except Exception:  # noqa: BLE001
            log.exception("could not salvage a bundle")
        reply = (
            "Something went wrong partway through: %s. Anything that was read "
            "before the failure is in the downloads below." % exc
        )

    downloads = [a.path for a in doc.artifacts if os.path.exists(a.path)]
    yield (
        reply or "Done.",
        _format_trace(doc),
        _artifact_rows(doc),
        gallery or None,
        downloads or None,
    )


with gr.Blocks(title="Document Agent (multi-line OCR)", fill_height=True) as demo:
    gr.Markdown("# Document Agent\n" + INTRO)

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.Files(
                label="Documents",
                file_types=ACCEPTED,
                file_count="multiple",
            )
            instruction = gr.Textbox(
                label="What do you want out of it?",
                placeholder="e.g. pull the line items into a spreadsheet",
                lines=2,
            )
            gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[instruction])
            go = gr.Button("Process", variant="primary")

            with gr.Accordion("Overrides", open=False):
                backend = gr.Radio(
                    ["auto", "glm_ocr", "trocr"],
                    value="auto",
                    label="OCR backend",
                    info="auto uses GLM-OCR for print and TrOCR for handwriting",
                )
                dpi = gr.Slider(
                    100, 400, value=200, step=25,
                    label="PDF render DPI",
                    info="higher is sharper and slower",
                )
                deskew = gr.Checkbox(value=True, label="Deskew and denoise pages")

        with gr.Column(scale=2):
            reply = gr.Markdown(label="Result", value="_Add a document to begin._")
            downloads = gr.Files(label="Download")
            with gr.Accordion("Agent trace", open=False):
                trace = gr.Markdown(value="_No run yet._")
            with gr.Accordion("Artifacts", open=False):
                artifacts = gr.Markdown(value="_No artifacts._")
            with gr.Accordion("Pages", open=False):
                gallery = gr.Gallery(label="Pages", columns=3, height=320)

    # api_visibility="private" keeps the UI handler out of the API and MCP
    # surfaces: it takes Gradio file objects and means nothing to an external
    # agent, which should call the tools in mcp_api instead.
    run_args = dict(
        fn=handle,
        inputs=[files, instruction, backend, dpi, deskew],
        outputs=[reply, trace, artifacts, gallery, downloads],
        api_visibility="private",
    )
    go.click(**run_args)
    instruction.submit(**run_args)

    gr.Markdown(
        "---\n"
        "**Models** - reading: `%s` - planning and summarising: `%s` - "
        "handwriting: `%s` (v%s)\n\n"
        "**MCP** - this Space is also an MCP server. Point a client at "
        "`/gradio_api/mcp/` (trailing slash) to call `process_document`, `ocr_document`, "
        "`extract_tables`, `extract_fields`, `summarize_document`, "
        "`classify_document` and `document_to_xlsx` directly.\n\n"
        "**GPU quota** - ZeroGPU gives each visitor about 5 minutes of GPU per "
        "day (2 minutes signed out). Documents whose pages already carry a text "
        "layer, such as digital PDFs and Office files, are handled entirely on "
        "CPU and cost nothing."
        % (OCR_MODEL_ID, BRAIN_MODEL_ID, TROCR_MODEL_ID, __version__)
    )

    # Publish the MCP tools. Registered inside the Blocks context, exposed by
    # launch(mcp_server=True).
    mcp_api.register()


if __name__ == "__main__":
    demo.launch(mcp_server=True)
