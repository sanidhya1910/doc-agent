"""Model loading, ZeroGPU wiring, and the GPU-session decorator.

ZeroGPU requires that models are moved to ``cuda`` at *module level*: a CUDA
emulation layer is active outside ``@spaces.GPU`` functions, and placements
done at import time are far cheaper than transfers made inside a GPU function.
So on a ZeroGPU Space we eagerly construct everything at import
(:func:`preload`), while local development stays lazy so that tests and CPU
runs do not pull ~12GB of weights.

Env vars
--------
``DOCAGENT_OCR_MODEL``    perception model (default ``zai-org/GLM-OCR``)
``DOCAGENT_BRAIN``        planner/summariser (default ``Qwen/Qwen3.5-4B``)
``DOCAGENT_TROCR_MODEL``  handwriting model (default ``microsoft/trocr-base-handwritten``)
``DOCAGENT_DISABLE_MODELS``  set to ``1`` to hard-disable model loading (tests)
``DOCAGENT_DEVICE``       force ``cpu`` / ``cuda``
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from typing import Any, Callable

log = logging.getLogger("docagent.models")

OCR_MODEL_ID = os.environ.get("DOCAGENT_OCR_MODEL", "zai-org/GLM-OCR")
BRAIN_MODEL_ID = os.environ.get("DOCAGENT_BRAIN", "Qwen/Qwen3.5-4B")
TROCR_MODEL_ID = os.environ.get("DOCAGENT_TROCR_MODEL", "microsoft/trocr-base-handwritten")

_MODELS_DISABLED = os.environ.get("DOCAGENT_DISABLE_MODELS", "") == "1"


#: Incremented every time a model actually runs. The agent trace snapshots it
#: around each tool call so the manifest reports GPU work that really happened
#: rather than work a tool might have done -- which is what makes the
#: "digital PDFs cost zero GPU" claim checkable.
_GPU_CALLS = 0


def note_model_run() -> None:
    """Record that a model was actually invoked."""
    global _GPU_CALLS
    _GPU_CALLS += 1


def model_run_count() -> int:
    return _GPU_CALLS


class ModelsUnavailable(RuntimeError):
    """Raised when a model is needed but cannot be loaded.

    Callers catch this and degrade rather than crash, so the app still starts
    and the CPU-only parts of the pipeline stay usable.
    """


# --------------------------------------------------------------------------
# environment detection
# --------------------------------------------------------------------------

def is_zerogpu() -> bool:
    """True when running on a Hugging Face ZeroGPU Space."""
    return bool(os.environ.get("SPACES_ZERO_GPU") or os.environ.get("ZERO_GPU"))


@functools.lru_cache(maxsize=1)
def device() -> str:
    forced = os.environ.get("DOCAGENT_DEVICE")
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available() or is_zerogpu():
            return "cuda"
    except Exception:  # pragma: no cover - torch always present in practice
        pass
    return "cpu"


@functools.lru_cache(maxsize=1)
def torch_dtype() -> Any:
    import torch

    return torch.bfloat16 if device() == "cuda" else torch.float32


# --------------------------------------------------------------------------
# GPU session decorator
# --------------------------------------------------------------------------

def gpu_task(duration: int | Callable[..., int] = 60) -> Callable:
    """Wrap a function in ``spaces.GPU`` when available, else pass through.

    ``duration`` may be a callable taking the same arguments as the wrapped
    function, which is how a multi-page run asks for more than the default 60s
    without over-reserving for a single page.
    """

    def decorator(fn: Callable) -> Callable:
        if not is_zerogpu():
            return fn
        try:
            import spaces
        except ImportError:  # pragma: no cover
            log.warning("spaces not importable; running %s without a GPU session", fn.__name__)
            return fn
        return spaces.GPU(duration=duration)(fn)

    return decorator


def estimate_duration(doc: Any, *_args: Any, **_kwargs: Any) -> int:
    """Seconds of GPU to reserve for one agent run.

    Scales with the number of pages that actually need OCR -- a digital PDF
    with a text layer costs nothing beyond the planner turn.  Capped at 300s
    because a longer reservation only worsens queue position.
    """
    try:
        pending = len(doc.pages_needing_ocr())
    except Exception:
        pending = 1
    return int(min(30 + 12 * max(pending, 1), 300))


# --------------------------------------------------------------------------
# the model hub
# --------------------------------------------------------------------------

class _Hub:
    """Lazily-built, process-wide singletons for the three models."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, Any] = {}
        self._failed: dict[str, str] = {}

    def _get(self, key: str, builder: Callable[[], Any]) -> Any:
        if _MODELS_DISABLED:
            raise ModelsUnavailable("model loading disabled via DOCAGENT_DISABLE_MODELS")
        if key in self._cache:
            return self._cache[key]
        if key in self._failed:
            raise ModelsUnavailable(self._failed[key])
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            try:
                log.info("loading %s ...", key)
                self._cache[key] = builder()
            except Exception as exc:
                msg = "could not load %s: %s" % (key, exc)
                log.warning(msg)
                self._failed[key] = msg
                raise ModelsUnavailable(msg) from exc
        return self._cache[key]

    def available(self, key: str) -> bool:
        """Whether a model can be used, without forcing a load attempt."""
        if _MODELS_DISABLED:
            return False
        return key in self._cache or key not in self._failed

    # -- perception ----------------------------------------------------

    def glm_ocr(self) -> tuple[Any, Any]:
        """(processor, model) for GLM-OCR, the primary perception model."""

        def build() -> tuple[Any, Any]:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            processor = AutoProcessor.from_pretrained(OCR_MODEL_ID)
            model = AutoModelForImageTextToText.from_pretrained(
                OCR_MODEL_ID, dtype=torch_dtype()
            )
            model.to(device())
            model.eval()
            return processor, model

        return self._get("glm_ocr", build)

    # -- handwriting ---------------------------------------------------

    def trocr(self) -> tuple[Any, Any]:
        """(processor, model) for TrOCR.

        Loaded from the Hub by id.  The original app called
        ``from_pretrained(".")``, which looked for weights in the repo root
        where none were ever committed -- that is why the Space crashed on
        boot.
        """

        def build() -> tuple[Any, Any]:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
            model = VisionEncoderDecoderModel.from_pretrained(
                TROCR_MODEL_ID, dtype=torch_dtype()
            )
            model.to(device())
            model.eval()
            return processor, model

        return self._get("trocr", build)

    # -- planner / summariser -------------------------------------------

    def brain(self) -> tuple[Any, Any]:
        """(processor, model) for the planning + summarising LLM."""

        def build() -> tuple[Any, Any]:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            processor = AutoProcessor.from_pretrained(BRAIN_MODEL_ID)
            model = AutoModelForImageTextToText.from_pretrained(
                BRAIN_MODEL_ID, dtype=torch_dtype()
            )
            model.to(device())
            model.eval()
            return processor, model

        return self._get("brain", build)


HUB = _Hub()


def preload(ocr: bool = True, brain: bool = True, trocr: bool = False) -> dict[str, bool]:
    """Build models at import time, as ZeroGPU wants.

    Failures are swallowed and reported in the return value: a Space that
    cannot reach one model should still boot and serve the parts that work.
    TrOCR is off by default since only handwriting pages need it, and it is
    cheap enough to fetch on first use.
    """
    status: dict[str, bool] = {}
    for name, enabled, getter in (
        ("glm_ocr", ocr, HUB.glm_ocr),
        ("brain", brain, HUB.brain),
        ("trocr", trocr, HUB.trocr),
    ):
        if not enabled:
            continue
        try:
            getter()
            status[name] = True
        except ModelsUnavailable:
            status[name] = False
    return status
