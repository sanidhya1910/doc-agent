"""Validate the Hugging Face Space configuration before it reaches the Hub.

The Hub validates README front matter in a pre-receive hook, so a bad value
does not fail at build time -- it rejects the push outright, after CI has run
and after the token has authenticated. These checks move that feedback here.

Every limit asserted below is one the Hub actually enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

#: Enforced by the Hub's pre-receive hook. Exceeding it rejects the push with
#: "short_description length must be less than or equal to 60 characters long".
MAX_SHORT_DESCRIPTION = 60


@pytest.fixture(scope="module")
def front_matter() -> dict:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "README.md must open with a YAML front-matter block"
    data = yaml.safe_load(match.group(1))
    assert isinstance(data, dict), "front matter must be a mapping"
    return data


def test_short_description_fits_the_hub_limit(front_matter):
    description = front_matter.get("short_description", "")
    assert description, "short_description is what the Space card shows"
    assert len(description) <= MAX_SHORT_DESCRIPTION, (
        "short_description is %d characters; the Hub rejects anything over %d"
        % (len(description), MAX_SHORT_DESCRIPTION)
    )


def test_required_space_keys_are_present(front_matter):
    for key in ("title", "emoji", "sdk", "app_file", "license"):
        assert front_matter.get(key), "front matter is missing %r" % key


def test_sdk_is_gradio_because_zerogpu_supports_nothing_else(front_matter):
    assert front_matter["sdk"] == "gradio"


def test_sdk_version_looks_like_a_version(front_matter):
    version = str(front_matter.get("sdk_version", ""))
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", version), (
        "sdk_version %r is not a version the Hub will accept" % version
    )


def test_python_version_is_one_zerogpu_supports(front_matter):
    """ZeroGPU images ship 3.12.12 and 3.10.13; anything else will not build."""
    assert str(front_matter.get("python_version", "")) in ("3.10", "3.12")


def test_app_file_exists(front_matter):
    assert (ROOT / front_matter["app_file"]).is_file()


def test_declared_requirements_file_exists():
    assert (ROOT / "requirements.txt").is_file()


# ---------------------------------------------------------------------------
# the sync workflow
# ---------------------------------------------------------------------------

WORKFLOW = ROOT / ".github" / "workflows" / "sync-to-hub.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW.is_file(), "the Space sync workflow is missing"
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_pushes_head_not_a_bare_branch_name():
    """``git push <url> main`` needs a local ref named main.

    actions/checkout only creates the branch that triggered the run, so a bare
    "main" refspec breaks manual dispatch from any other branch.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "HEAD:main" in text
    assert not re.search(r"^\s+main\s*$", text, re.MULTILINE)


def test_workflow_does_not_hardcode_a_token():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "secrets.HF_TOKEN" in text
    assert "hf_" not in text.lower().replace("hf_token", "")


def test_workflow_targets_the_right_space():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "huggingface.co/spaces/sanidhya1910/doc-agent" in text
    # The placeholder the original template shipped with.
    assert "HF_USERNAME" not in text and "SPACE_NAME" not in text


def test_workflow_fetches_full_history_and_lfs(workflow):
    """A shallow clone cannot be pushed to the Space, which is its own repo."""
    checkout = next(
        s for s in workflow["jobs"]["sync-to-hub"]["steps"]
        if str(s.get("uses", "")).startswith("actions/checkout")
    )
    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["lfs"] is True


def test_checkout_action_is_on_a_supported_node_runtime(workflow):
    """v4 targets Node 20, which GitHub deprecated."""
    checkout = next(
        s for s in workflow["jobs"]["sync-to-hub"]["steps"]
        if str(s.get("uses", "")).startswith("actions/checkout")
    )
    major = int(re.search(r"@v(\d+)", checkout["uses"]).group(1))
    assert major >= 5, "actions/checkout v%d runs on the deprecated Node 20" % major
