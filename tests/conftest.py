"""Shared test fixtures.

Models are disabled for the whole suite: these tests exercise ingest, table
parsing, routing, GPU accounting and every exporter, none of which need
weights.  That keeps CI from downloading ~12GB.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DOCAGENT_DISABLE_MODELS", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config: object) -> None:
    """Build the fixture files on first run so the suite is self-contained."""
    if not (FIXTURES / "digital.pdf").exists():
        from tests import make_fixtures

        make_fixtures.main()


@pytest.fixture
def fixture_path():
    def _get(name: str) -> str:
        path = FIXTURES / name
        if not path.exists():
            pytest.skip("fixture %s missing; run tests/make_fixtures.py" % name)
        return str(path)

    return _get


@pytest.fixture
def workdir(tmp_path):
    return str(tmp_path)
