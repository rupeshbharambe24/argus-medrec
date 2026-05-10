"""Shared pytest fixtures."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# Disable LLM for tests unless explicitly overridden — keeps CI hermetic
os.environ.setdefault("GEMINI_API_KEY", "")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def synthea_bundle(fixtures_dir: Path) -> dict:
    """Tiny Synthea-style bundle for unit testing."""
    with (fixtures_dir / "patient_bundle.json").open() as f:
        return json.load(f)


@pytest.fixture()
def temp_kb_path(tmp_path: Path, monkeypatch) -> Path:
    """Isolated reference KB for each test."""
    path = tmp_path / "reference.sqlite"
    monkeypatch.setenv("ARGUS_REFERENCE_KB_PATH", str(path))
    # Ensure get_settings cache is reset for this test
    from argus.config import get_settings

    get_settings.cache_clear()
    # Build the KB
    from argus.reference.build_kb import build

    build(path)
    yield path
    get_settings.cache_clear()


@pytest.fixture()
def sharp_context_dev():
    from argus.sharp_context import SharpContext

    return SharpContext(
        fhir_base_url="https://test.fhir.example/baseR4",
        fhir_token="test-token",
        patient_id="test-patient-1",
        encounter_id=None,
        user="tester",
    )
