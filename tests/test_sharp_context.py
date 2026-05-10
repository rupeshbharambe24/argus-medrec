"""Tests for SHARP-on-MCP context extraction.

Spec: https://www.sharponmcp.com/key-components.html
"""

from __future__ import annotations

from argus.sharp_context import extract_sharp_context


class TestCanonicalHeaders:
    def test_canonical_header_names(self):
        ctx = extract_sharp_context({
            "x-fhir-server-url": "https://example.org/fhir",
            "x-fhir-access-token": "abc123",
            "x-patient-id": "p-1",
        })
        assert ctx.fhir_base_url == "https://example.org/fhir"
        assert ctx.fhir_token == "abc123"
        assert ctx.patient_id == "p-1"
        assert ctx.has_token is True

    def test_canonical_with_bearer_prefix(self):
        ctx = extract_sharp_context({
            "x-fhir-server-url": "https://example.org/fhir",
            "x-fhir-access-token": "Bearer abc123",
        })
        assert ctx.fhir_token == "abc123"

    def test_canonical_encounter_and_user(self):
        ctx = extract_sharp_context({
            "x-fhir-server-url": "https://example.org/fhir",
            "x-fhir-access-token": "abc",
            "x-encounter-id": "e-42",
            "x-user": "dr.smith",
        })
        assert ctx.encounter_id == "e-42"
        assert ctx.user == "dr.smith"


class TestLegacyAliasesStillWork:
    def test_legacy_sharp_headers_still_work(self):
        ctx = extract_sharp_context({
            "x-sharp-fhir-base-url": "https://example.org/fhir",
            "x-sharp-fhir-token": "xyz",
            "x-sharp-patient": "leg-1",
            "x-sharp-encounter": "enc-1",
            "x-sharp-user": "legacy-user",
        })
        assert ctx.fhir_base_url == "https://example.org/fhir"
        assert ctx.fhir_token == "xyz"
        assert ctx.patient_id == "leg-1"
        assert ctx.encounter_id == "enc-1"
        assert ctx.user == "legacy-user"

    def test_canonical_preferred_over_legacy(self):
        ctx = extract_sharp_context({
            "x-fhir-server-url": "https://canonical.example/fhir",
            "x-sharp-fhir-base-url": "https://legacy.example/fhir",
            "x-fhir-access-token": "canon-token",
            "x-sharp-fhir-token": "legacy-token",
        })
        assert "canonical" in ctx.fhir_base_url
        assert ctx.fhir_token == "canon-token"

    def test_authorization_bearer_fallback(self):
        ctx = extract_sharp_context({
            "x-fhir-server-url": "https://example.org/fhir",
            "authorization": "Bearer fallback-token",
        })
        assert ctx.fhir_token == "fallback-token"


class TestExplicitOverride:
    def test_explicit_patient_id_beats_header(self):
        ctx = extract_sharp_context(
            {"x-patient-id": "from-header"},
            explicit_patient_id="from-arg",
        )
        assert ctx.patient_id == "from-arg"


class TestNormalization:
    def test_trailing_slash_stripped(self):
        ctx = extract_sharp_context({
            "x-fhir-server-url": "https://example.org/fhir/",
            "x-fhir-access-token": "abc",
        })
        assert ctx.fhir_base_url == "https://example.org/fhir"

    def test_case_insensitive_headers(self):
        ctx = extract_sharp_context({
            "X-FHIR-Server-URL": "https://example.org/fhir",
            "X-FHIR-Access-Token": "abc",
            "X-Patient-ID": "p-up",
        })
        assert ctx.fhir_base_url == "https://example.org/fhir"
        assert ctx.fhir_token == "abc"
        assert ctx.patient_id == "p-up"
