"""Integration test for renal_dose_check using a mocked FHIR server."""

from __future__ import annotations

import json

import httpx
import pytest

from argus.fhir_client import FhirClient
from argus.schemas import MedicationEntry, RenalCheckInput
from argus.sharp_context import SharpContext
from argus.tools.renal_check import run as run_renal


@pytest.fixture()
def mocked_fhir(monkeypatch, synthea_bundle):
    """Patch httpx to return pre-canned responses for the FHIR queries."""
    by_type: dict[str, list[dict]] = {}
    for entry in synthea_bundle["entry"]:
        res = entry["resource"]
        by_type.setdefault(res["resourceType"], []).append(res)

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        # Very small FHIR-search stub
        if "/Patient/test-patient-1" in url and "_count" not in url:
            return httpx.Response(200, json=by_type["Patient"][0])
        if "/Observation" in url:
            obs = by_type.get("Observation", [])
            # Filter by code param if present
            code_param = request.url.params.get("code", "")
            if code_param:
                loinc = code_param.split("|")[-1]
                obs = [o for o in obs if any(
                    c.get("code") == loinc for c in o.get("code", {}).get("coding", [])
                )]
            return httpx.Response(200, json={
                "resourceType": "Bundle", "type": "searchset",
                "entry": [{"resource": o} for o in obs]
            })
        if "/MedicationRequest" in url:
            return httpx.Response(200, json={
                "resourceType": "Bundle", "type": "searchset",
                "entry": [{"resource": r} for r in by_type.get("MedicationRequest", [])]
            })
        if "/MedicationStatement" in url:
            return httpx.Response(200, json={
                "resourceType": "Bundle", "type": "searchset",
                "entry": [{"resource": r} for r in by_type.get("MedicationStatement", [])]
            })
        if "/MedicationDispense" in url:
            return httpx.Response(200, json={"resourceType": "Bundle", "type": "searchset"})
        return httpx.Response(404, json={"error": "not mocked", "url": url})

    transport = httpx.MockTransport(_handler)
    original_cls = httpx.AsyncClient

    def _factory(*a, **kw):
        kw.pop("transport", None)
        return original_cls(transport=transport, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    return transport


class TestRenalCheck:
    @pytest.mark.asyncio
    async def test_egfr_computed_from_creatinine(
        self, temp_kb_path, sharp_context_dev, mocked_fhir, monkeypatch
    ):
        """Given the fixture patient (82M, Cr 2.1), eGFR should be in the CKD 3b range."""
        # Provide a pre-built medication list so we don't need RxNav
        meds = [
            MedicationEntry(
                rxnorm_ingredient_code="6809",
                rxnorm_ingredient_name="Metformin",
                rxnorm_clinical_drug_code="6809",
            ),
            MedicationEntry(
                rxnorm_ingredient_code="29046",
                rxnorm_ingredient_name="Lisinopril",
                rxnorm_clinical_drug_code="29046",
            ),
        ]
        req = RenalCheckInput(patient_id="test-patient-1", medication_list=meds)

        result = await run_renal(req, sharp_context_dev)

        assert result.renal_function.egfr_ml_min_1_73m2 is not None
        assert 25 <= result.renal_function.egfr_ml_min_1_73m2 <= 45
        assert result.renal_function.ckd_stage in ("3b", "4")

        # Metformin contraindication should be flagged
        metformin_recs = [r for r in result.recommendations if r.medication.get("rxnorm") == "6809"]
        assert len(metformin_recs) == 1
        assert metformin_recs[0].recommended_action in ("AVOID", "REDUCE")
