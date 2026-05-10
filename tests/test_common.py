"""Unit tests for argus.tools._common — FHIR parsing + eGFR."""

from __future__ import annotations

import pytest

from argus.tools._common import (
    ckd_stage,
    egfr_ckd_epi_2021,
    extract_authored_date,
    extract_dose,
    extract_frequency_text,
    extract_rxnorm_code,
    patient_age_years,
    patient_sex,
)


class TestEgfr:
    """CKD-EPI 2021 — compare against published values.

    Reference: Inker LA, et al. N Engl J Med 2021;385:1737–1749.
    """

    def test_normal_male(self):
        # 55M, Cr 1.0 → approximately 95 mL/min/1.73m²
        egfr = egfr_ckd_epi_2021(1.0, 55, "male")
        assert 85 < egfr < 105

    def test_normal_female(self):
        egfr = egfr_ckd_epi_2021(0.8, 55, "female")
        assert 80 < egfr < 105

    def test_ckd_stage_3b_male(self):
        # 70M, Cr 2.0 → ~35 mL/min (stage 3b)
        egfr = egfr_ckd_epi_2021(2.0, 70, "male")
        assert 30 <= egfr <= 45
        assert ckd_stage(egfr) == "3b"

    def test_stage_5_mapping(self):
        assert ckd_stage(10) == "5"
        assert ckd_stage(20) == "4"
        assert ckd_stage(35) == "3b"
        assert ckd_stage(50) == "3a"
        assert ckd_stage(75) == "2"
        assert ckd_stage(100) == "1"


class TestRxNormExtraction:
    def test_extracts_rxnorm_coding(self):
        res = {
            "medicationCodeableConcept": {
                "coding": [
                    {"system": "http://snomed.info/sct", "code": "xxx"},
                    {"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "5487", "display": "Lisinopril"},
                ]
            }
        }
        rxcui, display = extract_rxnorm_code(res)
        assert rxcui == "5487"
        assert display == "Lisinopril"

    def test_returns_text_when_no_rxnorm(self):
        res = {"medicationCodeableConcept": {"text": "Some drug"}}
        rxcui, display = extract_rxnorm_code(res)
        assert rxcui is None
        assert display == "Some drug"

    def test_missing_med(self):
        rxcui, display = extract_rxnorm_code({})
        assert rxcui is None
        assert display is None


class TestDose:
    def test_extracts_dose_quantity(self):
        res = {
            "dosageInstruction": [{
                "doseAndRate": [{"doseQuantity": {"value": 10, "unit": "mg"}}]
            }]
        }
        d = extract_dose(res)
        assert d is not None
        assert d.value == 10.0
        assert d.unit == "mg"

    def test_missing_dose_returns_none(self):
        assert extract_dose({}) is None
        assert extract_dose({"dosageInstruction": []}) is None

    def test_frequency_from_text(self):
        res = {"dosageInstruction": [{"text": "twice daily"}]}
        assert extract_frequency_text(res) == "twice daily"


class TestAuthoredDate:
    def test_authored_on(self):
        assert extract_authored_date({"authoredOn": "2026-01-15"}).year == 2026

    def test_authored_on_with_time(self):
        assert extract_authored_date({"authoredOn": "2026-01-15T10:30:00Z"}).month == 1

    def test_falls_back_to_effective_datetime(self):
        assert extract_authored_date({"effectiveDateTime": "2026-02-20"}).day == 20

    def test_missing_returns_none(self):
        assert extract_authored_date({}) is None


class TestPatientHelpers:
    def test_age_calculation(self, synthea_bundle):
        patient = synthea_bundle["entry"][0]["resource"]
        # Born 1943-09-12 — should be in 80s range by 2026
        age = patient_age_years(patient)
        assert age is not None
        assert 80 <= age <= 85

    def test_sex(self, synthea_bundle):
        patient = synthea_bundle["entry"][0]["resource"]
        assert patient_sex(patient) == "male"
