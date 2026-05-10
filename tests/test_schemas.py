"""Tests for argus.schemas — ensures tool I/O contracts don't drift."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from argus.schemas import (
    ActionItem,
    CheckInteractionsInput,
    CheckInteractionsOutput,
    Discrepancy,
    DiscrepancyType,
    DISCLAIMER_TEXT,
    Dose,
    DrugInteraction,
    FhirReference,
    GenerateNoteInput,
    GetMedicationsInput,
    GetMedicationsOutput,
    MedicationEntry,
    NoteAudience,
    ReconcileInput,
    ReconcileOutput,
    RenalCheckInput,
    RenalCheckOutput,
    RenalFunction,
    ScreenFinding,
    ScreenPatternsInput,
    ScreenPatternsOutput,
    ScreenType,
    Severity,
)


class TestFhirReference:
    def test_reference_property(self):
        ref = FhirReference(resource_type="Patient", resource_id="abc")
        assert ref.reference == "Patient/abc"

    def test_frozen(self):
        ref = FhirReference(resource_type="Patient", resource_id="abc")
        with pytest.raises(ValidationError):
            ref.resource_id = "xyz"  # type: ignore[misc]


class TestSeverity:
    def test_enum_values(self):
        assert Severity.MAJOR.value == "major"
        assert Severity("critical") == Severity.CRITICAL


class TestDose:
    def test_basic(self):
        d = Dose(value=10.0, unit="mg")
        assert d.value == 10.0

    def test_requires_fields(self):
        with pytest.raises(ValidationError):
            Dose(value=10.0)  # type: ignore[call-arg]


class TestMedicationEntry:
    def test_minimal(self):
        m = MedicationEntry()
        assert m.status == "active"
        assert m.sources == []

    def test_roundtrip(self):
        m = MedicationEntry(
            rxnorm_ingredient_code="5487",
            rxnorm_ingredient_name="Lisinopril",
            dose=Dose(value=10, unit="mg"),
        )
        d = m.model_dump()
        m2 = MedicationEntry.model_validate(d)
        assert m2.rxnorm_ingredient_code == "5487"


class TestBaseResponseEnvelope:
    def test_disclaimer_present(self):
        from datetime import date as _date
        out = GetMedicationsOutput(as_of=_date.today())
        assert out.disclaimer == DISCLAIMER_TEXT
        assert out.generated_at is not None

    def test_all_tool_outputs_have_disclaimer(self):
        from datetime import date as _date
        assert GetMedicationsOutput(as_of=_date.today()).disclaimer
        assert CheckInteractionsOutput().disclaimer
        assert RenalCheckOutput(renal_function=RenalFunction()).disclaimer
        assert ReconcileOutput().disclaimer
        assert ScreenPatternsOutput().disclaimer


class TestInputValidation:
    def test_lookback_days_range(self):
        with pytest.raises(ValidationError):
            GetMedicationsInput(lookback_days=0)
        with pytest.raises(ValidationError):
            GetMedicationsInput(lookback_days=99999)

    def test_severity_threshold_enum(self):
        with pytest.raises(ValidationError):
            CheckInteractionsInput(severity_threshold="invalid")  # type: ignore[arg-type]

    def test_note_input_defaults(self):
        req = GenerateNoteInput()
        assert req.audience == NoteAudience.PHYSICIAN
        assert req.language == "en"
        assert len(req.include_sections) == 5


class TestDiscrepancy:
    def test_intentionality_types(self):
        d = Discrepancy(
            type=DiscrepancyType.OMISSION,
            intentionality="likely_unintentional",
            intentionality_confidence=0.8,
            reasoning="test",
            clinical_significance=Severity.MODERATE,
            recommended_action="restart med",
        )
        assert d.intentionality == "likely_unintentional"

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            Discrepancy(
                type=DiscrepancyType.OMISSION,
                intentionality="likely_unintentional",
                intentionality_confidence=1.5,  # invalid
                reasoning="test",
                clinical_significance=Severity.MODERATE,
                recommended_action="x",
            )


class TestActionItem:
    def test_owner_role_enum(self):
        with pytest.raises(ValidationError):
            ActionItem(
                priority=Severity.MAJOR,
                action="x",
                reason="y",
                owner_role="janitor",  # type: ignore[arg-type]
            )


class TestScreenOutput:
    def test_finding_roundtrip(self):
        f = ScreenFinding(
            screen=ScreenType.BEERS,
            severity=Severity.MODERATE,
            title="test",
            description="test",
            medications_involved=[{"rxnorm": "3498", "name": "Diphenhydramine"}],
        )
        d = f.model_dump()
        f2 = ScreenFinding.model_validate(d)
        assert f2.screen == ScreenType.BEERS
