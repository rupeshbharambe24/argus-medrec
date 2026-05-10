"""Pydantic schemas for all tool inputs and outputs.

This file is the single source of truth for tool contracts. FastMCP auto-generates
JSON schemas from these models so agents can validate their calls.

Design principles:
- All outputs include a `citations` field referencing source FHIR resources.
- All outputs include `warnings` for data-quality transparency.
- All clinical claims are traceable to a FHIR resource ID.
- Every response carries a safety disclaimer (added by the server).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared enums & primitives
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Clinical severity rating used across tools."""

    TRIVIAL = "trivial"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class NoteAudience(str, Enum):
    PHYSICIAN = "physician"
    PHARMACIST = "pharmacist"
    NURSE = "nurse"
    PATIENT = "patient"


class NoteFormat(str, Enum):
    SOAP = "soap"
    NARRATIVE = "narrative"
    BULLETED = "bulleted"
    TABLE = "table"


class DiscrepancyType(str, Enum):
    OMISSION = "OMISSION"
    COMMISSION = "COMMISSION"
    DOSE_CHANGE = "DOSE_CHANGE"
    FREQUENCY_CHANGE = "FREQUENCY_CHANGE"
    ROUTE_CHANGE = "ROUTE_CHANGE"
    THERAPEUTIC_SUBSTITUTION = "THERAPEUTIC_SUBSTITUTION"


class ScreenType(str, Enum):
    BEERS = "beers"
    QTC = "qtc"
    OPIOID_BENZO = "opioid_benzo"
    SEROTONIN_SYNDROME = "serotonin_syndrome"
    ANTICHOLINERGIC_BURDEN = "anticholinergic_burden"
    PREGNANCY = "pregnancy"
    ADHERENCE_GAP = "adherence_gap"


# ---------------------------------------------------------------------------
# Shared objects
# ---------------------------------------------------------------------------


class Dose(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: float
    unit: str = Field(description="e.g. 'mg', 'mcg', 'mg/mL', 'unit'")


class FhirReference(BaseModel):
    """A citation back to a specific FHIR resource."""

    model_config = ConfigDict(frozen=True)
    resource_type: str
    resource_id: str
    label: str | None = None

    @property
    def reference(self) -> str:
        return f"{self.resource_type}/{self.resource_id}"


class Warning_(BaseModel):
    """Non-fatal data-quality warning surfaced to the caller."""

    model_config = ConfigDict(frozen=True)
    code: str
    message: str


class MedicationEntry(BaseModel):
    """A single normalized active medication."""

    rxnorm_ingredient_code: str | None = None
    rxnorm_ingredient_name: str | None = None
    rxnorm_clinical_drug_code: str | None = None
    clinical_drug_name: str | None = None
    dose: Dose | None = None
    frequency: str | None = None
    route: str | None = None
    status: Literal["active", "on-hold", "completed", "stopped", "unknown"] = "active"
    source_priority: Literal[
        "medication_request", "medication_statement", "medication_dispense"
    ] = "medication_request"
    sources: list[FhirReference] = Field(default_factory=list)
    first_seen: date | None = None
    last_confirmed: date | None = None
    therapeutic_class_atc: str | None = None


# ---------------------------------------------------------------------------
# Base response — every tool output extends this so safety metadata is uniform
# ---------------------------------------------------------------------------


DISCLAIMER_TEXT = (
    "AI-generated clinical decision support. All outputs require review by a "
    "licensed clinician before clinical action. Built on synthetic data; not a "
    "substitute for clinical judgment."
)


class BaseToolResponse(BaseModel):
    """All tool responses share this envelope."""

    tool: str
    patient_id: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    warnings: list[Warning_] = Field(default_factory=list)
    citations: list[FhirReference] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER_TEXT
    latency_ms: int | None = None


# ---------------------------------------------------------------------------
# Tool 1 — get_active_medications
# ---------------------------------------------------------------------------


class GetMedicationsInput(BaseModel):
    patient_id: str | None = Field(
        default=None,
        description=(
            "FHIR Patient logical id. If omitted, resolved from SHARP context."
        ),
    )
    as_of_date: date | None = None
    include_discontinued: bool = False
    lookback_days: int = Field(default=180, ge=1, le=3650)


class MedicationDataQuality(BaseModel):
    has_medication_statement: bool = False
    has_medication_request: bool = False
    has_medication_dispense: bool = False
    most_recent_reconciliation_age_days: int | None = None
    coverage_score: float = Field(default=1.0, ge=0.0, le=1.0)


class GetMedicationsOutput(BaseToolResponse):
    tool: Literal["get_active_medications"] = "get_active_medications"
    as_of: date
    medications: list[MedicationEntry] = Field(default_factory=list)
    data_quality: MedicationDataQuality = Field(default_factory=MedicationDataQuality)


# ---------------------------------------------------------------------------
# Tool 2 — check_drug_interactions
# ---------------------------------------------------------------------------


class InteractionFactor(BaseModel):
    factor: str
    direction: Literal["increases_risk", "decreases_risk", "neutral"]
    shap_value: float | None = None


class DrugInteraction(BaseModel):
    drug_a: dict[str, Any]
    drug_b: dict[str, Any]
    base_severity: Severity
    base_severity_source: str
    contextual_severity_score: float = Field(ge=0, le=5)
    contextual_severity_label: Severity
    mechanism: str
    patient_specific_factors: list[InteractionFactor] = Field(default_factory=list)
    recommended_action: str
    evidence_urls: list[str] = Field(default_factory=list)
    source_fhir_resources: list[FhirReference] = Field(default_factory=list)


class CheckInteractionsInput(BaseModel):
    patient_id: str | None = None
    medication_list: list[MedicationEntry] | None = Field(
        default=None, description="If omitted, auto-fetched via get_active_medications."
    )
    new_medication_candidate: MedicationEntry | None = Field(
        default=None, description="Test-before-prescribe: score interactions against current list."
    )
    severity_threshold: Severity = Severity.MODERATE


class CheckInteractionsOutput(BaseToolResponse):
    tool: Literal["check_drug_interactions"] = "check_drug_interactions"
    interactions: list[DrugInteraction] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)
    analyzed_pair_count: int = 0


# ---------------------------------------------------------------------------
# Tool 3 — renal_dose_check
# ---------------------------------------------------------------------------


class RenalFunction(BaseModel):
    egfr_ml_min_1_73m2: float | None = None
    ckd_epi_2021: bool = True
    ckd_stage: Literal["1", "2", "3a", "3b", "4", "5", "unknown"] = "unknown"
    source_creatinine_value: float | None = None
    source_creatinine_unit: str | None = None
    source_creatinine_collected_at: date | None = None
    source_creatinine_fhir_reference: FhirReference | None = None
    confidence: Literal["high", "medium", "low"] = "low"
    notes: list[str] = Field(default_factory=list)


class RenalRecommendation(BaseModel):
    medication: dict[str, Any]
    current_dose: Dose | None = None
    recommended_action: Literal["AVOID", "REDUCE", "MONITOR", "NO_CHANGE"]
    rationale: str
    suggested_dose: Dose | None = None
    severity: Severity
    guideline_source: str


class RenalCheckInput(BaseModel):
    patient_id: str | None = None
    medication_list: list[MedicationEntry] | None = None
    creatinine_lookback_days: int = Field(default=365, ge=1, le=3650)


class RenalCheckOutput(BaseToolResponse):
    tool: Literal["renal_dose_check"] = "renal_dose_check"
    renal_function: RenalFunction
    recommendations: list[RenalRecommendation] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool 4 — reconcile_home_vs_hospital
# ---------------------------------------------------------------------------


class Discrepancy(BaseModel):
    type: DiscrepancyType
    home_medication: MedicationEntry | None = None
    hospital_medication: MedicationEntry | None = None
    intentionality: Literal["likely_intentional", "likely_unintentional", "needs_review"]
    intentionality_confidence: float = Field(ge=0, le=1)
    reasoning: str
    clinical_significance: Severity
    recommended_action: str


class ReconcileInput(BaseModel):
    patient_id: str | None = None
    hospital_med_list: list[MedicationEntry] | None = None
    encounter_id: str | None = None


class ReconcileOutput(BaseToolResponse):
    tool: Literal["reconcile_home_vs_hospital"] = "reconcile_home_vs_hospital"
    encounter_id: str | None = None
    home_med_count: int = 0
    hospital_med_count: int = 0
    discrepancies: list[Discrepancy] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool 5 — generate_med_rec_note
# ---------------------------------------------------------------------------


class ActionItem(BaseModel):
    priority: Severity
    action: str
    reason: str
    owner_role: Literal["attending", "resident", "pharmacist", "nurse", "patient"]
    evidence: list[FhirReference] = Field(default_factory=list)


class GenerateNoteInput(BaseModel):
    patient_id: str | None = None
    audience: NoteAudience = NoteAudience.PHYSICIAN
    format: NoteFormat = NoteFormat.SOAP
    language: Literal["en", "es", "hi", "fr", "pt", "ar", "zh"] = "en"
    include_sections: list[
        Literal["interactions", "renal", "discrepancies", "high_risk", "recommendations"]
    ] = Field(
        default_factory=lambda: [
            "interactions",
            "renal",
            "discrepancies",
            "high_risk",
            "recommendations",
        ]
    )


class GenerateNoteOutput(BaseToolResponse):
    tool: Literal["generate_med_rec_note"] = "generate_med_rec_note"
    note_markdown: str
    structured_action_items: list[ActionItem] = Field(default_factory=list)
    llm_tokens_used: int | None = None
    generation_time_ms: int | None = None


# ---------------------------------------------------------------------------
# Tool 6 — screen_high_risk_patterns
# ---------------------------------------------------------------------------


class ScreenFinding(BaseModel):
    screen: ScreenType
    severity: Severity
    title: str
    description: str
    guideline: str | None = None
    medications_involved: list[dict[str, Any]] = Field(default_factory=list)
    recommended_alternative: str | None = None
    evidence: list[FhirReference] = Field(default_factory=list)


class ScreenPatternsInput(BaseModel):
    patient_id: str | None = None
    screens: list[ScreenType] = Field(
        default_factory=lambda: [
            ScreenType.BEERS,
            ScreenType.QTC,
            ScreenType.OPIOID_BENZO,
            ScreenType.ANTICHOLINERGIC_BURDEN,
            ScreenType.ADHERENCE_GAP,
        ]
    )


class ScreenPatternsOutput(BaseToolResponse):
    tool: Literal["screen_high_risk_patterns"] = "screen_high_risk_patterns"
    screens_run: list[ScreenType] = Field(default_factory=list)
    findings: list[ScreenFinding] = Field(default_factory=list)
