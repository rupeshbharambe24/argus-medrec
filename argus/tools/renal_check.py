"""Tool 3: renal_dose_check.

Flags medications needing dose adjustment or avoidance given the patient's
current renal function.

Steps:
    1. Fetch latest creatinine (LOINC 2160-0) within lookback window.
    2. Compute eGFR via CKD-EPI 2021 (race-free).
    3. Look up each active medication against the renal dosing KB.
    4. Return structured per-medication recommendations with guideline citations.

Data source:
    - Renal dosing rules are loaded from the reference KB SQLite, populated by
      argus/reference/build_kb.py from curated FDA label + KDIGO 2022 guidance.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import aiosqlite

from argus.config import get_settings
from argus.fhir_client import FhirClient
from argus.logging_setup import get_logger
from argus.schemas import (
    Dose,
    FhirReference,
    GetMedicationsInput,
    MedicationEntry,
    RenalCheckInput,
    RenalCheckOutput,
    RenalFunction,
    RenalRecommendation,
    Severity,
    Warning_,
)
from argus.sharp_context import SharpContext
from argus.tools._common import (
    ckd_stage,
    egfr_ckd_epi_2021,
    observation_date,
    observation_value_quantity,
    patient_age_years,
    patient_sex,
    pick_latest_observation,
    resource_ref,
)
from argus.tools.get_medications import run as run_get_medications

log = get_logger(__name__)

LOINC_SERUM_CREATININE = ["2160-0"]  # Creatinine [Mass/volume] in Serum or Plasma

SEVERITY_FROM_ACTION = {
    "AVOID": Severity.MAJOR,
    "REDUCE": Severity.MODERATE,
    "MONITOR": Severity.MINOR,
    "NO_CHANGE": Severity.TRIVIAL,
}


async def run(request: RenalCheckInput, context: SharpContext) -> RenalCheckOutput:
    start = time.perf_counter()
    patient_id = request.patient_id or context.patient_id

    if not patient_id:
        return RenalCheckOutput(
            renal_function=RenalFunction(),
            warnings=[Warning_(code="missing_patient_id", message="No patient_id provided.")],
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    async with FhirClient(context) as fhir:
        patient_task = fhir.get_patient(patient_id)
        cr_task = _fetch_creatinine_with_escalating_lookback(
            fhir, patient_id, request.creatinine_lookback_days
        )
        meds_task: Any
        if request.medication_list is None:
            meds_task = run_get_medications(
                GetMedicationsInput(patient_id=patient_id), context
            )
        else:
            meds_task = _wrap_medication_list(request.medication_list)

        patient, cr_observations, med_result = await asyncio.gather(
            patient_task, cr_task, meds_task, return_exceptions=True
        )

    warnings: list[Warning_] = []

    if isinstance(patient, Exception):
        warnings.append(Warning_(code="patient_fetch_failed", message=str(patient)))
        patient = {}
    if isinstance(cr_observations, Exception):
        warnings.append(Warning_(code="creatinine_fetch_failed", message=str(cr_observations)))
        cr_observations = []
    if isinstance(med_result, Exception):
        warnings.append(Warning_(code="medications_fetch_failed", message=str(med_result)))
        med_list: list[MedicationEntry] = []
    else:
        med_list = (
            med_result.medications if hasattr(med_result, "medications") else med_result
        )

    renal = _compute_renal_function(patient, cr_observations, warnings)

    recommendations: list[RenalRecommendation] = []
    citations: list[FhirReference] = []
    missing_data: list[str] = []

    if renal.egfr_ml_min_1_73m2 is None:
        missing_data.append("No recent serum creatinine available; cannot compute eGFR.")
    else:
        recommendations = await _lookup_renal_rules(med_list, renal.egfr_ml_min_1_73m2)
        citations = _citations_for_recommendations(recommendations, med_list)
        if renal.source_creatinine_fhir_reference:
            citations.append(renal.source_creatinine_fhir_reference)

    latency_ms = int((time.perf_counter() - start) * 1000)
    log.info(
        "renal_check.done",
        patient_id=patient_id,
        egfr=renal.egfr_ml_min_1_73m2,
        recommendations=len(recommendations),
        latency_ms=latency_ms,
    )

    return RenalCheckOutput(
        patient_id=patient_id,
        renal_function=renal,
        recommendations=recommendations,
        missing_data=missing_data,
        warnings=warnings,
        citations=citations,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Renal function
# ---------------------------------------------------------------------------


def _compute_renal_function(
    patient: dict[str, Any],
    cr_observations: list[dict[str, Any]],
    warnings: list[Warning_],
) -> RenalFunction:
    age = patient_age_years(patient) if patient else None
    sex = patient_sex(patient) if patient else None

    latest = pick_latest_observation(cr_observations) if cr_observations else None
    if not latest:
        return RenalFunction(confidence="low", notes=["No creatinine within lookback."])

    value, unit = observation_value_quantity(latest)
    obs_date = observation_date(latest)

    if value is None:
        warnings.append(
            Warning_(code="malformed_creatinine", message="Creatinine has no valueQuantity.")
        )
        return RenalFunction(confidence="low", notes=["Latest creatinine was malformed."])

    # Normalize to mg/dL if needed (UCUM: mg/dL == mg/dL; umol/L * 0.0113 ≈ mg/dL)
    if unit and unit.lower() in ("umol/l", "µmol/l", "micromol/l"):
        value = round(value * 0.0113, 2)
        unit = "mg/dL"

    if age is None or sex is None:
        return RenalFunction(
            source_creatinine_value=value,
            source_creatinine_unit=unit,
            source_creatinine_collected_at=obs_date,
            source_creatinine_fhir_reference=resource_ref(latest, "Serum creatinine"),
            confidence="low",
            notes=["Age or sex missing; cannot compute eGFR."],
        )

    egfr = egfr_ckd_epi_2021(value, age, sex)
    return RenalFunction(
        egfr_ml_min_1_73m2=round(egfr, 1),
        ckd_epi_2021=True,
        ckd_stage=ckd_stage(egfr),  # type: ignore[arg-type]
        source_creatinine_value=value,
        source_creatinine_unit=unit or "mg/dL",
        source_creatinine_collected_at=obs_date,
        source_creatinine_fhir_reference=resource_ref(latest, "Serum creatinine"),
        confidence="high" if obs_date else "medium",
        notes=(
            []
            if cr_observations and len(cr_observations) >= 2
            else ["Single Cr used; no trend available."]
        ),
    )


# ---------------------------------------------------------------------------
# Rule lookup
# ---------------------------------------------------------------------------


async def _lookup_renal_rules(
    med_list: list[MedicationEntry],
    egfr: float,
) -> list[RenalRecommendation]:
    settings = get_settings()
    recs: list[RenalRecommendation] = []

    ingredient_codes = [m.rxnorm_ingredient_code for m in med_list if m.rxnorm_ingredient_code]
    if not ingredient_codes:
        return recs

    async with aiosqlite.connect(str(settings.reference_kb_path)) as db:
        db.row_factory = aiosqlite.Row
        placeholders = ",".join(["?"] * len(ingredient_codes))
        query = f"""
            SELECT rxnorm_ingredient, egfr_threshold, action, adjusted_dose_pattern,
                   rationale, source
            FROM renal_dosing_rules
            WHERE rxnorm_ingredient IN ({placeholders})
            ORDER BY egfr_threshold DESC
        """
        async with db.execute(query, ingredient_codes) as cur:
            rows = await cur.fetchall()

    # Group rules per ingredient — pick the strictest applicable
    by_ingredient: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_ingredient.setdefault(row["rxnorm_ingredient"], []).append(dict(row))

    for med in med_list:
        code = med.rxnorm_ingredient_code
        if not code or code not in by_ingredient:
            continue
        applicable = [r for r in by_ingredient[code] if egfr < r["egfr_threshold"]]
        if not applicable:
            continue
        # Strictest = lowest threshold
        rule = min(applicable, key=lambda r: r["egfr_threshold"])
        action = rule["action"].upper()
        severity = SEVERITY_FROM_ACTION.get(action, Severity.MINOR)
        recs.append(
            RenalRecommendation(
                medication={
                    "rxnorm": code,
                    "name": med.rxnorm_ingredient_name or med.clinical_drug_name,
                },
                current_dose=med.dose,
                recommended_action=action,  # type: ignore[arg-type]
                rationale=(
                    f"eGFR {egfr:.0f} mL/min/1.73m². {rule['rationale']}"
                ),
                suggested_dose=_parse_adjusted_dose(rule.get("adjusted_dose_pattern")),
                severity=severity,
                guideline_source=rule.get("source") or "Unknown guideline",
            )
        )
    return recs


def _parse_adjusted_dose(pattern: str | None) -> Dose | None:
    """Parse a simple pattern like '500 mg daily' into a Dose."""
    if not pattern:
        return None
    parts = pattern.strip().split()
    if len(parts) >= 2:
        try:
            return Dose(value=float(parts[0]), unit=parts[1])
        except (ValueError, IndexError):
            return None
    return None


def _citations_for_recommendations(
    recs: list[RenalRecommendation], med_list: list[MedicationEntry]
) -> list[FhirReference]:
    """Cite the MedicationRequest/Statement behind each recommendation."""
    by_code = {m.rxnorm_ingredient_code: m for m in med_list if m.rxnorm_ingredient_code}
    cites: list[FhirReference] = []
    seen: set[str] = set()
    for r in recs:
        code = r.medication.get("rxnorm")
        med = by_code.get(code) if code else None
        if med:
            for s in med.sources:
                if s.reference not in seen:
                    cites.append(s)
                    seen.add(s.reference)
    return cites


async def _wrap_medication_list(
    med_list: list[MedicationEntry],
) -> list[MedicationEntry]:
    """Passthrough to keep the gather() types uniform."""
    return med_list


async def _fetch_creatinine_with_escalating_lookback(
    fhir: FhirClient,
    patient_id: str,
    initial_lookback_days: int,
) -> list[dict[str, Any]]:
    """Try the requested lookback first; if empty, escalate.

    Synthea generates labs at irregular intervals — for an elderly patient with
    stable CKD, the most recent creatinine may be 6-18 months old. Rather than
    returning "no creatinine" when the requested window is too narrow, we try
    increasing windows up to 5 years before giving up.
    """
    candidate_windows = [initial_lookback_days]
    for w in (365, 730, 1825):
        if w > initial_lookback_days:
            candidate_windows.append(w)

    last_observations: list[dict[str, Any]] = []
    for lookback in candidate_windows:
        observations = await fhir.get_observations(
            patient_id,
            loinc_codes=LOINC_SERUM_CREATININE,
            lookback_days=lookback,
        )
        last_observations = observations
        if observations:
            if lookback != initial_lookback_days:
                log.info(
                    "renal_check.escalated_lookback",
                    patient_id=patient_id,
                    initial_days=initial_lookback_days,
                    successful_days=lookback,
                    obs_count=len(observations),
                )
            return observations
    return last_observations
