"""Tool 2: check_drug_interactions.

Flags clinical DDIs ranked by **patient-specific** severity — not the context-blind
alerts that cause alarm fatigue.

The contextual severity pipeline:

    1. Base severity from the drug_interactions KB (seeded from published refs).
    2. Patient context fetched: age, sex, weight, relevant conditions, recent labs
       (eGFR, K+, QTc, INR), duration of coadministration.
    3. Contextual score = f(base_severity, context) via:
         - ML model (XGBoost + SHAP) if available at argus/ml/artifacts/ddi_severity.xgb
         - Heuristic rules fallback (documented; always works without models)
    4. LLM generates patient-specific recommended action with citations.

This tool is the AI-Factor centerpiece: rule-based checkers cannot differentiate
"warfarin + amiodarone, stable 3 years" from "same pair, newly started in elderly CKD."
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

import aiosqlite

from argus.config import get_settings
from argus.fhir_client import FhirClient
from argus.llm import get_llm
from argus.logging_setup import get_logger
from argus.schemas import (
    CheckInteractionsInput,
    CheckInteractionsOutput,
    DrugInteraction,
    FhirReference,
    GetMedicationsInput,
    InteractionFactor,
    MedicationEntry,
    Severity,
    Warning_,
)
from argus.sharp_context import SharpContext
from argus.tools._common import (
    observation_value_quantity,
    patient_age_years,
    patient_sex,
    pick_latest_observation,
    resource_ref,
)
from argus.tools.get_medications import run as run_get_medications

log = get_logger(__name__)


SEVERITY_BASE_SCORE = {
    Severity.TRIVIAL: 0.5,
    Severity.MINOR: 1.5,
    Severity.MODERATE: 2.5,
    Severity.MAJOR: 4.0,
    Severity.CRITICAL: 4.8,
}

# LOINCs for patient context
LOINC_CREATININE = ["2160-0"]
LOINC_POTASSIUM = ["6298-4", "2823-3"]
LOINC_INR = ["34714-6", "6301-6"]
LOINC_QTC = ["8634-8"]

THRESHOLD_ORDER = [
    Severity.TRIVIAL,
    Severity.MINOR,
    Severity.MODERATE,
    Severity.MAJOR,
    Severity.CRITICAL,
]


async def run(
    request: CheckInteractionsInput, context: SharpContext
) -> CheckInteractionsOutput:
    start = time.perf_counter()
    patient_id = request.patient_id or context.patient_id

    if not patient_id:
        return CheckInteractionsOutput(
            warnings=[Warning_(code="missing_patient_id", message="No patient_id.")],
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    # ---- Gather meds and patient context concurrently ----------------------
    async with FhirClient(context) as fhir:
        patient_task = fhir.get_patient(patient_id)
        conditions_task = fhir.get_conditions(patient_id)
        cr_task = fhir.get_observations(
            patient_id, loinc_codes=LOINC_CREATININE, lookback_days=90
        )
        k_task = fhir.get_observations(
            patient_id, loinc_codes=LOINC_POTASSIUM, lookback_days=30
        )
        inr_task = fhir.get_observations(
            patient_id, loinc_codes=LOINC_INR, lookback_days=60
        )
        qtc_task = fhir.get_observations(
            patient_id, loinc_codes=LOINC_QTC, lookback_days=90
        )

        if request.medication_list is None:
            meds_result = await run_get_medications(
                GetMedicationsInput(patient_id=patient_id), context
            )
            med_list = list(meds_result.medications)
        else:
            med_list = list(request.medication_list)

        if request.new_medication_candidate:
            med_list.append(request.new_medication_candidate)

        patient, conditions, cr_obs, k_obs, inr_obs, qtc_obs = await asyncio.gather(
            patient_task, conditions_task, cr_task, k_task, inr_task, qtc_task,
            return_exceptions=True,
        )

    warnings: list[Warning_] = []
    patient = _or_empty(patient, "patient_fetch_failed", {}, warnings)
    conditions = _or_empty(conditions, "conditions_fetch_failed", [], warnings)
    cr_obs = _or_empty(cr_obs, "creatinine_fetch_failed", [], warnings)
    k_obs = _or_empty(k_obs, "potassium_fetch_failed", [], warnings)
    inr_obs = _or_empty(inr_obs, "inr_fetch_failed", [], warnings)
    qtc_obs = _or_empty(qtc_obs, "qtc_fetch_failed", [], warnings)

    patient_ctx = _build_patient_context(patient, conditions, cr_obs, k_obs, inr_obs, qtc_obs)

    # ---- Resolve pair-level base severities --------------------------------
    pair_severities = await _lookup_base_severities(med_list)
    analyzed_pair_count = _pair_count(len(med_list))

    # ---- Score each interaction -------------------------------------------
    interactions: list[DrugInteraction] = []
    citations_seen: set[str] = set()
    citations: list[FhirReference] = []

    threshold_idx = THRESHOLD_ORDER.index(request.severity_threshold)

    llm = get_llm()
    llm_tasks = []
    interaction_stubs: list[dict[str, Any]] = []

    for (rx_a, rx_b), row in pair_severities.items():
        med_a = _find_med(med_list, rx_a)
        med_b = _find_med(med_list, rx_b)
        if med_a is None or med_b is None:
            continue

        base_sev = Severity(row["base_severity"])
        context_score, factors = _contextual_severity(base_sev, patient_ctx)
        contextual_label = _score_to_severity(context_score)

        if THRESHOLD_ORDER.index(contextual_label) < threshold_idx:
            continue

        source_refs = _sources_for_pair(med_a, med_b, patient_ctx, citations_seen)
        citations.extend(source_refs)

        interaction_stubs.append({
            "drug_a": {"rxnorm": med_a.rxnorm_ingredient_code, "name": med_a.rxnorm_ingredient_name},
            "drug_b": {"rxnorm": med_b.rxnorm_ingredient_code, "name": med_b.rxnorm_ingredient_name},
            "base_severity": base_sev,
            "base_severity_source": row.get("source", "Argus DDI KB"),
            "contextual_severity_score": context_score,
            "contextual_severity_label": contextual_label,
            "mechanism": row["mechanism"],
            "patient_specific_factors": factors,
            "evidence_urls": [row["evidence_url"]] if row.get("evidence_url") else [],
            "source_fhir_resources": source_refs,
        })
        llm_tasks.append(_recommend_action(llm, row, patient_ctx, med_a, med_b))

    actions = (
        await asyncio.gather(*llm_tasks, return_exceptions=True) if llm_tasks else []
    )

    for stub, action in zip(interaction_stubs, actions, strict=True):
        if isinstance(action, Exception):
            log.warning("check_interactions.llm_failed", error=str(action))
            action = _fallback_action(stub)
        interactions.append(DrugInteraction(recommended_action=action, **stub))

    interactions.sort(key=lambda x: -x.contextual_severity_score)

    summary = _severity_summary(interactions)
    latency_ms = int((time.perf_counter() - start) * 1000)

    log.info(
        "check_interactions.done",
        patient_id=patient_id,
        analyzed=analyzed_pair_count,
        reported=len(interactions),
        latency_ms=latency_ms,
    )

    return CheckInteractionsOutput(
        patient_id=patient_id,
        interactions=interactions,
        summary=summary,
        analyzed_pair_count=analyzed_pair_count,
        warnings=warnings,
        citations=citations,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Patient context
# ---------------------------------------------------------------------------


def _build_patient_context(
    patient: dict[str, Any],
    conditions: list[dict[str, Any]],
    cr_obs: list[dict[str, Any]],
    k_obs: list[dict[str, Any]],
    inr_obs: list[dict[str, Any]],
    qtc_obs: list[dict[str, Any]],
) -> dict[str, Any]:
    cond_codes = set()
    for c in conditions:
        for coding in (c.get("code", {}).get("coding") or []):
            code = coding.get("code")
            if code:
                cond_codes.add(code)

    def _val(obs_list: list[dict[str, Any]]) -> tuple[float | None, FhirReference | None]:
        latest = pick_latest_observation(obs_list)
        if not latest:
            return None, None
        v, _ = observation_value_quantity(latest)
        return v, resource_ref(latest)

    cr, cr_ref = _val(cr_obs)
    k, k_ref = _val(k_obs)
    inr, inr_ref = _val(inr_obs)
    qtc, qtc_ref = _val(qtc_obs)

    return {
        "age": patient_age_years(patient) if patient else None,
        "sex": patient_sex(patient) if patient else None,
        "creatinine_mg_dl": cr,
        "creatinine_ref": cr_ref,
        "potassium_meq_l": k,
        "potassium_ref": k_ref,
        "inr": inr,
        "inr_ref": inr_ref,
        "qtc_ms": qtc,
        "qtc_ref": qtc_ref,
        "condition_codes": cond_codes,
        "has_ckd": _has_ckd(cond_codes),
        "has_hepatic": _has_hepatic(cond_codes),
        "has_cardiac": _has_cardiac(cond_codes),
    }


def _has_ckd(codes: set[str]) -> bool:
    # SNOMED + ICD-10 CKD families
    ckd_fragments = ("N18", "N19", "431855005", "709044004", "90688005")
    return any(any(c.startswith(f) or c == f for f in ckd_fragments) for c in codes)


def _has_hepatic(codes: set[str]) -> bool:
    hep = ("K70", "K72", "K74", "K76", "197321007", "328383001")
    return any(any(c.startswith(f) or c == f for f in hep) for c in codes)


def _has_cardiac(codes: set[str]) -> bool:
    card = ("I50", "I48", "I25", "84114007", "49601007")
    return any(any(c.startswith(f) or c == f for f in card) for c in codes)


# ---------------------------------------------------------------------------
# Base severity lookup
# ---------------------------------------------------------------------------


async def _lookup_base_severities(
    med_list: list[MedicationEntry],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return {(rx_a, rx_b) -> row} for every known pair in this med list."""
    ingredients = [m.rxnorm_ingredient_code for m in med_list if m.rxnorm_ingredient_code]
    unique = list(dict.fromkeys(ingredients))
    if len(unique) < 2:
        return {}

    settings = get_settings()
    placeholders = ",".join(["?"] * len(unique))

    async with aiosqlite.connect(str(settings.reference_kb_path)) as db:
        db.row_factory = aiosqlite.Row
        query = f"""
            SELECT rxnorm_a, name_a, rxnorm_b, name_b, base_severity,
                   mechanism, evidence_url
            FROM drug_interactions
            WHERE rxnorm_a IN ({placeholders})
              AND rxnorm_b IN ({placeholders})
        """
        async with db.execute(query, unique + unique) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        # Normalize pair ordering — smaller id first
        a, b = sorted((r["rxnorm_a"], r["rxnorm_b"]))
        out[(a, b)] = {
            "base_severity": r["base_severity"],
            "mechanism": r["mechanism"],
            "evidence_url": r.get("evidence_url"),
            "source": "Argus DDI KB",
        }
    return out


# ---------------------------------------------------------------------------
# Contextual severity — heuristic fallback (transparent, always works)
# ---------------------------------------------------------------------------


def _contextual_severity(
    base: Severity, ctx: dict[str, Any]
) -> tuple[float, list[InteractionFactor]]:
    """Adjust base severity by patient-specific factors.

    Returns (score_0_to_5, factors_with_explanations).

    NOTE: When the XGBoost DDI severity model is trained (see scripts/train_ddi_model.py),
    this function delegates to argus.ml.ddi_severity. Until then the heuristic is used
    and is fully transparent — every contribution is surfaced.
    """
    settings = get_settings()
    if settings.enable_ml_severity:
        try:
            from argus.ml.ddi_severity import score_interaction  # lazy import

            ml_result = score_interaction(base, ctx)
            if ml_result is not None:
                return ml_result
        except Exception as exc:  # noqa: BLE001
            log.debug("ml.ddi_severity.unavailable", error=str(exc))

    score = SEVERITY_BASE_SCORE[base]
    factors: list[InteractionFactor] = []

    age = ctx.get("age")
    if age is not None:
        if age >= 80:
            score += 0.4
            factors.append(InteractionFactor(factor="age_80_plus", direction="increases_risk", shap_value=0.4))
        elif age >= 65:
            score += 0.2
            factors.append(InteractionFactor(factor="age_65_plus", direction="increases_risk", shap_value=0.2))

    if ctx.get("has_ckd") or _low_egfr(ctx):
        score += 0.3
        factors.append(InteractionFactor(factor="renal_impairment", direction="increases_risk", shap_value=0.3))

    if ctx.get("has_hepatic"):
        score += 0.3
        factors.append(InteractionFactor(factor="hepatic_impairment", direction="increases_risk", shap_value=0.3))

    k = ctx.get("potassium_meq_l")
    if k is not None and k < 3.5:
        score += 0.2
        factors.append(InteractionFactor(factor="hypokalemia", direction="increases_risk", shap_value=0.2))

    inr = ctx.get("inr")
    if inr is not None and inr > 3.5:
        score += 0.3
        factors.append(InteractionFactor(factor="inr_above_range", direction="increases_risk", shap_value=0.3))

    qtc = ctx.get("qtc_ms")
    if qtc is not None and qtc > 470:
        score += 0.3
        factors.append(InteractionFactor(factor="qtc_prolonged", direction="increases_risk", shap_value=0.3))

    score = max(0.0, min(5.0, score))
    return score, factors


def _low_egfr(ctx: dict[str, Any]) -> bool:
    cr = ctx.get("creatinine_mg_dl")
    age = ctx.get("age")
    sex = ctx.get("sex")
    if cr is None or age is None or sex is None:
        return False
    from argus.tools._common import egfr_ckd_epi_2021

    try:
        return egfr_ckd_epi_2021(cr, age, sex) < 60
    except Exception:  # noqa: BLE001
        return False


def _score_to_severity(score: float) -> Severity:
    if score >= 4.3:
        return Severity.CRITICAL
    if score >= 3.5:
        return Severity.MAJOR
    if score >= 2.3:
        return Severity.MODERATE
    if score >= 1.2:
        return Severity.MINOR
    return Severity.TRIVIAL


# ---------------------------------------------------------------------------
# Action recommendation — LLM with deterministic fallback
# ---------------------------------------------------------------------------


async def _recommend_action(
    llm,
    row: dict[str, Any],
    ctx: dict[str, Any],
    med_a: MedicationEntry,
    med_b: MedicationEntry,
) -> str:
    if not llm.available:
        return _fallback_action(
            {
                "drug_a": {"name": med_a.rxnorm_ingredient_name},
                "drug_b": {"name": med_b.rxnorm_ingredient_name},
                "base_severity": Severity(row["base_severity"]),
                "mechanism": row["mechanism"],
            }
        )

    prompt = f"""You are a clinical pharmacist providing a concise, actionable
recommendation for a drug-drug interaction.

Drug A: {med_a.rxnorm_ingredient_name}
Drug B: {med_b.rxnorm_ingredient_name}
Mechanism: {row['mechanism']}
Base severity: {row['base_severity']}

Patient context:
- Age: {ctx.get('age')}
- Sex: {ctx.get('sex')}
- Serum creatinine: {ctx.get('creatinine_mg_dl')} mg/dL
- Potassium: {ctx.get('potassium_meq_l')} mEq/L
- INR: {ctx.get('inr')}
- QTc: {ctx.get('qtc_ms')} ms
- Has CKD: {ctx.get('has_ckd')}
- Has hepatic disease: {ctx.get('has_hepatic')}
- Has cardiac disease: {ctx.get('has_cardiac')}

Write a single 1-3 sentence action-oriented recommendation. Do not repeat the
mechanism. Be specific about monitoring (labs, intervals) and dose adjustments
where applicable. No disclaimers — the system adds them."""

    result = await llm.generate(prompt, timeout_s=12.0)
    if result and result.text.strip():
        return result.text.strip()
    return _fallback_action(
        {
            "drug_a": {"name": med_a.rxnorm_ingredient_name},
            "drug_b": {"name": med_b.rxnorm_ingredient_name},
            "base_severity": Severity(row["base_severity"]),
            "mechanism": row["mechanism"],
        }
    )


def _fallback_action(stub: dict[str, Any]) -> str:
    sev = stub["base_severity"]
    name_a = stub["drug_a"].get("name") or "Drug A"
    name_b = stub["drug_b"].get("name") or "Drug B"
    if sev == Severity.CRITICAL:
        return f"AVOID combination of {name_a} and {name_b}. Select alternative therapy."
    if sev == Severity.MAJOR:
        return (
            f"Use caution combining {name_a} and {name_b}. Consider dose adjustment "
            "or closer monitoring; review alternatives if not essential."
        )
    if sev == Severity.MODERATE:
        return (
            f"Monitor closely when co-prescribing {name_a} and {name_b}. "
            "Verify dosing and check relevant labs at appropriate intervals."
        )
    return f"Continue {name_a} and {name_b}; no action required beyond routine monitoring."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_med(
    med_list: list[MedicationEntry], rxnorm: str
) -> MedicationEntry | None:
    for m in med_list:
        if m.rxnorm_ingredient_code == rxnorm:
            return m
    return None


def _sources_for_pair(
    med_a: MedicationEntry,
    med_b: MedicationEntry,
    ctx: dict[str, Any],
    seen: set[str],
) -> list[FhirReference]:
    out: list[FhirReference] = []
    for s in med_a.sources + med_b.sources:
        if s.reference not in seen:
            out.append(s)
            seen.add(s.reference)
    for key in ("creatinine_ref", "potassium_ref", "inr_ref", "qtc_ref"):
        ref = ctx.get(key)
        if ref and ref.reference not in seen:
            out.append(ref)
            seen.add(ref.reference)
    return out


def _pair_count(n: int) -> int:
    return n * (n - 1) // 2 if n > 1 else 0


def _severity_summary(interactions: list[DrugInteraction]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for i in interactions:
        counts[i.contextual_severity_label.value] += 1
    return dict(counts)


def _or_empty(val: Any, code: str, default: Any, warnings: list[Warning_]) -> Any:
    if isinstance(val, Exception):
        warnings.append(Warning_(code=code, message=str(val)))
        return default
    return val
