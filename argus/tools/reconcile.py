"""Tool 4: reconcile_home_vs_hospital.

Compares home medications (MedicationStatement, status=active) against the
current encounter's orders (MedicationRequest tied to an Encounter), categorizes
each discrepancy, and classifies it as **intentional** or **unintentional**
using contextual clinical reasoning.

Discrepancy types (Joint Commission / IHI framework):
    - OMISSION: home med not continued in hospital
    - COMMISSION: hospital med with no home equivalent
    - DOSE_CHANGE / FREQUENCY_CHANGE / ROUTE_CHANGE
    - THERAPEUTIC_SUBSTITUTION: different ingredient, same ATC class

Intentionality is determined by an LLM classifier with a structured rubric that
considers:
    - Admission / surgery / NPO status
    - Renal / hepatic impairment driving dose reduction
    - Formulary substitution patterns
    - Contraindications from new labs or conditions
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from pydantic import BaseModel, Field

from argus.fhir_client import FhirClient
from argus.llm import get_llm
from argus.logging_setup import get_logger
from argus.schemas import (
    Discrepancy,
    DiscrepancyType,
    FhirReference,
    GetMedicationsInput,
    MedicationEntry,
    ReconcileInput,
    ReconcileOutput,
    Severity,
    Warning_,
)
from argus.sharp_context import SharpContext
from argus.tools._common import (
    extract_authored_date,
    extract_dose,
    extract_frequency_text,
    extract_route,
    extract_rxnorm_code,
    resource_ref,
)
from argus.tools.get_medications import run as run_get_medications

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM schema for intentionality classification
# ---------------------------------------------------------------------------


class _IntentionalityVerdict(BaseModel):
    intentionality: str = Field(
        description="'likely_intentional', 'likely_unintentional', or 'needs_review'"
    )
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    clinical_significance: str = Field(
        description="'trivial', 'minor', 'moderate', 'major', 'critical'"
    )


class _BatchedVerdicts(BaseModel):
    """Wrapper for a single LLM call that classifies many discrepancies at once."""

    verdicts: list[_IntentionalityVerdict]


async def run(request: ReconcileInput, context: SharpContext) -> ReconcileOutput:
    start = time.perf_counter()
    patient_id = request.patient_id or context.patient_id

    if not patient_id:
        return ReconcileOutput(
            warnings=[Warning_(code="missing_patient_id", message="No patient_id.")],
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    # ---- Gather data --------------------------------------------------------
    async with FhirClient(context) as fhir:
        patient_task = fhir.get_patient(patient_id)
        conditions_task = fhir.get_conditions(patient_id)
        encounter_task = fhir.get_current_encounter(patient_id, request.encounter_id)
        home_task = run_get_medications(
            GetMedicationsInput(patient_id=patient_id, include_discontinued=False),
            context,
        )

        patient, conditions, encounter, home_result = await asyncio.gather(
            patient_task, conditions_task, encounter_task, home_task,
            return_exceptions=True,
        )

        warnings: list[Warning_] = []
        patient = _or_empty(patient, "patient_fetch_failed", {}, warnings)
        conditions = _or_empty(conditions, "conditions_fetch_failed", [], warnings)
        encounter = _or_empty(encounter, "encounter_fetch_failed", None, warnings)
        home_meds = (
            list(home_result.medications)
            if hasattr(home_result, "medications")
            else []
        )

        if request.hospital_med_list is not None:
            hospital_meds = list(request.hospital_med_list)
        else:
            encounter_id = encounter.get("id") if encounter else request.encounter_id
            hospital_meds = await _fetch_hospital_meds(
                fhir, patient_id, encounter_id, warnings
            )

    # ---- Categorize discrepancies ------------------------------------------
    raw_discrepancies = _categorize(home_meds, hospital_meds)
    if not raw_discrepancies:
        log.info("reconcile.no_discrepancies", patient_id=patient_id)
        return ReconcileOutput(
            patient_id=patient_id,
            encounter_id=(encounter.get("id") if encounter else request.encounter_id),
            home_med_count=len(home_meds),
            hospital_med_count=len(hospital_meds),
            discrepancies=[],
            summary={"total_discrepancies": 0},
            warnings=warnings,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    # ---- Classify intentionality (one batched LLM call for all discrepancies) -
    patient_ctx = {
        "conditions": _condition_labels(conditions),
        "encounter_type": _encounter_type(encounter) if encounter else None,
        "encounter_reason": _encounter_reason(encounter) if encounter else None,
    }
    llm = get_llm()

    # Batching collapses what used to be N LLM calls (one per discrepancy)
    # into a single call — critical for staying under Gemini's free-tier
    # quota when reconciling polypharmacy patients.
    classifications = await _classify_intentionality_batched(
        llm, raw_discrepancies, patient_ctx
    )

    discrepancies: list[Discrepancy] = []
    citations_seen: set[str] = set()
    citations: list[FhirReference] = []
    for raw, verdict in zip(raw_discrepancies, classifications, strict=True):
        discrepancies.append(
            _build_discrepancy(raw, verdict, citations_seen, citations)
        )

    summary = _summarize(discrepancies)
    latency_ms = int((time.perf_counter() - start) * 1000)
    log.info(
        "reconcile.done",
        patient_id=patient_id,
        discrepancies=len(discrepancies),
        latency_ms=latency_ms,
    )

    return ReconcileOutput(
        patient_id=patient_id,
        encounter_id=(encounter.get("id") if encounter else request.encounter_id),
        home_med_count=len(home_meds),
        hospital_med_count=len(hospital_meds),
        discrepancies=discrepancies,
        summary=summary,
        warnings=warnings,
        citations=citations,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------


async def _fetch_hospital_meds(
    fhir: FhirClient,
    patient_id: str,
    encounter_id: str | None,
    warnings: list[Warning_],
) -> list[MedicationEntry]:
    """Pull MedicationRequest tied to the current encounter and normalize
    every entry's RxCUI to ingredient level via RxNav.

    Without this normalization, hospital meds keep their raw clinical-drug
    RxCUIs (e.g. ``312961`` = "simvastatin 20 MG Oral Tablet") while the home
    list — which goes through ``get_active_medications`` — gets normalized to
    the ingredient code (``36567`` = "simvastatin"). The two lists then never
    match, and every med shows up as both an OMISSION and a COMMISSION.
    """
    from argus.rxnorm import RxNormClient

    try:
        params: dict[str, Any] = {"patient": patient_id, "status": "active"}
        if encounter_id:
            params["encounter"] = encounter_id
        resources = await fhir.search("MedicationRequest", params)
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            Warning_(code="hospital_meds_fetch_failed", message=str(exc))
        )
        return []

    raw_entries = [_resource_to_entry(res, "MedicationRequest") for res in resources]

    rxcuis = [e.rxnorm_clinical_drug_code for e in raw_entries if e.rxnorm_clinical_drug_code]
    if not rxcuis:
        return raw_entries

    async with RxNormClient() as rx:
        norm_map = await rx.normalize_batch(list(set(rxcuis)))

    normalized: list[MedicationEntry] = []
    for entry in raw_entries:
        norm = norm_map.get(entry.rxnorm_clinical_drug_code or "")
        if norm and norm.ingredients:
            ing = norm.ingredients[0]
            normalized.append(
                entry.model_copy(
                    update={
                        "rxnorm_ingredient_code": ing.rxcui,
                        "rxnorm_ingredient_name": ing.name,
                        "therapeutic_class_atc": (norm.atc[0] if norm.atc else None),
                    }
                )
            )
        else:
            normalized.append(entry)
    return normalized


def _resource_to_entry(res: dict[str, Any], resource_type: str) -> MedicationEntry:
    rxcui, display = extract_rxnorm_code(res)
    return MedicationEntry(
        rxnorm_ingredient_code=None,  # set after RxNav normalization in _fetch_hospital_meds
        rxnorm_ingredient_name=display,
        rxnorm_clinical_drug_code=rxcui,
        clinical_drug_name=display,
        dose=extract_dose(res),
        frequency=extract_frequency_text(res),
        route=extract_route(res),
        status=res.get("status", "active"),
        source_priority="medication_request",
        sources=[resource_ref(res)],
        first_seen=extract_authored_date(res),
        last_confirmed=extract_authored_date(res),
    )


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------


def _categorize(
    home: list[MedicationEntry],
    hospital: list[MedicationEntry],
) -> list[dict[str, Any]]:
    home_by_ing = {m.rxnorm_ingredient_code: m for m in home if m.rxnorm_ingredient_code}
    hosp_by_ing = {m.rxnorm_ingredient_code: m for m in hospital if m.rxnorm_ingredient_code}

    raw: list[dict[str, Any]] = []

    # Omissions — home without hospital counterpart
    for code, home_med in home_by_ing.items():
        if code not in hosp_by_ing:
            # Therapeutic substitution?
            sub = _find_therapeutic_substitute(home_med, hospital)
            if sub is not None:
                raw.append({
                    "type": DiscrepancyType.THERAPEUTIC_SUBSTITUTION,
                    "home": home_med,
                    "hospital": sub,
                })
            else:
                raw.append({"type": DiscrepancyType.OMISSION, "home": home_med, "hospital": None})

    # Commissions — hospital without home counterpart (and not already paired)
    paired_hospital_codes = {
        d["hospital"].rxnorm_ingredient_code
        for d in raw
        if d.get("hospital")
    }
    for code, hosp_med in hosp_by_ing.items():
        if code in home_by_ing:
            continue
        if code in paired_hospital_codes:
            continue
        raw.append({"type": DiscrepancyType.COMMISSION, "home": None, "hospital": hosp_med})

    # Dose / freq / route changes (same ingredient)
    for code in set(home_by_ing) & set(hosp_by_ing):
        hm = home_by_ing[code]
        sp = hosp_by_ing[code]
        if hm.dose and sp.dose and (
            hm.dose.value != sp.dose.value or hm.dose.unit != sp.dose.unit
        ):
            raw.append({"type": DiscrepancyType.DOSE_CHANGE, "home": hm, "hospital": sp})
        elif (hm.frequency or "") != (sp.frequency or "") and (hm.frequency or sp.frequency):
            raw.append({"type": DiscrepancyType.FREQUENCY_CHANGE, "home": hm, "hospital": sp})
        elif (hm.route or "") != (sp.route or "") and (hm.route or sp.route):
            raw.append({"type": DiscrepancyType.ROUTE_CHANGE, "home": hm, "hospital": sp})

    return raw


def _find_therapeutic_substitute(
    home_med: MedicationEntry, hospital: list[MedicationEntry]
) -> MedicationEntry | None:
    """Same ATC therapeutic class, different ingredient."""
    if not home_med.therapeutic_class_atc:
        return None
    home_prefix = home_med.therapeutic_class_atc[:4]  # ATC class level
    for hm in hospital:
        if (
            hm.therapeutic_class_atc
            and hm.therapeutic_class_atc[:4] == home_prefix
            and hm.rxnorm_ingredient_code != home_med.rxnorm_ingredient_code
        ):
            return hm
    return None


# ---------------------------------------------------------------------------
# Intentionality classification
# ---------------------------------------------------------------------------


async def _classify_intentionality_batched(
    llm,
    raws: list[dict[str, Any]],
    patient_ctx: dict[str, Any],
) -> list[_IntentionalityVerdict]:
    """Classify all discrepancies in a single LLM call.

    For a polypharmacy patient with 18 discrepancies, the per-call approach
    fires 18 LLM requests, which trivially exhausts Gemini's free-tier quota
    (10 RPM / 250 RPD on flash, 15 RPM / 1000 RPD on flash-lite). Batching
    keeps the whole reconciliation to one call.

    If the LLM is unavailable or the call fails, every discrepancy falls
    back to the deterministic heuristic.
    """
    if not raws:
        return []
    if not llm.available:
        return [_heuristic_verdict(r, patient_ctx) for r in raws]

    items = [
        {
            "id": i,
            "type": r["type"].value,
            "home": _med_summary(r.get("home")),
            "hospital": _med_summary(r.get("hospital")),
        }
        for i, r in enumerate(raws)
    ]
    prompt = f"""You are a clinical pharmacist classifying medication-reconciliation
discrepancies for a single patient. For EACH discrepancy below, decide whether the
change is INTENTIONAL (clinically justified) or UNINTENTIONAL (likely an oversight).

Patient conditions: {', '.join(patient_ctx.get('conditions') or []) or 'none reported'}
Encounter type: {patient_ctx.get('encounter_type') or 'unknown'}
Encounter reason: {patient_ctx.get('encounter_reason') or 'unknown'}

Discrepancies (process every one, return verdicts in the SAME order):
{json.dumps(items, indent=2)}

Guidance:
- NPO, surgical, or procedural encounters justify omission of oral meds.
- Formulary substitutions within the same therapeutic class are usually intentional.
- Dose reductions aligned with acute renal/hepatic injury are usually intentional.
- Omissions of chronic disease meds (antihypertensives, statins, anticoagulants)
  without documented contraindication are usually UNintentional.
- When uncertain, choose 'needs_review'.

Respond with this JSON (verdicts list MUST have exactly {len(items)} entries, one
per discrepancy id, in the input order):
{{
  "verdicts": [
    {{
      "intentionality": "likely_intentional" | "likely_unintentional" | "needs_review",
      "confidence": 0.0-1.0,
      "reasoning": "one or two sentences, clinician-level",
      "clinical_significance": "trivial" | "minor" | "moderate" | "major" | "critical"
    }}
  ]
}}"""

    result = await llm.generate_json(prompt, _BatchedVerdicts, timeout_s=45.0)
    if result is None or len(result.verdicts) != len(raws):
        if result is not None:
            log.warning(
                "reconcile.batch_verdict_count_mismatch",
                expected=len(raws),
                got=len(result.verdicts),
            )
        return [_heuristic_verdict(r, patient_ctx) for r in raws]
    return result.verdicts


async def _classify_intentionality(
    llm,
    raw: dict[str, Any],
    patient_ctx: dict[str, Any],
) -> _IntentionalityVerdict:
    if not llm.available:
        return _heuristic_verdict(raw, patient_ctx)

    home = raw.get("home")
    hosp = raw.get("hospital")

    prompt = f"""You are a clinical pharmacist classifying a medication reconciliation
discrepancy. Decide whether the change is INTENTIONAL (clinically justified) or
UNINTENTIONAL (likely an oversight).

Discrepancy type: {raw['type'].value}
Home medication: {_med_summary(home)}
Hospital medication: {_med_summary(hosp)}

Patient conditions: {', '.join(patient_ctx.get('conditions') or []) or 'none reported'}
Encounter type: {patient_ctx.get('encounter_type') or 'unknown'}
Encounter reason: {patient_ctx.get('encounter_reason') or 'unknown'}

Guidance:
- NPO, surgical, or procedural encounters justify omission of oral meds
- Formulary substitutions within the same therapeutic class are usually intentional
- Dose reductions aligned with acute renal/hepatic injury are usually intentional
- Omissions of chronic disease meds (antihypertensives, statins, anticoagulants) without
  documented contraindication are usually UNintentional
- When uncertain, choose 'needs_review'

Respond with this JSON:
{{
  "intentionality": "likely_intentional" | "likely_unintentional" | "needs_review",
  "confidence": 0.0-1.0,
  "reasoning": "one or two sentences, clinician-level",
  "clinical_significance": "trivial" | "minor" | "moderate" | "major" | "critical"
}}"""

    result = await llm.generate_json(prompt, _IntentionalityVerdict, timeout_s=15.0)
    if result is None:
        return _heuristic_verdict(raw, patient_ctx)
    return result


def _heuristic_verdict(
    raw: dict[str, Any], patient_ctx: dict[str, Any]
) -> _IntentionalityVerdict:
    """Rule-based fallback — conservative; always routes to 'needs_review' when unclear."""
    dtype = raw["type"]
    if dtype == DiscrepancyType.THERAPEUTIC_SUBSTITUTION:
        return _IntentionalityVerdict(
            intentionality="likely_intentional",
            confidence=0.7,
            reasoning="Therapeutic substitutions within a class are routine formulary practice.",
            clinical_significance="minor",
        )
    if dtype == DiscrepancyType.OMISSION:
        return _IntentionalityVerdict(
            intentionality="needs_review",
            confidence=0.5,
            reasoning="Omission without documented rationale — clinician should confirm.",
            clinical_significance="moderate",
        )
    if dtype == DiscrepancyType.DOSE_CHANGE:
        return _IntentionalityVerdict(
            intentionality="needs_review",
            confidence=0.5,
            reasoning="Dose change — confirm whether adjustment is tied to renal/hepatic status.",
            clinical_significance="moderate",
        )
    return _IntentionalityVerdict(
        intentionality="needs_review",
        confidence=0.4,
        reasoning="Discrepancy flagged for clinician verification.",
        clinical_significance="minor",
    )


def _build_discrepancy(
    raw: dict[str, Any],
    verdict: _IntentionalityVerdict,
    citations_seen: set[str],
    citations: list[FhirReference],
) -> Discrepancy:
    # Collect evidence references
    for med in (raw.get("home"), raw.get("hospital")):
        if med is not None:
            for ref in med.sources:
                if ref.reference not in citations_seen:
                    citations.append(ref)
                    citations_seen.add(ref.reference)

    sig = verdict.clinical_significance.lower()
    if sig not in ("trivial", "minor", "moderate", "major", "critical"):
        sig = "moderate"
    intent = verdict.intentionality
    if intent not in ("likely_intentional", "likely_unintentional", "needs_review"):
        intent = "needs_review"

    return Discrepancy(
        type=raw["type"],
        home_medication=raw.get("home"),
        hospital_medication=raw.get("hospital"),
        intentionality=intent,  # type: ignore[arg-type]
        intentionality_confidence=min(max(verdict.confidence, 0.0), 1.0),
        reasoning=verdict.reasoning,
        clinical_significance=Severity(sig),
        recommended_action=_recommend_from_verdict(raw, verdict),
    )


def _recommend_from_verdict(
    raw: dict[str, Any], verdict: _IntentionalityVerdict
) -> str:
    dtype = raw["type"]
    if verdict.intentionality == "likely_intentional":
        return "No action — document the clinical rationale in the chart."
    if dtype == DiscrepancyType.OMISSION:
        home = raw.get("home")
        name = home.rxnorm_ingredient_name if home else "the medication"
        return f"Consider re-adding {name} unless a specific contraindication exists."
    if dtype == DiscrepancyType.COMMISSION:
        hosp = raw.get("hospital")
        name = hosp.rxnorm_ingredient_name if hosp else "the new medication"
        return f"Verify indication for {name} and patient awareness."
    if dtype in (
        DiscrepancyType.DOSE_CHANGE,
        DiscrepancyType.FREQUENCY_CHANGE,
        DiscrepancyType.ROUTE_CHANGE,
    ):
        return "Verify the change is intentional and documented; confirm with patient."
    return "Review with the prescribing clinician."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _med_summary(med: MedicationEntry | None) -> str:
    if med is None:
        return "none"
    parts = [med.rxnorm_ingredient_name or med.clinical_drug_name or "unknown"]
    if med.dose:
        parts.append(f"{med.dose.value} {med.dose.unit}")
    if med.frequency:
        parts.append(med.frequency)
    if med.route:
        parts.append(med.route)
    return " | ".join(parts)


def _condition_labels(conditions: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for c in conditions:
        text = (c.get("code") or {}).get("text")
        if text:
            out.append(text)
            continue
        coding = (c.get("code") or {}).get("coding") or []
        if coding:
            display = coding[0].get("display") or coding[0].get("code")
            if display:
                out.append(display)
    return out[:20]  # cap


def _encounter_type(enc: dict[str, Any]) -> str | None:
    cls = enc.get("class") or {}
    return cls.get("display") or cls.get("code")


def _encounter_reason(enc: dict[str, Any]) -> str | None:
    rc = enc.get("reasonCode") or []
    if rc:
        return rc[0].get("text") or (rc[0].get("coding") or [{}])[0].get("display")
    return None


def _summarize(discrepancies: list[Discrepancy]) -> dict[str, int]:
    return {
        "total_discrepancies": len(discrepancies),
        "likely_unintentional": sum(
            1 for d in discrepancies if d.intentionality == "likely_unintentional"
        ),
        "likely_intentional": sum(
            1 for d in discrepancies if d.intentionality == "likely_intentional"
        ),
        "needs_review": sum(
            1 for d in discrepancies if d.intentionality == "needs_review"
        ),
    }


def _or_empty(val: Any, code: str, default: Any, warnings: list[Warning_]) -> Any:
    if isinstance(val, Exception):
        warnings.append(Warning_(code=code, message=str(val)))
        return default
    return val
