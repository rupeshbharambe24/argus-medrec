"""Tool 1: get_active_medications.

Produces the canonical, deduplicated, RxNorm-normalized current medication list
for a patient. Every other tool depends on this.

Steps:
    1. Fan out FHIR queries for MedicationRequest, MedicationStatement,
       MedicationDispense.
    2. Extract RxNorm codes and dose/freq/route from each resource.
    3. Normalize each RxNorm code to ingredient level via RxNav (cached).
    4. Deduplicate by (ingredient_rxcui, route), keeping highest-priority source.
    5. Compute data-quality signals.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from datetime import date
from typing import Any

from argus.fhir_client import FhirClient
from argus.logging_setup import get_logger
from argus.rxnorm import RxNormClient, RxNormNorm
from argus.schemas import (
    FhirReference,
    GetMedicationsInput,
    GetMedicationsOutput,
    MedicationDataQuality,
    MedicationEntry,
    Warning_,
)
from argus.sharp_context import SharpContext
from argus.tools._common import (
    extract_authored_date,
    extract_dose,
    extract_frequency_text,
    extract_route,
    extract_rxnorm_code,
)

log = get_logger(__name__)

SOURCE_PRIORITY = {
    "MedicationRequest": 3,
    "MedicationStatement": 2,
    "MedicationDispense": 1,
}

# Mapped to schema's Literal
PRIORITY_LABEL = {
    "MedicationRequest": "medication_request",
    "MedicationStatement": "medication_statement",
    "MedicationDispense": "medication_dispense",
}


async def run(
    request: GetMedicationsInput,
    context: SharpContext,
) -> GetMedicationsOutput:
    start = time.perf_counter()
    patient_id = request.patient_id or context.patient_id
    if not patient_id:
        return GetMedicationsOutput(
            as_of=request.as_of_date or date.today(),
            warnings=[
                Warning_(
                    code="missing_patient_id",
                    message="No patient_id provided in request or SHARP context.",
                )
            ],
            data_quality=MedicationDataQuality(coverage_score=0.0),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    async with FhirClient(context) as fhir, RxNormClient() as rx:
        try:
            requests_task = fhir.get_active_medication_requests(patient_id)
            statements_task = fhir.get_active_medication_statements(patient_id)
            dispenses_task = fhir.get_medication_dispenses(
                patient_id, lookback_days=min(request.lookback_days, 365)
            )
            med_requests, med_statements, med_dispenses = await asyncio.gather(
                requests_task, statements_task, dispenses_task, return_exceptions=True
            )
        except Exception as exc:  # noqa: BLE001
            log.error("get_medications.fhir_fetch_failed", error=str(exc))
            return _empty_result_with_warning(
                patient_id,
                request,
                start,
                code="fhir_fetch_error",
                message=str(exc),
            )

        warnings: list[Warning_] = []

        med_requests_list = _coerce_list(med_requests, "MedicationRequest", warnings)
        med_statements_list = _coerce_list(med_statements, "MedicationStatement", warnings)
        med_dispenses_list = _coerce_list(med_dispenses, "MedicationDispense", warnings)

        raw_candidates = _collect_candidates(
            med_requests_list, med_statements_list, med_dispenses_list, warnings
        )

        # Batch RxNorm normalization
        unique_rxcuis = list({c["rxcui"] for c in raw_candidates if c.get("rxcui")})
        norm_map = await rx.normalize_batch(unique_rxcuis) if unique_rxcuis else {}

        entries = _build_entries(raw_candidates, norm_map, warnings)
        dedup = _deduplicate(entries)

        coverage = _coverage_score(raw_candidates, norm_map)

        dq = MedicationDataQuality(
            has_medication_request=bool(med_requests_list),
            has_medication_statement=bool(med_statements_list),
            has_medication_dispense=bool(med_dispenses_list),
            most_recent_reconciliation_age_days=_reconciliation_age(
                med_statements_list, request.as_of_date or date.today()
            ),
            coverage_score=coverage,
        )

        citations: list[FhirReference] = []
        for e in dedup:
            citations.extend(e.sources)

        latency_ms = int((time.perf_counter() - start) * 1000)

        result = GetMedicationsOutput(
            patient_id=patient_id,
            as_of=request.as_of_date or date.today(),
            medications=dedup,
            data_quality=dq,
            warnings=warnings,
            citations=citations,
            latency_ms=latency_ms,
        )
        log.info(
            "get_medications.done",
            patient_id=patient_id,
            med_count=len(dedup),
            latency_ms=latency_ms,
            coverage=coverage,
        )
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_list(
    val: Any, resource_type: str, warnings: list[Warning_]
) -> list[dict[str, Any]]:
    """If the gathered value was an exception, record a warning and return []."""
    if isinstance(val, Exception):
        warnings.append(
            Warning_(
                code=f"{resource_type.lower()}_fetch_failed",
                message=f"Could not fetch {resource_type}: {val}",
            )
        )
        return []
    return list(val or [])


def _collect_candidates(
    med_requests: list[dict[str, Any]],
    med_statements: list[dict[str, Any]],
    med_dispenses: list[dict[str, Any]],
    warnings: list[Warning_],
) -> list[dict[str, Any]]:
    """Normalize FHIR resources into a uniform intermediate dict."""
    out: list[dict[str, Any]] = []

    for res in med_requests:
        out.append(_as_candidate(res, "MedicationRequest"))
    for res in med_statements:
        out.append(_as_candidate(res, "MedicationStatement"))
    for res in med_dispenses:
        out.append(_as_candidate(res, "MedicationDispense"))

    unmapped = sum(1 for c in out if not c.get("rxcui"))
    if unmapped:
        warnings.append(
            Warning_(
                code="unmapped_medications",
                message=f"{unmapped} medication resource(s) lacked an RxNorm code.",
            )
        )
    return out


def _as_candidate(res: dict[str, Any], resource_type: str) -> dict[str, Any]:
    rxcui, display = extract_rxnorm_code(res)
    return {
        "resource_type": resource_type,
        "resource_id": res.get("id", ""),
        "rxcui": rxcui,
        "display": display,
        "dose": extract_dose(res),
        "frequency": extract_frequency_text(res),
        "route": extract_route(res),
        "authored_on": extract_authored_date(res),
        "status": res.get("status", "unknown"),
        "prescriber_display": _prescriber_display(res),
    }


def _prescriber_display(res: dict[str, Any]) -> str | None:
    for key in ("requester", "informationSource", "performer"):
        val = res.get(key)
        if isinstance(val, dict):
            return val.get("display")
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, dict):
                d = first.get("actor") or first
                return d.get("display") if isinstance(d, dict) else None
    return None


def _build_entries(
    candidates: list[dict[str, Any]],
    norm_map: dict[str, RxNormNorm | None],
    warnings: list[Warning_],
) -> list[MedicationEntry]:
    entries: list[MedicationEntry] = []
    for c in candidates:
        norm = norm_map.get(c.get("rxcui") or "")
        ingredient_code: str | None = None
        ingredient_name: str | None = None
        atc: str | None = None
        if norm and norm.ingredients:
            ingredient_code = norm.ingredients[0].rxcui
            ingredient_name = norm.ingredients[0].name
            atc = norm.atc[0] if norm.atc else None

        source = FhirReference(
            resource_type=c["resource_type"],
            resource_id=c["resource_id"],
            label=(
                f"{c['resource_type']} authored {c['authored_on']}"
                if c.get("authored_on")
                else c["resource_type"]
            ),
        )
        status_raw = c.get("status", "unknown")
        if status_raw not in ("active", "on-hold", "completed", "stopped"):
            status_raw = "unknown"

        entries.append(
            MedicationEntry(
                rxnorm_ingredient_code=ingredient_code,
                rxnorm_ingredient_name=ingredient_name,
                rxnorm_clinical_drug_code=(
                    norm.clinical_drug.rxcui
                    if norm and norm.clinical_drug
                    else c.get("rxcui")
                ),
                clinical_drug_name=(
                    norm.clinical_drug.name if norm and norm.clinical_drug else c.get("display")
                ),
                dose=c.get("dose"),
                frequency=c.get("frequency"),
                route=c.get("route"),
                status=status_raw,  # type: ignore[arg-type]
                source_priority=PRIORITY_LABEL.get(
                    c["resource_type"], "medication_request"
                ),  # type: ignore[arg-type]
                sources=[source],
                first_seen=c.get("authored_on"),
                last_confirmed=c.get("authored_on"),
                therapeutic_class_atc=atc,
            )
        )

    # Flag entries that couldn't be normalized at all
    unmapped = sum(1 for e in entries if not e.rxnorm_ingredient_code)
    if unmapped and norm_map:
        warnings.append(
            Warning_(
                code="rxnorm_normalization_gaps",
                message=(
                    f"{unmapped} medication(s) could not be resolved to a canonical "
                    "ingredient; returned with raw code only."
                ),
            )
        )
    return entries


def _deduplicate(entries: list[MedicationEntry]) -> list[MedicationEntry]:
    """Group by (ingredient_code, route); keep highest source priority."""
    buckets: dict[tuple[str | None, str | None], list[MedicationEntry]] = defaultdict(list)
    for e in entries:
        key = (e.rxnorm_ingredient_code, e.route)
        buckets[key].append(e)

    deduped: list[MedicationEntry] = []
    for group in buckets.values():
        # Prefer highest priority source; tiebreak on most recent
        group.sort(
            key=lambda x: (
                -SOURCE_PRIORITY.get(
                    x.sources[0].resource_type if x.sources else "", 0
                ),
                -(x.last_confirmed.toordinal() if x.last_confirmed else 0),
            )
        )
        primary = group[0]
        # Merge sources from all group members so the citation trail is complete
        merged_sources: list[FhirReference] = []
        seen = set()
        for e in group:
            for s in e.sources:
                if s.reference not in seen:
                    merged_sources.append(s)
                    seen.add(s.reference)
        primary = primary.model_copy(update={"sources": merged_sources})
        deduped.append(primary)
    return deduped


def _coverage_score(
    candidates: list[dict[str, Any]], norm_map: dict[str, RxNormNorm | None]
) -> float:
    if not candidates:
        return 1.0  # vacuous success — no meds to map
    mapped = sum(
        1 for c in candidates if c.get("rxcui") and norm_map.get(c["rxcui"]) is not None
    )
    return round(mapped / len(candidates), 3)


def _reconciliation_age(
    statements: list[dict[str, Any]], as_of: date
) -> int | None:
    latest: date | None = None
    for s in statements:
        d = extract_authored_date(s)
        if d and (latest is None or d > latest):
            latest = d
    if not latest:
        return None
    return (as_of - latest).days


def _empty_result_with_warning(
    patient_id: str,
    request: GetMedicationsInput,
    start: float,
    *,
    code: str,
    message: str,
) -> GetMedicationsOutput:
    return GetMedicationsOutput(
        patient_id=patient_id,
        as_of=request.as_of_date or date.today(),
        warnings=[Warning_(code=code, message=message)],
        data_quality=MedicationDataQuality(coverage_score=0.0),
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
