"""Tool 6: screen_high_risk_patterns.

Multi-screen safety sweep for patterns the other tools don't cover:

    - Beers (AGS 2023): potentially inappropriate medications in age ≥65
    - QTc: multiple QTc-prolonging drugs + electrolyte context
    - Opioid + benzodiazepine coprescription (CDC 2022)
    - Anticholinergic burden (ACB scale, flagged ≥3)
    - Pregnancy: Category D/X meds in females 15-50
    - Adherence gap: MPR < 0.8 from MedicationDispense history
    - Serotonin syndrome risk: 2+ serotonergic agents

Each screen is independent and fast; all run concurrently.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

import aiosqlite

from argus.config import get_settings
from argus.fhir_client import FhirClient
from argus.logging_setup import get_logger
from argus.schemas import (
    FhirReference,
    GetMedicationsInput,
    MedicationEntry,
    ScreenFinding,
    ScreenPatternsInput,
    ScreenPatternsOutput,
    ScreenType,
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


# LOINCs used by QTc screen
LOINC_POTASSIUM = ["6298-4", "2823-3"]       # K+ serum / plasma
LOINC_MAGNESIUM = ["2601-3", "19123-9"]      # Mg
LOINC_QTC = ["8634-8", "8633-0"]             # Corrected QT

# Age boundaries
GERIATRIC_AGE = 65
REPRO_AGE_MIN = 15
REPRO_AGE_MAX = 50


async def run(
    request: ScreenPatternsInput, context: SharpContext
) -> ScreenPatternsOutput:
    start = time.perf_counter()
    patient_id = request.patient_id or context.patient_id
    if not patient_id:
        return ScreenPatternsOutput(
            warnings=[Warning_(code="missing_patient_id", message="No patient_id.")],
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    async with FhirClient(context) as fhir:
        patient_task = fhir.get_patient(patient_id)
        meds_task = run_get_medications(GetMedicationsInput(patient_id=patient_id), context)
        potassium_task = fhir.get_observations(
            patient_id, loinc_codes=LOINC_POTASSIUM, lookback_days=30
        )
        magnesium_task = fhir.get_observations(
            patient_id, loinc_codes=LOINC_MAGNESIUM, lookback_days=30
        )
        qtc_task = fhir.get_observations(
            patient_id, loinc_codes=LOINC_QTC, lookback_days=90
        )
        dispenses_task = fhir.get_medication_dispenses(patient_id, lookback_days=365)

        results = await asyncio.gather(
            patient_task,
            meds_task,
            potassium_task,
            magnesium_task,
            qtc_task,
            dispenses_task,
            return_exceptions=True,
        )

    warnings: list[Warning_] = []
    patient, med_result, k_obs, mg_obs, qtc_obs, dispenses = _unpack(results, warnings)

    med_list: list[MedicationEntry] = (
        med_result.medications if hasattr(med_result, "medications") else []
    )

    age = patient_age_years(patient) if patient else None
    sex = patient_sex(patient) if patient else None

    findings: list[ScreenFinding] = []
    screens_run: list[ScreenType] = []

    settings = get_settings()
    async with aiosqlite.connect(str(settings.reference_kb_path)) as db:
        db.row_factory = aiosqlite.Row

        for screen in request.screens:
            try:
                if screen == ScreenType.BEERS:
                    screens_run.append(screen)
                    if age is not None and age >= GERIATRIC_AGE:
                        findings.extend(await _screen_beers(db, med_list))
                elif screen == ScreenType.QTC:
                    screens_run.append(screen)
                    findings.extend(
                        await _screen_qtc(db, med_list, k_obs, mg_obs, qtc_obs)
                    )
                elif screen == ScreenType.OPIOID_BENZO:
                    screens_run.append(screen)
                    findings.extend(_screen_opioid_benzo(med_list))
                elif screen == ScreenType.ANTICHOLINERGIC_BURDEN:
                    screens_run.append(screen)
                    findings.extend(await _screen_anticholinergic(db, med_list))
                elif screen == ScreenType.PREGNANCY:
                    screens_run.append(screen)
                    if (
                        sex
                        and sex.lower() == "female"
                        and age is not None
                        and REPRO_AGE_MIN <= age <= REPRO_AGE_MAX
                    ):
                        findings.extend(await _screen_pregnancy(db, med_list))
                elif screen == ScreenType.ADHERENCE_GAP:
                    screens_run.append(screen)
                    findings.extend(_screen_adherence(med_list, dispenses))
                elif screen == ScreenType.SEROTONIN_SYNDROME:
                    screens_run.append(screen)
                    findings.extend(await _screen_serotonin(db, med_list))
            except Exception as exc:  # noqa: BLE001
                log.warning("screen.failed", screen=screen.value, error=str(exc))
                warnings.append(
                    Warning_(code=f"screen_{screen.value}_failed", message=str(exc))
                )

    citations = _collect_citations(findings)
    latency_ms = int((time.perf_counter() - start) * 1000)

    log.info(
        "screen_patterns.done",
        patient_id=patient_id,
        screens=len(screens_run),
        findings=len(findings),
        latency_ms=latency_ms,
    )

    return ScreenPatternsOutput(
        patient_id=patient_id,
        screens_run=screens_run,
        findings=findings,
        warnings=warnings,
        citations=citations,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Individual screens
# ---------------------------------------------------------------------------


async def _screen_beers(
    db: aiosqlite.Connection, med_list: list[MedicationEntry]
) -> list[ScreenFinding]:
    ingredients = [m.rxnorm_ingredient_code for m in med_list if m.rxnorm_ingredient_code]
    if not ingredients:
        return []
    placeholders = ",".join(["?"] * len(ingredients))
    async with db.execute(
        f"""
        SELECT rxnorm_ingredient, pim_category, rationale, alternative, severity
        FROM beers_criteria
        WHERE rxnorm_ingredient IN ({placeholders})
        """,
        ingredients,
    ) as cur:
        rows = await cur.fetchall()

    by_code = {m.rxnorm_ingredient_code: m for m in med_list if m.rxnorm_ingredient_code}
    findings: list[ScreenFinding] = []
    for row in rows:
        med = by_code.get(row["rxnorm_ingredient"])
        if not med:
            continue
        sev = Severity(row["severity"]) if row["severity"] else Severity.MODERATE
        findings.append(
            ScreenFinding(
                screen=ScreenType.BEERS,
                severity=sev,
                title="Potentially Inappropriate Medication in Elderly",
                description=(
                    f"{med.rxnorm_ingredient_name} — {row['rationale']} "
                    f"(Beers category: {row['pim_category']})"
                ),
                guideline="AGS Beers 2023",
                medications_involved=[
                    {"rxnorm": med.rxnorm_ingredient_code, "name": med.rxnorm_ingredient_name}
                ],
                recommended_alternative=row["alternative"],
                evidence=list(med.sources),
            )
        )
    return findings


async def _screen_qtc(
    db: aiosqlite.Connection,
    med_list: list[MedicationEntry],
    k_obs: list[dict[str, Any]],
    mg_obs: list[dict[str, Any]],
    qtc_obs: list[dict[str, Any]],
) -> list[ScreenFinding]:
    ingredients = [m.rxnorm_ingredient_code for m in med_list if m.rxnorm_ingredient_code]
    if not ingredients:
        return []
    placeholders = ",".join(["?"] * len(ingredients))
    async with db.execute(
        f"""
        SELECT rxnorm_ingredient, risk_category
        FROM qtc_drugs
        WHERE rxnorm_ingredient IN ({placeholders})
        """,
        ingredients,
    ) as cur:
        rows = await cur.fetchall()

    if len(rows) < 2:
        return []

    by_code = {m.rxnorm_ingredient_code: m for m in med_list if m.rxnorm_ingredient_code}
    involved = [by_code[row["rxnorm_ingredient"]] for row in rows if row["rxnorm_ingredient"] in by_code]

    latest_k = pick_latest_observation(k_obs)
    k_val, _ = observation_value_quantity(latest_k) if latest_k else (None, None)

    latest_mg = pick_latest_observation(mg_obs)
    mg_val, _ = observation_value_quantity(latest_mg) if latest_mg else (None, None)

    latest_qtc = pick_latest_observation(qtc_obs)
    qtc_val, _ = observation_value_quantity(latest_qtc) if latest_qtc else (None, None)

    hypo_k = k_val is not None and k_val < 3.5
    hypo_mg = mg_val is not None and mg_val < 1.8
    long_qtc = qtc_val is not None and qtc_val > 470

    severity = Severity.MODERATE
    if len(rows) >= 3 or hypo_k or hypo_mg or long_qtc:
        severity = Severity.MAJOR

    description_parts = [
        f"{len(rows)} QTc-prolonging medications concurrently: "
        + ", ".join(m.rxnorm_ingredient_name for m in involved if m.rxnorm_ingredient_name)
    ]
    if hypo_k:
        description_parts.append(f"K+ low ({k_val} mEq/L) — proarrhythmic")
    if hypo_mg:
        description_parts.append(f"Mg low ({mg_val} mg/dL) — proarrhythmic")
    if long_qtc:
        description_parts.append(f"Most recent QTc {qtc_val} ms (>470 prolonged)")
    if not qtc_obs:
        description_parts.append("No ECG on file in past 90 days")

    evidence: list[FhirReference] = []
    for m in involved:
        evidence.extend(m.sources)
    for obs in (latest_k, latest_mg, latest_qtc):
        if obs:
            evidence.append(resource_ref(obs))

    return [
        ScreenFinding(
            screen=ScreenType.QTC,
            severity=severity,
            title="Multiple QTc-prolonging medications",
            description=". ".join(description_parts),
            guideline="CredibleMeds KnownRisk",
            medications_involved=[
                {"rxnorm": m.rxnorm_ingredient_code, "name": m.rxnorm_ingredient_name}
                for m in involved
            ],
            recommended_alternative=(
                "Obtain ECG; replete K+ to ≥4.0 and Mg ≥2.0; consider substitution "
                "of lowest-priority QTc offender with alternative agent."
            ),
            evidence=evidence,
        )
    ]


def _screen_opioid_benzo(med_list: list[MedicationEntry]) -> list[ScreenFinding]:
    opioid_atc_prefixes = ("N02A",)  # opioids
    benzo_atc_prefixes = ("N05BA", "N05CD")  # benzos + BZ-like hypnotics

    opioids = [m for m in med_list if m.therapeutic_class_atc and m.therapeutic_class_atc.startswith(opioid_atc_prefixes)]
    benzos = [m for m in med_list if m.therapeutic_class_atc and m.therapeutic_class_atc.startswith(benzo_atc_prefixes)]

    # Fallback ingredient-level matching if ATC not populated
    opioid_names = {"oxycodone", "hydrocodone", "morphine", "fentanyl", "tramadol", "codeine", "hydromorphone", "oxymorphone", "methadone", "tapentadol"}
    benzo_names = {"alprazolam", "diazepam", "lorazepam", "clonazepam", "temazepam", "midazolam", "triazolam", "oxazepam", "chlordiazepoxide"}

    if not opioids:
        opioids = [m for m in med_list if (m.rxnorm_ingredient_name or "").lower() in opioid_names]
    if not benzos:
        benzos = [m for m in med_list if (m.rxnorm_ingredient_name or "").lower() in benzo_names]

    if not opioids or not benzos:
        return []

    evidence: list[FhirReference] = []
    for m in opioids + benzos:
        evidence.extend(m.sources)

    return [
        ScreenFinding(
            screen=ScreenType.OPIOID_BENZO,
            severity=Severity.MAJOR,
            title="Concurrent opioid + benzodiazepine",
            description=(
                f"Opioid(s): {', '.join(m.rxnorm_ingredient_name or '?' for m in opioids)}. "
                f"Benzodiazepine(s): {', '.join(m.rxnorm_ingredient_name or '?' for m in benzos)}. "
                "CDC 2022 opioid guideline discourages this combination due to "
                "respiratory depression risk."
            ),
            guideline="CDC 2022 Opioid Prescribing Guideline",
            medications_involved=[
                {"rxnorm": m.rxnorm_ingredient_code, "name": m.rxnorm_ingredient_name}
                for m in opioids + benzos
            ],
            recommended_alternative="Taper lower-priority agent; if both indicated, naloxone co-prescription.",
            evidence=evidence,
        )
    ]


async def _screen_anticholinergic(
    db: aiosqlite.Connection, med_list: list[MedicationEntry]
) -> list[ScreenFinding]:
    ingredients = [m.rxnorm_ingredient_code for m in med_list if m.rxnorm_ingredient_code]
    if not ingredients:
        return []
    placeholders = ",".join(["?"] * len(ingredients))
    async with db.execute(
        f"""
        SELECT rxnorm_ingredient, acb_score
        FROM anticholinergic_burden
        WHERE rxnorm_ingredient IN ({placeholders})
        """,
        ingredients,
    ) as cur:
        rows = await cur.fetchall()

    if not rows:
        return []

    by_code = {m.rxnorm_ingredient_code: m for m in med_list if m.rxnorm_ingredient_code}
    total_score = 0
    involved: list[MedicationEntry] = []
    for row in rows:
        med = by_code.get(row["rxnorm_ingredient"])
        if med:
            total_score += int(row["acb_score"])
            involved.append(med)

    if total_score < 3:
        return []

    evidence: list[FhirReference] = []
    for m in involved:
        evidence.extend(m.sources)

    return [
        ScreenFinding(
            screen=ScreenType.ANTICHOLINERGIC_BURDEN,
            severity=Severity.MAJOR if total_score >= 5 else Severity.MODERATE,
            title=f"High anticholinergic burden (ACB score {total_score})",
            description=(
                f"Cumulative anticholinergic burden from {len(involved)} medications: "
                f"{', '.join(m.rxnorm_ingredient_name or '?' for m in involved)}. "
                "Associated with cognitive decline, delirium, and falls, particularly in elderly."
            ),
            guideline="Anticholinergic Cognitive Burden Scale (Boustani 2008)",
            medications_involved=[
                {"rxnorm": m.rxnorm_ingredient_code, "name": m.rxnorm_ingredient_name}
                for m in involved
            ],
            recommended_alternative="Review each agent for necessity; substitute alternatives where possible.",
            evidence=evidence,
        )
    ]


async def _screen_pregnancy(
    db: aiosqlite.Connection, med_list: list[MedicationEntry]
) -> list[ScreenFinding]:
    ingredients = [m.rxnorm_ingredient_code for m in med_list if m.rxnorm_ingredient_code]
    if not ingredients:
        return []
    placeholders = ",".join(["?"] * len(ingredients))
    async with db.execute(
        f"""
        SELECT rxnorm_ingredient, category, pllr_summary
        FROM pregnancy_categories
        WHERE rxnorm_ingredient IN ({placeholders})
          AND category IN ('D', 'X')
        """,
        ingredients,
    ) as cur:
        rows = await cur.fetchall()

    by_code = {m.rxnorm_ingredient_code: m for m in med_list if m.rxnorm_ingredient_code}
    findings: list[ScreenFinding] = []
    for row in rows:
        med = by_code.get(row["rxnorm_ingredient"])
        if not med:
            continue
        sev = Severity.CRITICAL if row["category"] == "X" else Severity.MAJOR
        findings.append(
            ScreenFinding(
                screen=ScreenType.PREGNANCY,
                severity=sev,
                title=f"Pregnancy Category {row['category']} medication in reproductive-age patient",
                description=f"{med.rxnorm_ingredient_name}: {row['pllr_summary']}",
                guideline="FDA PLLR / Category",
                medications_involved=[
                    {"rxnorm": med.rxnorm_ingredient_code, "name": med.rxnorm_ingredient_name}
                ],
                recommended_alternative="Confirm non-pregnancy or discuss alternatives; ensure contraception if teratogenic.",
                evidence=list(med.sources),
            )
        )
    return findings


def _screen_adherence(
    med_list: list[MedicationEntry], dispenses: list[dict[str, Any]]
) -> list[ScreenFinding]:
    """Crude MPR: fraction of last-6-months covered by dispenses."""
    if not dispenses:
        return []

    by_rxcui: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in dispenses:
        cc = (d.get("medicationCodeableConcept") or {}).get("coding") or []
        for c in cc:
            if "rxnorm" in (c.get("system") or "").lower():
                by_rxcui[c.get("code")].append(d)
                break

    findings: list[ScreenFinding] = []

    for med in med_list:
        if not med.rxnorm_clinical_drug_code:
            continue
        disp_list = by_rxcui.get(med.rxnorm_clinical_drug_code, [])
        if len(disp_list) < 2:
            continue
        covered_days = 0
        seen_refs: list[FhirReference] = []
        for d in disp_list:
            qty = d.get("daysSupply", {}).get("value")
            hand = d.get("whenHandedOver")
            if qty and hand:
                try:
                    covered_days += int(qty)
                except (TypeError, ValueError):
                    continue
                seen_refs.append(resource_ref(d, f"Dispense {hand[:10]}"))
        mpr = covered_days / 180.0
        if mpr < 0.8:
            findings.append(
                ScreenFinding(
                    screen=ScreenType.ADHERENCE_GAP,
                    severity=Severity.MODERATE if mpr > 0.5 else Severity.MAJOR,
                    title=f"Possible non-adherence: {med.rxnorm_ingredient_name}",
                    description=(
                        f"Medication Possession Ratio (MPR) {mpr:.2f} over last 180 days "
                        f"(threshold 0.80). {len(disp_list)} dispenses covered {covered_days} days."
                    ),
                    guideline="MPR < 0.8 commonly used adherence threshold",
                    medications_involved=[
                        {"rxnorm": med.rxnorm_ingredient_code, "name": med.rxnorm_ingredient_name}
                    ],
                    recommended_alternative="Patient education; simplify regimen; confirm cost/access barriers.",
                    evidence=seen_refs,
                )
            )
    return findings


async def _screen_serotonin(
    db: aiosqlite.Connection, med_list: list[MedicationEntry]
) -> list[ScreenFinding]:
    ingredients = [m.rxnorm_ingredient_code for m in med_list if m.rxnorm_ingredient_code]
    if not ingredients:
        return []
    placeholders = ",".join(["?"] * len(ingredients))
    async with db.execute(
        f"""
        SELECT rxnorm_ingredient, mechanism
        FROM serotonergic_drugs
        WHERE rxnorm_ingredient IN ({placeholders})
        """,
        ingredients,
    ) as cur:
        rows = await cur.fetchall()
    if len(rows) < 2:
        return []
    by_code = {m.rxnorm_ingredient_code: m for m in med_list if m.rxnorm_ingredient_code}
    involved = [by_code[r["rxnorm_ingredient"]] for r in rows if r["rxnorm_ingredient"] in by_code]
    ev: list[FhirReference] = []
    for m in involved:
        ev.extend(m.sources)
    return [
        ScreenFinding(
            screen=ScreenType.SEROTONIN_SYNDROME,
            severity=Severity.MAJOR,
            title="Multiple serotonergic agents — serotonin syndrome risk",
            description=(
                "Concurrent serotonergic medications: "
                + ", ".join(m.rxnorm_ingredient_name or "?" for m in involved)
            ),
            guideline="Hunter serotonin toxicity criteria",
            medications_involved=[
                {"rxnorm": m.rxnorm_ingredient_code, "name": m.rxnorm_ingredient_name}
                for m in involved
            ],
            recommended_alternative="Reduce overlap where clinically possible; monitor for tremor, hyperreflexia, autonomic instability.",
            evidence=ev,
        )
    ]


# ---------------------------------------------------------------------------


def _unpack(results, warnings: list[Warning_]):
    def _or_empty(r, code: str, default):
        if isinstance(r, Exception):
            warnings.append(Warning_(code=code, message=str(r)))
            return default
        return r

    patient, med_result, k_obs, mg_obs, qtc_obs, dispenses = results
    return (
        _or_empty(patient, "patient_fetch_failed", {}),
        _or_empty(med_result, "medications_fetch_failed", None),
        _or_empty(k_obs, "potassium_fetch_failed", []),
        _or_empty(mg_obs, "magnesium_fetch_failed", []),
        _or_empty(qtc_obs, "qtc_fetch_failed", []),
        _or_empty(dispenses, "dispenses_fetch_failed", []),
    )


def _collect_citations(findings: list[ScreenFinding]) -> list[FhirReference]:
    cites: list[FhirReference] = []
    seen: set[str] = set()
    for f in findings:
        for ref in f.evidence:
            if ref.reference not in seen:
                cites.append(ref)
                seen.add(ref.reference)
    return cites
