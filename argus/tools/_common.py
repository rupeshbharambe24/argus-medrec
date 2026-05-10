"""Helpers shared across tool implementations.

Resource parsing from FHIR JSON to our internal dataclasses/Pydantic models,
calculation utilities (eGFR, age), and common plumbing.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from argus.schemas import Dose, FhirReference

# ---------------------------------------------------------------------------
# Patient helpers
# ---------------------------------------------------------------------------


def patient_age_years(patient: dict[str, Any], as_of: date | None = None) -> int | None:
    """Compute age in whole years from a FHIR Patient resource."""
    bd = patient.get("birthDate")
    if not bd:
        return None
    try:
        birth = date.fromisoformat(bd)
    except ValueError:
        return None
    ref = as_of or date.today()
    return ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))


def patient_sex(patient: dict[str, Any]) -> str | None:
    return patient.get("gender")


def patient_display_name(patient: dict[str, Any]) -> str | None:
    names = patient.get("name") or []
    if not names:
        return None
    n = names[0]
    given = " ".join(n.get("given", [])) if n.get("given") else ""
    family = n.get("family", "")
    display = f"{given} {family}".strip()
    return display or None


# ---------------------------------------------------------------------------
# Medication helpers
# ---------------------------------------------------------------------------


def extract_rxnorm_code(med_resource: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull the RxNorm code + display from a MedicationRequest/Statement/Dispense.

    FHIR allows either an inline `medicationCodeableConcept` or a `medicationReference`
    to a Medication resource. We prefer the inline concept when present.

    Returns:
        (rxcui, display) — either may be None if not available.
    """
    # Inline concept is the common case in Synthea output
    cc = med_resource.get("medicationCodeableConcept") or {}
    codings = cc.get("coding") or []
    for coding in codings:
        system = (coding.get("system") or "").lower()
        if "rxnorm" in system:
            return coding.get("code"), coding.get("display") or cc.get("text")
    # Fallback — text only
    if cc.get("text"):
        return None, cc.get("text")
    return None, None


def extract_dose(med_resource: dict[str, Any]) -> Dose | None:
    """Pull the dose from the first dosageInstruction."""
    di_list = med_resource.get("dosageInstruction") or med_resource.get("dosage") or []
    if not di_list:
        return None
    di = di_list[0]
    doses = di.get("doseAndRate") or []
    if not doses:
        return None
    qty = doses[0].get("doseQuantity") or {}
    if "value" in qty and "unit" in qty:
        return Dose(value=float(qty["value"]), unit=str(qty["unit"]))
    return None


def extract_frequency_text(med_resource: dict[str, Any]) -> str | None:
    """Human-readable frequency, e.g. 'twice daily'. Falls back to timing-based text."""
    di_list = med_resource.get("dosageInstruction") or med_resource.get("dosage") or []
    if not di_list:
        return None
    di = di_list[0]
    if di.get("text"):
        return di["text"]
    timing = (di.get("timing") or {}).get("repeat") or {}
    freq = timing.get("frequency")
    period = timing.get("period")
    period_unit = timing.get("periodUnit")
    if freq is not None and period is not None and period_unit:
        return f"{freq} every {period} {period_unit}"
    return None


def extract_route(med_resource: dict[str, Any]) -> str | None:
    di_list = med_resource.get("dosageInstruction") or med_resource.get("dosage") or []
    if not di_list:
        return None
    route = di_list[0].get("route") or {}
    codings = route.get("coding") or []
    if codings:
        return codings[0].get("display") or codings[0].get("code")
    return route.get("text")


def extract_authored_date(med_resource: dict[str, Any]) -> date | None:
    for key in ("authoredOn", "effectiveDateTime", "dateAsserted", "whenHandedOver"):
        val = med_resource.get(key)
        if val:
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00")).date()
            except ValueError:
                continue
    return None


def resource_ref(res: dict[str, Any], label: str | None = None) -> FhirReference:
    return FhirReference(
        resource_type=res.get("resourceType", "Resource"),
        resource_id=res.get("id", ""),
        label=label,
    )


# ---------------------------------------------------------------------------
# eGFR
# ---------------------------------------------------------------------------


def egfr_ckd_epi_2021(
    creatinine_mg_dl: float,
    age_years: int,
    sex: str,
) -> float:
    """CKD-EPI 2021 race-free creatinine equation.

    Reference: Inker LA, et al. N Engl J Med 2021;385:1737–1749.
    Formula:
        eGFR = 142 * min(Scr/κ, 1)^α * max(Scr/κ, 1)^(-1.200) * 0.9938^age * (1.012 if female)

    Where κ = 0.7 (female) or 0.9 (male), α = -0.241 (female) or -0.302 (male).
    """
    female = sex.lower() in ("female", "f")
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = creatinine_mg_dl / kappa
    return (
        142.0
        * (min(ratio, 1) ** alpha)
        * (max(ratio, 1) ** -1.200)
        * (0.9938 ** age_years)
        * (1.012 if female else 1.0)
    )


def ckd_stage(egfr: float) -> str:
    if egfr >= 90:
        return "1"
    if egfr >= 60:
        return "2"
    if egfr >= 45:
        return "3a"
    if egfr >= 30:
        return "3b"
    if egfr >= 15:
        return "4"
    return "5"


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def observation_value_quantity(obs: dict[str, Any]) -> tuple[float | None, str | None]:
    q = obs.get("valueQuantity") or {}
    if "value" in q:
        return float(q["value"]), q.get("unit")
    return None, None


def observation_date(obs: dict[str, Any]) -> date | None:
    for key in ("effectiveDateTime", "issued"):
        val = obs.get(key)
        if val:
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00")).date()
            except ValueError:
                continue
    return None


def pick_latest_observation(
    obs_list: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    latest_date: date | None = None
    for obs in obs_list:
        d = observation_date(obs)
        if d and (latest_date is None or d > latest_date):
            latest = obs
            latest_date = d
    return latest
