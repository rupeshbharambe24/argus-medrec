"""Scan a folder of Synthea FHIR bundles and rank patients by clinical richness.

Picks the best demo candidates for Argus by counting:
  - active MedicationRequest entries
  - Observation count (proxy for lab availability — needed for renal_dose_check)
  - Condition count
  - patient age (older = more likely to trigger Beers / polypharmacy patterns)

Usage:
    python scripts/rank_synthea_bundles.py path/to/synthea/output/fhir
    python scripts/rank_synthea_bundles.py path/to/synthea/output/fhir --top 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path


def _age(birth_date: str | None) -> int | None:
    if not birth_date:
        return None
    try:
        bd = datetime.fromisoformat(birth_date).date()
    except ValueError:
        return None
    today = date.today()
    return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))


def _has_creatinine(observations: list[dict]) -> bool:
    for obs in observations:
        codings = (obs.get("code") or {}).get("coding") or []
        for c in codings:
            code = (c.get("code") or "").lower()
            display = (c.get("display") or "").lower()
            if code in {"2160-0", "38483-4"} or "creatinine" in display:
                return True
    return False


def score_bundle(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        bundle = json.load(f)
    entries = bundle.get("entry") or []

    patient = None
    med_requests_active = 0
    med_statements = 0
    observations: list[dict] = []
    conditions = 0
    encounters = 0

    for e in entries:
        res = e.get("resource") or {}
        rt = res.get("resourceType")
        if rt == "Patient" and patient is None:
            patient = res
        elif rt == "MedicationRequest":
            if (res.get("status") or "").lower() == "active":
                med_requests_active += 1
        elif rt == "MedicationStatement":
            if (res.get("status") or "").lower() in {"active", "intended"}:
                med_statements += 1
        elif rt == "Observation":
            observations.append(res)
        elif rt == "Condition":
            conditions += 1
        elif rt == "Encounter":
            encounters += 1

    age = _age((patient or {}).get("birthDate"))
    has_cr = _has_creatinine(observations)

    # Composite score — favors polypharmacy, elderly, with available labs.
    score = (
        med_requests_active * 5
        + med_statements * 3
        + (10 if has_cr else 0)
        + (15 if age is not None and age >= 65 else 0)
        + min(conditions, 10)
    )

    name = "?"
    if patient:
        name_obj = (patient.get("name") or [{}])[0]
        family = name_obj.get("family", "")
        given = " ".join(name_obj.get("given") or [])
        name = f"{given} {family}".strip() or "?"

    return {
        "file": path.name,
        "name": name,
        "age": age,
        "sex": (patient or {}).get("gender", "?"),
        "active_med_requests": med_requests_active,
        "med_statements": med_statements,
        "observations": len(observations),
        "has_creatinine": has_cr,
        "conditions": conditions,
        "encounters": encounters,
        "score": score,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="Folder containing Synthea *.json bundles")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument(
        "--min-meds",
        type=int,
        default=1,
        help="Minimum active medication requests to include (default 1)",
    )
    args = ap.parse_args()

    if not args.folder.exists():
        print(f"[error] folder not found: {args.folder}")
        return 1

    files = sorted(args.folder.glob("*.json"))
    if not files:
        print(f"[error] no .json bundles in {args.folder}")
        return 1

    print(f"Scanning {len(files)} bundles...")
    results = []
    for f in files:
        # Skip Synthea aux files like hospitalInformation, practitionerInformation
        name = f.name.lower()
        if name.startswith(("hospitalinformation", "practitionerinformation")):
            continue
        try:
            results.append(score_bundle(f))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {f.name}: {exc}")

    results = [r for r in results if r["active_med_requests"] >= args.min_meds]
    results.sort(key=lambda r: r["score"], reverse=True)

    print(
        f"\nTop {min(args.top, len(results))} of {len(results)} candidates "
        f"(filter: active_med_requests >= {args.min_meds}):\n"
    )
    print(
        f"{'#':>3}  {'score':>5}  {'age':>3} {'sx':>2}  "
        f"{'meds':>4} {'stmt':>4} {'obs':>4} {'cr':>2}  {'cond':>4}  "
        f"name / file"
    )
    print("-" * 110)
    for i, r in enumerate(results[: args.top], 1):
        cr = "Y" if r["has_creatinine"] else "-"
        age = r["age"] if r["age"] is not None else "?"
        print(
            f"{i:>3}  {r['score']:>5}  {age:>3} {r['sex'][:1]:>2}  "
            f"{r['active_med_requests']:>4} {r['med_statements']:>4} "
            f"{r['observations']:>4} {cr:>2}  {r['conditions']:>4}  "
            f"{r['name']}  ({r['file']})"
        )

    if results:
        best = results[0]
        print(
            f"\nBest demo candidate: {best['name']}  ->  "
            f"upload `{args.folder / best['file']}` to Prompt Opinion."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
