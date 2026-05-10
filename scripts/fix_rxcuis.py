"""Look up canonical ingredient RxCUIs from RxNav and rewrite seed CSVs.

The original seed files were drafted from memory and contain incorrect /
duplicated RxCUI codes. This script:

1. Reads every reference CSV
2. Collects unique drug names
3. Queries RxNav for the canonical ingredient RxCUI of each
4. Rewrites the CSVs with correct codes
5. Drops rows whose names cannot be resolved (with a warning)

Run once when seed data is updated. Idempotent.
"""

from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import httpx

DATA = Path(__file__).resolve().parent.parent / "argus" / "reference" / "data"
RXNAV = "https://rxnav.nlm.nih.gov/REST"

# Files that have a single rxnorm_ingredient + ingredient_name column
SINGLE_INGREDIENT_FILES = [
    "renal_rules.csv",
    "beers_2023.csv",
    "qtc_drugs.csv",
    "anticholinergic_burden.csv",
    "pregnancy_categories.csv",
    "serotonergic_drugs.csv",
]
# DDI file has rxnorm_a/name_a/rxnorm_b/name_b
DDI_FILE = "drug_interactions.csv"


def normalize_name(name: str) -> str:
    """Strip qualifiers like '(as first-line HTN)' to improve match rate."""
    name = re.sub(r"\([^)]*\)", "", name).strip()
    # Common variants
    name = name.replace(" oral/patch", "")
    name = name.replace(" IR", "")
    return name.strip()


def lookup_rxcui(client: httpx.Client, name: str) -> str | None:
    """Query RxNav for the ingredient RxCUI of a drug name. Returns None if not found."""
    nname = normalize_name(name)
    if not nname:
        return None

    # Try exact ingredient match
    try:
        resp = client.get(f"{RXNAV}/rxcui.json", params={"name": nname, "search": "2"})
        if resp.status_code == 200:
            ids = (resp.json().get("idGroup") or {}).get("rxnormId") or []
            if ids:
                return _resolve_to_ingredient(client, ids[0])
    except httpx.HTTPError:
        pass

    # Fall back to approximateTerm
    try:
        resp = client.get(
            f"{RXNAV}/approximateTerm.json", params={"term": nname, "maxEntries": "1"}
        )
        if resp.status_code == 200:
            cands = (resp.json().get("approximateGroup") or {}).get("candidate") or []
            if cands:
                rxcui = cands[0].get("rxcui")
                if rxcui:
                    return _resolve_to_ingredient(client, rxcui)
    except httpx.HTTPError:
        pass

    return None


def _resolve_to_ingredient(client: httpx.Client, rxcui: str) -> str | None:
    """Given any RxCUI, return its ingredient-level (TTY=IN) RxCUI."""
    try:
        resp = client.get(f"{RXNAV}/rxcui/{rxcui}/related.json", params={"tty": "IN"})
        if resp.status_code != 200:
            return rxcui
        groups = (resp.json().get("relatedGroup") or {}).get("conceptGroup") or []
        for g in groups:
            if g.get("tty") == "IN":
                props = g.get("conceptProperties") or []
                if props:
                    return props[0].get("rxcui") or rxcui
        return rxcui
    except httpx.HTTPError:
        return rxcui


def rewrite_single_ingredient(client: httpx.Client, fname: str) -> tuple[int, int]:
    path = DATA / fname
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return 0, 0

    # Cache lookups by name
    name_to_rxcui: dict[str, str | None] = {}
    fixed = []
    dropped = 0

    for row in rows:
        name = (row.get("ingredient_name") or "").strip()
        if not name:
            dropped += 1
            continue
        if name not in name_to_rxcui:
            name_to_rxcui[name] = lookup_rxcui(client, name)
            time.sleep(0.05)  # be polite to RxNav
        rxcui = name_to_rxcui[name]
        if not rxcui:
            print(f"  ⚠ {fname}: could not resolve '{name}' — dropping row")
            dropped += 1
            continue
        row["rxnorm_ingredient"] = rxcui
        fixed.append(row)

    # Write back — strip any None keys that DictReader produced from raw rows
    fieldnames = [k for k in rows[0].keys() if k is not None]
    cleaned = [{k: v for k, v in r.items() if k is not None} for r in fixed]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)

    return len(fixed), dropped


def rewrite_ddi(client: httpx.Client) -> tuple[int, int]:
    path = DATA / DDI_FILE
    rows = list(csv.DictReader(path.open()))
    name_cache: dict[str, str | None] = {}
    fixed = []
    dropped = 0

    for row in rows:
        name_a = (row.get("name_a") or "").strip()
        name_b = (row.get("name_b") or "").strip()
        if not name_a or not name_b:
            dropped += 1
            continue
        for n in (name_a, name_b):
            if n not in name_cache:
                name_cache[n] = lookup_rxcui(client, n)
                time.sleep(0.05)
        a, b = name_cache[name_a], name_cache[name_b]
        if not a or not b:
            print(f"  ⚠ {DDI_FILE}: could not resolve {name_a} × {name_b}")
            dropped += 1
            continue
        # Normalize ordering
        if a > b:
            a, b = b, a
            row["rxnorm_a"], row["name_a"], row["rxnorm_b"], row["name_b"] = (
                a, name_b, b, name_a,
            )
        else:
            row["rxnorm_a"] = a
            row["rxnorm_b"] = b
        fixed.append(row)

    # Dedupe (a, b) pairs after normalization
    seen = set()
    deduped = []
    for r in fixed:
        key = (r["rxnorm_a"], r["rxnorm_b"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    fieldnames = [k for k in rows[0].keys() if k is not None]
    cleaned = [{k: v for k, v in r.items() if k is not None} for r in deduped]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)
    return len(deduped), dropped + (len(fixed) - len(deduped))


def main() -> int:
    print(f"Querying RxNav and rewriting CSVs in {DATA}\n")
    with httpx.Client(timeout=10.0) as client:
        for fname in SINGLE_INGREDIENT_FILES:
            print(f"→ {fname}")
            kept, dropped = rewrite_single_ingredient(client, fname)
            print(f"  ✓ kept {kept}  dropped {dropped}\n")
        print(f"→ {DDI_FILE}")
        kept, dropped = rewrite_ddi(client)
        print(f"  ✓ kept {kept}  dropped {dropped}")
    print("\nDone. Rebuild KB: python -m argus.reference.build_kb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
