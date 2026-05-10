"""Strip conditional Practitioner/Organization references from a Synthea bundle.

Synthea emits transaction bundles whose patient resources point to Practitioners
and Organizations via conditional references (e.g.
``Practitioner?identifier=http://hl7.org/fhir/sid/us-npi|9999904995``). Target
FHIR servers that don't already have those resources pre-loaded reject the
bundle with `not-found` errors.

This script makes a Synthea patient bundle self-contained by:
  1. Collecting every conditional Practitioner / Organization reference in the
     bundle.
  2. Adding a minimal inline `Practitioner` / `Organization` stub for each
     unique identifier, with a deterministic UUID.
  3. Rewriting every conditional reference to point at the local stub URN.

Output: a NEW file alongside the input, suffixed `.sanitized.json`. The
original file is untouched.

Usage:
    python scripts/sanitize_synthea_bundle.py path/to/Dewey930_..._.json
    python scripts/sanitize_synthea_bundle.py path/to/folder --batch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

# Patterns we rewrite — Synthea always produces these two.
COND_RE = re.compile(
    r"^(?P<type>Practitioner|Organization|Location)\?identifier="
    r"(?P<system>[^|]+)\|(?P<value>.+)$"
)


def _stub_uuid(resource_type: str, system: str, value: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{resource_type}|{system}|{value}"))


def _stub_resource(resource_type: str, system: str, value: str, ref_uuid: str) -> dict:
    suffix = value[-6:] if len(value) >= 6 else value
    if resource_type == "Practitioner":
        return {
            "resourceType": "Practitioner",
            "id": ref_uuid,
            "identifier": [{"system": system, "value": value}],
            "active": True,
            "name": [{"family": f"Synthetic-{suffix}", "given": ["Practitioner"]}],
        }
    if resource_type == "Location":
        return {
            "resourceType": "Location",
            "id": ref_uuid,
            "identifier": [{"system": system, "value": value}],
            "status": "active",
            "name": f"Synthetic Location {suffix}",
        }
    return {
        "resourceType": "Organization",
        "id": ref_uuid,
        "identifier": [{"system": system, "value": value}],
        "active": True,
        "name": f"Synthetic Organization {suffix}",
    }


def _walk_and_rewrite(node: Any, stubs: dict[tuple[str, str, str], str]) -> None:
    """Walk the JSON tree mutating every {'reference': 'Type?identifier=sys|val'}.

    Side effect: populates `stubs` with the unique (type, system, value) tuples
    encountered, mapped to deterministic UUIDs.
    """
    if isinstance(node, dict):
        ref = node.get("reference")
        if isinstance(ref, str):
            m = COND_RE.match(ref)
            if m:
                key = (m["type"], m["system"], m["value"])
                stub_id = stubs.setdefault(key, _stub_uuid(*key))
                node["reference"] = f"urn:uuid:{stub_id}"
        for v in node.values():
            _walk_and_rewrite(v, stubs)
    elif isinstance(node, list):
        for v in node:
            _walk_and_rewrite(v, stubs)


def sanitize(path: Path, bundle_type: str = "batch") -> Path:
    """Sanitize a Synthea bundle for ingestion.

    Args:
        path: input bundle path.
        bundle_type: one of "batch" (default — what PO accepts), "transaction",
            or "collection".

    "batch" is the most forgiving for ingestion UIs: each entry is processed
    independently, so a single malformed entry doesn't fail the whole import.
    """
    if bundle_type not in {"batch", "transaction", "collection"}:
        raise ValueError(f"Unsupported bundle_type: {bundle_type}")

    with path.open(encoding="utf-8") as f:
        bundle = json.load(f)

    if bundle.get("resourceType") != "Bundle":
        raise ValueError(f"Not a Bundle: {path}")

    entries = bundle.get("entry") or []
    stubs: dict[tuple[str, str, str], str] = {}

    for entry in entries:
        _walk_and_rewrite(entry.get("resource"), stubs)
        _walk_and_rewrite(entry.get("request"), stubs)

    # Append one stub entry per unique (type, system, value).
    # NOTE: no `ifNoneExist` — that's a conditional-create only allowed in
    # transaction bundles; PromptOpinion's batch upload rejects it.
    for (rtype, system, value), stub_id in stubs.items():
        resource = _stub_resource(rtype, system, value, stub_id)
        entry = {
            "fullUrl": f"urn:uuid:{stub_id}",
            "resource": resource,
            "request": {"method": "POST", "url": rtype},
        }
        entries.append(entry)

    bundle["type"] = bundle_type

    if bundle_type == "collection":
        for entry in entries:
            entry.pop("request", None)
            entry.pop("response", None)
    else:
        # batch / transaction — every entry must have a request block.
        # Synthea already supplies these, but stub-only entries we created
        # already have them. Make sure no `response` blocks linger.
        for entry in entries:
            entry.pop("response", None)
            if "request" not in entry:
                resource = entry.get("resource") or {}
                rt = resource.get("resourceType")
                if rt:
                    entry["request"] = {"method": "POST", "url": rt}

    bundle["entry"] = entries

    suffix = f".{bundle_type}.json"
    out = path.with_suffix(suffix)
    with out.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print(
        f"  rewrote {len(stubs)} conditional refs -> stubs;"
        f" type={bundle['type']!r}; entries={len(entries)}"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="A Synthea patient bundle (.json) or a folder")
    ap.add_argument("--batch", action="store_true", help="Sanitize every *.json in the folder")
    ap.add_argument(
        "--type",
        choices=["batch", "transaction", "collection"],
        default="batch",
        help=(
            "Bundle type to emit. Default 'batch' is what PromptOpinion's "
            "FHIR uploader expects (each entry processed independently)."
        ),
    )
    args = ap.parse_args()
    bundle_type = args.type

    if not args.path.exists():
        print(f"[error] not found: {args.path}")
        return 1

    if args.path.is_dir() or args.batch:
        files = sorted(args.path.glob("*.json"))
        files = [
            f
            for f in files
            if not f.name.lower().startswith(
                ("hospitalinformation", "practitionerinformation")
            )
            and not f.name.endswith(".sanitized.json")
        ]
        for f in files:
            print(f"-> {f.name}")
            sanitize(f, bundle_type=bundle_type)
        print(f"\nSanitized {len(files)} bundles.")
    else:
        out = sanitize(args.path, bundle_type=bundle_type)
        print(f"\nSanitized -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
