"""Upload Synthea-generated FHIR Bundle JSON files to a Prompt Opinion workspace
(or any FHIR server).

Usage:
    # Env-driven — uses ARGUS_FALLBACK_FHIR_BASE_URL + ARGUS_FALLBACK_FHIR_TOKEN
    python scripts/upload_to_prompt_opinion.py

    # Or explicit:
    python scripts/upload_to_prompt_opinion.py \\
        --base-url https://workspace.promptopinion.ai/fhir \\
        --token YOUR_TOKEN \\
        --input-dir data/synthea_output/fhir
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from argus.config import get_settings
from argus.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


async def upload_bundle(
    client: httpx.AsyncClient, base_url: str, token: str | None, bundle_path: Path
) -> bool:
    """POST a transaction Bundle to the FHIR server root."""
    headers = {
        "Content-Type": "application/fhir+json",
        "Accept": "application/fhir+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = bundle_path.read_bytes()
    try:
        resp = await client.post(
            f"{base_url.rstrip('/')}/",
            content=payload,
            headers=headers,
            timeout=60.0,
        )
        resp.raise_for_status()
        return True
    except httpx.HTTPStatusError as exc:
        log.error(
            "upload.http_error",
            file=bundle_path.name,
            status=exc.response.status_code,
            body=exc.response.text[:200],
        )
        return False
    except httpx.HTTPError as exc:
        log.error("upload.transport_error", file=bundle_path.name, error=str(exc))
        return False


async def upload_directory(
    base_url: str, token: str | None, input_dir: Path, limit: int | None, concurrency: int
) -> dict[str, int]:
    files = sorted(input_dir.glob("*.json"))
    # Filter out hospital/practitioner info files if user passed whole fhir dir
    patient_files = [f for f in files if not f.name.startswith(("hospital", "practitioner"))]
    if limit:
        patient_files = patient_files[:limit]

    log.info(
        "upload.starting",
        base_url=base_url,
        file_count=len(patient_files),
        concurrency=concurrency,
    )

    sem = asyncio.Semaphore(concurrency)
    success = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        async def _one(path: Path) -> None:
            nonlocal success, failed
            async with sem:
                ok = await upload_bundle(client, base_url, token, path)
                if ok:
                    success += 1
                else:
                    failed += 1
                if (success + failed) % 25 == 0:
                    log.info("upload.progress", done=success + failed, success=success)

        await asyncio.gather(*(_one(p) for p in patient_files))

    # Upload hospital + practitioner resources first if present (they're referenced)
    return {"success": success, "failed": failed, "total": len(patient_files)}


def main() -> int:
    configure_logging()
    settings = get_settings()

    ap = argparse.ArgumentParser(description="Upload Synthea bundles to a FHIR server")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--input-dir", type=Path, default=Path("data/synthea_output/fhir"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    base_url = args.base_url or (
        str(settings.fallback_fhir_base_url) if settings.fallback_fhir_base_url else None
    )
    token = args.token or (
        settings.fallback_fhir_token.get_secret_value()
        if settings.fallback_fhir_token
        else None
    )

    if not base_url:
        print("Error: --base-url (or ARGUS_FALLBACK_FHIR_BASE_URL) required", file=sys.stderr)
        return 2
    if not args.input_dir.exists():
        print(f"Error: input-dir {args.input_dir} not found", file=sys.stderr)
        return 2

    result = asyncio.run(
        upload_directory(base_url, token, args.input_dir, args.limit, args.concurrency)
    )
    print(f"\n✓ Uploaded {result['success']}/{result['total']} bundles "
          f"({result['failed']} failed)")
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
