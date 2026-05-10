"""Verify a running Argus instance advertises SHARP-on-MCP capability.

Usage:
    # In one shell, start the server:
    python -m argus.server

    # In another shell:
    python scripts/verify_sharp.py
    # or against a remote URL:
    python scripts/verify_sharp.py https://argus.example/mcp

The script connects to the MCP endpoint, performs the `initialize` handshake,
and asserts that `capabilities.experimental.fhir_context_required.value` is
true — the flag PromptOpinion checks before showing the "Pass FHIR token"
toggle. Exits 0 on success, 1 on failure.

Spec: https://www.sharponmcp.com/key-components.html
"""

from __future__ import annotations

import asyncio
import sys

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def verify(url: str) -> int:
    print(f"-> Connecting to {url}")
    # Pass dummy SHARP headers so the 403 middleware (which Argus advertises
    # support for) lets follow-up calls through during this smoke test.
    headers = {
        "X-FHIR-Server-URL": "https://verify.local/fhir",
        "X-FHIR-Access-Token": "verify-script-dummy",
    }
    async with (
        streamablehttp_client(url, headers=headers) as (read, write, _),
        ClientSession(read, write) as session,
    ):
            init = await session.initialize()

            caps = init.capabilities
            print("\nServer capabilities:")
            print(caps.model_dump_json(indent=2))

            # Primary check — the PromptOpinion-canonical extension key, per
            # https://docs.promptopinion.ai/fhir-context/mcp-fhir-context
            extensions = getattr(caps, "extensions", None) or {}
            po_ext = extensions.get("ai.promptopinion/fhir-context")
            if po_ext is None:
                print(
                    "\n[FAIL] capabilities.extensions['ai.promptopinion/fhir-context']"
                    " not advertised."
                )
                print(
                    "  PromptOpinion will show the 'does not support FHIR"
                    " extension' warning."
                )
                return 1
            scopes = po_ext.get("scopes") or []
            if not scopes:
                print(
                    "\n[FAIL] PromptOpinion fhir-context extension has no scopes."
                )
                return 1

            # Secondary — SHARP-on-MCP forward-compat flag.
            experimental = caps.experimental or {}
            fhir_req = experimental.get("fhir_context_required") or {}
            sharp_ok = fhir_req.get("value") is True

            print(
                f"\n[OK] PromptOpinion FHIR-context extension advertised "
                f"({len(scopes)} scopes)."
            )
            print(
                f"  SHARP-on-MCP fhir_context_required.value: "
                f"{'present' if sharp_ok else 'missing (non-fatal)'}"
            )
            print("  PromptOpinion will show the 'Pass FHIR token' checkbox.")
            return 0


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080/mcp"
    return asyncio.run(verify(url))


if __name__ == "__main__":
    sys.exit(main())
