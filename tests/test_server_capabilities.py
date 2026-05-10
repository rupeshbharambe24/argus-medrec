"""Verify Argus advertises SHARP-on-MCP capability.

Boots the FastMCP server in-memory (no HTTP) and asserts the `initialize`
response contains `experimental.fhir_context_required.value = true`.

Spec: https://www.sharponmcp.com/key-components.html
"""

from __future__ import annotations

import pytest
from fastmcp import Client


@pytest.mark.asyncio
async def test_initialize_advertises_promptopinion_fhir_extension():
    """Per https://docs.promptopinion.ai/fhir-context/mcp-fhir-context the
    server must advertise capabilities.extensions['ai.promptopinion/fhir-context']
    with a non-empty scopes array. Without this, PromptOpinion shows the
    'does not support FHIR extension' warning."""
    from argus.server import mcp as argus_mcp

    async with Client(argus_mcp) as client:
        init = client.initialize_result
        assert init is not None, "initialize_result should be set after connect"

        caps = init.capabilities
        extensions = getattr(caps, "extensions", None) or {}
        assert "ai.promptopinion/fhir-context" in extensions, (
            "MCP initialize response missing extensions['ai.promptopinion/fhir-context'] — "
            "PromptOpinion will show the 'does not support FHIR extension' warning. "
            f"Got extensions: {list(extensions)}"
        )
        po_ext = extensions["ai.promptopinion/fhir-context"]
        scopes = po_ext.get("scopes") or []
        assert len(scopes) > 0, "PromptOpinion fhir-context extension must declare at least one scope"
        for scope in scopes:
            assert "name" in scope and "required" in scope, (
                f"Each scope must have 'name' and 'required' fields, got {scope!r}"
            )


@pytest.mark.asyncio
async def test_sharp_on_mcp_forward_compat_flag():
    """Forward-compat: keep advertising experimental.fhir_context_required
    from sharponmcp.com so non-PO clients can also detect us."""
    from argus.server import mcp as argus_mcp

    async with Client(argus_mcp) as client:
        init = client.initialize_result
        caps = init.capabilities
        experimental = caps.experimental or {}
        assert experimental.get("fhir_context_required", {}).get("value") is True


@pytest.mark.asyncio
async def test_all_six_tools_registered():
    from argus.server import mcp as argus_mcp

    async with Client(argus_mcp) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        expected = {
            "get_active_medications",
            "check_drug_interactions",
            "renal_dose_check",
            "reconcile_home_vs_hospital",
            "generate_med_rec_note",
            "screen_high_risk_patterns",
        }
        assert expected.issubset(names), f"Missing tools: {expected - names}"


def test_required_sharp_headers_constant():
    """The 403 middleware enforces these two canonical headers."""
    from argus.server import REQUIRED_SHARP_HEADERS

    assert REQUIRED_SHARP_HEADERS == ("x-fhir-server-url", "x-fhir-access-token")
