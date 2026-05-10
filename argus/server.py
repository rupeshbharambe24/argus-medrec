"""Argus MCP server — entry point.

Registers all six tools with FastMCP and serves them over streamable HTTP.
Each tool wrapper:
    1. Extracts SHARP context from request headers.
    2. Validates the input against the tool's Pydantic input schema.
    3. Delegates to the tool module.
    4. Returns a JSON-serializable dict.

Run locally:
    python -m argus.server

Expose to Prompt Opinion:
    ngrok http 8080
    In PO workspace: add MCP at <ngrok-url>/mcp, enable "Pass FHIR token".
"""

from __future__ import annotations

import sys
from typing import Any

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from argus.config import get_settings
from argus.logging_setup import configure_logging, get_logger
from argus.schemas import (
    CheckInteractionsInput,
    GenerateNoteInput,
    GetMedicationsInput,
    ReconcileInput,
    RenalCheckInput,
    ScreenPatternsInput,
)
from argus.sharp_context import (
    HEADER_FHIR_BASE_URL,
    HEADER_FHIR_TOKEN,
    extract_sharp_context,
)
from argus.tools.check_interactions import run as run_check_interactions
from argus.tools.generate_note import run as run_generate_note
from argus.tools.get_medications import run as run_get_medications
from argus.tools.reconcile import run as run_reconcile
from argus.tools.renal_check import run as run_renal_check
from argus.tools.screen_patterns import run as run_screen_patterns

log = get_logger(__name__)

# FastMCP provides helpers for accessing HTTP headers in streamable-http mode.
# We resolve the import defensively — FastMCP has moved this helper between
# minor versions, so we try several paths and fall back to empty headers.
try:
    from fastmcp.server.dependencies import get_http_headers  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    try:
        from fastmcp.server.http import get_http_headers  # type: ignore[no-redef]
    except ImportError:
        def get_http_headers() -> dict[str, str]:  # type: ignore[no-redef]
            return {}


mcp = FastMCP(
    name="Argus",
    instructions=(
        "Argus — Medication reconciliation and safety suite.\n\n"
        "Provides six composable tools for healthcare agents: medication listing, "
        "drug-interaction analysis, renal dose checking, home-vs-hospital "
        "reconciliation, high-risk pattern screening, and reconciliation note "
        "generation. All outputs are FHIR-cited and require clinician review."
    ),
)


# ---------------------------------------------------------------------------
# PromptOpinion FHIR-context extension declaration
# ---------------------------------------------------------------------------
#
# Per https://docs.promptopinion.ai/fhir-context/mcp-fhir-context the canonical
# capability declaration that PromptOpinion looks for in the MCP `initialize`
# response is:
#
#     capabilities.extensions["ai.promptopinion/fhir-context"] = {
#         "scopes": [{"name": "<smart-scope>", "required": true}, ...]
#     }
#
# When this is present, PromptOpinion:
#   - Hides the "does not support PromptOpinion's FHIR extension" warning
#   - Exposes a "Pass FHIR token" toggle in the MCP server config dialog
#   - Sends X-FHIR-Server-URL / X-FHIR-Access-Token / X-Patient-ID on every
#     subsequent tool call
#
# We also keep the SHARP-on-MCP `experimental.fhir_context_required` flag from
# https://www.sharponmcp.com/key-components.html as a forward-compatibility
# signal — harmless if the platform ignores it.
#
# Argus reads patient demographics, medications (request/statement/dispense),
# observations (labs), conditions, encounters, and allergies — so we declare
# specific patient-scope read+search SMART scopes for transparency.

PO_FHIR_EXTENSION_KEY = "ai.promptopinion/fhir-context"

PO_FHIR_EXTENSION_VALUE: dict[str, Any] = {
    "scopes": [
        {"name": "patient/Patient.rs", "required": True},
        {"name": "patient/MedicationRequest.rs", "required": True},
        {"name": "patient/MedicationStatement.rs", "required": True},
        {"name": "patient/MedicationDispense.rs", "required": False},
        {"name": "patient/Observation.rs", "required": True},
        {"name": "patient/Condition.rs", "required": True},
        {"name": "patient/Encounter.rs", "required": False},
        {"name": "patient/AllergyIntolerance.rs", "required": False},
    ],
}

SHARP_EXPERIMENTAL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "fhir_context_required": {"value": True},
}


def _install_sharp_capability(server: FastMCP) -> None:
    """Inject FHIR-context capability declarations into the MCP initialize response.

    Patches the FastMCP low-level server in two complementary places:

    1. `get_capabilities` — adds the PromptOpinion `ai.promptopinion/fhir-context`
       extension under `capabilities.extensions`. This is the signal PO
       *actually* checks before showing the FHIR-token toggle.
    2. `create_initialization_options` — injects the SHARP-on-MCP
       `experimental.fhir_context_required` flag for forward compatibility.

    Patching survives across all transports (http/streamable-http/stdio) and
    across lifespan restarts.
    """
    low = server._mcp_server  # the underlying mcp.server.lowlevel.Server

    original_get_caps = low.get_capabilities

    def patched_get_caps(notification_options, experimental_capabilities):
        caps = original_get_caps(notification_options, experimental_capabilities)
        existing_ext: dict[str, Any] = getattr(caps, "extensions", None) or {}
        caps.extensions = {**existing_ext, PO_FHIR_EXTENSION_KEY: PO_FHIR_EXTENSION_VALUE}
        return caps

    low.get_capabilities = patched_get_caps  # type: ignore[method-assign]

    original_cio = low.create_initialization_options

    def patched_cio(notification_options=None, experimental_capabilities=None, **kwargs):
        merged = dict(SHARP_EXPERIMENTAL_CAPABILITIES)
        if experimental_capabilities:
            merged.update(experimental_capabilities)
        return original_cio(
            notification_options=notification_options,
            experimental_capabilities=merged,
            **kwargs,
        )

    low.create_initialization_options = patched_cio  # type: ignore[method-assign]


_install_sharp_capability(mcp)


# ---------------------------------------------------------------------------
# SHARP-on-MCP 403 enforcement middleware
# ---------------------------------------------------------------------------

REQUIRED_SHARP_HEADERS: tuple[str, ...] = (HEADER_FHIR_BASE_URL, HEADER_FHIR_TOKEN)


class SharpFhirContextMiddleware(BaseHTTPMiddleware):
    """Reject tools/call requests missing the SHARP-on-MCP context headers.

    Per https://www.sharponmcp.com/key-components.html section 3:
        "If FHIR context is required and the client does not include one or
        more of the required headers, the MCP server should respond with a
        403 Forbidden response."

    The MCP `initialize` handshake passes through unchecked because canonical
    SHARP headers are part of per-call context, not session-level auth, and
    PromptOpinion sends them only on follow-up calls. We use the presence of
    `Mcp-Session-Id` to discriminate: if it's set, this is a follow-up call
    and must carry the FHIR headers.

    In dev mode (ARGUS_ENV=dev) enforcement is bypassed so local smoke tests
    work without a real PromptOpinion in front.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or not request.url.path.startswith("/mcp"):
            return await call_next(request)

        if get_settings().env == "dev":
            return await call_next(request)

        # Initialize requests have no Mcp-Session-Id yet; let them through.
        if not request.headers.get("mcp-session-id"):
            return await call_next(request)

        missing = [h for h in REQUIRED_SHARP_HEADERS if not request.headers.get(h)]
        if missing:
            log.warning(
                "sharp.middleware.reject_403",
                missing=missing,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "fhir_context_required",
                    "message": (
                        "This MCP server requires SHARP-on-MCP FHIR context. "
                        "Headers required on every tools/call request: "
                        + ", ".join(REQUIRED_SHARP_HEADERS)
                    ),
                    "missing_headers": missing,
                    "spec": "https://www.sharponmcp.com/key-components.html",
                },
            )
        return await call_next(request)


def _headers() -> dict[str, str]:
    try:
        return dict(get_http_headers() or {})
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Tool 1
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_active_medications(
    patient_id: str | None = None,
    as_of_date: str | None = None,
    include_discontinued: bool = False,
    lookback_days: int = 180,
) -> dict[str, Any]:
    """Return the deduplicated, RxNorm-normalized active medication list for a patient.

    Args:
        patient_id: FHIR Patient logical id. If omitted, resolved from SHARP
            context.
        as_of_date: ISO date (YYYY-MM-DD) to anchor the "active at this date"
            determination. Defaults to today.
        include_discontinued: Include medications with status=stopped/completed.
        lookback_days: How far back to search MedicationStatement / Dispense.

    Returns:
        GetMedicationsOutput as a dict — see schemas.py for structure.
    """
    from datetime import date as _date
    req = GetMedicationsInput(
        patient_id=patient_id,
        as_of_date=_date.fromisoformat(as_of_date) if as_of_date else None,
        include_discontinued=include_discontinued,
        lookback_days=lookback_days,
    )
    ctx = extract_sharp_context(_headers(), explicit_patient_id=patient_id)
    result = await run_get_medications(req, ctx)
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool 2
# ---------------------------------------------------------------------------

@mcp.tool()
async def check_drug_interactions(
    patient_id: str | None = None,
    severity_threshold: str = "moderate",
    new_medication_candidate_rxnorm: str | None = None,
    new_medication_candidate_name: str | None = None,
) -> dict[str, Any]:
    """Analyze drug-drug interactions with patient-specific contextual severity ranking.

    Args:
        patient_id: FHIR Patient id. If omitted, from SHARP context.
        severity_threshold: Minimum severity to report — minor | moderate | major.
        new_medication_candidate_rxnorm: (Test-before-prescribe mode) RxNorm code
            of a proposed new medication to screen against the current list.
        new_medication_candidate_name: Display name for the candidate — used when
            RxNorm code resolution is not needed.

    Returns:
        CheckInteractionsOutput — interactions ranked by contextual severity,
        each with SHAP-style patient-specific factors and recommended actions.
    """
    from argus.schemas import MedicationEntry, Severity

    candidate: MedicationEntry | None = None
    if new_medication_candidate_rxnorm or new_medication_candidate_name:
        candidate = MedicationEntry(
            rxnorm_ingredient_code=new_medication_candidate_rxnorm,
            rxnorm_ingredient_name=new_medication_candidate_name,
        )
    req = CheckInteractionsInput(
        patient_id=patient_id,
        severity_threshold=Severity(severity_threshold),
        new_medication_candidate=candidate,
    )
    ctx = extract_sharp_context(_headers(), explicit_patient_id=patient_id)
    result = await run_check_interactions(req, ctx)
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool 3
# ---------------------------------------------------------------------------

@mcp.tool()
async def renal_dose_check(
    patient_id: str | None = None,
    creatinine_lookback_days: int = 90,
) -> dict[str, Any]:
    """Check medications for renal dose adjustments based on eGFR (CKD-EPI 2021).

    Args:
        patient_id: FHIR Patient id. If omitted, from SHARP context.
        creatinine_lookback_days: How far back to look for a recent serum creatinine.

    Returns:
        RenalCheckOutput — eGFR value, CKD stage, and per-medication recommendations
        with guideline citations (FDA label, KDIGO 2022, Beers 2023).
    """
    req = RenalCheckInput(
        patient_id=patient_id,
        creatinine_lookback_days=creatinine_lookback_days,
    )
    ctx = extract_sharp_context(_headers(), explicit_patient_id=patient_id)
    result = await run_renal_check(req, ctx)
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool 4
# ---------------------------------------------------------------------------

@mcp.tool()
async def reconcile_home_vs_hospital(
    patient_id: str | None = None,
    encounter_id: str | None = None,
) -> dict[str, Any]:
    """Compare home medications against the current encounter's orders; classify each
    discrepancy as intentional or unintentional.

    Args:
        patient_id: FHIR Patient id. If omitted, from SHARP context.
        encounter_id: Specific FHIR Encounter id. If omitted, uses the currently
            in-progress encounter, or all hospital MedicationRequests if none.

    Returns:
        ReconcileOutput — categorized discrepancies (omission, commission, dose
        change, therapeutic substitution) with LLM-classified intentionality.
    """
    req = ReconcileInput(patient_id=patient_id, encounter_id=encounter_id)
    ctx = extract_sharp_context(_headers(), explicit_patient_id=patient_id)
    result = await run_reconcile(req, ctx)
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool 5
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_med_rec_note(
    patient_id: str | None = None,
    audience: str = "physician",
    format: str = "soap",
    language: str = "en",
    include_sections: list[str] | None = None,
) -> dict[str, Any]:
    """Produce a clinician-ready medication reconciliation note with FHIR citations.

    Orchestrates tools 1-4 + tool 6 and synthesizes the results into a structured
    Markdown note. Every clinical claim is traceable to a FHIR resource ID.

    Args:
        patient_id: FHIR Patient id. If omitted, from SHARP context.
        audience: physician | pharmacist | nurse | patient.
        format: soap | narrative | bulleted | table.
        language: en | es | hi | fr | pt | ar | zh.
        include_sections: Subsets of interactions | renal | discrepancies |
            high_risk | recommendations. Defaults to all.

    Returns:
        GenerateNoteOutput — Markdown note, structured action items, citations.
    """
    from argus.schemas import NoteAudience, NoteFormat

    if include_sections is None:
        include_sections = ["interactions", "renal", "discrepancies", "high_risk", "recommendations"]
    req = GenerateNoteInput(
        patient_id=patient_id,
        audience=NoteAudience(audience),
        format=NoteFormat(format),
        language=language,  # type: ignore[arg-type]
        include_sections=include_sections,  # type: ignore[arg-type]
    )
    ctx = extract_sharp_context(_headers(), explicit_patient_id=patient_id)
    result = await run_generate_note(req, ctx)
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tool 6
# ---------------------------------------------------------------------------

@mcp.tool()
async def screen_high_risk_patterns(
    patient_id: str | None = None,
    screens: list[str] | None = None,
) -> dict[str, Any]:
    """Run multi-screen safety sweep for high-risk prescribing patterns.

    Args:
        patient_id: FHIR Patient id. If omitted, from SHARP context.
        screens: Subset of: beers | qtc | opioid_benzo | serotonin_syndrome |
            anticholinergic_burden | pregnancy | adherence_gap. Defaults to a
            clinically useful core set.

    Returns:
        ScreenPatternsOutput — flagged findings with guideline attribution.
    """
    from argus.schemas import ScreenType

    if screens is None:
        screens_list = [
            ScreenType.BEERS,
            ScreenType.QTC,
            ScreenType.OPIOID_BENZO,
            ScreenType.ANTICHOLINERGIC_BURDEN,
            ScreenType.ADHERENCE_GAP,
        ]
    else:
        screens_list = [ScreenType(s) for s in screens]
    req = ScreenPatternsInput(patient_id=patient_id, screens=screens_list)
    ctx = extract_sharp_context(_headers(), explicit_patient_id=patient_id)
    result = await run_screen_patterns(req, ctx)
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    configure_logging()
    settings = get_settings()

    log.info(
        "argus.starting",
        host=settings.host,
        port=settings.port,
        env=settings.env,
        model=settings.gemini_model,
        llm_available=bool(settings.gemini_api_key),
    )

    # Ensure reference KB exists — safe to call repeatedly; cheap if already built.
    try:
        from argus.reference.build_kb import build

        build(settings.reference_kb_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("argus.kb_build_warning", error=str(exc))

    # FastMCP streamable HTTP — the transport Prompt Opinion expects.
    # The SharpFhirContextMiddleware enforces the SHARP-on-MCP 403 contract on
    # every tools/call request (skipped in dev mode and on the initialize
    # handshake; see middleware docstring).
    mcp.run(
        transport="http",
        host=settings.host,
        port=settings.port,
        middleware=[Middleware(SharpFhirContextMiddleware)],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
