"""SHARP-on-MCP context handling.

Argus implements the SHARP-on-MCP specification
(https://www.sharponmcp.com/key-components.html) to receive FHIR session
credentials from the calling agent platform on each tool call.

Compliant agent platforms (PromptOpinion, etc.) attach context via canonical
HTTP headers on every `tools/call` request:

    X-FHIR-Server-URL:    https://workspace/fhir
    X-FHIR-Access-Token:  <bearer token>
    X-Patient-ID:         <FHIR Patient logical id>     (optional)
    X-Encounter-ID:       <FHIR Encounter logical id>   (optional)

For backwards compatibility with earlier Argus deployments and pre-canonical
PromptOpinion drafts, a small set of legacy `x-sharp-*` aliases is also accepted
(canonical names always win when both are present).

In development, when running without an upstream platform in front, we fall back
to settings values (`ARGUS_FALLBACK_FHIR_BASE_URL` / `ARGUS_FALLBACK_FHIR_TOKEN`).

References:
    - https://www.sharponmcp.com/key-components.html
    - https://github.com/prompt-opinion/po-community-mcp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from argus.config import get_settings
from argus.logging_setup import get_logger

log = get_logger(__name__)


# Canonical SHARP-on-MCP header names
# (https://www.sharponmcp.com/key-components.html)
HEADER_FHIR_BASE_URL = "x-fhir-server-url"
HEADER_FHIR_TOKEN = "x-fhir-access-token"   # noqa: S105 — header name, not a secret
HEADER_PATIENT = "x-patient-id"
HEADER_ENCOUNTER = "x-encounter-id"
HEADER_USER = "x-user"

# Aliases — canonical name first so it wins when both are present.
# Legacy `x-sharp-*` retained for backwards compatibility with older clients
# and pre-canonical PromptOpinion drafts.
_ALIAS_BASE_URL = (
    "x-fhir-server-url",         # SHARP-on-MCP canonical
    "x-sharp-fhir-base-url",     # legacy
    "x-fhir-base-url",           # generic fallback
    "fhir-base-url",             # generic fallback
)
_ALIAS_TOKEN = (
    "x-fhir-access-token",       # SHARP-on-MCP canonical
    "x-sharp-fhir-token",        # legacy
    "x-fhir-token",              # generic fallback
    "authorization",             # bearer-style fallback
)
_ALIAS_PATIENT = (
    "x-patient-id",              # SHARP-on-MCP canonical
    "x-sharp-patient",           # legacy
)
_ALIAS_ENCOUNTER = (
    "x-encounter-id",            # SHARP-on-MCP canonical
    "x-sharp-encounter",         # legacy
)
_ALIAS_USER = (
    "x-user",                    # SHARP-on-MCP canonical
    "x-sharp-user",              # legacy
)


@dataclass(frozen=True)
class SharpContext:
    """Parsed per-request FHIR context."""

    fhir_base_url: str
    fhir_token: str | None
    patient_id: str | None
    encounter_id: str | None
    user: str | None

    @property
    def has_token(self) -> bool:
        return bool(self.fhir_token)


def _lower_headers(headers: dict[str, Any] | None) -> dict[str, str]:
    if not headers:
        return {}
    return {str(k).lower(): str(v) for k, v in headers.items()}


def _first_alias(headers: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        val = headers.get(alias)
        if val:
            return val
    return None


def _strip_bearer(raw: str | None) -> str | None:
    """`Authorization: Bearer xyz` → `xyz`. Plain token → passthrough."""
    if not raw:
        return None
    parts = raw.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return raw.strip()


def extract_sharp_context(
    headers: dict[str, Any] | None = None,
    *,
    explicit_patient_id: str | None = None,
) -> SharpContext:
    """Build a SharpContext from MCP request headers, falling back to env settings.

    Args:
        headers: Raw request headers from the MCP `Context`. May be None during tests.
        explicit_patient_id: Patient id supplied in the tool call arguments, which
            takes precedence over the one in SHARP headers.

    Returns:
        SharpContext — always returns an instance; uses fallback settings when no
        request-scoped context is available (dev mode).
    """
    settings = get_settings()
    lower = _lower_headers(headers)

    base_url = _first_alias(lower, _ALIAS_BASE_URL)
    token_raw = _first_alias(lower, _ALIAS_TOKEN)
    token = _strip_bearer(token_raw)

    patient_id = explicit_patient_id or _first_alias(lower, _ALIAS_PATIENT)
    encounter_id = _first_alias(lower, _ALIAS_ENCOUNTER)
    user = _first_alias(lower, _ALIAS_USER)

    # Dev fallback
    if not base_url and settings.fallback_fhir_base_url:
        base_url = str(settings.fallback_fhir_base_url).rstrip("/")
        log.debug("sharp_context.fallback_base_url", base_url=base_url)

    if not token and settings.fallback_fhir_token:
        token = settings.fallback_fhir_token.get_secret_value()
        log.debug("sharp_context.fallback_token_used")

    if not base_url:
        # Last-resort public sandbox for smoke tests — explicitly logged
        base_url = "https://hapi.fhir.org/baseR4"
        log.warning(
            "sharp_context.no_fhir_base_url",
            message="Neither SHARP headers nor fallback set; using HAPI public sandbox",
        )

    ctx = SharpContext(
        fhir_base_url=base_url.rstrip("/"),
        fhir_token=token,
        patient_id=patient_id,
        encounter_id=encounter_id,
        user=user,
    )

    log.debug(
        "sharp_context.built",
        has_token=ctx.has_token,
        patient_id=ctx.patient_id,
        encounter_id=ctx.encounter_id,
    )
    return ctx
