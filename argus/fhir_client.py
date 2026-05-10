"""Async FHIR R4 client used by every tool.

Design goals:
- **Async everywhere** — tools often fan out to 3-5 FHIR queries in parallel.
- **Typed helpers** for the resources we actually care about (MedicationRequest,
  MedicationStatement, MedicationDispense, Observation, Patient, Condition,
  AllergyIntolerance, Encounter) — rather than dumping raw JSON at the tool layer.
- **Paging** handled automatically up to a configurable cap.
- **Retry with jitter** on 5xx / network errors (tenacity).
- **Never caches clinical data** — patient data freshness is critical.

The client is stateless and cheap to instantiate; each tool call builds one from
the request's SharpContext.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from argus.logging_setup import get_logger
from argus.sharp_context import SharpContext

log = get_logger(__name__)


DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)
MAX_PAGES = 10          # cap to avoid runaway queries
DEFAULT_PAGE_COUNT = 50  # _count per FHIR page

RETRYABLE_STATUSES = {500, 502, 503, 504}


class FhirError(RuntimeError):
    """Base exception for FHIR client failures."""

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class FhirNotFoundError(FhirError):
    """Requested resource does not exist."""


class FhirAuthError(FhirError):
    """401/403 from the FHIR server."""


def _is_retryable_http_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code in RETRYABLE_STATUSES
    ) or isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout))


class FhirClient:
    """Per-request FHIR client bound to a SharpContext."""

    def __init__(
        self,
        context: SharpContext,
        *,
        timeout: httpx.Timeout | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._context = context
        self._timeout = timeout or DEFAULT_TIMEOUT
        self._owned_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=self._timeout,
            base_url=context.fhir_base_url,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }
        if self._context.fhir_token:
            headers["Authorization"] = f"Bearer {self._context.fhir_token}"
        return headers

    async def aclose(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    async def __aenter__(self) -> FhirClient:
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.aclose()

    # -----------------------------------------------------------------------
    # Low-level
    # -----------------------------------------------------------------------

    async def _request(
        self, method: str, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Issue a request with retries. Raises typed FhirError subclasses."""
        url = path if path.startswith("http") else path
        log.debug("fhir.request", method=method, url=url, params=params)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=0.5, max=3.0),
            retry=retry_if_exception_type(
                (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout)
            ),
            reraise=True,
        ):
            with attempt:
                try:
                    resp = await self._client.request(method, url, params=params)
                except httpx.HTTPError as exc:
                    log.warning("fhir.transport_error", error=str(exc))
                    raise

                if resp.status_code in (401, 403):
                    raise FhirAuthError(
                        f"Auth error from FHIR server: {resp.status_code}",
                        status_code=resp.status_code,
                        body=resp.text[:500],
                    )
                if resp.status_code == 404:
                    raise FhirNotFoundError(
                        f"Not found: {url}", status_code=404, body=resp.text[:500]
                    )
                if resp.status_code >= 400:
                    if resp.status_code in RETRYABLE_STATUSES and attempt.retry_state.attempt_number < 3:
                        log.info(
                            "fhir.retryable_error",
                            status_code=resp.status_code,
                            attempt=attempt.retry_state.attempt_number,
                        )
                        resp.raise_for_status()  # lets tenacity retry
                    raise FhirError(
                        f"FHIR {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code,
                        body=resp.text[:500],
                    )

                return resp.json()

        # Unreachable; tenacity would have raised or returned.
        raise FhirError("Retry loop exited without return")  # pragma: no cover

    # -----------------------------------------------------------------------
    # Generic search with paging
    # -----------------------------------------------------------------------

    async def search(
        self,
        resource_type: str,
        params: dict[str, Any] | None = None,
        *,
        max_pages: int = MAX_PAGES,
    ) -> list[dict[str, Any]]:
        """Execute a FHIR search and follow pagination. Returns list of resources."""
        params = dict(params or {})
        params.setdefault("_count", DEFAULT_PAGE_COUNT)

        resources: list[dict[str, Any]] = []
        next_url: str | None = f"/{resource_type}"
        pages = 0

        while next_url and pages < max_pages:
            if pages == 0:
                bundle = await self._request("GET", next_url, params=params)
            else:
                bundle = await self._request("GET", next_url)

            for entry in bundle.get("entry", []):
                res = entry.get("resource")
                if res:
                    resources.append(res)

            next_url = None
            for link in bundle.get("link", []):
                if link.get("relation") == "next":
                    next_url = link.get("url")
                    break
            pages += 1

        log.debug(
            "fhir.search.done",
            resource_type=resource_type,
            count=len(resources),
            pages=pages,
        )
        return resources

    async def read(self, resource_type: str, resource_id: str) -> dict[str, Any]:
        """GET {resource}/{id}."""
        return await self._request("GET", f"/{resource_type}/{resource_id}")

    # -----------------------------------------------------------------------
    # Typed helpers — convenience wrappers around search() for the resources
    # our tools actually use. Keeps the tool code readable.
    # -----------------------------------------------------------------------

    async def get_patient(self, patient_id: str) -> dict[str, Any]:
        return await self.read("Patient", patient_id)

    async def get_active_medication_requests(
        self, patient_id: str, *, include_on_hold: bool = True
    ) -> list[dict[str, Any]]:
        statuses = ["active"]
        if include_on_hold:
            statuses.append("on-hold")
        return await self.search(
            "MedicationRequest",
            {"patient": patient_id, "status": ",".join(statuses)},
        )

    async def get_active_medication_statements(
        self, patient_id: str
    ) -> list[dict[str, Any]]:
        return await self.search(
            "MedicationStatement",
            {"patient": patient_id, "status": "active"},
        )

    async def get_medication_dispenses(
        self, patient_id: str, *, lookback_days: int = 90
    ) -> list[dict[str, Any]]:
        # whenhandedover prefix for recent dispenses
        from datetime import date, timedelta

        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        return await self.search(
            "MedicationDispense",
            {"patient": patient_id, "whenhandedover": f"ge{cutoff}"},
        )

    async def get_observations(
        self,
        patient_id: str,
        *,
        loinc_codes: list[str] | None = None,
        lookback_days: int | None = None,
        status: str = "final",
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"patient": patient_id, "status": status}
        if loinc_codes:
            params["code"] = ",".join(f"http://loinc.org|{c}" for c in loinc_codes)
        if lookback_days:
            from datetime import date, timedelta

            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            params["date"] = f"ge{cutoff}"
        return await self.search("Observation", params)

    async def get_conditions(self, patient_id: str) -> list[dict[str, Any]]:
        return await self.search(
            "Condition",
            {"patient": patient_id, "clinical-status": "active,recurrence,relapse"},
        )

    async def get_allergies(self, patient_id: str) -> list[dict[str, Any]]:
        return await self.search(
            "AllergyIntolerance",
            {"patient": patient_id, "clinical-status": "active"},
        )

    async def get_current_encounter(
        self, patient_id: str, encounter_id: str | None = None
    ) -> dict[str, Any] | None:
        if encounter_id:
            try:
                return await self.read("Encounter", encounter_id)
            except FhirNotFoundError:
                return None
        # Fallback — latest in-progress encounter
        encs = await self.search(
            "Encounter",
            {"patient": patient_id, "status": "in-progress,arrived,triaged"},
            max_pages=1,
        )
        return encs[0] if encs else None
