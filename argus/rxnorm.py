"""RxNav (NLM) client with SQLite-backed persistent cache.

Why a cache: Tool 1 calls RxNav once per unique RxNorm code encountered. RxNav is
free and unauthenticated, but it's slow (500-1500ms per call) and polite usage
means not hammering it. With cache, 95%+ of lookups hit SQLite in <1ms.

The cache is stored in the reference KB SQLite file (a single `rxnorm_cache` table)
so operationally there's just one DB to manage.

Free API docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite
import httpx

from argus.config import get_settings
from argus.logging_setup import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RxNormIngredient:
    """Normalized ingredient-level identity for a medication."""

    rxcui: str
    name: str


@dataclass(frozen=True)
class RxNormNorm:
    """Full normalization result for a given RxNorm code."""

    input_rxcui: str
    ingredients: list[RxNormIngredient]
    clinical_drug: RxNormIngredient | None  # TTY SCD/SBD when available
    atc: list[str]                           # ATC codes for therapeutic class


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rxnorm_cache (
    rxcui TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rxnorm_cache_fetched ON rxnorm_cache(fetched_at);
"""


class RxNormClient:
    """Async client for RxNav with persistent SQLite cache."""

    def __init__(
        self,
        *,
        db_path: str | None = None,
        base_url: str | None = None,
        ttl_days: int | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self._db_path = str(db_path or settings.reference_kb_path)
        self._base_url = base_url or settings.rxnav_base
        self._ttl = timedelta(days=ttl_days or settings.rxnav_cache_ttl_days)
        self._owned_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=10.0, write=3.0, pool=3.0),
        )
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(self._db_path) as db:
                await db.executescript(_CREATE_TABLE_SQL)
                await db.commit()
            self._initialized = True

    async def aclose(self) -> None:
        if self._owned_http:
            await self._http.aclose()

    async def __aenter__(self) -> RxNormClient:
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.aclose()

    # -----------------------------------------------------------------------

    async def normalize(self, rxcui: str) -> RxNormNorm | None:
        """Return full normalization for a RxNorm code. Cached."""
        if not rxcui:
            return None

        await self._ensure_schema()
        cached = await self._read_cache(rxcui)
        if cached is not None:
            return cached

        norm = await self._fetch_normalization(rxcui)
        if norm is not None:
            await self._write_cache(rxcui, norm)
        return norm

    async def normalize_batch(self, rxcuis: list[str]) -> dict[str, RxNormNorm | None]:
        """Normalize a batch of RxCUIs concurrently. Returns {rxcui: RxNormNorm|None}."""
        unique = list(dict.fromkeys(r for r in rxcuis if r))
        results = await asyncio.gather(
            *(self.normalize(r) for r in unique), return_exceptions=True
        )
        out: dict[str, RxNormNorm | None] = {}
        for rxcui, res in zip(unique, results, strict=True):
            if isinstance(res, Exception):
                log.warning("rxnorm.normalize_failed", rxcui=rxcui, error=str(res))
                out[rxcui] = None
            else:
                out[rxcui] = res  # type: ignore[assignment]
        return out

    # -----------------------------------------------------------------------
    # Cache layer
    # -----------------------------------------------------------------------

    async def _read_cache(self, rxcui: str) -> RxNormNorm | None:
        async with aiosqlite.connect(self._db_path) as db, db.execute(
            "SELECT payload, fetched_at FROM rxnorm_cache WHERE rxcui = ?",
            (rxcui,),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        payload_str, fetched_str = row
        fetched = datetime.fromisoformat(fetched_str)
        if datetime.now(UTC) - fetched > self._ttl:
            return None
        try:
            payload = json.loads(payload_str)
            return _deserialize(payload)
        except Exception as e:  # noqa: BLE001
            log.warning("rxnorm.cache_deserialize_failed", rxcui=rxcui, error=str(e))
            return None

    async def _write_cache(self, rxcui: str, norm: RxNormNorm) -> None:
        payload = _serialize(norm)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO rxnorm_cache(rxcui, payload, fetched_at) "
                "VALUES (?, ?, ?)",
                (rxcui, json.dumps(payload), datetime.now(UTC).isoformat()),
            )
            await db.commit()

    # -----------------------------------------------------------------------
    # RxNav calls
    # -----------------------------------------------------------------------

    async def _fetch_normalization(self, rxcui: str) -> RxNormNorm | None:
        """Call RxNav to resolve ingredients + ATC.

        Note on the tty parameter: RxNav's documented format is
        ``tty=IN+PIN+SCD+SBD`` (literal `+` as separator). httpx URL-encodes a
        literal `+` in a params dict value as ``%2B``, which RxNav rejects and
        returns an HTML error page (breaking JSON parsing). Passing the values
        as space-separated lets httpx encode the spaces as `+`, producing the
        canonical URL.
        """
        related_url = f"{self._base_url}/rxcui/{rxcui}/related.json"
        try:
            resp = await self._http.get(related_url, params={"tty": "IN PIN SCD SBD"})
            resp.raise_for_status()
            related = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("rxnorm.related_fetch_failed", rxcui=rxcui, error=str(exc))
            return None

        ingredients: list[RxNormIngredient] = []
        clinical_drug: RxNormIngredient | None = None

        groups = ((related.get("relatedGroup") or {}).get("conceptGroup")) or []
        for grp in groups:
            tty = grp.get("tty")
            props = grp.get("conceptProperties") or []
            for p in props:
                entry = RxNormIngredient(rxcui=p.get("rxcui", ""), name=p.get("name", ""))
                if tty in ("IN", "PIN"):
                    ingredients.append(entry)
                elif tty in ("SCD", "SBD") and clinical_drug is None:
                    clinical_drug = entry

        # If RxNav gave us nothing in 'IN', fall back to treating the input as ingredient.
        if not ingredients:
            name_url = f"{self._base_url}/rxcui/{rxcui}/property.json"
            try:
                rp = await self._http.get(name_url, params={"propName": "RxNorm Name"})
                if rp.status_code == 200:
                    body = rp.json()
                    props = (body.get("propConceptGroup") or {}).get("propConcept") or []
                    if props:
                        ingredients = [
                            RxNormIngredient(rxcui=rxcui, name=props[0].get("propValue", ""))
                        ]
            except httpx.HTTPError:
                pass

        atc = await self._fetch_atc(rxcui)

        return RxNormNorm(
            input_rxcui=rxcui,
            ingredients=ingredients,
            clinical_drug=clinical_drug,
            atc=atc,
        )

    async def _fetch_atc(self, rxcui: str) -> list[str]:
        """Pull ATC therapeutic classes for the drug (via RxClass)."""
        url = f"{self._base_url}/rxclass/class/byRxcui.json"
        try:
            resp = await self._http.get(url, params={"rxcui": rxcui, "relaSource": "ATC"})
            if resp.status_code != 200:
                return []
            body = resp.json()
            items = ((body.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo")) or []
            codes: list[str] = []
            for it in items:
                cls = (it or {}).get("rxclassMinConceptItem") or {}
                code = cls.get("classId")
                if code:
                    codes.append(code)
            # dedupe preserving order
            return list(dict.fromkeys(codes))
        except httpx.HTTPError as exc:
            log.debug("rxnorm.atc_fetch_failed", rxcui=rxcui, error=str(exc))
            return []


# ---------------------------------------------------------------------------
# (De)serialization helpers
# ---------------------------------------------------------------------------


def _serialize(n: RxNormNorm) -> dict[str, Any]:
    return {
        "input_rxcui": n.input_rxcui,
        "ingredients": [{"rxcui": i.rxcui, "name": i.name} for i in n.ingredients],
        "clinical_drug": (
            {"rxcui": n.clinical_drug.rxcui, "name": n.clinical_drug.name}
            if n.clinical_drug
            else None
        ),
        "atc": list(n.atc),
    }


def _deserialize(payload: dict[str, Any]) -> RxNormNorm:
    cd = payload.get("clinical_drug")
    return RxNormNorm(
        input_rxcui=payload["input_rxcui"],
        ingredients=[RxNormIngredient(**i) for i in payload.get("ingredients", [])],
        clinical_drug=(RxNormIngredient(**cd) if cd else None),
        atc=list(payload.get("atc", [])),
    )
