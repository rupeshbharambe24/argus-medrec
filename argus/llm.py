"""Gemini LLM client wrapper.

Used by Tool 2 (recommended_action), Tool 4 (intentionality classification),
and Tool 5 (note generation).

Behavior:
- When no API key is configured, calls return `None` and the caller falls back
  to a deterministic template. This keeps the server usable offline / in CI.
- All prompts enforce JSON structured output where possible, with a Pydantic
  schema validation pass on the response.
- Single timeout / retry policy; no streaming.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from argus.config import get_settings
from argus.logging_setup import get_logger

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult:
    text: str
    raw: dict[str, Any]
    tokens_used: int | None = None


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM is requested but not configured."""


class LLMClient:
    """Thin async wrapper around google-generativeai with JSON schema helpers."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model_name = self._settings.gemini_model
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        return bool(self._settings.gemini_api_key)

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.available:
            raise LLMUnavailableError("GEMINI_API_KEY not configured")
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailableError(f"google-generativeai not installed: {exc}") from exc

        genai.configure(api_key=self._settings.gemini_api_key.get_secret_value())  # type: ignore[union-attr]
        self._model = genai.GenerativeModel(self._model_name)
        return self._model

    # -----------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
    ) -> LLMResult | None:
        """Plain text generation. Returns None if LLM unavailable.

        Retries on 429 / quota / rate-limit errors with exponential backoff
        (2s, 5s). After max_retries the call returns None and the caller
        falls back to its deterministic path.
        """
        if not self.available:
            return None
        try:
            model = self._ensure_model()
        except LLMUnavailableError as exc:
            log.info("llm.unavailable", reason=str(exc))
            return None

        backoffs = [2.0, 5.0]
        attempt = 0
        while True:
            try:
                resp = await asyncio.wait_for(
                    asyncio.to_thread(model.generate_content, prompt),
                    timeout=timeout_s,
                )
                break
            except TimeoutError:
                log.warning("llm.timeout", timeout=timeout_s)
                return None
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                is_quota = (
                    "429" in msg
                    or "quota" in msg
                    or "rate" in msg
                    or "resourceexhausted" in msg
                )
                if is_quota and attempt < max_retries:
                    delay = backoffs[min(attempt, len(backoffs) - 1)]
                    log.warning(
                        "llm.quota_retry",
                        attempt=attempt + 1,
                        delay_s=delay,
                        error=str(exc)[:200],
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                log.warning(
                    "llm.generate_failed",
                    quota_exhausted=is_quota,
                    error=str(exc)[:300],
                )
                return None

        text = getattr(resp, "text", None) or ""
        usage = getattr(resp, "usage_metadata", None)
        tokens = getattr(usage, "total_token_count", None) if usage else None
        return LLMResult(
            text=text,
            raw={"candidates": str(getattr(resp, "candidates", None))},
            tokens_used=tokens,
        )

    async def generate_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        timeout_s: float = 30.0,
        max_retries: int = 2,
    ) -> T | None:
        """Generation expecting JSON output validated against a Pydantic schema.

        The prompt should instruct the model to emit JSON only (no prose).
        We extract the first ``{...}`` block from the text to be tolerant of
        Markdown fences.
        """
        result = await self.generate(
            prompt + "\n\nRespond with a single JSON object only. No prose, no markdown.",
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        if result is None:
            return None
        payload = _extract_first_json_object(result.text)
        if payload is None:
            log.warning("llm.json.no_json_found", snippet=result.text[:200])
            return None
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            log.warning("llm.json.schema_invalid", error=str(exc)[:300])
            return None


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Find the first balanced {...} block in text and parse it as JSON."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# Convenience singleton
_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
