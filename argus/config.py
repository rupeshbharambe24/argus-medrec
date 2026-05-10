"""Configuration loaded from environment variables (.env in dev).

All runtime knobs live here. No hardcoded values elsewhere in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for Argus. Populated from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Server ---------------------------------------------------------------
    host: str = Field(default="0.0.0.0", alias="ARGUS_HOST")
    port: int = Field(default=8080, alias="ARGUS_PORT")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="ARGUS_LOG_LEVEL"
    )
    env: Literal["dev", "staging", "prod"] = Field(default="dev", alias="ARGUS_ENV")

    # ---- LLM ------------------------------------------------------------------
    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    # gemini-2.5-flash-lite has ~4x the free-tier daily quota of gemini-2.5-flash
    # (~1000 RPD vs ~250 RPD) at minor quality cost — better default for the
    # hackathon. Override with GEMINI_MODEL=gemini-2.5-flash for higher quality.
    gemini_model: str = Field(default="gemini-2.5-flash-lite", alias="GEMINI_MODEL")

    # ---- FHIR fallback (dev only — real requests get token from SHARP context) -
    fallback_fhir_base_url: HttpUrl | None = Field(
        default=None, alias="ARGUS_FALLBACK_FHIR_BASE_URL"
    )
    fallback_fhir_token: SecretStr | None = Field(
        default=None, alias="ARGUS_FALLBACK_FHIR_TOKEN"
    )

    # ---- Reference KB ---------------------------------------------------------
    reference_kb_path: Path = Field(
        default=Path("argus/reference/reference.sqlite"),
        alias="ARGUS_REFERENCE_KB_PATH",
    )

    # ---- ML artifacts ---------------------------------------------------------
    model_dir: Path = Field(default=Path("argus/ml/artifacts"), alias="ARGUS_MODEL_DIR")

    # ---- External APIs --------------------------------------------------------
    rxnav_base_url: HttpUrl = Field(
        default="https://rxnav.nlm.nih.gov/REST",  # type: ignore[assignment]
        alias="RXNAV_BASE_URL",
    )
    rxnav_cache_ttl_days: int = Field(default=7, alias="RXNAV_CACHE_TTL_DAYS")

    # ---- Feature flags --------------------------------------------------------
    enable_llm_reasoning: bool = Field(default=True, alias="ARGUS_ENABLE_LLM_REASONING")
    enable_ml_severity: bool = Field(default=True, alias="ARGUS_ENABLE_ML_SEVERITY")
    default_severity_threshold: Literal["minor", "moderate", "major"] = Field(
        default="moderate", alias="ARGUS_DEFAULT_SEVERITY_THRESHOLD"
    )

    # ---- Derived --------------------------------------------------------------
    @property
    def is_prod(self) -> bool:
        return self.env == "prod"

    @property
    def rxnav_base(self) -> str:
        # Strip trailing slash for predictable URL joining
        return str(self.rxnav_base_url).rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (lazy, memoized)."""
    return Settings()  # type: ignore[call-arg]
