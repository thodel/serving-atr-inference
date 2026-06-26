"""Gateway configuration.

All settings come from environment variables (prefix ``ATR_``) with sane
defaults, so the gateway runs out of the box for local development. On the
server, set at least ``ATR_API_KEY``.

The two VMs share a hard-coded API key (private university network, behind the
same firewall, no TLS) — see README §Security.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATR_", env_file=".env", extra="ignore")

    # ── HTTP ──────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Auth ──────────────────────────────────────────────────────────────
    # Static shared key. Sent by clients in the `X-API-Key` header.
    # Default is dev-only; ALWAYS override on the server via ATR_API_KEY.
    api_key: str = "dev-insecure-change-me"
    # When False, /models and /recognize are open (dev convenience only).
    require_auth: bool = True

    # ── Registry ──────────────────────────────────────────────────────────
    models_config: Path = REPO_ROOT / "config" / "models.yaml"

    # ── Engine backends (gateway -> engine services over localhost) ───────
    # Phase 0 only records them; routes that use them arrive in later phases.
    kraken_url: str = "http://127.0.0.1:8101"
    trocr_url: str = "http://127.0.0.1:8102"
    party_url: str = "http://127.0.0.1:8103"
    # vLLM instances are dynamic (one per resident VLM); discovered via the
    # ModelManager in Phase 3, not statically configured here.

    def engine_urls(self) -> dict[str, str]:
        return {"kraken": self.kraken_url, "trocr": self.trocr_url, "party": self.party_url}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
