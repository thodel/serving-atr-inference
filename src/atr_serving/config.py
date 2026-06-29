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

# Sentinel default key — safe only for local dev. Override via ATR_API_KEY.
DEFAULT_INSECURE_KEY = "dev-insecure-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATR_", env_file=".env", extra="ignore")

    # ── HTTP ──────────────────────────────────────────────────────────────
    # :8000/:8080/:9000/:11434/:80 are already taken on asterAIx — see
    # docs/asteraix-environment.md. Gateway lives on :8200, engines on :820x.
    host: str = "0.0.0.0"
    port: int = 8200

    # ── Auth ──────────────────────────────────────────────────────────────
    # Static shared key. Sent by clients in the `X-API-Key` header.
    # Default is dev-only; ALWAYS override on the server via ATR_API_KEY.
    api_key: str = DEFAULT_INSECURE_KEY
    # When False, /models and /recognize are open (dev convenience only).
    require_auth: bool = True

    # ── Registry ──────────────────────────────────────────────────────────
    models_config: Path = REPO_ROOT / "config" / "models.yaml"

    # ── Engine backends (gateway -> engine services over localhost) ───────
    # Phase 0 only records them; routes that use them arrive in later phases.
    kraken_url: str = "http://127.0.0.1:8201"
    trocr_url: str = "http://127.0.0.1:8202"
    party_url: str = "http://127.0.0.1:8203"
    # vLLM instances are dynamic (one per resident VLM); discovered via the
    # ModelManager in Phase 3, not statically configured here.

    def engine_urls(self) -> dict[str, str]:
        return {"kraken": self.kraken_url, "trocr": self.trocr_url, "party": self.party_url}

    # ── vLLM (managed as subprocesses by the ModelManager, not systemd) ───────
    # asterAIx: GPU 1 only (GPU 0 is the shared RAG GPU); one 8B resident at a time.
    vllm_python: Path = REPO_ROOT / ".venvs" / "vllm" / "bin" / "vllm"
    vllm_gpu: int = 1
    vllm_port_base: int = 8210
    # Budget for resident vLLM models on vllm_gpu. ~30 GB lets LightOnOCR (3 GB,
    # pinned) + one 8B (18 GB) co-reside while leaving headroom for the small
    # engine services that also sit on GPU 1. A 2nd 8B is evicted (LRU).
    vllm_vram_budget_mb: int = 30000
    vllm_gpu_memory_utilization: float = 0.45
    vllm_trust_remote_code: bool = True
    vllm_max_model_len: int | None = None
    vllm_startup_timeout_s: int = 180
    vllm_max_new_tokens: int = 512


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
