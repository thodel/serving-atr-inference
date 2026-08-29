"""Gateway configuration.

All settings come from environment variables (prefix ``ATR_``) with sane
defaults, so the gateway runs out of the box for local development. On the
server, set at least ``ATR_API_KEY``.

The two VMs share a hard-coded API key (private university network, behind the
same firewall, no TLS) — see README §Security.
"""

from __future__ import annotations

from pathlib import Path

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
    #: Locally trained models, written by the trainer's register stage and
    #: **gitignored** — the tracked registry above stays a reviewed artifact. Only
    #: entries the promotion gate has proven servable (``enabled: true``) are
    #: merged; a missing file is the normal state of a box that has not trained.
    models_overlay: Path = REPO_ROOT / "config" / "models.local.yaml"

    # ── Engine backends (gateway -> engine services over localhost) ───────
    # Phase 0 only records them; routes that use them arrive in later phases.
    kraken_url: str = "http://127.0.0.1:8201"
    trocr_url: str = "http://127.0.0.1:8202"
    party_url: str = "http://127.0.0.1:8203"
    # The training service (#34). Not a recognition engine — it is reached only by
    # the /train/* proxy (#35), which is the ONLY way in: atr-train binds
    # 127.0.0.1 and the ufw rule opens :8200 alone to the client host.
    train_url: str = "http://127.0.0.1:8204"
    # vLLM instances are dynamic (one per resident VLM); discovered via the
    # ModelManager in Phase 3, not statically configured here.

    def engine_urls(self) -> dict[str, str]:
        """Recognition engines only — ``get_engine_client`` indexes this."""
        return {"kraken": self.kraken_url, "trocr": self.trocr_url, "party": self.party_url}

    def service_urls(self) -> dict[str, str]:
        """Everything /health reports, recognition engines plus the trainer.

        Kept separate from :meth:`engine_urls` on purpose: that mapping is indexed
        by ``ENGINE_IMAGE_FIELD`` for multipart recognition calls, and the trainer
        has no image field. Merging them would put a KeyError one typo away.
        """
        return {**self.engine_urls(), "train": self.train_url}

    # ── vLLM (managed as subprocesses by the ModelManager, not systemd) ───────
    # asterAIx: GPU 1 only (GPU 0 is the shared RAG GPU); one 8B resident at a time.
    vllm_python: Path = REPO_ROOT / ".venvs" / "vllm" / "bin" / "vllm"
    vllm_gpu: int = 1
    vllm_port_base: int = 8210
    # Budget for resident vLLM models on vllm_gpu. ~30 GB lets LightOnOCR (3 GB,
    # pinned) + one 8B (18 GB) co-reside while leaving headroom for the small
    # engine services that also sit on GPU 1. A 2nd 8B is evicted (LRU).
    vllm_vram_budget_mb: int = 30000
    # GPU 1 is essentially free (~45 GB). At 0.45, weights (16.6 GB) + Qwen3-VL's
    # profiling overhead left NEGATIVE KV cache. 0.70 (~32 GB) leaves ~8 GB for KV
    # and still ~14 GB for the small engines (kraken/trocr) on GPU 1.
    vllm_gpu_memory_utilization: float = 0.70
    vllm_trust_remote_code: bool = True
    # Qwen3-VL defaults to 262k context; the KV cache for that won't fit alongside
    # 17 GB of weights on GPU 1. Cap it — OCR/HTR needs nothing close.
    vllm_max_model_len: int | None = 16384
    vllm_startup_timeout_s: int = 300  # 8B load + CUDA graph capture can exceed 180s
    #: How many line crops are recognised at once in the line pipeline. The loop was
    #: strictly sequential: a 79-line page cost 79 round trips at ~0.58s, ~46s, which
    #: measurement made the largest single item in an ensemble page
    #: (agentic_historian#404). Bounded rather than unbounded — one GPU serves these,
    #: and flooding it trades latency for queueing plus a memory risk.
    line_concurrency: int = 6
    vllm_max_new_tokens: int = 512
    # The Qwen3-VL / LightOnOCR models are LoRA adapters whose adaptation includes
    # the vision tower, which vLLM can't serve as a runtime LoRA. scripts/merge_loras.py
    # bakes each adapter into its base here; the launcher serves the merged dir if present.
    vllm_merged_dir: Path = Path.home() / "atr-cache" / "vllm-merged"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
