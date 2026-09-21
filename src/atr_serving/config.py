"""Gateway configuration.

All settings come from environment variables (prefix ``ATR_``) with sane
defaults, so the gateway runs out of the box for local development. On the
server, set at least ``ATR_API_KEY``.

The two VMs share a hard-coded API key (private university network, behind the
same firewall, no TLS) — see README §Security.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

# Sentinel default key — safe only for local dev. Override via ATR_API_KEY.
DEFAULT_INSECURE_KEY = "dev-insecure-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATR_", env_file=".env", extra="ignore")

    # ── HTTP ──────────────────────────────────────────────────────────────
    # :8000/:8080/:9000/:11434/:80 are already taken on asterAIx — see
    # docs/idhefix-environment.md. Gateway lives on :8200, engines on :820x.
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
    #: The registry directory on the research share (#138), e.g.
    #: ``/mnt/wbkolleg_dh_1/Textrecognition_Training/registry``. The gateway
    #: publishes ``models_config`` there as ``models.yaml`` on startup and serves
    #: the trainer's ``trained/<id>.yaml`` alongside ``models_overlay``, reloading
    #: both without a restart. None = off, and the gateway reads exactly what it
    #: read before. The trainer's ``ATR_TRAIN_MODELS_CONFIG`` must name
    #: ``<this>/models.yaml`` — one more value the two machines have to agree on.
    #:
    #: Absolute and from the environment, not REPO_ROOT-relative like the two
    #: above: relative to a checkout, two checkouts would each have their own
    #: registry, which is the problem the shared directory exists to remove.
    registry_root: Path | None = None
    #: At most one look at ``trained/`` per this many seconds, triggered by
    #: requests. A listing on CIFS is cheap but not free, and the client's
    #: attribute cache makes a change visible late anyway.
    registry_reload_interval_s: float = 5.0

    @field_validator("registry_root", mode="before")
    @classmethod
    def _registry_root_is_absolute(cls, value):
        # `ATR_REGISTRY_ROOT=` in .env reads as "off", not as the current directory.
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if not Path(value).is_absolute():
            raise ValueError(f"registry_root must be an absolute path, got {value!r}")
        return value

    # ── Engine backends (gateway -> engine services over localhost) ───────
    # Phase 0 only records them; routes that use them arrive in later phases.
    kraken_url: str = "http://127.0.0.1:8201"
    trocr_url: str = "http://127.0.0.1:8202"
    party_url: str = "http://127.0.0.1:8203"
    #: Party transcribes every image alongside the requested engine
    #: (config/models.yaml calls it "always-on"). It costs real latency — its
    #: decoder generates token by token, measured at ~8 s per line — so this is a
    #: switch rather than a constant: a slow second opinion must be removable
    #: without a redeploy.
    party_second_opinion: bool = True
    #: How long the requested engine's result may wait for party once it is ready
    #: (#149). Counted from the primary result, not from the request: the two run
    #: concurrently, so a party page that finishes inside the primary's own time
    #: costs nothing, and this only bounds how long a finished answer is held.
    #: Measured on 17.09.: one party page takes 30-80 s, and with pages queued
    #: behind each other the n-th request of a burst waited for n-1 of them.
    party_second_opinion_grace_s: float = 30.0
    # The training service (#34). Not a recognition engine — it is reached only by
    # the /train/* proxy (#35), and callers (the Discord bot, the ATR-MCP) know
    # nothing but :8200: ufw opens that port alone to tei.
    #
    # Since the split (#137) the trainer lives on asteraix; on idhefix this is
    # ``http://130.92.59.242:8204`` since 16.09.2026. Nothing about this box's
    # cards depends on it any more (#139): /train/gpu is always the trainer's
    # reading, the vLLM launcher asks no trainer, and GET /gpu reads this box.
    # Whether the host part is loopback only decides how a trainer's 5xx text is
    # labelled (see :func:`is_loopback_url`).
    train_url: str = "http://127.0.0.1:8204"
    #: Sent as ``X-API-Key`` on every call to ``train_url``. **Shared** with the
    #: trainer, which reads the same variable name (ATR_TRAIN_API_KEY) and refuses
    #: every route but /health without it once it listens beyond loopback: on
    #: asteraix ufw does not filter high ports, so the application is the only
    #: guard. Deliberately NOT ``api_key``: that one authenticates callers of this
    #: gateway, this one authenticates the gateway to the trainer, and one leaked
    #: value should not open both directions (training-atr-models#9).
    #: Empty = no header, which is what the in-repo trainer on 127.0.0.1 expects.
    #: ``repr=False`` keeps it out of any log line that prints the settings.
    train_api_key: str = Field("", repr=False)
    #: How long the proxy waits for the trainer to start answering, per call.
    #: **Must stay below the callers' own timeout** — agentic_historian's
    #: atr_status.TIMEOUT_S is 30 s — or the 504 naming the trainer's URL is never
    #: seen: the caller's clock starts before the gateway even connects, so with
    #: 30 s on both sides the bot gave up first, every time, and reported a
    #: ReadTimeout against idhefix's :8200 (reproduced in the #137 review against a
    #: trainer that accepted connections and never answered). The connect phase
    #: has its own 5 s (clients.TRAINER_CONNECT_TIMEOUT_S), so a call gives up
    #: within 25 s — and each route the bot reads makes one call. (A submit adds
    #: the 5 s /health check, but its caller, agent_a's training_client, waits
    #: ATR_HTTP_TIMEOUT = 300 s.) The ~1 MB /train/jobs body does not need more:
    #: httpx's read timeout counts the gap between chunks, not the transfer, so
    #: only the trainer's time to first byte counts against it.
    train_timeout_s: float = 20.0
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
    # Budget for resident vLLM models on vllm_gpu, in registry vram_mb; a launch
    # beyond it evicts the least recently used lazy model. Unset (the default),
    # it is read at launch time: the card less what the engines hold less
    # vllm_vram_reserve_mb — 28190 MiB on idhefix GPU 1 on 16.09.2026, see
    # manager.vram_budget. Set, it is an override. The old default, 30000, was
    # more than that card has left once the engines (15830 MiB) are on it.
    vllm_vram_budget_mb: int | None = None

    @field_validator("vllm_vram_budget_mb", mode="before")
    @classmethod
    def _empty_budget_is_unset(cls, value):
        # `ATR_VLLM_VRAM_BUDGET_MB=` in .env reads as "derive it", not as an error.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # GPU 1 is essentially free (~45 GB). At 0.45, weights (16.6 GB) + Qwen3-VL's
    # profiling overhead left NEGATIVE KV cache. 0.70 (~32 GB) leaves ~8 GB for KV
    # and still ~14 GB for the small engines (kraken/trocr) on GPU 1.
    vllm_gpu_memory_utilization: float = 0.70
    # …and 0.70 is a constant that knows neither the model nor the card. On
    # 2026-09-14 a 12 GB model failed to start three times running on a card that
    # had 45 GB total and 19 free, because 0.70 of the total is 31 GB. With
    # autosizing the launcher asks the registry how big the model is and the
    # driver how much is free, and computes the fraction per launch; the constant
    # above stays as the fallback for a model with no vram_mb and for a host where
    # nvidia-smi cannot be read. See manager.plan_gpu_budget.
    vllm_autosize: bool = True
    # vram_mb is the weights; vLLM wants a KV cache, activation scratch and CUDA
    # graphs out of the same allocation.
    vllm_vram_headroom: float = 1.6
    # Left to the card on top of the model's share: this process's CUDA context,
    # fragmentation, and growth in the small engines sharing the card.
    vllm_vram_reserve_mb: int = 2048
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
    #: Generation ceiling for a **line** crop. Ample there, and short for a page —
    #: which is why it is no longer the answer for both (#131). A page-level model
    #: gets ``vllm_max_new_tokens_page`` unless its registry entry declares its own
    #: ``max_new_tokens``.
    #: Send each vLLM model its images at the pixel budget its fine-tune used.
    #: Off sends the scan as it is, which is what this gateway did until a page
    #: model answered in fragments because the image arrived at eight times the
    #: scale it was trained on.
    vllm_visual_budget: bool = True
    vllm_max_new_tokens: int = 512
    #: Generation ceiling for a **page**. A dense page of nineteenth-century German
    #: runs well past 512 tokens, and hitting the ceiling is a normal ``200`` whose
    #: transcription stops mid-sentence — visible since #123, but only to someone
    #: who looks. asterAIx had 4096 set by hand in `.env`; every other deployment
    #: and every fresh checkout got 512. It is a fallback: a model that knows its
    #: own length says so in the registry.
    vllm_max_new_tokens_page: int = 4096
    #: Kept free of generated tokens inside ``vllm_max_model_len`` for the prompt
    #: and, at page level, the image — which is most of it. A ceiling that leaves
    #: no room for the input is not a ceiling, it is a failed request.
    vllm_prompt_reserve_tokens: int = 4096
    # The Qwen3-VL / LightOnOCR models are LoRA adapters whose adaptation includes
    # the vision tower, which vLLM can't serve as a runtime LoRA. scripts/merge_loras.py
    # bakes each adapter into its base here; the launcher serves the merged dir if present.
    vllm_merged_dir: Path = Path.home() / "atr-cache" / "vllm-merged"


def is_loopback_url(url: str) -> bool:
    """Whether ``url`` names this machine: ``localhost`` or a loopback address.

    The one test for "is the trainer on this box" (#137). Since #139 it only
    decides whether a trainer's 5xx text is prefixed with the trainer's URL and
    whether a missing ATR_TRAIN_API_KEY is warned about; nothing about a GPU
    depends on it. Textual, no DNS: the configured values are literal
    addresses, and a lookup is a new way to hang. Anything else — a hostname,
    this box's own public address, a URL without a scheme — counts as remote,
    which errs towards naming the machine.
    """
    host = urlsplit(url).hostname
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
