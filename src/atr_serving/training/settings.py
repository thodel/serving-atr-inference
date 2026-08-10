"""Trainer-service configuration (env prefix ``ATR_TRAIN_``).

Kept separate from the gateway's :class:`atr_serving.config.Settings`: the
trainer owns paths and guards the gateway has no business knowing about, and it
runs in its own venv.

Both classes use ``extra="ignore"``, which matters because the prefixes overlap —
the gateway reads ``ATR_TRAIN_URL`` as its ``train_url`` (#35) and this class
would otherwise see it as an unknown ``url``.

One instance of this service supervises **every** training backend; which
interpreter and runner module a job gets is looked up per engine in
:mod:`atr_serving.training.backends`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from atr_serving.training.backends import runner_python

REPO_ROOT = Path(__file__).resolve().parents[3]


class TrainerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATR_TRAIN_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8204

    # ── layout ────────────────────────────────────────────────────────────
    #: One directory per job; all job state lives here (see jobstore).
    jobs_root: Path = Path.home() / "atr-cache" / "training"
    #: Promoted weights, one directory per trained model.
    trained_root: Path = Path.home() / "atr-cache" / "trained"
    #: The gitignored registry overlay trained models are registered in.
    overlay_path: Path = REPO_ROOT / "config" / "models.local.yaml"
    #: Checkpoints go to LOCAL disk, not the job directory on the share. Lightning
    #: saves them via a temp file + rename; with the target on CIFS and the temp
    #: local that rename is cross-device, and the fsspec version datasets<4 pins
    #: (2025.3.0) cannot fall back to a copy — "Upgrade fsspec to enable
    #: cross-device local checkpoints". Local is also plainly right: kraken keeps
    #: the top 10 checkpoints and rewrites them every epoch, which is a lot of
    #: traffic to push over SMB for files we discard once the best is converted.
    checkpoint_root: Path = Path.home() / "atr-cache" / "checkpoints"
    #: Cache the downloaded ground truth in the standard HF cache, or stream it.
    #:
    #: **False (default) — stream from the hub, keeping nothing.** This is the
    #: right default for the selections this trainer exists for: a ~1 TB page
    #: selection's Arrow generation cache is not something this box wants to
    #: materialise, and in cached mode ``datasets`` downloads and converts the
    #: *entire* selection before yielding the first row — 11½ hours with zero
    #: pages written and no progress reported, on the run that exposed it (#60).
    #: Streaming passes rows straight into the kraken page format, so pages
    #: appear within minutes and the page count is a real progress signal.
    #:
    #: True — download and convert once, reuse across runs. Correct at project
    #: scale (a 116 MB dataset fetched repeatedly is waste), and the reasoning
    #: that made it the old default. It inverts entirely at terabyte scale, which
    #: is why the default moved rather than the option disappearing.
    cache_datasets: bool = False

    # ── executables ───────────────────────────────────────────────────────
    ketos: Path = REPO_ROOT / ".venvs" / "kraken-train" / "bin" / "ketos"
    #: Where the per-engine venvs live. Each job is spawned with *its own*
    #: engine's interpreter (see runner_python) — this service never imports an
    #: engine package, so it does not matter which venv it happens to run in.
    venvs_root: Path = REPO_ROOT / ".venvs"

    def runner_python(self, engine: str) -> Path:
        """Interpreter for ``engine``'s detached runner."""
        return runner_python(engine, self.venvs_root)

    # ── the promotion gate (#36) ──────────────────────────────────────────
    #: The gate posts one held-out page here. Through the gateway, not straight
    #: to the engine: "can this box serve it" is a question about the path real
    #: clients take.
    gateway_url: str = "http://127.0.0.1:8200"
    #: Same shared key the gateway already requires. Empty disables the gate,
    #: which leaves models registered-but-disabled rather than wrongly advertised.
    gateway_api_key: str = ""

    # ── guards (docs/TRAINING_PLAN.md §5) ─────────────────────────────────
    #: PHYSICAL GPU index. GPU 0 is the shared RAG GPU and stays untouched;
    #: nvidia-smi enumerates physically and ignores CUDA_VISIBLE_DEVICES, so this
    #: is the number preflight queries. The child gets CUDA_VISIBLE_DEVICES=<gpu>,
    #: which makes it cuda:0 inside the process.
    gpu: int = 1
    #: Headroom a kraken run needs (batch 256 through 3× Lbx256).
    min_free_vram_mb: int = 12000
    #: A QLoRA fine-tune of an 8B Qwen3-VL is a different order of appetite: ~6 GB
    #: of 4-bit weights, plus activations for a 4 k-token page sample and paged
    #: optimizer state. Checked instead of ``min_free_vram_mb`` for vllm jobs, so a
    #: VLM job queues rather than OOMing on a card that would have fit a kraken run.
    vlm_min_free_vram_mb: int = 24000
    #: `/` is ~80 % full on asterAIx — never materialize a dataset into the last
    #: of it.
    min_free_disk_gb: int = 50
    #: Pages materialized before a chunk is compiled and deleted (#39). 0 = off,
    #: which materializes the whole selection first — right for the 238-page test
    #: case, impossible for the full corpus: 548,322 pages is ~6.96 TB of pages on
    #: top of a ~6.6 TB hub cache, on a share with ~6.2 TB free. With chunking on,
    #: peak page-disk is one chunk instead of the whole selection.
    chunk_pages: int = 0

    #: Training and inference do not share the card politely; one job at a time.
    max_concurrent: int = 1
    #: How often the scheduler reconciles jobs and starts a queued one.
    poll_interval_s: int = 10
    #: Lines of a stage log kept on a failed job record.
    log_tail_lines: int = 50

    def min_free_vram_for(self, engine: str) -> int:
        """VRAM a job of this engine must find free before it may start."""
        return self.vlm_min_free_vram_mb if engine == "vllm" else self.min_free_vram_mb

    def env_for_child(self) -> dict[str, str]:
        """Environment overrides for a spawned training process."""
        return {"CUDA_VISIBLE_DEVICES": str(self.gpu)}


_settings: TrainerSettings | None = None


def get_settings() -> TrainerSettings:
    global _settings
    if _settings is None:
        _settings = TrainerSettings()
    return _settings
