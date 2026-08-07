"""Trainer-service configuration (env prefix ``ATR_TRAIN_``).

Kept separate from the gateway's :class:`atr_serving.config.Settings`: the
trainer owns paths and guards the gateway has no business knowing about, and it
runs in its own venv.

Both classes use ``extra="ignore"``, which matters because the prefixes overlap —
the gateway reads ``ATR_TRAIN_URL`` as its ``train_url`` (#35) and this class
would otherwise see it as an unknown ``url``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    #: Keep downloaded ground truth in the standard HF cache (whose ``hub/`` is
    #: symlinked to the research share on asterAIx), so the same dataset is
    #: fetched once and reused — by us and by lassberg/vlm_training alike.
    #: False = stream from the hub every run, keeping nothing.
    cache_datasets: bool = True

    # ── executables ───────────────────────────────────────────────────────
    ketos: Path = REPO_ROOT / ".venvs" / "kraken-train" / "bin" / "ketos"
    #: Interpreter used to spawn the detached runner — this venv's own by default.
    python: Path = Path(sys.executable)

    # ── guards (docs/TRAINING_PLAN.md §5) ─────────────────────────────────
    #: PHYSICAL GPU index. GPU 0 is the shared RAG GPU and stays untouched;
    #: nvidia-smi enumerates physically and ignores CUDA_VISIBLE_DEVICES, so this
    #: is the number preflight queries. The child gets CUDA_VISIBLE_DEVICES=<gpu>,
    #: which makes it cuda:0 inside the process.
    gpu: int = 1
    min_free_vram_mb: int = 12000
    #: `/` is ~80 % full on asterAIx — never materialize a dataset into the last
    #: of it.
    min_free_disk_gb: int = 50
    #: Training and inference do not share the card politely; one job at a time.
    max_concurrent: int = 1
    #: How often the scheduler reconciles jobs and starts a queued one.
    poll_interval_s: int = 10
    #: Lines of a stage log kept on a failed job record.
    log_tail_lines: int = 50

    def env_for_child(self) -> dict[str, str]:
        """Environment overrides for a spawned training process."""
        return {"CUDA_VISIBLE_DEVICES": str(self.gpu)}


_settings: TrainerSettings | None = None


def get_settings() -> TrainerSettings:
    global _settings
    if _settings is None:
        _settings = TrainerSettings()
    return _settings
