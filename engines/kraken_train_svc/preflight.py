"""Resource guards — refuse a job that cannot succeed instead of discovering it
three hours in.

Two hard limits on asterAIx (``docs/asteraix-environment.md``):

* **GPU 1 is shared with the serving engines** (kraken/trocr/party ≈ 10 GB) and,
  when a vLLM model is resident, with an 18 GB 8 B model. Training into whatever
  is left is how both sides OOM.
* **``/`` is ~80 % full**, ~356 GB free, and the ground-truth dataset is ~6.6 TB.

Disk is checked at submit (it will not fix itself); VRAM is checked at start,
because a busy GPU is exactly what a queue is for.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PreflightError", "GpuInfo", "free_disk_gb", "query_gpus", "check_disk", "check_vram"]


class PreflightError(RuntimeError):
    """Raised when the host cannot host the job."""


@dataclass(frozen=True)
class GpuInfo:
    index: int
    free_mb: int
    total_mb: int


def free_disk_gb(path: str | Path) -> float:
    """Free space on the filesystem holding ``path`` (the nearest existing parent
    — the job directory itself may not exist yet)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def query_gpus(nvidia_smi: str = "nvidia-smi", timeout: float = 10.0) -> list[GpuInfo]:
    """All GPUs and their free VRAM, by **physical** index.

    ``nvidia-smi`` enumerates physically and does not honour
    ``CUDA_VISIBLE_DEVICES``, so the indices here match ``TrainerSettings.gpu``
    rather than the ``cuda:0`` the training process sees.
    """
    cmd = [nvidia_smi, "--query-gpu=index,memory.free,memory.total",
           "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except FileNotFoundError as exc:
        raise PreflightError(f"{nvidia_smi} not found — cannot verify free VRAM") from exc
    except subprocess.CalledProcessError as exc:
        raise PreflightError(
            f"{nvidia_smi} failed ({exc.returncode}): {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise PreflightError(f"{nvidia_smi} timed out after {timeout}s") from exc

    gpus = []
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append(GpuInfo(int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    if not gpus:
        raise PreflightError(f"could not parse any GPU from {nvidia_smi} output: {out.stdout!r}")
    return gpus


def check_disk(path: str | Path, min_free_gb: float) -> None:
    free = free_disk_gb(path)
    if free < min_free_gb:
        raise PreflightError(
            f"only {free:.1f} GB free at {path}; this job needs {min_free_gb:.0f} GB of "
            "headroom. Delete old job directories or lower ATR_TRAIN_MIN_FREE_DISK_GB."
        )


def check_vram(gpu: int, min_free_mb: int, gpus: list[GpuInfo] | None = None) -> GpuInfo:
    """Verify GPU ``gpu`` has ``min_free_mb`` free. Returns the GPU's state."""
    gpus = query_gpus() if gpus is None else gpus
    by_index = {g.index: g for g in gpus}
    if gpu not in by_index:
        raise PreflightError(
            f"GPU {gpu} does not exist (nvidia-smi reports {sorted(by_index)})"
        )
    info = by_index[gpu]
    if info.free_mb < min_free_mb:
        raise PreflightError(
            f"GPU {gpu} has {info.free_mb} MB free, need {min_free_mb} MB. Something else "
            "is resident — check the gateway's vLLM residency (/health) before training."
        )
    return info
