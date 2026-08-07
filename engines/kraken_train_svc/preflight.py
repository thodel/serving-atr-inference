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

__all__ = ["PreflightError", "GpuInfo", "free_disk_gb", "query_gpus", "check_disk",
           "check_vram", "check_tmpdir", "mount_fstype", "NETWORK_FS"]

#: Filesystems where POSIX delete semantics do not hold well enough for the
#: temp-directory churn that ketos/lightning/datasets do.
NETWORK_FS = frozenset({"cifs", "smb3", "smbfs", "nfs", "nfs4", "fuse.sshfs", "9p", "afs"})


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


def mount_fstype(path: str | Path, mounts_file: str | Path = "/proc/mounts") -> tuple[str, str] | None:
    """(mountpoint, fstype) for the filesystem holding ``path``, or None.

    Longest-prefix match over /proc/mounts, which is how the kernel resolves it.
    """
    try:
        lines = Path(mounts_file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    target = Path(path).expanduser().resolve()
    best: tuple[str, str] | None = None
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mountpoint = parts[1].replace("\\040", " ")  # /proc/mounts escapes spaces
        fstype = parts[2]
        mp = Path(mountpoint)
        if target == mp or mp in target.parents:
            if best is None or len(mountpoint) > len(best[0]):
                best = (mountpoint, fstype)
    return best


def check_tmpdir(path: str | Path, mounts_file: str | Path = "/proc/mounts") -> None:
    """Refuse a temp directory on a network filesystem.

    Found the hard way: with TMPDIR on the CIFS research share, ``ketos compile``
    died three minutes in with ``OSError: [Errno 39] Directory not empty`` from
    ``shutil.rmtree`` — SMB does not release directory entries promptly enough for
    the create/delete churn of temporary directories. Local scratch is also simply
    faster. Failing here turns a confusing mid-stage crash into an immediate,
    explicable rejection.
    """
    hit = mount_fstype(path, mounts_file)
    if hit and hit[1] in NETWORK_FS:
        raise PreflightError(
            f"TMPDIR {path} is on a {hit[1]} filesystem ({hit[0]}). Temporary "
            "directories there fail to clean up (ENOTEMPTY in shutil.rmtree during "
            "ketos compile). Point TMPDIR at local disk in .env."
        )
