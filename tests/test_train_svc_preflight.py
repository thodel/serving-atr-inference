"""Resource guards (#34) — nvidia-smi parsing and the disk check."""

from pathlib import Path

import pytest

from atr_serving.training import preflight
from atr_serving.training.preflight import (
    GpuInfo,
    PreflightError,
    check_disk,
    check_tmpdir,
    check_vram,
    free_disk_gb,
    query_gpus,
)

# `nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits`
# on asterAIx: GPU 0 shared with the RAG service, GPU 1 ours.
SMI_OUTPUT = "0, 35000, 46068\n1, 40000, 46068\n"


class FakeCompleted:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def test_query_gpus_parses_physical_indices(monkeypatch):
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: FakeCompleted(SMI_OUTPUT))
    assert query_gpus() == [GpuInfo(0, 35000, 46068), GpuInfo(1, 40000, 46068)]


def test_missing_nvidia_smi_is_reported_not_ignored(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(preflight.subprocess, "run", boom)
    with pytest.raises(PreflightError, match="not found"):
        query_gpus()


def test_unparsable_output_is_an_error(monkeypatch):
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: FakeCompleted("no GPUs\n"))
    with pytest.raises(PreflightError, match="could not parse"):
        query_gpus()


def test_check_vram_passes_when_the_card_is_free():
    gpus = [GpuInfo(0, 35000, 46068), GpuInfo(1, 40000, 46068)]
    assert check_vram(1, 12000, gpus).free_mb == 40000


def test_check_vram_refuses_a_busy_card():
    """An 8B vLLM model resident on GPU 1 leaves no room to train beside it."""
    gpus = [GpuInfo(0, 35000, 46068), GpuInfo(1, 6000, 46068)]
    with pytest.raises(PreflightError, match="6000 MB free, need 12000"):
        check_vram(1, 12000, gpus)


def test_check_vram_on_a_nonexistent_gpu():
    with pytest.raises(PreflightError, match="does not exist"):
        check_vram(3, 12000, [GpuInfo(0, 1, 2)])


def test_free_disk_gb_walks_up_to_an_existing_parent(tmp_path: Path):
    """The job directory does not exist yet when the guard runs."""
    assert free_disk_gb(tmp_path / "not" / "created" / "yet") > 0


def test_check_disk_refuses_when_the_box_is_full(tmp_path: Path):
    check_disk(tmp_path, 0.0)  # passes
    with pytest.raises(PreflightError, match="GB free"):
        check_disk(tmp_path, 10**9)


# ── TMPDIR must be local (a CIFS TMPDIR broke ketos compile) ────────────────
MOUNTS = """\
/dev/nvme0n1p2 / ext4 rw,relatime 0 0
proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0
//resstore.unibe.ch/wbkolleg_dh_1 /mnt/wbkolleg_dh_1 cifs rw,relatime 0 0
tmpfs /run tmpfs rw,nosuid,nodev 0 0
"""


@pytest.fixture
def mounts(tmp_path: Path) -> Path:
    f = tmp_path / "mounts"
    f.write_text(MOUNTS, encoding="utf-8")
    return f


def test_mount_fstype_longest_prefix_wins(mounts):
    from atr_serving.training.preflight import mount_fstype

    assert mount_fstype("/mnt/wbkolleg_dh_1/x/y", mounts) == ("/mnt/wbkolleg_dh_1", "cifs")
    assert mount_fstype("/home/tobias", mounts) == ("/", "ext4")


def test_check_tmpdir_rejects_the_research_share(mounts):
    """ketos compile died with ENOTEMPTY in shutil.rmtree with TMPDIR here."""
    with pytest.raises(PreflightError, match="cifs"):
        check_tmpdir("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/tmp", mounts)


def test_check_tmpdir_accepts_local_disk(mounts):
    check_tmpdir("/home/tobias/atr-cache/tmp", mounts)
    check_tmpdir("/tmp", mounts)


def test_check_tmpdir_is_silent_without_proc_mounts(tmp_path: Path):
    """Not Linux, or /proc unavailable — do not invent a failure."""
    check_tmpdir("/anything", tmp_path / "nope")
