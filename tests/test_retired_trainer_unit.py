"""The in-repo trainer stays retired on idhefix (#139).

Training moved to asteraix on 16.09.2026. ``scripts/install_user_units.sh`` used
to copy, enable and start ``atr-train`` with the engines, so the next routine
deploy of this repo would have brought back a trainer with none of
training-atr-models#15's job-ownership rules, pointed at the shared job store.
These tests run the real script against a fake ``systemctl`` and a throwaway
home, so what they pin is what the script does, not what it says.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "install_user_units.sh"


def _fake_bin(tmp_path: Path, trainer_enabled: bool) -> Path:
    """A PATH entry whose systemctl/loginctl log their arguments and do nothing."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    enabled = "0" if trainer_enabled else "1"
    systemctl = f"""#!/usr/bin/env bash
echo "systemctl $*" >> "{log}"
case "$*" in
  *"is-enabled atr-train.service"*) exit {enabled} ;;
  *"is-active atr-train.service"*) exit 1 ;;
esac
exit 0
"""
    loginctl = f"""#!/usr/bin/env bash
echo "loginctl $*" >> "{log}"
echo "Linger=yes"
"""
    for name, body in (("systemctl", systemctl), ("loginctl", loginctl)):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def _run(tmp_path: Path, trainer_enabled: bool = False) -> tuple[subprocess.CompletedProcess, str]:
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = _fake_bin(tmp_path, trainer_enabled)
    env = {"PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin", "HOME": str(home), "USER": "tester"}
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True,
                            text=True, timeout=30)
    log = (tmp_path / "calls.log").read_text()
    return result, log


def test_the_installer_never_installs_the_retired_trainer(tmp_path):
    result, log = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    installed = sorted(p.name for p in (tmp_path / "home/.config/systemd/user").iterdir())
    assert "atr-train.service" not in installed
    assert "atr-gateway.service" in installed
    assert "enable atr-train" not in log
    assert "start atr-train" not in log


def test_the_trainer_unit_is_not_shipped_any_more():
    assert not (REPO / "deploy" / "systemd" / "atr-train.service").exists()


def test_a_trainer_still_enabled_here_is_named_not_silently_kept(tmp_path):
    result, log = _run(tmp_path, trainer_enabled=True)
    assert result.returncode == 0, result.stderr
    assert "atr-train.service is retired" in result.stdout
    assert "disable --now atr-train.service" in result.stdout
    # Named, not acted on: disabling a unit on a live box is the operator's call.
    assert "disable" not in log
