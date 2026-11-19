"""scripts/eval_env.sh — the venv for one measurement in one call (#180).

The real uv, torch and vLLM are not installed here: a fake ``uv`` records every
call and lays down what the real one would (a CPython with ``Python.h``, a venv,
a ``vllm`` that answers ``--version``). What is pinned: the pins are the ones
make_venvs.sh uses, the headers reach ``CPATH`` in ``serve.sh``, a shim uv
leaves in ``~/.local/bin`` is removed and nothing else there, a second call
builds nothing, and a build whose vLLM does not start is not stamped.

The header half was also run for real on 19.11.2026 with uv 0.11.32: CPython
3.12.13 into ``--install-dir`` with ``--no-bin``, ``Python.h`` present, a venv on
it sees the same include dir, and ``gcc`` finds ``Python.h`` with ``CPATH`` and
fails without it — the asteraix failure, reproduced and fixed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

FAKE_UV = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_UV_LOG"
case "$1 $2" in
  "python install")
    ver=$3; dir=""
    shift 3
    while [ $# -gt 0 ]; do
      case "$1" in --install-dir) dir=$2; shift 2 ;; *) shift ;; esac
    done
    root="$dir/cpython-$ver.13-linux-x86_64-gnu"
    mkdir -p "$root/bin" "$root/include/python$ver"
    [ -n "${FAKE_UV_NOHEADERS:-}" ] || touch "$root/include/python$ver/Python.h"
    printf '#!/usr/bin/env bash\necho %s\n' "$root/include/python$ver" > "$root/bin/python$ver"
    chmod +x "$root/bin/python$ver"
    if [ -n "${FAKE_UV_SHIM:-}" ]; then
      mkdir -p "$HOME/.local/bin"; ln -sf "$root/bin/python$ver" "$HOME/.local/bin/python$ver"
    fi
    ;;
  "venv --python")
    target="${@: -1}"
    mkdir -p "$target/bin"; printf '#!/bin/sh\n' > "$target/bin/python"; chmod +x "$target/bin/python"
    ;;
  "pip install")
    if printf '%s\n' "$@" | grep -qx -- -r; then
      venv=$(dirname "$(dirname "$4")")
      cat > "$venv/bin/vllm" <<'EOS'
#!/usr/bin/env bash
if [ "${1:-}" = "--version" ]; then [ -z "${FAKE_VLLM_BROKEN:-}" ] || exit 1; echo 0.11.0; exit 0; fi
echo "ARGS $*"; echo "GPU=$CUDA_VISIBLE_DEVICES"; echo "CPATH=$CPATH"
EOS
      chmod +x "$venv/bin/vllm"
    fi
    ;;
esac
"""


@pytest.fixture
def repo(tmp_path):
    """The script in a repo of its own, so a test may change the pins."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "eval_env.sh", root / "scripts" / "eval_env.sh")
    for variant in ("vllm", "vllm-next"):
        (root / "engines" / variant).mkdir(parents=True)
        shutil.copy(REPO / "engines" / variant / "requirements.txt",
                    root / "engines" / variant / "requirements.txt")
    uv = tmp_path / "fake-uv"
    uv.write_text(FAKE_UV, encoding="utf-8")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "home").mkdir()
    return root


def _run(repo, *args, **env):
    tmp = repo.parent
    full = {"PATH": os.environ["PATH"], "HOME": str(tmp / "home"),
            "EVAL_ENV_UV": str(tmp / "fake-uv"), "FAKE_UV_LOG": str(tmp / "uv.log"), **env}
    return subprocess.run(["bash", str(repo / "scripts" / "eval_env.sh"), *args],
                          capture_output=True, text=True, env=full, timeout=60)


def _calls(repo):
    log = repo.parent / "uv.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


# ── a fresh build ────────────────────────────────────────────────────────────

def test_a_fresh_build_pins_like_make_venvs_and_stamps_last(repo, tmp_path):
    out = tmp_path / "env"
    r = _run(repo, str(out))
    assert r.returncode == 0, r.stderr
    calls = _calls(repo)
    assert calls[0].startswith("python install 3.12 --install-dir") and calls[0].endswith("--no-bin")
    assert calls[1].startswith("venv --python") and "--clear" in calls[1]
    assert "torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128" in calls[2]
    assert calls[3].endswith(f"-r {repo}/engines/vllm/requirements.txt")
    assert "--extra-index-url" not in calls[3]
    stamp = json.loads((out / "eval_env.json").read_text(encoding="utf-8"))
    assert stamp["variant"] == "0.11" and stamp["include"].endswith("include/python3.12")
    assert sorted(p.name for p in out.iterdir()) == ["eval_env.json", "serve.sh", "uvpy", "venv"]


def test_next_takes_the_cu129_builds_and_the_vllm_next_pins(repo, tmp_path):
    r = _run(repo, str(tmp_path / "env"), "--vllm", "next")
    assert r.returncode == 0, r.stderr
    calls = _calls(repo)
    assert "torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129" in calls[2]
    assert f"-r {repo}/engines/vllm-next/requirements.txt" in calls[3]
    assert calls[3].endswith(" --extra-index-url https://wheels.vllm.ai/0.29.0/cu129"
                             " --extra-index-url https://download.pytorch.org/whl/cu129")


def test_the_torch_pins_are_the_ones_make_venvs_uses():
    """Two scripts, one set of pins: a measurement on other torch than the
    gateway serves with measures something else."""
    mk = (REPO / "scripts" / "make_venvs.sh").read_text(encoding="utf-8")
    ev = (REPO / "scripts" / "eval_env.sh").read_text(encoding="utf-8")

    def block(name):
        return mk.split(f"if wanted {name}; then", 1)[1].split("\nfi\n", 1)[0]
    vllm_torch = re.search(r"install (torch==\S+) --index-url (\S+)", block("vllm"))
    next_torch = re.search(r"install (torch==\S+ torchvision==\S+) \\\s+--index-url (\S+)",
                           block("vllm-next"))
    assert f"TORCH=({vllm_torch[1]})" in ev and f"TORCH_INDEX={vllm_torch[2]}" in ev
    assert f"TORCH=({next_torch[1]})" in ev and f"TORCH_INDEX={next_torch[2]}" in ev
    assert "wheels.vllm.ai/0.29.0/cu129" in block("vllm-next") and "wheels.vllm.ai/0.29.0/cu129" in ev


# ── idempotent, and nothing mixed ────────────────────────────────────────────

def test_a_second_call_checks_and_builds_nothing(repo, tmp_path):
    out = tmp_path / "env"
    assert _run(repo, str(out)).returncode == 0
    n = len(_calls(repo))
    r = _run(repo, str(out))
    assert r.returncode == 0 and "nothing to do" in r.stdout
    assert len(_calls(repo)) == n


def test_a_directory_built_for_the_other_variant_is_refused(repo, tmp_path):
    out = tmp_path / "env"
    assert _run(repo, str(out)).returncode == 0
    n = len(_calls(repo))
    r = _run(repo, str(out), "--vllm", "next")
    assert r.returncode == 1 and "built for --vllm 0.11" in r.stderr
    assert len(_calls(repo)) == n


def test_changed_pins_are_not_called_built(repo, tmp_path):
    out = tmp_path / "env"
    assert _run(repo, str(out)).returncode == 0
    reqs = repo / "engines" / "vllm" / "requirements.txt"
    reqs.write_text(reqs.read_text(encoding="utf-8") + "pillow==11.0.0\n", encoding="utf-8")
    r = _run(repo, str(out))
    assert r.returncode == 1 and "other pins" in r.stderr and f"rm -rf {out}" in r.stderr


def test_a_stamped_directory_whose_vllm_is_gone_is_not_called_built(repo, tmp_path):
    out = tmp_path / "env"
    assert _run(repo, str(out)).returncode == 0
    (out / "venv" / "bin" / "vllm").unlink()
    r = _run(repo, str(out))
    assert r.returncode == 1 and "gone" in r.stderr


def test_a_stamped_directory_whose_headers_are_gone_is_not_called_built(repo, tmp_path):
    """The headers are what the directory is for: vLLM would start, load the
    weights, and die in Triton's gcc call."""
    out = tmp_path / "env"
    assert _run(repo, str(out)).returncode == 0
    include = json.loads((out / "eval_env.json").read_text(encoding="utf-8"))["include"]
    (Path(include) / "Python.h").unlink()
    r = _run(repo, str(out))
    assert r.returncode == 1 and "gone" in r.stderr


def test_a_directory_with_foreign_content_is_refused(repo, tmp_path):
    out = tmp_path / "env"
    out.mkdir()
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    r = _run(repo, str(out))
    assert r.returncode == 1 and "notes.txt" in r.stderr
    assert _calls(repo) == [] and sorted(p.name for p in out.iterdir()) == ["notes.txt"]


@pytest.mark.parametrize("args, code", [(["--vllm", "latest"], 2), ([], 2), (["a", "b"], 2)])
def test_bad_arguments(repo, args, code):
    r = _run(repo, *args)
    assert r.returncode == code and _calls(repo) == []


# ── a build that does not start is not built ─────────────────────────────────

def test_a_vllm_that_does_not_start_is_not_stamped_and_the_next_call_builds_again(repo, tmp_path):
    out = tmp_path / "env"
    r = _run(repo, str(out), FAKE_VLLM_BROKEN="1")
    assert r.returncode == 1 and "not stamping" in r.stderr
    assert not (out / "eval_env.json").exists()
    n = len(_calls(repo))
    assert _run(repo, str(out)).returncode == 0
    assert len(_calls(repo)) > n and (out / "eval_env.json").exists()


# ── the shim ─────────────────────────────────────────────────────────────────

def test_the_shim_uv_leaves_is_removed_and_nothing_else(repo, tmp_path):
    bin_ = tmp_path / "home" / ".local" / "bin"
    bin_.mkdir(parents=True)
    (bin_ / "python3").write_text("#!/bin/sh\n", encoding="utf-8")       # somebody's own
    (bin_ / "black").write_text("#!/bin/sh\n", encoding="utf-8")
    r = _run(repo, str(tmp_path / "env"), FAKE_UV_SHIM="1")
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in bin_.iterdir()) == ["black", "python3"]
    assert "removing the shim" in r.stdout


# ── serve.sh ─────────────────────────────────────────────────────────────────

def test_serve_sh_sets_cpath_the_card_and_loopback(repo, tmp_path):
    out = tmp_path / "env"
    assert _run(repo, str(out)).returncode == 0
    include = json.loads((out / "eval_env.json").read_text(encoding="utf-8"))["include"]
    env = {"PATH": os.environ["PATH"], "GPU": "2", "PORT": "8399", "CPATH": "/x"}
    r = subprocess.run([str(out / "serve.sh"), "/models/m", "--max-model-len", "8192"],
                       capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "ARGS serve /models/m --host 127.0.0.1 --port 8399 --max-model-len 8192" in r.stdout
    assert "GPU=2" in r.stdout
    assert f"CPATH={include}:/x" in r.stdout


@pytest.mark.parametrize("missing", ["GPU", "PORT"])
def test_serve_sh_refuses_without_a_claimed_card(repo, tmp_path, missing):
    out = tmp_path / "env"
    assert _run(repo, str(out)).returncode == 0
    env = {"PATH": os.environ["PATH"], "GPU": "2", "PORT": "8399"}
    del env[missing]
    r = subprocess.run([str(out / "serve.sh"), "/models/m"], capture_output=True, text=True,
                       env=env, timeout=30)
    assert r.returncode != 0 and f"{missing} is not set" in r.stderr


def test_a_python_without_headers_stops_before_the_venv(repo, tmp_path):
    """The whole point: Triton would fail at vLLM's start, minutes in."""
    r = _run(repo, str(tmp_path / "env"), FAKE_UV_NOHEADERS="1")
    assert r.returncode == 1 and "no Python.h" in r.stderr
    assert [c.split()[0] for c in _calls(repo)] == ["python"]
