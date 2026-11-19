"""scripts/measure_model.sh — one measurement is one command (#179).

Run as real bash with the real card lease, workspace and trap
(scripts/eval_cleanup.sh) against a fake ``nvidia-smi``. The four heavy steps
are stand-ins: an eval_env.sh whose serve.sh starts a tiny HTTP server, an
eval_pages.py and a merge_loras.py that lay down what the real ones would, and
an eval_granularity.py in a fake training repo that writes its arguments as
the report. Pinned: the order, the arguments each step gets, that the report
and the selection leave the workspace, that a refusal before the claim leaves
nothing, and that a failed step or a server that dies is kept and said.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FAKE_SMI = """#!/bin/bash
case "$1" in
  --query-gpu=uuid) echo "GPU-0";;
  --query-gpu=*) echo "0, NVIDIA A40, 46068, 100, 45968, 0, 0";;
  --query-compute-apps=*)
    [ -f "$FAKE_GPU_PIDS" ] && while read -r p; do
      kill -0 "$p" 2>/dev/null && echo "$p, 9000, GPU-0"
    done < "$FAKE_GPU_PIDS";;
esac
exit 0
"""

# serve.sh as eval_env.sh writes it, minus vLLM: a server that answers /health.
FAKE_EVAL_ENV = """#!/bin/bash
set -euo pipefail
echo "eval_env $*" >> "$STEPS"
mkdir -p "$1/venv/bin"
ln -sf "$PY" "$1/venv/bin/python"
cat > "$1/serve.sh" <<'EOS'
#!/bin/bash
: "${GPU:?GPU}"; : "${PORT:?PORT}"
echo "serve $* gpu=$GPU" >> "$STEPS"
exec "$PY" "$FAKE_SERVER" "$PORT"
EOS
chmod +x "$1/serve.sh"
"""

FAKE_SERVER = """import http.server, os, sys
with open(os.environ["FAKE_GPU_PIDS"], "a") as fh:
    fh.write(f"{os.getpid()}\\n")
if os.environ.get("FAKE_SERVER_DIES"):
    print("CUDA out of memory (fake)", file=sys.stderr)
    sys.exit(3)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404); self.end_headers()
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""

FAKE_PAGES = """import json, os, sys
open(os.environ["STEPS"], "a").write("eval_pages " + " ".join(sys.argv[1:]) + "\\n")
out = sys.argv[sys.argv.index("--out") + 1]
os.makedirs(out)
json.dump({"jsonl": "eval_pages.jsonl", "pages": ["a.xml"]}, open(os.path.join(out, "selection.json"), "w"))
"""

FAKE_MERGE = """import os, sys
open(os.environ["STEPS"], "a").write(f"merge {' '.join(sys.argv[1:])} python={sys.executable}\\n")
model = sys.argv[sys.argv.index("--only") + 1]
d = os.path.join(os.environ["ATR_VLLM_MERGED_DIR"], model)
os.makedirs(d)
open(os.path.join(d, "config.json"), "w").write("{}")
"""

FAKE_GRANULARITY = """import json, os, sys
open(os.environ["STEPS"], "a").write("granularity " + " ".join(sys.argv[1:]) + "\\n")
if os.environ.get("FAKE_GRAN_FAILS"):
    sys.exit(4)
out = sys.argv[sys.argv.index("--out") + 1]
json.dump({"argv": sys.argv[1:]}, open(out, "w"))
"""


def _setup(tmp_path):
    def put(path, text, mode=0o755):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        path.chmod(mode)
        return path
    put(tmp_path / "bin" / "nvidia-smi", FAKE_SMI)
    trepo = tmp_path / "trepo"
    put(trepo / "scripts" / "eval_granularity.py", FAKE_GRANULARITY)
    put(trepo / "src" / "atr_training" / "__init__.py", "")
    put(trepo / "src" / "atr_training" / "textmetrics.py", "def score_pairs(*a): pass\n")
    env = dict(
        os.environ,
        PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        HOME=str(tmp_path / "home"),
        PY=sys.executable, ATR_EVAL_PYTHON=sys.executable, ATR_TRAINING_PYTHON=sys.executable,
        ATR_TRAINING_REPO=str(trepo),
        ATR_EVAL_ROOT=str(tmp_path / "atr-eval"), ATR_EVAL_LEASE_DIR=str(tmp_path / "leases"),
        ATR_EVAL_TERM_GRACE_S="2", ATR_EVAL_CARD_WAIT_S="10",
        ATR_MEASURE_EVAL_ENV=str(put(tmp_path / "fake_eval_env.sh", FAKE_EVAL_ENV)),
        ATR_MEASURE_EVAL_PAGES=str(put(tmp_path / "fake_pages.py", FAKE_PAGES)),
        ATR_MEASURE_MERGE=str(put(tmp_path / "fake_merge.py", FAKE_MERGE)),
        ATR_MEASURE_POLL_S="0.2", ATR_MEASURE_START_S="30",
        ATR_MEASURE_REPORTS=str(tmp_path / "reports"),
        FAKE_SERVER=str(put(tmp_path / "fake_server.py", FAKE_SERVER)),
        FAKE_GPU_PIDS=str(tmp_path / "gpu-pids"), STEPS=str(tmp_path / "steps"),
    )
    (tmp_path / "home").mkdir()
    (tmp_path / "gpu-pids").write_text("")
    return env


def _run(tmp_path, *args, **extra):
    env = _setup(tmp_path)
    env.update(extra)
    for k in [k for k, v in env.items() if v is None]:
        del env[k]
    done = subprocess.run(["bash", str(ROOT / "scripts" / "measure_model.sh"), *args],
                          env=env, capture_output=True, text=True, timeout=120)
    steps = (tmp_path / "steps").read_text().splitlines() if (tmp_path / "steps").exists() else []
    return done, steps


def _workspaces(tmp_path):
    root = tmp_path / "atr-eval"
    return list(root.iterdir()) if root.exists() else []


def _server_pids_alive(tmp_path):
    alive = []
    for p in (tmp_path / "gpu-pids").read_text().split():
        try:
            os.kill(int(p), 0)
            stat = Path(f"/proc/{p}/stat").read_text()
            if stat.rsplit(")", 1)[1].split()[0] != "Z":
                alive.append(p)
        except (ProcessLookupError, OSError):
            pass
    return alive


def test_a_measurement_runs_every_step_in_order_and_leaves_only_the_report(tmp_path):
    done, steps = _run(tmp_path, "m-1", "--n", "5", "--seed", "7", "--", "--max-pages", "3")
    assert done.returncode == 0, done.stderr
    assert [s.split()[0] for s in steps] == ["eval_env", "eval_pages", "merge", "serve", "granularity"]
    env_step, pages, merge, serve, gran = steps
    assert env_step.endswith("/env --vllm 0.11")
    assert pages.startswith("eval_pages m-1 --n 5 --seed 7 --out ") and pages.endswith("/pages")
    assert merge.startswith("merge --only m-1 python=") and merge.endswith("/env/venv/bin/python")
    assert "/merged/m-1 --served-model-name m-1 --gpu-memory-utilization 0.85" in serve
    reports = list((tmp_path / "reports").iterdir())
    report = next(r for r in reports if r.name.endswith("-m-1.json"))
    argv = json.loads(report.read_text())["argv"]
    assert argv[argv.index("--jsonl") + 1] == "eval_pages.jsonl"
    assert argv[argv.index("--model") + 1] == "m-1"
    assert argv[argv.index("--base-url") + 1].startswith("http://127.0.0.1:")
    assert "--no-auth" in argv and argv[-2:] == ["--max-pages", "3"]
    assert json.loads(Path(f"{report}.selection.json").read_text())["jsonl"] == "eval_pages.jsonl"
    assert _workspaces(tmp_path) == [], "the workspace is deleted after a success"
    assert _server_pids_alive(tmp_path) == []
    assert not list((tmp_path / "leases").glob("gpu*.lock"))


def test_the_next_variant_and_a_given_model_skip_what_they_replace(tmp_path):
    archive = tmp_path / "prepared"
    archive.mkdir()
    (archive / "selection.json").write_text(json.dumps({"jsonl": "held.jsonl"}))
    done, steps = _run(tmp_path, "m-2", "--vllm", "next", "--pages", str(archive),
                       "--serve", "org/m-2")
    assert done.returncode == 0, done.stderr
    assert [s.split()[0] for s in steps] == ["eval_env", "serve", "granularity"]
    assert steps[0].endswith("--vllm next")
    assert steps[1].startswith("serve org/m-2 --served-model-name m-2")
    assert f"--root {archive} --jsonl held.jsonl" in steps[2]


@pytest.mark.parametrize("change, message", [
    ({"ATR_TRAINING_REPO": None}, "ATR_TRAINING_REPO is not set"),
    ({"ATR_TRAINING_REPO": "/nonexistent"}, "no /nonexistent/scripts/eval_granularity.py"),
    ({"ATR_TRAINING_PYTHON": "/bin/false"}, "cannot run eval_granularity.py"),
])
def test_the_measuring_side_is_refused_before_a_card_is_claimed(tmp_path, change, message):
    done, steps = _run(tmp_path, "m-3", **change)
    assert done.returncode == 1 and message in done.stderr
    assert steps == [] and _workspaces(tmp_path) == []
    assert not (tmp_path / "leases").exists() or not list((tmp_path / "leases").iterdir())


def test_an_existing_report_is_not_overwritten(tmp_path):
    (tmp_path / "r.json").write_text("{}")
    done, steps = _run(tmp_path, "m-4", "--report", str(tmp_path / "r.json"))
    assert done.returncode == 1 and "does not overwrite" in done.stderr
    assert steps == [] and _workspaces(tmp_path) == []


def test_a_directory_without_selection_is_not_an_archive(tmp_path):
    (tmp_path / "pages").mkdir()
    done, steps = _run(tmp_path, "m-5", "--pages", str(tmp_path / "pages"))
    assert done.returncode == 1 and "without selection.json" in done.stderr
    assert steps == [] and _workspaces(tmp_path) == []


def test_a_failed_measurement_keeps_the_workspace_and_stops_the_server(tmp_path):
    done, steps = _run(tmp_path, "m-6", FAKE_GRAN_FAILS="1")
    assert done.returncode == 4
    assert steps[-1].startswith("granularity")
    kept = _workspaces(tmp_path)
    assert len(kept) == 1 and (kept[0] / "server.log").exists()
    assert "KEPT" in done.stderr
    assert _server_pids_alive(tmp_path) == []
    assert not list((tmp_path / "reports").glob("*.json"))


def test_a_server_that_dies_is_said_with_its_log(tmp_path):
    done, steps = _run(tmp_path, "m-7", FAKE_SERVER_DIES="1")
    assert done.returncode == 1
    assert "exited before it answered /health" in done.stderr
    assert "CUDA out of memory (fake)" in done.stderr
    assert not any(s.startswith("granularity") for s in steps)
    assert len(_workspaces(tmp_path)) == 1


def test_a_merge_that_produces_nothing_names_the_way_out(tmp_path):
    fake = tmp_path / "nomerge.py"
    fake.write_text("import sys\n")
    done, steps = _run(tmp_path, "m-8", ATR_MEASURE_MERGE=str(fake))
    assert done.returncode == 1 and "pass --serve" in done.stderr
    assert not any(s.startswith("serve") for s in steps)


def test_the_merge_python_can_be_the_training_venv(tmp_path):
    done, steps = _run(tmp_path, "m-9", ATR_MERGE_PYTHON=sys.executable)
    assert done.returncode == 0, done.stderr
    merge = next(s for s in steps if s.startswith("merge"))
    assert merge.endswith(f"python={sys.executable}")


@pytest.mark.parametrize("args", [[], ["a", "b"], ["m", "--vllm", "latest"], ["m", "--bogus"]])
def test_bad_arguments_claim_nothing(tmp_path, args):
    done, steps = _run(tmp_path, *args)
    assert done.returncode == 2 and steps == [] and _workspaces(tmp_path) == []


def test_the_docs_skeleton_names_the_command():
    text = (ROOT / "docs" / "MEASURING.md").read_text(encoding="utf-8")
    assert "scripts/measure_model.sh" in text


def test_keep_keeps_a_successful_workspace_but_never_the_server(tmp_path):
    done, _ = _run(tmp_path, "m-10", "--keep")
    assert done.returncode == 0, done.stderr
    assert len(_workspaces(tmp_path)) == 1
    assert _server_pids_alive(tmp_path) == []
