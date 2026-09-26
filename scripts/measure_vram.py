#!/usr/bin/env python3
"""Measure what a served model actually holds, and write it into the registry.

`config/models.yaml` carried one comment for a year — *vram_mb values are rough
estimates* — and for most of that year it was harmless: the field fed the
resident-set budget, and a wrong number cost an unnecessary eviction. Since #127
it decides `--gpu-memory-utilization` for every vLLM launch, so the estimate is
load-bearing in both directions (#130):

- **too low** — vLLM starts, spends a minute loading the weights, and dies
  computing a KV cache. The message names neither the model nor the cause. That
  is the failure of 2026-09-14: a hand-set 0.35 came up 30 MiB short.
- **too high** — the launch is refused before it starts, or a resident model is
  evicted that did not need to be.

None of the values were ever measured. `qwen3vl-german-xix-v1` declares 12000
against 8.3 GB of merged bf16 weights on disk; the two 8B entries declare 18000;
every kraken entry declares 500 regardless of model.

## What this measures, and what it deliberately does not

The obvious method — `nvidia-smi --query-compute-apps=pid,used_memory` against
the vLLM pid — measures the wrong thing on its own. vLLM **claims**
`gpu_memory_utilization` of the card and fills whatever is left after the weights
with KV cache, so that number is the grant the launcher just computed. Measuring
it and feeding it back into `vram_mb` would close a circle around the launcher's
own arithmetic and look like a measurement while doing so.

The figure that belongs in `vram_mb` is the **weights**, which are fixed by the
checkpoint. vLLM reports them itself while profiling, together with the KV cache
it got for them. So this tool reads both:

- `nvidia-smi`, through the gateway's own `GET /gpu`, for the resident total —
  a cross-check on the grant, recorded but never written into `vram_mb`.
- the gateway's journal, for vLLM's memory-profiling line — the weights/KV split.

The ratio between them is what calibrates `manager.KV_HEADROOM`, the 1.6 that was
fitted to a single model. Note what the ratio is *not*: a successful launch's KV
cache is what was **left over**, an upper bound on what vLLM needed, not the need
itself. Only a failed launch states the need ("2.25 GiB KV cache is needed"). A
model whose measured KV cache is thin is worth re-running at a lower utilisation
to find its floor; this tool will not pretend to have found it.

## Running it

On the serving box, against the running gateway, one model at a time (they are
`lazy` and LRU-evicted, so a second would displace the first):

    python scripts/measure_vram.py --model qwen3vl-german-xix-v1
    python scripts/measure_vram.py --all --write     # every vllm entry, in turn
    python scripts/measure_vram.py --report          # what is measured so far
    python scripts/measure_vram.py --check           # CI gate, reads nothing live

`--write` patches `config/models.yaml` in place, textually, so the file keeps its
comments. `--check` needs no GPU and no gateway: it re-reads the measurements
already in the registry and fails when one of them no longer matches the
`vram_mb` beside it — the drift that would otherwise be discovered by a launch.

## A caveat worth reading before trusting the split

The journal patterns below were written from vLLM's source and from the failure
messages this repo has recorded (docs/GERMAN_XIX_MODELS.md), **not** verified
against a live run of the vLLM versions in `.venvs/`. If a pattern does not
match, the tool records `weights_mib: null` and says so rather than guessing a
split. A number invented here would be worse than the estimate it replaces,
because it would carry the word "measured".
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "config" / "models.yaml"

#: The unit whose journal carries vLLM's stdout. `manager._launch` runs
#: `subprocess.Popen(cmd, env=env)` without capturing, so the child's output is
#: inherited by the gateway and lands in the gateway's journal.
GATEWAY_UNIT = "atr-gateway.service"

#: Seconds to wait for a lazy model to become resident after the warm request.
WARM_TIMEOUT_S = 300

#: `manager.KV_HEADROOM`, restated rather than imported: this script runs from a
#: checkout without the package installed, and the number is what is being
#: judged, so it should be visible in the file that judges it.
KV_HEADROOM = 1.6

#: How far `vram_mb` may sit from the measured weights before `--check` fails.
#: Mirrors `manager.VRAM_DRIFT_TOLERANCE_MB`.
DRIFT_TOLERANCE_MB = 256


# ── parsing vLLM's own account of its memory ────────────────────────────────
GIB = 1024

#: vLLM >= 0.9 logs one sentence with the whole split. The wording has been
#: stable across 0.9–0.11; the numbers are GiB.
_SENTENCE = re.compile(
    r"total_gpu_memory\s*\(([\d.]+)GiB\)\s*x\s*gpu_memory_utilization\s*\(([\d.]+)\)"
    r".*?model weights take\s*([\d.]+)GiB"
    r".*?KV Cache is\s*([\d.]+)GiB",
    re.IGNORECASE | re.DOTALL,
)

#: The key=value form of the same thing, emitted by the v1 engine's profiler.
_KV_PAIRS = re.compile(
    r"total_gpu_memory=([\d.]+)GiB.*?"
    r"kv_cache_size=([\d.]+)GiB.*?"
    r"gpu_memory_utilization=([\d.]+)",
    re.IGNORECASE | re.DOTALL,
)

#: Two separate lines, the oldest form still worth reading.
_WEIGHTS_ONLY = re.compile(r"Model loading took\s*([\d.]+)\s*GiB", re.IGNORECASE)
_KV_ONLY = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB", re.IGNORECASE)

_MAX_MODEL_LEN = re.compile(r"--max-model-len[=\s]+(\d+)")


@dataclass
class Profile:
    """The split vLLM reported, in MiB. Any field may be None."""

    weights_mib: int | None = None
    kv_cache_mib: int | None = None
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    #: Which pattern matched, so a surprising number can be traced to its line.
    source: str | None = None


def _mib(gib: str) -> int:
    return int(round(float(gib) * GIB))


def parse_profile(log: str) -> Profile:
    """The weights/KV split from a stretch of gateway journal, if it is in there.

    Returns an empty :class:`Profile` rather than raising: a vLLM whose wording
    we do not know is a gap in this tool, not a failed measurement of the card,
    and the caller still has the `nvidia-smi` total to record.
    """
    profile = Profile()
    if match := _MAX_MODEL_LEN.search(log):
        profile.max_model_len = int(match.group(1))

    if match := _SENTENCE.search(log):
        profile.gpu_memory_utilization = float(match.group(2))
        profile.weights_mib = _mib(match.group(3))
        profile.kv_cache_mib = _mib(match.group(4))
        profile.source = "sentence"
        return profile

    if match := _KV_PAIRS.search(log):
        profile.kv_cache_mib = _mib(match.group(2))
        profile.gpu_memory_utilization = float(match.group(3))
        profile.source = "key=value"
        # The pair form does not carry the weights; the separate line may.
        if weights := _WEIGHTS_ONLY.search(log):
            profile.weights_mib = _mib(weights.group(1))
        return profile

    weights, kv = _WEIGHTS_ONLY.search(log), _KV_ONLY.search(log)
    if weights or kv:
        profile.weights_mib = _mib(weights.group(1)) if weights else None
        profile.kv_cache_mib = _mib(kv.group(1)) if kv else None
        profile.source = "separate lines"
    return profile


# ── talking to the running gateway ──────────────────────────────────────────
def _client(gateway: str, api_key: str | None):
    import httpx

    headers = {"X-API-Key": api_key} if api_key else {}
    return httpx.Client(base_url=gateway, headers=headers, timeout=WARM_TIMEOUT_S)


def warm(client, model_id: str) -> None:
    """Make the model resident by asking it to read one small image.

    A recognition rather than a bare load: the KV cache is allocated when the
    engine starts, but CUDA graphs are captured on the first real forward pass,
    and those are inside the same allocation `vram_mb` is meant to cover.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (512, 64), "white").save(buf, format="PNG")
    buf.seek(0)
    response = client.post(
        "/recognize",
        files={"file": ("warm.png", buf, "image/png")},
        data={"model": model_id},
    )
    if response.status_code >= 400:
        raise SystemExit(
            f"warm request for {model_id} failed: {response.status_code} "
            f"{response.text[:400]}"
        )


def resident_total_mib(client) -> tuple[int, int | None, list[int]]:
    """``(MiB held by this gateway's vLLM children, gpu index, pids)``.

    From `GET /gpu`, which already separates the gateway's own vLLM children from
    the engines on the same card — the distinction this measurement stands or
    falls on, and one this script must not reimplement.
    """
    payload = client.get("/gpu").json()
    vllm = payload.get("vllm", {})
    pids = set(vllm.get("pids") or [])
    gpu = vllm.get("gpu")
    total = 0
    for card in payload.get("cards", []):
        if gpu is not None and card.get("index") != gpu:
            continue
        for process in card.get("processes", []):
            if process.get("pid") in pids:
                total += int(process.get("used_mib") or 0)
    return total, gpu, sorted(pids)


def journal_since(when: datetime, unit: str = GATEWAY_UNIT) -> str:
    """The gateway's journal from ``when``, or '' when journalctl cannot say."""
    stamp = when.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    try:
        out = subprocess.run(  # noqa: S603
            ["journalctl", "-u", unit, "--since", stamp, "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=30, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        print(f"  journal unreadable ({type(exc).__name__}: {exc}); "
              "the weights/KV split will be missing", file=sys.stderr)
        return ""
    return out.stdout


# ── the registry ────────────────────────────────────────────────────────────
def load_registry() -> list[dict]:
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["models"]


def write_measurement(text: str, model_id: str, measurement: dict) -> str:
    """Put ``vram_measured`` under ``model_id`` in the raw YAML, comments intact.

    Textual rather than a load/dump round trip. `config/models.yaml` is as much
    commentary as data — every non-obvious field carries the incident that put it
    there — and PyYAML would drop all of it. Twelve entries do not justify a new
    dependency, and a rewrite that silently deletes the file's reasoning is
    exactly the kind of quiet loss this issue is about.
    """
    lines = text.splitlines(keepends=True)
    start = _entry_start(lines, model_id)
    if start is None:
        raise KeyError(f"no entry '- id: {model_id}' in {REGISTRY.name}")
    end = _entry_end(lines, start)

    indent = _field_indent(lines, start, end)
    block = [f"{indent}vram_measured:\n"]
    for key, value in measurement.items():
        if value is None:
            continue
        block.append(f"{indent}  {key}: {_scalar(value)}\n")

    # Drop an earlier measurement of this model before inserting the new one, so
    # a re-run replaces rather than stacks.
    body = list(range(start + 1, end))
    if span := _measured_span(lines, start, end):
        body = [i for i in body if not span[0] <= i < span[1]]

    kept = [lines[i] for i in body]
    anchor = _insert_after(kept, indent)
    return "".join(lines[:start + 1] + kept[:anchor] + block + kept[anchor:] + lines[end:])


def _entry_start(lines: list[str], model_id: str) -> int | None:
    for i, line in enumerate(lines):
        if re.match(rf"\s*-\s+id:\s*{re.escape(model_id)}\s*$", line.rstrip()):
            return i
    return None


def _entry_end(lines: list[str], start: int) -> int:
    for i in range(start + 1, len(lines)):
        if re.match(r"\s*-\s+id:\s", lines[i]):
            return i
    return len(lines)


def _field_indent(lines: list[str], start: int, end: int) -> str:
    for line in lines[start + 1:end]:
        if line.strip() and not line.lstrip().startswith("#"):
            return line[: len(line) - len(line.lstrip())]
    # `- id: x` alone: fields sit two past the dash.
    dash = lines[start].index("-")
    return " " * (dash + 2)


def _measured_span(lines: list[str], start: int, end: int) -> tuple[int, int] | None:
    for i in range(start + 1, end):
        if re.match(r"\s*vram_measured:\s*$", lines[i]):
            indent = len(lines[i]) - len(lines[i].lstrip())
            for j in range(i + 1, end):
                stripped = lines[j].strip()
                if stripped and len(lines[j]) - len(lines[j].lstrip()) <= indent:
                    return (i, j)
            return (i, end)
    return None


def _insert_after(kept: list[str], indent: str) -> int:
    """Index to insert at: right after ``vram_mb``, else at the end of the entry."""
    for i, line in enumerate(kept):
        if re.match(rf"{re.escape(indent)}vram_mb:\s", line):
            return i + 1
    trailing = len(kept)
    while trailing > 0 and not kept[trailing - 1].strip():
        trailing -= 1
    return trailing


def _scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    return text if re.fullmatch(r"[A-Za-z0-9_.:+\-]+", text) else json.dumps(text)


# ── the three modes ─────────────────────────────────────────────────────────
def measure(model_id: str, gateway: str, api_key: str | None, gpu_hint: int | None) -> dict:
    started = datetime.now(timezone.utc)
    with _client(gateway, api_key) as client:
        print(f"  warming {model_id} …")
        warm(client, model_id)
        # The engine reports its profile while starting; the journal needs a
        # moment to have it before we read back.
        time.sleep(2)
        total, gpu, pids = resident_total_mib(client)

    profile = parse_profile(journal_since(started))
    if profile.source is None:
        print("  no vLLM memory-profiling line in the journal — "
              "recording the total only, weights/KV left null", file=sys.stderr)

    return {
        "total_mib": total,
        "weights_mib": profile.weights_mib,
        "kv_cache_mib": profile.kv_cache_mib,
        "max_model_len": profile.max_model_len,
        "gpu_memory_utilization": profile.gpu_memory_utilization,
        "host": socket.gethostname(),
        "measured_at": started.date().isoformat(),
        "gpu": gpu if gpu is not None else gpu_hint,
        "note": None if profile.source else "no profiling line in the journal",
        "_pids": pids,
    }


def report(models: list[dict]) -> None:
    print(f"{'model':<38} {'vram_mb':>8} {'weights':>8} {'kv':>8} {'ratio':>6}  source")
    for spec in models:
        measured = spec.get("vram_measured")
        if not measured:
            print(f"{spec['id']:<38} {spec.get('vram_mb', 0):>8} "
                  f"{'—':>8} {'—':>8} {'—':>6}  estimate, never measured")
            continue
        weights = measured.get("weights_mib")
        kv = measured.get("kv_cache_mib")
        ratio = f"{(weights + kv) / weights:.2f}" if weights and kv else "—"
        print(f"{spec['id']:<38} {spec.get('vram_mb', 0):>8} "
              f"{weights or '—':>8} {kv or '—':>8} {ratio:>6}  "
              f"{measured.get('measured_at')} {measured.get('host')}")
    print(f"\nKV_HEADROOM in manager.py is {KV_HEADROOM}. The ratio column is "
          "(weights + KV) / weights as it came out on a successful launch — an\n"
          "upper bound on what the model needed, not the need. A ratio well under "
          f"{KV_HEADROOM} means the launch was granted more than it used;\na ratio "
          "at or above it means the multiplier has no margin left for that model.")


def check(models: list[dict]) -> int:
    """Fail when a measured entry and its ``vram_mb`` have drifted apart.

    Deliberately silent about unmeasured entries. Failing on those would mean
    failing every build until somebody gets to the GPU box, and a gate that is
    red for a reason nobody can act on today stops being read — which is how the
    blanket "rough estimates" comment survived a year.
    """
    problems: list[str] = []
    measured_count = 0
    for spec in models:
        measured = spec.get("vram_measured")
        if not measured:
            continue
        measured_count += 1
        weights = measured.get("weights_mib")
        declared = spec.get("vram_mb", 0)
        if weights is None:
            continue
        drift = weights - declared
        if abs(drift) > DRIFT_TOLERANCE_MB:
            direction = ("under-declared — every launch sizes this model from a "
                         "number smaller than its weights"
                         if drift > 0 else "over-declared — launches are refused "
                         "and residents evicted for memory this model does not use")
            problems.append(
                f"  {spec['id']}: vram_mb {declared}, measured weights {weights} "
                f"({drift:+d} MiB) — {direction}")

    if problems:
        print("vram_mb no longer matches what was measured:")
        print("\n".join(problems))
        print("\nRe-run `python scripts/measure_vram.py --model <id>` on the "
              "serving box, or correct vram_mb to the measured weights.")
        return 1
    print(f"{measured_count} measured entr{'y' if measured_count == 1 else 'ies'}, "
          f"all within {DRIFT_TOLERANCE_MB} MiB of their vram_mb "
          f"({len(models) - measured_count} still estimated)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="one registry id")
    parser.add_argument("--all", action="store_true",
                        help="every vllm entry, one after another")
    parser.add_argument("--write", action="store_true",
                        help="patch config/models.yaml with the result")
    parser.add_argument("--report", action="store_true",
                        help="what is measured so far; reads nothing live")
    parser.add_argument("--check", action="store_true",
                        help="CI gate: measured entries still match their vram_mb")
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--gpu", type=int, default=None,
                        help="card index to record when /gpu does not name one")
    args = parser.parse_args()

    models = load_registry()
    if args.check:
        return check(models)
    if args.report:
        report(models)
        return 0

    if args.all:
        targets = [m["id"] for m in models if m.get("engine") == "vllm"]
    elif args.model:
        targets = [args.model]
    else:
        parser.error("pass --model, --all, --report or --check")
        return 2

    results = {}
    for model_id in targets:
        print(f"{model_id}:")
        result = measure(model_id, args.gateway, args.api_key, args.gpu)
        pids = result.pop("_pids")
        results[model_id] = result
        print(f"  resident total {result['total_mib']} MiB across pids {pids}")
        if result["weights_mib"]:
            print(f"  weights {result['weights_mib']} MiB, "
                  f"KV cache {result['kv_cache_mib']} MiB")

        if args.write:
            text = REGISTRY.read_text(encoding="utf-8")
            REGISTRY.write_text(
                write_measurement(text, model_id, result), encoding="utf-8")
            print(f"  written into {REGISTRY.name}")

    if not args.write:
        print("\n" + json.dumps(results, indent=2))
        print("\nNothing written. Re-run with --write to put these in the registry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
