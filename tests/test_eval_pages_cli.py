"""The eval_pages command itself (#181, part 3)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from test_eval_pages import MODEL, make_corpus

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "scripts" / "eval_pages.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("eval_pages_cli", CLI_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_cli(ep_cli, argv):
    old = sys.argv[1:]
    sys.argv[1:] = argv
    try:
        return ep_cli.main()
    finally:
        sys.argv[1:] = old


def test_cli_runs_end_to_end(tmp_path, capsys):
    ep_cli = _load_cli()
    trained, jobs, artefacts, _ = make_corpus(tmp_path)
    out = tmp_path / "cli-out"
    code = _run_cli(ep_cli, [MODEL, "--n", "3", "--out", str(out), "--seed", "5",
                             "--trained-root", str(trained), "--jobs-root", str(jobs),
                             "--artefacts-root", str(artefacts)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "selection.json" in printed
    assert "--root" in printed
    assert (out / "selection.json").is_file()


def test_cli_reports_the_error(tmp_path, capsys):
    ep_cli = _load_cli()
    trained, jobs, artefacts, _ = make_corpus(tmp_path)
    code = _run_cli(ep_cli, ["no-such-model", "--out", str(tmp_path / "o"),
                             "--trained-root", str(trained), "--jobs-root", str(jobs),
                             "--artefacts-root", str(artefacts)])
    assert code == 1
    assert "does not exist" in capsys.readouterr().err


def test_cli_passes_seed_and_jsonl_and_prints_the_follow_up_command(tmp_path, capsys):
    """The flags reach the archive, and the printed line is what eval_granularity takes."""
    import json

    ep_cli = _load_cli()
    trained, jobs, artefacts, _ = make_corpus(tmp_path)
    out = tmp_path / "cli-out"
    code = _run_cli(ep_cli, [MODEL, "--n", "2", "--out", str(out), "--seed", "7",
                             "--jsonl", "held.jsonl",
                             "--trained-root", str(trained), "--jobs-root", str(jobs),
                             "--artefacts-root", str(artefacts)])
    assert code == 0
    record = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert record["seed"] == 7 and record["n"] == 2 and record["jsonl"] == "held.jsonl"
    assert (out / "data" / "held.jsonl").is_file()
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].startswith(f"{record['lines']} lines from {len(record['pages'])} pages")
    assert printed[-1] == f"eval_granularity: --root {out} --jsonl held.jsonl"


def test_cli_refuses_a_directory_that_is_not_empty(tmp_path, capsys):
    ep_cli = _load_cli()
    trained, jobs, artefacts, _ = make_corpus(tmp_path)
    out = tmp_path / "busy"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    code = _run_cli(ep_cli, [MODEL, "--n", "2", "--out", str(out),
                             "--trained-root", str(trained), "--jobs-root", str(jobs),
                             "--artefacts-root", str(artefacts)])
    assert code == 1
    assert "not empty" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]


def test_cli_refuses_more_pages_than_were_held_out(tmp_path, capsys):
    """The default n is 15; the fixture holds out 12. A trained page would be
    measured otherwise — the refusal is the point of #181, and the command
    must surface it as an error, not write a smaller archive."""
    ep_cli = _load_cli()
    trained, jobs, artefacts, _ = make_corpus(tmp_path)
    out = tmp_path / "o"
    code = _run_cli(ep_cli, [MODEL, "--out", str(out),
                             "--trained-root", str(trained), "--jobs-root", str(jobs),
                             "--artefacts-root", str(artefacts)])
    assert code == 1
    assert "would test memory" in capsys.readouterr().err
    assert not (out / "selection.json").exists()
