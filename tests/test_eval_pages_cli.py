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
