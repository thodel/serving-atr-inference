"""The venv version check (#53).

`scripts/check_requirements.py` is what makes `check_venvs.sh` able to fail on the
transformers 5.x incident. Imports could not: `import transformers` worked on 5.14.1,
and so did `TrainingArguments(...)`.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.check_requirements import check, main, requirements_in  # noqa: E402

VLM_REQUIREMENTS = """\
# a comment
--index-url https://download.pytorch.org/whl/cu128

transformers>=4.57,<5
torch==2.8.0
uvicorn[standard]>=0.29
wandb
"""


@pytest.fixture
def reqs(tmp_path: Path) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text(VLM_REQUIREMENTS, encoding="utf-8")
    return path


def installed(mapping: dict[str, str]):
    """Stand in for importlib.metadata.version."""
    from importlib.metadata import PackageNotFoundError

    def lookup(name: str) -> str:
        try:
            return mapping[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    return lookup


# ── parsing ─────────────────────────────────────────────────────────────────
def test_comments_and_pip_directives_are_not_requirements(reqs):
    names = [r.name for r in requirements_in(reqs)]
    assert names == ["transformers", "torch", "uvicorn", "wandb"]


def test_an_unparsable_line_warns_but_does_not_abort(tmp_path, capsys):
    path = tmp_path / "r.txt"
    path.write_text("transformers>=4.57\n===nonsense===\ntorch==2.8.0\n", encoding="utf-8")
    assert [r.name for r in requirements_in(path)] == ["transformers", "torch"]
    assert "cannot parse" in capsys.readouterr().out


# ── the incident this exists for ────────────────────────────────────────────
def test_transformers_5x_is_caught(reqs, monkeypatch, capsys):
    """The whole point: 5.14.1 imports fine and constructs TrainingArguments fine."""
    monkeypatch.setattr("scripts.check_requirements.installed_version",
                        installed({"transformers": "5.14.1", "torch": "2.8.0",
                                   "uvicorn": "0.29.0", "wandb": "0.28.1"}))
    assert check(reqs) == 1
    out = capsys.readouterr().out
    assert "MISMATCH  transformers 5.14.1" in out
    assert "requires <5,>=4.57" in out or "requires >=4.57,<5" in out


def test_the_intended_version_passes(reqs, monkeypatch):
    monkeypatch.setattr("scripts.check_requirements.installed_version",
                        installed({"transformers": "4.57.6", "torch": "2.8.0",
                                   "uvicorn": "0.29.0", "wandb": "0.28.1"}))
    assert check(reqs) == 0


# ── PEP 440 details that would otherwise produce false alarms ───────────────
def test_a_local_version_tag_satisfies_an_exact_pin(reqs, monkeypatch):
    """torch reports 2.8.0+cu128 against a `torch==2.8.0` pin. That is compatible —
    a local identifier is allowed unless the specifier names one — and flagging it
    would make the check cry wolf on every GPU venv this repo builds."""
    monkeypatch.setattr("scripts.check_requirements.installed_version",
                        installed({"transformers": "4.57.6", "torch": "2.8.0+cu128",
                                   "uvicorn": "0.29.0", "wandb": "0.28.1"}))
    assert check(reqs) == 0


def test_a_different_torch_is_still_caught(reqs, monkeypatch, capsys):
    monkeypatch.setattr("scripts.check_requirements.installed_version",
                        installed({"transformers": "4.57.6", "torch": "2.10.0",
                                   "uvicorn": "0.29.0", "wandb": "0.28.1"}))
    assert check(reqs) == 1
    assert "MISMATCH  torch 2.10.0" in capsys.readouterr().out


def test_an_unpinned_requirement_cannot_be_violated(reqs, monkeypatch):
    """`wandb` has no specifier; any version satisfies it."""
    monkeypatch.setattr("scripts.check_requirements.installed_version",
                        installed({"transformers": "4.57.6", "torch": "2.8.0",
                                   "uvicorn": "0.29.0", "wandb": "0.1"}))
    assert check(reqs) == 0


def test_extras_do_not_confuse_the_lookup(reqs, monkeypatch):
    """`uvicorn[standard]>=0.29` is installed as `uvicorn`."""
    monkeypatch.setattr("scripts.check_requirements.installed_version",
                        installed({"transformers": "4.57.6", "torch": "2.8.0",
                                   "uvicorn": "0.52.1", "wandb": "0.28.1"}))
    assert check(reqs) == 0


# ── missing packages ────────────────────────────────────────────────────────
def test_a_missing_package_is_reported_not_skipped(reqs, monkeypatch, capsys):
    monkeypatch.setattr("scripts.check_requirements.installed_version",
                        installed({"transformers": "4.57.6", "torch": "2.8.0"}))
    assert check(reqs) == 2
    out = capsys.readouterr().out
    assert "MISSING   uvicorn" in out and "MISSING   wandb" in out


# ── the CLI contract check_venvs.sh depends on ──────────────────────────────
def test_exit_code_is_zero_only_when_everything_is_satisfied(reqs, monkeypatch):
    monkeypatch.setattr("scripts.check_requirements.installed_version",
                        installed({"transformers": "4.57.6", "torch": "2.8.0",
                                   "uvicorn": "0.29.0", "wandb": "0.28.1"}))
    assert main([str(reqs)]) == 0
    monkeypatch.setattr("scripts.check_requirements.installed_version",
                        installed({"transformers": "5.14.1", "torch": "2.8.0",
                                   "uvicorn": "0.29.0", "wandb": "0.28.1"}))
    assert main([str(reqs)]) == 1


def test_a_missing_requirements_file_is_a_failure_not_a_pass(tmp_path, capsys):
    """Silently passing when the file is absent would make the whole check
    vacuous the moment a path is mistyped."""
    assert main([str(tmp_path / "nope.txt")]) == 1
    assert "requirements file not found" in capsys.readouterr().out
