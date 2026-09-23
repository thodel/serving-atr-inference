"""#155: an upload that uploads nothing must not exit 0.

On 20.09., publishing the v1 cards from UBELIX, `/rs_scratch` was not bound in
the container. `scan_trained("/scratch/.../runs/trained")` returned an empty
scan — no models, no skips, no hint that the path did not exist — and the
script printed "trained models under …: 0" and exited 0. It surfaced only
because a later line indexed into the empty list.

Three modes, three answers: `--list` on a box that never trained says so and
exits 0; a publish with nothing to publish is an error; a `--trained-root` the
caller named and that does not exist is an error in either mode.

Offline: no hub, no network — every case stops before an uploader is built.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_to_hub.py"


def _main():
    spec = importlib.util.spec_from_file_location("publish_to_hub_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


@pytest.fixture
def main():
    return _main()


def test_a_named_trained_root_that_does_not_exist_is_an_error(main, capsys, tmp_path):
    assert main(["--trained-root", str(tmp_path / "not-bound")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_that_holds_for_list_too(main, capsys, tmp_path):
    """The mount was missing in a `--list` run as well, and it read as empty."""
    assert main(["--trained-root", str(tmp_path / "not-bound"), "--list"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_publishing_from_an_empty_root_is_an_error_naming_the_path(main, capsys, tmp_path):
    empty = tmp_path / "trained"
    empty.mkdir()

    assert main(["--trained-root", str(empty)]) == 2

    err = capsys.readouterr().err
    assert "nothing to publish" in err
    assert str(empty) in err, "the message must name the path it looked in"


def test_listing_an_empty_root_is_not_an_error(main, capsys, tmp_path):
    """A box that has never finished a training run is a normal state, and
    `--list` is the mode that may answer 'nothing here'."""
    empty = tmp_path / "trained"
    empty.mkdir()

    assert main(["--trained-root", str(empty), "--list"]) == 0

    assert "no trained models on this box yet" in capsys.readouterr().out


def test_listing_a_missing_default_root_is_not_an_error(main, capsys, tmp_path, monkeypatch):
    """Same, for the default root: no --trained-root given, nothing trained yet."""
    monkeypatch.setenv("ATR_TRAIN_TRAINED_ROOT", str(tmp_path / "never-trained"))

    assert main(["--list"]) == 0

    assert "no trained models on this box yet" in capsys.readouterr().out
