"""The backend registry, and the capability it declares on each runner's behalf (#85).

`Backend.supports_chunked_prepare` mirrors a ClassVar on the runner class. The
supervising service cannot read that ClassVar — each runner lives in its own venv
and this service imports neither — so it is declared here and pinned there.

Two sources of truth is a smell, and this file is the price of it. Without the
declaration the size guard read ATR_TRAIN_CHUNK_PAGES alone and cleared a 293 GB
vllm corpus on the strength of a setting the VLM backend ignores.
"""

import re
from pathlib import Path

import pytest

from atr_serving.training.backends import BACKENDS, UnknownBackend, backend_for

ROOT = Path(__file__).resolve().parents[1]

#: Where each backend's runner class is defined, for the drift check below.
RUNNERS = {
    "kraken": ROOT / "engines" / "kraken_train_svc" / "runner.py",
    "vllm": ROOT / "engines" / "vlm_train_svc" / "runner.py",
    "trocr": ROOT / "engines" / "trocr_train_svc" / "runner.py",
}


def declared_in_source(path: Path) -> bool:
    """Read `supports_chunked_prepare = True` out of the runner without importing it.

    Importing would need that backend's venv — torch, kraken, transformers — which
    the test suite deliberately does not have.
    """
    if not path.is_file():
        pytest.skip(f"{path.name} not present")
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^\s*supports_chunked_prepare\s*=\s*(True|False)",
                      text, re.MULTILINE)
    return match.group(1) == "True" if match else False


@pytest.mark.parametrize("engine", sorted(RUNNERS))
def test_the_declared_capability_matches_the_runner(engine):
    """The one assertion this file exists for."""
    assert BACKENDS[engine].supports_chunked_prepare == declared_in_source(RUNNERS[engine])


def test_only_kraken_chunks_today():
    """A guard rail on the guard: if this changes, the size guard's behaviour changes."""
    chunking = {e for e, b in BACKENDS.items() if b.supports_chunked_prepare}
    assert chunking == {"kraken"}


def test_every_backend_names_a_venv_and_a_requirements_file():
    for engine, backend in BACKENDS.items():
        assert backend.engine == engine
        assert backend.venv and backend.requirements.endswith("requirements.txt")


def test_an_unknown_engine_lists_the_ones_that_exist():
    with pytest.raises(UnknownBackend, match="Trainable:"):
        backend_for("nonesuch")
