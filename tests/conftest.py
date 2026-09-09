"""Test-wide safety rails.

The one thing in here exists because the artefact cache (#109) defaults to a
directory under ``$HOME``, and a pipeline test that runs the whole lifecycle
will happily write there. A test suite that leaves 40 GB — or, as it did the
first time, five stray directories — in the developer's home is a bug in the
suite, not a quirk of it.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _artefact_cache_never_touches_home(tmp_path_factory, monkeypatch):
    """Point every ``TrainerSettings`` in the suite at a throwaway cache root.

    Set through the environment rather than the fixtures, so it holds for the
    settings objects constructed inside the code under test as well as the ones
    the tests build themselves.
    """
    root = tmp_path_factory.mktemp("artefact-cache")
    monkeypatch.setenv("ATR_TRAIN_ARTEFACT_CACHE_ROOT", str(root))
