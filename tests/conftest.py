"""Test-wide safety rails.

Both exist because a setting that points at a real place makes the suite write
there.

The artefact cache (#109) defaults to a directory under ``$HOME``, and a
pipeline test that runs the whole lifecycle will happily write there. A test
suite that leaves 40 GB — or, as it did the first time, five stray directories —
in the developer's home is a bug in the suite, not a quirk of it.

The shared registry (#138) is switched on by ``ATR_REGISTRY_ROOT`` in the
checkout's ``.env``, and ``Settings`` reads ``.env`` from the working directory —
the checkout the README says to run pytest from. There, every ``create_app`` in
the suite, and the module-level ``app`` that importing ``atr_serving.app``
builds, would publish that checkout's ``config/models.yaml`` to the live share
(edits and branch included) and read the live ``trained/``. Reproduced in the
#138 review: a ``.env`` with only that line published the curated registry on
import, and a merge_loras test picked up a registration from the share.

The trainer's address (#137) is the same trap. After the cutover idhefix's
``.env`` names asteraix in ``ATR_TRAIN_URL``; every ``Settings()`` in the suite
would then treat the trainer as remote, and the #129 launch-guard tests would
pass or fail for a reason that has nothing to do with the code. Its key would be
sent wherever a test pointed a client. A test that wants a remote trainer or a
key passes ``train_url``/``train_api_key`` to ``Settings`` itself.
"""

from __future__ import annotations

import os

import pytest

# An environment variable beats .env in pydantic-settings, and the validator
# reads "" as off. Set at import: conftest is loaded before any test module, so
# this is in place before one of them imports atr_serving.app.
os.environ["ATR_REGISTRY_ROOT"] = ""
TRAIN_URL_DEFAULT = "http://127.0.0.1:8204"
os.environ["ATR_TRAIN_URL"] = TRAIN_URL_DEFAULT
os.environ["ATR_TRAIN_API_KEY"] = ""


@pytest.fixture(autouse=True)
def _artefact_cache_never_touches_home(tmp_path_factory, monkeypatch):
    """Point every ``TrainerSettings`` in the suite at a throwaway cache root.

    Set through the environment rather than the fixtures, so it holds for the
    settings objects constructed inside the code under test as well as the ones
    the tests build themselves.
    """
    root = tmp_path_factory.mktemp("artefact-cache")
    monkeypatch.setenv("ATR_TRAIN_ARTEFACT_CACHE_ROOT", str(root))


@pytest.fixture(autouse=True)
def _shared_registry_is_off_unless_a_test_turns_it_on(monkeypatch):
    """Again per test: a test that deletes the variable gets it back, rather
    than handing the ``.env`` value to every test after it. A test that wants
    the feature passes ``registry_root`` to ``Settings`` itself."""
    monkeypatch.setenv("ATR_REGISTRY_ROOT", "")


@pytest.fixture(autouse=True)
def _trainer_is_local_and_keyless_unless_a_test_says_otherwise(monkeypatch):
    monkeypatch.setenv("ATR_TRAIN_URL", TRAIN_URL_DEFAULT)
    monkeypatch.setenv("ATR_TRAIN_API_KEY", "")
