"""Which commit the running code came from (#147).

The training service is imported from a git checkout on every host it runs on —
an editable install on idhefix, ``PYTHONPATH=$REPO/src`` inside the UBELIX
container — so the checkout's ``HEAD`` *is* the code. It is read once per
process: a checkout that changes underneath a running process does not change
the modules already imported, so re-reading it later would describe code the
process is not running.

Never raises. A host without git, or code that is not in a checkout, yields a
:class:`CodeVersion` with ``commit=None``, and callers record that as unknown.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from atr_serving.training.contracts import CodeVersion

__all__ = ["current_code", "describe_drift", "read_code_version"]

#: src/atr_serving/training/codeversion.py -> the repository root
_REPO = Path(__file__).resolve().parents[3]


def _git(repo: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def read_code_version(repo: Path) -> CodeVersion:
    """``HEAD`` and whether tracked files differ from it, for the checkout at ``repo``."""
    top = _git(repo, "rev-parse", "--show-toplevel")
    if top is None or Path(top).resolve() != repo.resolve():
        # Not a checkout of *this* repository: an installed package sitting
        # inside some unrelated git tree must not report that tree's commit.
        return CodeVersion()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit is None:
        return CodeVersion()
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return CodeVersion(commit=commit, dirty=None if status is None else bool(status))


@lru_cache(maxsize=1)
def current_code() -> CodeVersion:
    """The code this process is running. Read once, then cached."""
    return read_code_version(_REPO)


def describe_drift(created: CodeVersion | None, running: CodeVersion | None) -> str | None:
    """Why a stage's code differs from the job's, or ``None`` when it does not.

    Only a *known* difference counts. A job created before #147 has no recorded
    code, and a host without git has no current one; neither is a drift.
    """
    if not created or not running or not created.commit or not running.commit:
        return None
    if created.commit == running.commit and not running.dirty:
        return None
    if created.commit == running.commit:
        return f"running {running.short()}: the job's commit, with uncommitted changes"
    return (f"running {running.short()}, but the job was created with {created.short()} "
            "— this stage does not run the code the job was submitted with")
