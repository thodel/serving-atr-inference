#!/usr/bin/env python3
"""Assert that what is installed in THIS interpreter satisfies a requirements file.

    <venv>/bin/python scripts/check_requirements.py engines/vlm_train_svc/requirements.txt

Exists because importing a package does not tell you which one you imported (#53).
The transformers 5.x incident passed every import check there was: ``import
transformers`` worked, and ``TrainingArguments(...)`` constructed happily on 5.14.1
when the code was written against 4.57. It was found by printing ``__version__``.

The *repair* had the same shape and would have fooled the same checks: the downgrade
to 4.57.6 died with EPERM, pip exited non-zero, and the venv silently kept 5.14.1. A
smoke test that only imports passes identically before and after a fix that never
happened.

Every expectation here is read from the requirements file the venv was built from, so
there is no second list to drift out of step with the first — which is the failure mode
this is guarding against, one level up.
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path

# `packaging` implements PEP 440, which matters more than it looks: torch reports
# `2.8.0+cu128` against a `torch==2.8.0` pin, and a string comparison would call that
# a mismatch. A local version identifier is compatible unless the specifier names one.
try:
    from packaging.requirements import InvalidRequirement, Requirement
except ModuleNotFoundError:  # pragma: no cover - every venv has pip, not every one has packaging
    try:
        from pip._vendor.packaging.requirements import (  # type: ignore[no-redef]
            InvalidRequirement,
            Requirement,
        )
    except ModuleNotFoundError:
        print("SKIP  no `packaging` and no pip-vendored copy — cannot check versions")
        raise SystemExit(0) from None

#: Lines that configure pip rather than name a requirement.
_DIRECTIVES = ("-r", "-c", "-e", "--index-url", "--extra-index-url", "--find-links",
               "--no-binary", "--only-binary", "--pre", "--trusted-host")


def requirements_in(path: Path) -> list[Requirement]:
    out: list[Requirement] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or line.startswith(_DIRECTIVES):
            continue
        try:
            out.append(Requirement(line))
        except InvalidRequirement:
            print(f"WARN  cannot parse requirement: {line!r}")
    return out


def check(path: Path, verbose: bool = False) -> int:
    problems = 0
    for req in requirements_in(path):
        try:
            found = installed_version(req.name)
        except PackageNotFoundError:
            print(f"MISSING   {req.name}: not installed (requires {req.specifier or 'any'})")
            problems += 1
            continue

        # An unpinned requirement cannot be violated, so report it only on request.
        if not req.specifier:
            if verbose:
                print(f"ok        {req.name} {found} (unpinned)")
            continue

        if req.specifier.contains(found, prereleases=True):
            if verbose:
                print(f"ok        {req.name} {found} (requires {req.specifier})")
        else:
            print(f"MISMATCH  {req.name} {found} installed, but {path.name} "
                  f"requires {req.specifier}")
            problems += 1
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("requirements", type=Path, nargs="+")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="also print the requirements that are satisfied")
    args = parser.parse_args(argv)

    problems = 0
    for path in args.requirements:
        if not path.is_file():
            print(f"MISSING   requirements file not found: {path}")
            problems += 1
            continue
        problems += check(path, args.verbose)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
