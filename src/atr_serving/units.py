"""Restart an engine's systemd unit, from the gateway that fronts it.

The clients reach idhefix on exactly one port — ``:8200``, with ``X-API-Key``,
over the VPN (``docs/INFRASTRUCTURE.md`` §284/§308). There is no SSH from tei.
So the repair step the troubleshooting tables name, ``systemctl --user restart
atr-kraken``, is unreachable from every caller, and an engine that falls over
waits for a human with a shell on the box. On 09.10.2026 that wait cost a
``missiven`` run 16 of its 20 recognitions (agentic_historian#595).

The gateway can do it without privileges: the engines are **user** units
(``scripts/install_user_units.sh`` — idhefix has no passwordless sudo), and the
gateway runs in the same user session.

Two rules live here rather than in the route, because they are the ones worth a
unit test:

**``reset-failed`` before ``restart``.** ``Restart=on-failure`` in the unit
covers neither of the cases that actually happen. systemd does not restart a
process that exits 0, and once a unit has burned through its start-limit burst
systemd gives up and leaves it in ``failed`` — permanently, until something
clears it. In that state a bare ``restart`` can report success without starting
anything. That is the 09.10. case, so it is the case this module must handle
first.

**The unit name is derived, never taken from the caller.** ``unit_of`` resolves
only names the gateway already serves as engines; a path segment never becomes a
unit name. Nothing here can name ``atr-gateway``: restarting the process that is
answering the request would cut off its own answer, and that is not a primitive
this API should offer.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

__all__ = ["UNIT_PREFIX", "TIMEOUT_S", "RestartResult", "unit_of", "run_systemctl",
           "restart_unit"]

#: ``kraken`` → ``atr-kraken.service``. The authority for the spelling is
#: ``scripts/install_user_units.sh``'s ``UNITS``; ``tests/test_209_engine_restart.py``
#: asserts that every engine this gateway serves has a unit file under
#: ``deploy/systemd`` under exactly this name, so a renamed engine fails CI here
#: rather than at 02:00 on the box.
UNIT_PREFIX = "atr-"

#: How long one ``systemctl`` call may take. It talks to the user manager over a
#: socket and returns as soon as the job is queued, so this is generous; it
#: exists so a wedged manager cannot hold a request open for ever.
TIMEOUT_S = 20.0


@dataclass(frozen=True)
class RestartResult:
    """What ``systemctl`` did. Deliberately not "is the engine up".

    ``ok`` says the restart command exited 0 — that the unit was told to start,
    nothing more. Whether it answers again is a separate measurement, and the
    route makes it by probing ``/health``. Collapsing the two is how a value
    comes to claim more than was measured.
    """

    ok: bool
    detail: str = ""
    #: Each ``systemctl`` invocation, in the order it ran, as argv tails
    #: (``("reset-failed", "atr-kraken.service")``). The route does not use it;
    #: it is what lets a test assert the ORDER, which is the whole point of
    #: ``reset-failed`` being here at all.
    calls: tuple[tuple[str, ...], ...] = ()


def unit_of(engine: str, engines: dict[str, str]) -> str | None:
    """The unit name for ``engine``, or ``None`` when it is not one of ours.

    ``engines`` is ``Settings.engine_urls()`` — the recognition engines and
    nothing else. The trainer is not in it (it has lived on asteraix since
    16.09.2026, and a restart there belongs on that machine), and neither is the
    gateway. So both are refused by construction rather than by a blocklist that
    someone has to remember to extend.
    """
    if engine not in engines:
        return None
    return f"{UNIT_PREFIX}{engine}.service"


def run_systemctl(*args: str) -> tuple[int, str]:
    """``systemctl --user <args>`` → (returncode, combined output).

    The seam the tests replace: there is no systemd in CI, and standing one up
    to check that ``reset-failed`` runs before ``restart`` would be absurd.
    """
    binary = shutil.which("systemctl")
    if binary is None:
        return 127, "systemctl not found on PATH"
    try:
        done = subprocess.run(  # noqa: S603 - fixed binary, no shell, args are derived
            [binary, "--user", *args],
            capture_output=True, text=True, timeout=TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"systemctl --user {' '.join(args)} did not finish in {TIMEOUT_S:.0f}s"
    except OSError as exc:
        return 126, f"{type(exc).__name__}: {exc}"
    return done.returncode, (done.stdout + done.stderr).strip()


def restart_unit(unit: str) -> RestartResult:
    """``reset-failed`` then ``restart``, in that order.

    A non-zero ``reset-failed`` is **not** fatal and does not appear in
    ``detail`` on success: on a unit that is not in a failed state it is a no-op
    that some systemd versions still complain about, and refusing to restart
    because the clearing step was unnecessary would defeat the purpose. The
    ``restart`` decides.
    """
    calls: list[tuple[str, ...]] = []

    reset = ("reset-failed", unit)
    calls.append(reset)
    reset_code, reset_out = run_systemctl(*reset)

    start = ("restart", unit)
    calls.append(start)
    code, out = run_systemctl(*start)

    if code == 0:
        return RestartResult(ok=True, detail=out, calls=tuple(calls))

    # The restart failed, so the reset's output is now evidence rather than
    # noise — a unit that does not exist fails both, and saying so once is
    # clearer than saying "restart failed" about a unit that was never there.
    detail = f"systemctl --user restart {unit} exited {code}: {out or '(no output)'}"
    if reset_code != 0 and reset_out:
        detail += f" (reset-failed exited {reset_code}: {reset_out})"
    return RestartResult(ok=False, detail=detail, calls=tuple(calls))
