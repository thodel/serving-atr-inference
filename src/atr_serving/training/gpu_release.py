"""Ask the gateway to give the card back before a run gets on it (#129).

A training job spends its first hour or two in ``prepare`` and ``compile``,
which are disk, network and CPU. The card is idle throughout — v5 sat at 0 %
utilisation for 74 minutes — and for that whole time the gateway may serve
recognition requests, including ones that have to load a model first.

What it must not do is still be holding that model when the run reaches
``train``. That is how v4 died: a vLLM instance launched at 08:53 during
prepare was still resident at 09:52 when training began, and the run hit
OutOfMemoryError three minutes later.

The first answer was to refuse every launch from the moment a job started,
which is safe and costs an hour of inference on an empty GPU per run. This is
the second: the gateway keeps serving through the CPU stages, and the trainer
asks it to let go at the boundary. A failure to reach the gateway is reported
and not fatal — the trainer's own VRAM preflight still stands behind it, and it
is the one that actually refuses to start.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

__all__ = ["ReleaseResult", "release_gpu"]


@dataclass(frozen=True)
class ReleaseResult:
    """What the gateway let go of, and what it kept."""

    reached: bool
    dropped: list[str] = field(default_factory=list)
    #: Models the gateway refuses to evict — ``residency: pinned`` in the
    #: registry. Named rather than counted: if one of these is what the run then
    #: fails to fit around, the log has to say which.
    kept: list[str] = field(default_factory=list)
    detail: str = ""

    def __str__(self) -> str:
        if not self.reached:
            return f"gateway not reached ({self.detail})"
        parts = [f"released {', '.join(self.dropped)}" if self.dropped
                 else "nothing resident to release"]
        if self.kept:
            parts.append(f"kept pinned: {', '.join(self.kept)}")
        return "; ".join(parts)


def release_gpu(gateway_url: str, api_key: str, timeout: float = 60.0
                ) -> ReleaseResult:
    """Ask the gateway to unload its evictable vLLM models. Never raises."""
    import httpx  # trainer venv only

    url = f"{gateway_url.rstrip('/')}/admin/release-gpu"
    try:
        response = httpx.post(url, headers={"X-API-Key": api_key}, timeout=timeout)
        response.raise_for_status()
        body = response.json()
    except Exception as exc:  # noqa: BLE001 — a gateway that cannot be asked is
        # not a reason to abandon a run; the VRAM preflight is what decides.
        logger.warning("could not ask the gateway to release GPU memory ({}): {}",
                       url, exc)
        return ReleaseResult(reached=False, detail=f"{type(exc).__name__}: {exc}")

    result = ReleaseResult(reached=True,
                           dropped=list(body.get("dropped") or []),
                           kept=list(body.get("kept") or []))
    logger.info("gateway: {}", result)
    return result
