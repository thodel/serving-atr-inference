"""The gateway client: ``GET /health``, ``GET /models``, ``POST /segment``,
``/recognize``, ``/ocr``, every request carrying ``X-API-Key``.

What it knows, and why it is the place that knows it:

* **The timeout is the gateway's engine budget, 300 s.** A client with a
  tighter budget hangs up on work the gateway is still doing and blames the
  engine: agentic_historian wrote off ``trocr-medieval-escriptmask`` as
  broken on 2026-07-16 because its client allowed 120 s.
* **``/ocr`` reads kraken and trocr only; everything else is ``/recognize``.**
  ``/ocr`` is the page-level convenience that auto-segments; a party or vllm
  model sent there answers ``400: use /recognize``. Unknown engines go to
  ``/recognize`` (agentic_historian #540): guessing ``/ocr`` for a new engine
  reproduces the bug that made every vLLM model unreachable.
* **An error carries its status.** ``AtrClientError.status_code`` is the
  gateway's HTTP status, or ``None`` when it never answered. A batch runner
  retries ``None`` and 5xx; a 404 (unknown model) or 401 fails identically for
  ever, and retrying it only delays the report that says so.
* **A 4xx body is never parsed as a transcription.** Before the status check,
  a 422 validation payload read as an empty page.

What it does not do: retry. Whether a 502 is worth a second attempt is the
caller's policy (a batch runner's yes, a nightly gate's no), not the client's.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

#: The gateway's per-engine budget (``HTTPEngineClient(timeout=300.0)`` in
#: ``atr_serving.clients``). Never default to less than the server allows.
DEFAULT_TIMEOUT = 300.0

#: The engines ``/ocr`` accepts. ``vlm`` is deliberately absent and is not a
#: typo for ``vllm``: ``vlm`` is a consumer's GPUStack path that never touches
#: the gateway; ``vllm`` is the gateway's own engine.
OCR_ENGINES = frozenset({"kraken", "trocr"})

ENV_URL = "ATR_GATEWAY_URL"
ENV_KEY = "ATR_API_KEY"
ENV_TIMEOUT = "ATR_HTTP_TIMEOUT"

ImageLike = Path | bytes | io.BytesIO


def endpoint_for(engine: str | None) -> str:
    """Which gateway endpoint can read a model of this engine."""
    return "/ocr" if str(engine or "").lower() in OCR_ENGINES else "/recognize"


class AtrClientError(RuntimeError):
    """The gateway is unreachable or rejected the request.

    ``status_code`` is the gateway's HTTP status when it answered at all and
    ``None`` when it never did (unreachable host, reset, timeout).
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AtrResult:
    """A recognition result as the gateway returned it.

    The first four fields are the ``/ocr`` shape every caller reads. The rest
    is what ``/recognize`` additionally returns and defaults to empty, so a
    result built from an ``/ocr`` response is unchanged.
    """

    text: str
    confidence: float
    model: str
    service_version: str = "?"
    #: Which engine answered, as reported by the gateway, never inferred.
    engine: str = ""
    #: Per-line readings (order, baseline/bbox, text, confidence). Empty for
    #: page-level engines that expose no lines.
    lines: list[dict] = field(default_factory=list)
    #: What produced the line geometry when the engine did not segment itself.
    segmented_by: str | None = None
    #: Gateway-measured duration in milliseconds, ``None`` when it did not
    #: report one — not ``0``: zero milliseconds is not a measurement
    #: (agentic_historian #595).
    timing_ms: int | None = None
    #: The model stopped at its token ceiling, so this reading is cut off.
    #: False also when the gateway cannot tell (serving-atr-inference #123).
    truncated: bool = False
    #: Party's reading of the same image, attached by the gateway to every
    #: non-party result; ``None`` when off or when party is the engine.
    second_opinion: dict | None = None

    @property
    def model_used(self) -> str:
        """The name agentic_historian's callers read; same value as ``model``."""
        return self.model


def _timing_of(data: dict) -> int | None:
    raw = data.get("timing_ms")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class AtrClient:
    """Thin synchronous client for serving-atr-inference.

    Use it as a context manager for a batch (one connection pool for the
    run), or call methods directly for one-off requests; the session is
    created on first use and closed by ``close()``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.timeout = DEFAULT_TIMEOUT if timeout is None else timeout
        self._client: httpx.Client | None = http_client
        self._owns_client = http_client is None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, **overrides: Any) -> AtrClient:
        """A client from ``ATR_GATEWAY_URL``, ``ATR_API_KEY``, ``ATR_HTTP_TIMEOUT``."""
        env = os.environ if environ is None else environ
        kwargs: dict[str, Any] = {
            "base_url": env.get(ENV_URL, ""),
            "api_key": env.get(ENV_KEY, ""),
            "timeout": float(env.get(ENV_TIMEOUT) or DEFAULT_TIMEOUT),
        }
        kwargs.update(overrides)
        return cls(kwargs.pop("base_url"), **kwargs)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def __enter__(self) -> AtrClient:
        self._get_client()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the session this client created; a borrowed one is left open."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    # ── public API ───────────────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """Gateway health: status, version, model_count, resident models, engines."""
        return self._get("/health").json()

    def list_models(self) -> list[dict[str, Any]]:
        """The gateway's model registry (id, engine, scripts, centuries, …)."""
        models = self._get("/models").json().get("models", [])
        return models if isinstance(models, list) else []

    def list_model_ids(self) -> list[str]:
        return [m["id"] for m in self.list_models() if isinstance(m, dict) and "id" in m]

    def segment(self, image: ImageLike, seg_mode: str = "baseline") -> dict[str, Any]:
        """Segment a page without recognising its text."""
        return self._post("/segment", files=self._image_part(image),
                          data={"seg_mode": seg_mode}).json()

    def recognize(self, image: ImageLike, model: str, *, engine: str | None = None,
                  lines: list[dict[str, Any]] | None = None) -> AtrResult:
        """Read a page through ``/recognize``: every engine, the full result.

        ``lines`` — ``[{"order", "baseline", "bbox", "boundary", "regions"}, …]``
        in the image's pixels — are the page's own lines. A kraken model reads on
        exactly those, in that order, instead of segmenting again
        (``segmented_by: "given"``); the answer's lines match them one for one.
        """
        data = {"model": model}
        if lines:
            data["lines"] = json.dumps(list(lines))
        response = self._post("/recognize", files=self._image_part(image), data=data)
        return self._result(response, model=model, engine=engine or "")

    def ocr(self, image: ImageLike, model: str, *, seg_mode: str = "baseline",
            engine: str | None = None) -> AtrResult:
        """Read a page through ``/ocr`` (kraken and trocr only; auto-segments)."""
        response = self._post("/ocr", files=self._image_part(image),
                              data={"model": model, "seg_mode": seg_mode})
        return self._result(response, model=model, engine=engine or "")

    def read(self, image: ImageLike, model: str, engine: str | None,
             seg_mode: str = "baseline") -> AtrResult:
        """Read *image* with *model* on whichever endpoint its engine needs."""
        if endpoint_for(engine) == "/ocr":
            return self.ocr(image, model, seg_mode=seg_mode, engine=engine)
        return self.recognize(image, model, engine=engine)

    def transcribe(self, image: ImageLike, model: str, *, engine: str = "kraken",
                   seg_mode: str = "baseline") -> AtrResult:
        """``read`` with the engine defaulting to kraken — the historical name."""
        return self.read(image, model, engine, seg_mode=seg_mode)

    # ── internals ────────────────────────────────────────────────────────────

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            if not self.base_url:
                raise AtrClientError(f"ATR gateway URL is not configured; set {ENV_URL}")
            headers = {"X-API-Key": self.api_key} if self.api_key else {}
            self._client = httpx.Client(base_url=self.base_url, headers=headers,
                                        timeout=self.timeout, follow_redirects=True)
        return self._client

    def _get(self, path: str) -> httpx.Response:
        try:
            response = self._get_client().get(path)
        except httpx.RequestError as exc:
            raise AtrClientError(
                f"ATR gateway unreachable at {self.base_url}{path}: {exc}") from exc
        return self._check(response, path)

    def _post(self, path: str, files: dict, data: dict | None = None) -> httpx.Response:
        try:
            response = self._get_client().post(path, files=files, data=data)
        except httpx.RequestError as exc:
            raise AtrClientError(
                f"ATR gateway unreachable at {self.base_url}{path}: {exc}") from exc
        return self._check(response, path)

    def _check(self, response: httpx.Response, path: str) -> httpx.Response:
        if response.status_code >= 400:
            raise AtrClientError(
                f"ATR gateway returned {response.status_code} at {path}: {response.text[:400]}",
                status_code=response.status_code,
            )
        return response

    @staticmethod
    def _image_part(image: ImageLike) -> dict[str, tuple[str, bytes, str]]:
        if isinstance(image, Path):
            return {"image": (image.name, image.read_bytes(), "application/octet-stream")}
        if isinstance(image, bytes):
            return {"image": ("image", image, "application/octet-stream")}
        image.seek(0)
        return {"image": ("image", image.read(), "application/octet-stream")}

    @staticmethod
    def _result(response: httpx.Response, *, model: str, engine: str) -> AtrResult:
        data = response.json()
        return AtrResult(
            text=data.get("text", ""),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            model=data.get("model", model),
            service_version=data.get("version", "?"),
            engine=data.get("engine", engine) or engine,
            lines=list(data.get("lines") or []),
            segmented_by=data.get("segmented_by"),
            timing_ms=_timing_of(data),
            truncated=bool(data.get("truncated", False)),
            second_opinion=data.get("second_opinion"),
        )
