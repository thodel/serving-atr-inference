"""Gateway -> engine-service HTTP clients.

The gateway is dependency-free of ML libraries (IMPLEMENTATION_PLAN.md §3). It
forwards recognition/segmentation work to the per-engine FastAPI services over
``127.0.0.1`` via httpx. Phase 1 wires kraken only; ISSUE #8 generalizes this to
a registry of engine clients (trocr, party, vllm).

Tests monkeypatch ``KrakenEngineClient`` (or its ``_client``) so gateway routing
and the legacy ``/ocr`` alias are exercised without a live engine.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
from loguru import logger

from atr_serving.api.schemas import Line, RecognitionResult, SegmentResponse


class EngineError(Exception):
    """Raised when an engine service is unreachable or returns an error."""


class KrakenEngineClient:
    """Thin async httpx wrapper around the kraken engine service."""

    #: Every other engine client here uses 300 s. 120 s was below a cold kraken
    #: model load, measured at 91 s and 128 s on srv (#81), so any request that
    #: triggered a model swap was a coin flip — and it surfaced as "unreachable".
    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _apost(self, path: str, *, files, data) -> dict:
        # A fresh client per call keeps the gateway stateless and test-friendly
        # (tests patch this method or httpx.AsyncClient directly).
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, files=files, data=data)
        except httpx.TimeoutException as exc:
            # Distinct from a connection failure on purpose. Reporting a busy
            # engine as "unreachable" sent two diagnoses at the wrong host (#81):
            # the service was up and answering 200s throughout, loading a model.
            # httpx timeouts also carry an empty str(), so the old message ended
            # in a bare colon with nothing after it.
            logger.error("kraken engine timed out after {}s at {}", self.timeout, url)
            raise EngineError(
                f"kraken engine did not answer within {self.timeout:.0f}s at {url} "
                f"— it is running but busy (a cold model load takes ~90-130s)"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("kraken engine unreachable at {}: {}", url, exc)
            raise EngineError(f"kraken engine unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise EngineError(
                f"kraken engine error {resp.status_code} at {url}: {resp.text}"
            )
        return resp.json()

    async def segment(
        self, image: bytes, filename: str, content_type: str, mode: str = "baseline"
    ) -> SegmentResponse:
        data = await self._apost(
            "/segment",
            files={"image": (filename, image, content_type)},
            data={"mode": mode},
        )
        return SegmentResponse(**data)

    async def recognize(
        self,
        image: bytes,
        filename: str,
        content_type: str,
        model: str,
        lines: list[Line] | None = None,
    ) -> RecognitionResult:
        form: dict[str, str] = {"model": model}
        if lines is not None:
            form["lines"] = json.dumps([ln.model_dump() for ln in lines])
        data = await self._apost(
            "/recognize",
            files={"image": (filename, image, content_type)},
            data=form,
        )
        return RecognitionResult(**data)


def get_kraken_client(settings) -> KrakenEngineClient:
    """Factory used by routes; a seam for tests to monkeypatch."""
    return KrakenEngineClient(settings.kraken_url)


# ── generic multipart engine client (kraken / trocr / party) ────────────────
# Engine services disagree on the image form field and (trocr) on the line
# schema, so the client knows the field name and coerces responses tolerantly.
ENGINE_IMAGE_FIELD = {"kraken": "image", "trocr": "file", "party": "file"}


def _coerce_line(idx: int, ln: dict[str, Any]) -> Line:
    bbox = ln.get("bbox")
    baseline = ln.get("baseline")
    # trocr returns baseline as a BBox dict {x0,y0,x1,y1}; map it to bbox
    if isinstance(baseline, dict):
        if bbox is None:
            bbox = [baseline.get(k, 0) for k in ("x0", "y0", "x1", "y1")]
        baseline = None
    if not (isinstance(baseline, list) and baseline and isinstance(baseline[0], (list, tuple))):
        baseline = None
    if not (isinstance(bbox, list) and len(bbox) == 4):
        bbox = None
    return Line(
        order=ln.get("order", idx), text=ln.get("text"),
        confidence=ln.get("confidence"), bbox=bbox, baseline=baseline,
    )


def coerce_result(data: dict[str, Any], engine: str, fallback_model: str) -> RecognitionResult:
    """Build a gateway RecognitionResult from a (possibly divergent) engine JSON."""
    raw_lines = data.get("lines") or []
    lines = [_coerce_line(i, ln) for i, ln in enumerate(raw_lines) if isinstance(ln, dict)]
    return RecognitionResult(
        model=data.get("model") or fallback_model,
        engine=data.get("engine") or engine,
        text=data.get("text") or "",
        lines=lines,
        confidence=data.get("confidence"),
        timing_ms=data.get("timing_ms") or 0,
        segmented_by=data.get("segmented_by"),
        version=data.get("version") or "?",
    )


class EngineHTTPClient:
    """Generic async client for a multipart engine ``/recognize`` endpoint."""

    def __init__(self, base_url: str, engine: str, image_field: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.engine = engine
        self.image_field = image_field
        self.timeout = timeout

    async def recognize(
        self, image: bytes, filename: str, content_type: str,
        model: str, lines: list[Line] | None = None,
    ) -> RecognitionResult:
        form: dict[str, str] = {"model": model}
        if lines is not None and self.engine == "kraken":
            form["lines"] = json.dumps([ln.model_dump() for ln in lines])
        url = f"{self.base_url}/recognize"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    url, files={self.image_field: (filename, image, content_type)}, data=form
                )
        except httpx.RequestError as exc:
            raise EngineError(f"{self.engine} engine unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise EngineError(f"{self.engine} engine error {resp.status_code} at {url}: {resp.text}")
        return coerce_result(resp.json(), self.engine, model)


class TrocrClient(EngineHTTPClient):
    """Async client for the TrOCR engine, with a GPU-batched batch endpoint."""

    async def recognize_batch(
        self, images: list[bytes], filenames: list[str], content_type: str,
        model: str,
    ) -> tuple[list[str], list[Line]]:
        """
        N line images in one GPU-batched forward pass (#95 step 2).

        images and filenames are parallel lists.  Returns (texts, lines) where
        texts[i] is the reading for images[i] and lines[i].order == i.
        """
        url = f"{self.base_url}/recognize_batch"
        files = [("files", (fn, img, content_type))
                 for img, fn in zip(images, filenames)]
        form = {"model": model}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, files=files, data=form)
        except httpx.RequestError as exc:
            raise EngineError(f"{self.engine} engine unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise EngineError(f"{self.engine} engine error {resp.status_code} at {url}: {resp.text}")
        data = resp.json()
        texts: list[str] = data.get("texts") or []
        raw_lines: list[dict] = data.get("lines") or []
        out_lines = [
            Line(order=ln.get("index", i), text=ln.get("text", ""),
                 confidence=ln.get("confidence"))
            for i, ln in enumerate(raw_lines)
        ]
        return texts, out_lines


def get_engine_client(engine: str, settings) -> EngineHTTPClient:
    """Factory used by routes; a seam for tests to monkeypatch."""
    return EngineHTTPClient(settings.engine_urls()[engine], engine, ENGINE_IMAGE_FIELD[engine])


# ── vLLM (OpenAI-compatible) ────────────────────────────────────────────────
def _data_url(image: bytes, content_type: str) -> str:
    mime = content_type if content_type and content_type.startswith("image/") else "image/png"
    return f"data:{mime};base64,{base64.b64encode(image).decode()}"


def build_image_content(image: bytes, content_type: str, prompt: str | None) -> list[dict[str, Any]]:
    """OpenAI chat ``content`` for one image (+ optional text instruction)."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": _data_url(image, content_type)}}
    ]
    if prompt:
        content.append({"type": "text", "text": prompt})
    return content


class VllmClient:
    """Async client for a running vLLM OpenAI-compatible server (one instance)."""

    def __init__(self, port: int, timeout: float = 300.0) -> None:
        self.base_url = f"http://127.0.0.1:{port}"
        self.timeout = timeout

    async def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise EngineError(f"vLLM unreachable at {url}: {exc}") from exc
        if resp.status_code >= 400:
            raise EngineError(f"vLLM error {resp.status_code} at {url}: {resp.text}")
        return resp.json()

    async def transcribe_image_detail(
        self, model: str, image: bytes, content_type: str, prompt: str | None, max_tokens: int
    ) -> tuple[str, str | None]:
        """``(text, finish_reason)`` for one image.

        ``finish_reason`` is the only place the server says *why* generation
        stopped: ``"stop"`` when the model ended the text, ``"length"`` when it
        ran into ``max_tokens``. Dropping it — which ``transcribe_image`` did, and
        still does for callers that do not need it — makes a truncated reading
        indistinguishable from a complete one.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": build_image_content(image, content_type, prompt)}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        data = await self.chat(payload)
        choice = data["choices"][0]
        return choice["message"]["content"], choice.get("finish_reason")

    async def transcribe_image(
        self, model: str, image: bytes, content_type: str, prompt: str | None, max_tokens: int
    ) -> str:
        text, _ = await self.transcribe_image_detail(
            model, image, content_type, prompt, max_tokens
        )
        return text


def get_vllm_client(port: int) -> VllmClient:
    """Factory used by routes; a seam for tests to monkeypatch."""
    return VllmClient(port)


# ── training service (#34/#35) ──────────────────────────────────────────────
class TrainerError(EngineError):
    """The trainer answered with an error status.

    Carries the status and body so the proxy can pass both through unchanged.
    The trainer's errors are *actionable* — 507 names a full filesystem, 500 a
    network TMPDIR, 409 an already-terminal job — and collapsing them into a
    generic 502 would throw away the part that tells the caller what to fix.

    ``detail`` is the JSON value as the trainer sent it — a string, or for a 422
    the list of field errors — not its ``str()``. Flattening a list turned it into
    a Python repr nobody can index (#137).

    ``service`` is the trainer's base URL. The proxy needs it because a 5xx
    detail describes the trainer's machine ("the vlm-train venv is not built on
    this box") while the caller reads it under the gateway's address.
    """

    def __init__(self, status_code: int, detail: Any, *, service: str) -> None:
        super().__init__(f"trainer error {status_code} at {service}: {detail}")
        self.status_code = status_code
        self.detail = detail
        self.service = service


class TrainerTimeout(EngineError):
    """The trainer did not answer in time. The proxy makes this a 504.

    Kept apart from "unreachable" for the reason #81 gave for the engines: a slow
    answer and a closed port send the reader to different places, and across the
    network (#137) a slow answer is the likelier of the two.
    """


class TrainerAuthError(EngineError):
    """The trainer refused the **gateway** (401/403). The proxy makes this a 502.

    Not a :class:`TrainerError`, so it is never passed through: the caller's own
    key was accepted by this gateway before anything was forwarded, and a 401
    handed back reads as "your key is wrong" — agentic_historian's atr_status.py
    says exactly that on a 401. What is wrong is this gateway's configuration.
    """


#: The fields of a pydantic error worth passing on. ``input`` echoes the whole
#: normalised request body — for a kraken job that includes the injected VGSL
#: spec — and ``ctx``/``url`` add nothing a caller can act on.
_ERROR_FIELDS = ("type", "loc", "msg")


def readable_detail(detail: Any) -> Any:
    """A trainer error body, with pydantic error lists reduced to what reads.

    Strings and dicts pass unchanged. A list keeps its shape — one entry per
    field error — so a caller can still index ``detail[0]["loc"]``.
    """
    if not isinstance(detail, list):
        return detail
    return [{k: item[k] for k in _ERROR_FIELDS if k in item}
            if isinstance(item, dict) else item
            for item in detail]


def _scrub(value: Any, secret: str) -> Any:
    """``value`` with ``secret`` blanked wherever it appears as text.

    The trainer does not echo the key; this makes that a property of the gateway
    rather than a promise of the other side, because anything here ends up in an
    HTTP response or a log line.
    """
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "***")
    if isinstance(value, list):
        return [_scrub(v, secret) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v, secret) for k, v in value.items()}
    return value


#: asteraix is on this box's /24, so a TCP connection that is not up in 5 s is
#: not coming. Bounded apart from the read because httpx applies a bare float to
#: *each* phase: 20 s would allow 20 to connect plus 20 to answer — past the
#: callers' 30 s that ``Settings.train_timeout_s`` exists to stay under.
TRAINER_CONNECT_TIMEOUT_S = 5.0


class TrainerClient:
    """Async client for the training service. Used only by the /train/* proxy.

    The key travels in the ``X-API-Key`` header and nowhere else: not in the URL,
    not in a log line, not in an exception message. ``transport`` is a test seam
    (``httpx.MockTransport``), so the whole path from route to wire is testable
    without a trainer. ``timeout`` defaults to ``Settings.train_timeout_s``'s
    value; see there for why it is not 30.
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 20.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._api_key = api_key
        self._transport = transport

    def __repr__(self) -> str:
        return f"TrainerClient({self.base_url!r}, key={'set' if self._api_key else 'unset'})"

    async def _request(self, method: str, path: str, *, timeout: float | None = None,
                       **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        timeout = self.timeout if timeout is None else timeout
        limits = httpx.Timeout(timeout, connect=min(timeout, TRAINER_CONNECT_TIMEOUT_S))
        headers = {"X-API-Key": self._api_key} if self._api_key else {}
        try:
            async with httpx.AsyncClient(timeout=limits, headers=headers,
                                         transport=self._transport) as client:
                resp = await client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            # Before RequestError, which it subclasses. httpx timeouts stringify
            # to "" more often than not, so the message is built, not forwarded.
            if isinstance(exc, httpx.ConnectTimeout):
                what, waited = "could not connect", limits.connect
            else:
                what, waited = "did not answer", timeout
            logger.error("trainer {} within {}s at {}", what, waited, url)
            raise TrainerTimeout(
                f"training service {what} within {waited:g}s at {url}"
            ) from exc
        except httpx.RequestError as exc:
            reason = _scrub(str(exc), self._api_key)
            logger.error("trainer unreachable at {}: {}", url, reason)
            raise EngineError(f"training service unreachable at {url}: {reason}") from exc
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except ValueError:  # a non-JSON error body is still a body
                body = None
            detail = body.get("detail", resp.text) if isinstance(body, dict) else resp.text
            detail = _scrub(readable_detail(detail), self._api_key)
            if resp.status_code in (401, 403):
                message = self._refusal(resp.status_code, url, detail)
                logger.error("{}", message)
                raise TrainerAuthError(message)
            raise TrainerError(resp.status_code, detail, service=self.base_url)
        # Below: an answer that is not the trainer's. Unguarded, both reached
        # resp.json(), whose ValueError is no EngineError — a bare 500 with no URL
        # on every route, and on submit the /health courtesy check failed the job
        # it is documented to skip for (#137 review).
        if not resp.is_success:
            # httpx does not follow redirects, so a 3xx lands here with an empty
            # body. The trainer never redirects; an http->https front or another
            # service at ATR_TRAIN_URL does.
            location = resp.headers.get("location")
            where = (f"redirect to {_scrub(location, self._api_key)}" if location
                     else "no Location given")
            message = (f"training service at {url} answered {resp.status_code} "
                       f"({where}); ATR_TRAIN_URL must name the trainer itself")
            logger.error("{}", message)
            raise EngineError(message)
        try:
            return resp.json()
        except ValueError as exc:
            # A 2xx that is not JSON is some other HTTP service: asteraix runs
            # several on neighbouring ports (config.py), one port-typo away.
            # Scrubbed before it is cut, so a cut cannot leave half a key behind.
            body = _scrub(resp.text, self._api_key)[:120]
            message = (f"training service at {url} answered {resp.status_code} with a "
                       f"non-JSON body ({body!r}); is ATR_TRAIN_URL the trainer?")
            logger.error("{}", message)
            raise EngineError(message) from exc

    def _refusal(self, status: int, url: str, detail: Any) -> str:
        """Why the trainer turned the gateway away, in terms of the setting to fix."""
        if status == 401:
            cause = ("this gateway's ATR_TRAIN_API_KEY is "
                     + ("set but does not match the trainer's" if self._api_key
                        else "empty")
                     + " — both machines must hold the same value")
        else:
            cause = ("this gateway's address is not in the trainer's "
                     "ATR_TRAIN_ALLOWED_CLIENTS")
        return (f"training service at {url} refused the gateway with {status} "
                f"({detail}): {cause}. Not the caller's key — the gateway accepted "
                "that before forwarding; this is the gateway's own configuration.")

    async def submit(self, body: dict[str, Any]) -> dict:
        return await self._request("POST", "/jobs", json=body)

    async def list_jobs(self) -> dict:
        return await self._request("GET", "/jobs")

    async def get(self, job_id: str) -> dict:
        return await self._request("GET", f"/jobs/{job_id}")

    async def log(self, job_id: str, stage: str, lines: int) -> dict:
        return await self._request("GET", f"/jobs/{job_id}/log",
                                   params={"stage": stage, "lines": lines})

    async def cancel(self, job_id: str) -> dict:
        return await self._request("POST", f"/jobs/{job_id}/cancel")

    async def delete(self, job_id: str) -> dict:
        return await self._request("DELETE", f"/jobs/{job_id}")

    async def health(self, timeout: float | None = None) -> dict:
        return await self._request("GET", "/health", timeout=timeout)

    async def gpu(self) -> dict:
        """The trainer's own reading of its cards (#137). Older trainers answer 404."""
        return await self._request("GET", "/gpu")

    async def curve(self, job_id: str) -> dict:
        return await self._request("GET", f"/jobs/{job_id}/curve")

    async def verify(self, body: dict[str, Any]) -> dict:
        """Check a request's dataset against the hub. Queues nothing."""
        return await self._request("POST", "/jobs/verify", json=body)


def get_trainer_client(settings) -> TrainerClient:
    """Factory used by routes; a seam for tests to monkeypatch."""
    return TrainerClient(settings.train_url, api_key=settings.train_api_key,
                         timeout=settings.train_timeout_s)
