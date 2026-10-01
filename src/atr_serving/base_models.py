"""Can this candidate be a ``base_model`` — and if not, why not? (#115)

Discovery ranked by download count and text match, so
``microsoft/trocr-small-handwritten`` (450 981 downloads) sat in the same
undifferentiated table as any ``vision-encoder-decoder`` repo that matched the
string ``trocr``. The report answered "does this exist and does it mention
HTR", which is not the question worth a weekly notification.

This module answers the one that is, from the repository's **own files and
config** rather than from its card prose:

====== ========================================= =============================
kraken a ``.mlmodel``                            no weights file
TrOCR  a ``VisionEncoderDecoder`` and a tokenizer encoder-only; no tokenizer
VLM    a chat template, a vision tower, and a    above the QLoRA band
       parameter count inside the band
====== ========================================= =============================

## The band cites its measurement

``QLORA_CEILING`` is not a guess. Qwen3-VL-8B under QLoRA reached **28.4 GiB
steady on a 46 GiB A40 at batch_size 1** — so ~8B is near the ceiling, not
comfortably inside it, and anything materially larger is not a candidate on
this hardware however good it is. ``QLORA_MEASUREMENT`` carries that sentence
so that "too big" is a fact with a provenance and not an opinion, and so that
the number moves when the hardware does rather than when somebody's intuition
does.

The band has a ceiling and no floor, deliberately: there is a measurement for
"does not fit" and none for "too small to be worth the plumbing", and inventing
the second would put a guess next to a measurement where a reader cannot tell
them apart.

## What may be rejected automatically, and what may not

:attr:`Judgement.automatic` is the line. A reason that is a **fact about the
repository** — no weights, encoder-only, above the band — is written into the
catalogue as ``rejected`` once and never re-asked, which is what #113's
catalogue is for. A reason that is a fact about **this run** — the hub did not
answer, the parameter count is not published, nothing matched — is reported and
left for a person. Otherwise a rate limit on a Monday morning becomes a
permanent verdict, and the catalogue's whole value is that a verdict sticks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "BACKENDS",
    "ENCODER_ONLY",
    "Judgement",
    "QLORA_CEILING",
    "QLORA_MEASUREMENT",
    "RepoFacts",
    "judge",
]

BACKENDS = ("kraken", "trocr", "vllm")

#: Anything that can hold weights. A repo with none of these is a model card.
WEIGHT_SUFFIXES = (".mlmodel", ".safetensors", ".bin", ".pt", ".pth", ".gguf",
                   ".onnx", ".msgpack")
#: kraken loads one of these and nothing else.
KRAKEN_SUFFIX = ".mlmodel"
#: Any of these is a tokenizer we can resize for a new alphabet.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "spiece.model", "sentencepiece.bpe.model", "vocab.txt")
#: TrOCR's shape: an encoder and a decoder in one config.
ENCODER_DECODER = ("visionencoderdecoder",)
#: Encoder-only backbones that match "vision" searches and cannot produce text.
ENCODER_ONLY = ("vitmodel", "beitmodel", "swinmodel", "deitmodel",
                "donutswinmodel", "dinov2model", "clipvisionmodel",
                "segformerfor", "resnetmodel")

#: The measurement behind the band, carried so "too big" has a provenance.
QLORA_MEASUREMENT = (
    "qwen3-vl-8b under QLoRA: 28.4 GiB steady on a 46 GiB A40 at batch_size 1")
#: Parameters. 9e9 rather than 8e9 because the measurement says ~8B is *near*
#: the ceiling: a 8.4B variant is still worth a look, a 13B is not.
QLORA_CEILING = 9_000_000_000


@dataclass
class RepoFacts:
    """What the hub says about one repository. Files and config, not prose."""

    id: str
    files: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    #: Total parameters, when the repo publishes them (``safetensors.total``).
    #: ``None`` is "not published", which is not "small".
    parameters: int | None = None
    config: dict = field(default_factory=dict)
    #: False when the hub did not answer. Then nothing here is evidence of
    #: anything, and :attr:`Judgement.automatic` stays False.
    fetched: bool = True


@dataclass
class Judgement:
    """Which backend could load this, or the recorded reason none can."""

    backend: str | None = None
    #: What in the repository decided it, for the report and the catalogue.
    evidence: str = ""
    reason: str | None = None
    #: Whether :attr:`reason` may be written as a verdict without a person.
    automatic: bool = False

    @property
    def usable(self) -> bool:
        return self.backend is not None


def _lower(values) -> tuple[str, ...]:
    return tuple(str(v).lower() for v in values or ())


def _has_weights(facts: RepoFacts) -> bool:
    return any(f.lower().endswith(WEIGHT_SUFFIXES) for f in facts.files)


def _kraken_model(facts: RepoFacts) -> str | None:
    for name in facts.files:
        if name.lower().endswith(KRAKEN_SUFFIX):
            return name
    return None


def _tokenizer(facts: RepoFacts) -> str | None:
    names = {f.rsplit("/", 1)[-1].lower() for f in facts.files}
    for candidate in TOKENIZER_FILES:
        if candidate in names:
            return candidate
    return None


def _architectures(facts: RepoFacts) -> tuple[str, ...]:
    """Architectures from the facts, or from the config if only it has them."""
    if facts.architectures:
        return _lower(facts.architectures)
    return _lower((facts.config or {}).get("architectures") or ())


def _vision_tower(facts: RepoFacts) -> str | None:
    config = facts.config or {}
    for key in ("vision_config", "vision_tower", "visual"):
        # Present, not truthy: the key is the evidence. An empty vision_config
        # is a repository that declares a vision tower and says nothing else
        # about it, which is still a vision tower.
        if config.get(key) is not None:
            return key
    for tag in _lower(facts.tags):
        if tag in ("image-text-to-text", "visual-question-answering"):
            return f"tag {tag}"
    return None


def _chat_template(facts: RepoFacts) -> bool:
    names = {f.rsplit("/", 1)[-1].lower() for f in facts.files}
    if "chat_template.json" in names or "chat_template.jinja" in names:
        return True
    return bool((facts.config or {}).get("chat_template"))


def judge(facts: RepoFacts) -> Judgement:
    """Which backend could load ``facts``, or why none can.

    The order is cheapest-and-most-certain first: a repository with no weights
    is not a candidate for anything, and a ``.mlmodel`` is unambiguous.
    """
    if not facts.fetched:
        return Judgement(
            reason="the hub did not answer for this repository, so it was not "
                   "judged — not a verdict, and deliberately not recorded as one")

    if not _has_weights(facts):
        return Judgement(
            reason="no weights file, only a model card", automatic=True)

    kraken = _kraken_model(facts)
    if kraken:
        return Judgement(backend="kraken", evidence=f"a .mlmodel: {kraken}")

    architectures = _architectures(facts)

    if any(arch.startswith(ENCODER_DECODER) for arch in architectures):
        tokenizer = _tokenizer(facts)
        if tokenizer:
            return Judgement(backend="trocr",
                             evidence=f"VisionEncoderDecoder and {tokenizer}")
        return Judgement(
            reason="a VisionEncoderDecoder without a tokenizer: there is no "
                   "alphabet to resize, so it cannot be fine-tuned here",
            automatic=True)

    encoder_only = next((arch for arch in architectures
                         if arch.startswith(ENCODER_ONLY)), None)
    if encoder_only:
        return Judgement(
            reason=f"encoder-only ({encoder_only}): no decoder, so it produces "
                   "features and not text",
            automatic=True)

    tower = _vision_tower(facts)
    if tower and _chat_template(facts):
        if facts.parameters is None:
            return Judgement(
                reason="a VLM whose parameter count the repository does not "
                       f"publish, so the band cannot be applied ({QLORA_MEASUREMENT})")
        if facts.parameters > QLORA_CEILING:
            billions = facts.parameters / 1e9
            return Judgement(
                reason=f"{billions:.1f}B parameters, above the band this box can "
                       f"fine-tune — {QLORA_MEASUREMENT}",
                automatic=True)
        return Judgement(backend="vllm",
                         evidence=f"{facts.parameters / 1e9:.1f}B, {tower}, "
                                  "chat template")

    if tower:
        return Judgement(
            reason="a vision tower but no chat template: it may be servable, "
                   "but not by the VLM backend as it is configured here")

    return Judgement(
        reason="nothing identifies a backend that could load it — weights, but "
               "no .mlmodel, no VisionEncoderDecoder and no vision tower")
