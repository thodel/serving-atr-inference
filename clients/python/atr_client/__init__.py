"""Client for the serving-atr-inference recognition gateway.

The one client both consumers install (outremer ADR 0002, #87): the gateway
owns the wire contract, so the client lives next to it and is tested against
the real app in this repository's suite. Consumers pin a commit SHA of this
repository and keep no copy.
"""

from .client import (
    DEFAULT_TIMEOUT,
    OCR_ENGINES,
    AtrClient,
    AtrClientError,
    AtrResult,
    endpoint_for,
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "OCR_ENGINES",
    "AtrClient",
    "AtrClientError",
    "AtrResult",
    "endpoint_for",
]
