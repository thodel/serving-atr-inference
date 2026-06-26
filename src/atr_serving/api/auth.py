"""Static API-key auth.

The two VMs share a hard-coded key on a private, firewalled network (no TLS).
Clients send it in the ``X-API-Key`` header.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    settings = request.app.state.settings
    if not settings.require_auth:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key",
        )
