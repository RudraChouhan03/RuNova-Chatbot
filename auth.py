from __future__ import annotations

import logging
import os
import time
from typing import Any

import jwt  # type: ignore[import-not-found]
from fastapi import Header, HTTPException  # type: ignore[import-not-found]
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

logger = logging.getLogger(__name__)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
SESSION_TTL = 60 * 60 * 24 * 7  # 7 days

# --- Session secret: fail hard unless explicitly running in dev mode ---
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
if not SESSION_SECRET or SESSION_SECRET == "dev-only-change-me":
    if os.getenv("ALLOW_DEV_LOGIN") == "1":
        SESSION_SECRET = SESSION_SECRET or "dev-only-change-me"  # dev only
    else:
        raise RuntimeError(
            "SESSION_SECRET must be set to a strong random value in production. "
            "Set ALLOW_DEV_LOGIN=1 to explicitly allow the insecure dev default."
        )
elif len(SESSION_SECRET) < 32:
    raise RuntimeError(
        "SESSION_SECRET is too short — use at least 32 random bytes "
        "(e.g. `python -c \"import secrets; print(secrets.token_urlsafe(32))\"`)."
    )

_google_request = google_requests.Request()


def verify_google_token(token: str) -> dict[str, Any]:
    """Verify a Google ID token's signature + audience. Returns claims or raises."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(500, "GOOGLE_CLIENT_ID is not configured.")
    try:
        claims = google_id_token.verify_oauth2_token(
            token, _google_request, GOOGLE_CLIENT_ID
        )
    except Exception as error:
        # Don't leak internal exception text (token fragments, library internals) to the client.
        logger.warning("Google token verification failed: %s", error)
        raise HTTPException(401, "Invalid Google token.")
    if claims.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise HTTPException(401, "Untrusted token issuer.")
    if not claims.get("email_verified", False):
        raise HTTPException(401, "Google email not verified.")
    return claims


def issue_session(claims: dict[str, Any]) -> str:
    """Mint our own short-lived session JWT keyed to the verified Google sub."""
    sub = claims.get("sub")
    if not sub:
        raise HTTPException(401, "Google token missing subject claim.")
    now = int(time.time())
    payload = {
        "sub": f"google:{sub}",
        "email": claims.get("email", ""),
        "name": claims.get("name", claims.get("email", "User")),
        "iat": now,
        "exp": now + SESSION_TTL,
    }
    return jwt.encode(payload, SESSION_SECRET, algorithm="HS256")


def user_from_session(authorization: str = Header(default="")) -> dict[str, Any]:
    """FastAPI dependency: pull + verify our session JWT from the Bearer header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token.")
    token = authorization[7:]
    try:
        return jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Session expired.")
    except Exception:
        raise HTTPException(401, "Invalid session.")