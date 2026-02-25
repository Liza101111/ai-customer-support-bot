from __future__ import annotations
import os
from fastapi import Header, HTTPException


def require_admin(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")
) -> None:

    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=500, detail="ADMIN_TOKEN not set")

    if x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")
