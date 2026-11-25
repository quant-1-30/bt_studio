#! /usr/bin/env python3

from __future__ import annotations

from typing import Any


def sid_to_str(sid: Any) -> str:
    return sid.decode() if isinstance(sid, (bytes, bytearray)) else str(sid)


def sid_to_bytes(sid: Any) -> bytes:
    return sid.encode() if isinstance(sid, str) else bytes(sid)
