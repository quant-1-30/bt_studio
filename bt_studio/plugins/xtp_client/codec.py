from __future__ import annotations

from typing import Any, Optional

import msgpack


EOF = b"eof"


def encode_request(method: str, params: dict[str, Any]) -> bytes:
    return msgpack.packb({"method": method, "params": params}, use_bin_type=True)


def encode_response(result: Any = None, error: Optional[dict] = None) -> bytes:
    payload: dict[str, Any] = {}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    return msgpack.packb(payload, use_bin_type=True)


def decode_payload(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False)
