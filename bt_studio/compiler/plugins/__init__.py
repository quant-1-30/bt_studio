"""Agent operator plugins.

Importing this package mounts every bundled plugin into the agent core:
- ``talib_hook``: registers the TA-Lib whitelist into ``SAFE_OPS`` (compile-
  time gated on talib availability) and exposes ``talib_validate_hook``.
"""

from . import talib_hook
from .talib_hook import (
    register_talib_whitelist,
    make_talib_fn,
    talib_validate_hook,
)

__all__ = ["talib_hook", "register_talib_whitelist",
           "make_talib_fn", "talib_validate_hook"]