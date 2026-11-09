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
# 自动挂载: 导入插件包时自动将 TA-Lib 白名单算子注册到全局注册表 SAFE_OPS 中
try:
    register_talib_whitelist()
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Auto-registering TA-Lib whitelist failed: {e}")
