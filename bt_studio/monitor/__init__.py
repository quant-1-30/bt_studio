"""产物监控域: SQLite 索引 + 查询 (被 server 与 visual 共享)。

``MonitorIndex`` 是 server (REST 查询) 与 visual (Streamlit 工作台) 共用的
监控数据模型, 故独立于 ``bt_studio.server`` 放在中立位置, 避免 visual → server
的反向耦合。
"""

from .indexer import MonitorIndex

__all__ = ["MonitorIndex"]
