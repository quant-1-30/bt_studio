"""
bt_studio API Server (thin results gateway)

提供 RESTful API 和 WebSocket 接口,支持跨平台客户端访问。

主要功能:
- 常驻轮询 result 目录 (models / scores / collapse / llm_runs / features)
- 分析新增产物并推送实时事件到 iOS / Qt 客户端
- 产物注册表查询 (REST) 与实时快照 (WebSocket)

执行类端点 (HPO/inference 提交) 已移除: 编排逻辑上移至核心包
``bt_studio.engine``, 脚本经进程内 get_task_manager() 直连;
本服务与核心进程以文件系统为唯一契约。

启动方式:
    uvicorn bt_studio.plugins.api_server.main:app --host 0.0.0.0 --port 8000
"""

from .main import app, get_application

__all__ = ["app", "get_application"]