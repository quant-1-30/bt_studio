from __future__ import annotations

import copy
from typing import Any, Dict, Optional

# 透传至 exp_config 顶层的控制字段白名单
_CONTROL_KEYS = frozenset({"run_name", "precomputed_features", "ast_recipe", "feature_col"})


def _deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    轻量且安全的递归合并：将 override 递归合并至 base，返回全新的独立字典。
    """
    if not override:
        return copy.deepcopy(base)

    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
# Production baseline — 通用实验基线参数 (禁止分叉)
# ---------------------------------------------------------------------------

PRODUCTION_COMMON_PARAMS: Dict[str, Any] = {
    # 样本区间与基准
    "start_date": 20100101,
    "end_date": 20201231,
    "benchmark": "1A0001",

    # 样本过滤
    "top_k_ratio": 0.25,          # 选股分位数比例
    "days_since_ipo": 120,        # 次新股剔除天数

    # Walk-forward 训练/滚动窗口
    "train_window": 12,
    "oss_step": 6,

    # 市场状态过滤
    "regime_filter": {"ma_window": 20},

    # T+1 未来收益目标
    "T1_rets": {
        "open_5m": 5,
        "open_15m": 15,
        "open_30m": 30,
    },
    "decay_minutes": 15,          # 分钟衰减权重

    # 微观结构 / OFI 权重
    "eps": 1e-8,
    "min_factor_weight": 0.05,

    # 模式匹配与曲线重叠设置 (Stumpy / DTW)
    "exclude_bars": 10,           # 尾盘剔除 bar 数 (14:50 之后)
    "dtw_window_frac": 0.1,

    # 宏观排名与触发阈值
    "ranking_window": 5,
    "ranking_ratio": 0.25,
    "topk": 5,
    "trigger": 30,

    # 统计显著性
    "alternative": "greater",
    "u_pval": 0.10,

    # 并发与可复现性
    "num_workers": 12,
    "n_startup_trials": 40,
    "num_trials": 500,            # 移至通用参数：优化预算
    "seed": 42,
    "storage_path": "/tmp/ray_results",
}

# 超参数搜索空间定义（严格遵循 key -> [bounds / discrete choices]）
PRODUCTION_SEARCH_BOUNDS: Dict[str, Any] = {
    "downsample": [3, 4, 5],              # 降采样倍率 (离散网格)
    "motif_minutes": [30, 45, 60, 90],    # 模式窗口长度 (离散网格)
    "threshold_r": [0.65, 0.85],          # 相似度相关系数区间 [min, max]
}


def build_exp_config(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    基于生产基准构建完整的实验配置 (exp_config)。

    Parameters
    ----------
    user_config : Optional[Dict[str, Any]]
        用户覆盖配置，支持包含:
        - common_params: 覆盖通用运行参数
        - search_bounds: 覆盖超参数搜索空间
        - _CONTROL_KEYS 中的控制字段 (run_name, ast_recipe 等)

    Returns
    -------
    Dict[str, Any]
        深拷贝隔离的完整配置字典。
    """
    if user_config is None:
        user_config = {}
    elif not isinstance(user_config, dict):
        raise TypeError(f"build_exp_config: user_config 必须为 dict 或 None, 收到 {type(user_config).__name__}")

    # 1. 严格白名单校验
    illegal = [
        k for k in user_config
        if k not in ("common_params", "search_bounds") and k not in _CONTROL_KEYS
    ]
    if illegal:
        raise ValueError(
            f"build_exp_config: 检测到未知的顶层配置键 {illegal}。\n"
            f"允许的键: 'common_params', 'search_bounds', 以及控制字段 {sorted(_CONTROL_KEYS)}。\n"
            "通用运行参数请放入 common_params，超参搜索范围请放入 search_bounds。"
        )

    # 2. 提取与类型宽容校验
    common = user_config.get("common_params") or {}
    search = user_config.get("search_bounds") or {}

    if not isinstance(common, dict) or not isinstance(search, dict):
        raise TypeError("common_params 与 search_bounds 必须为 dict 类型")

    # 3. 递归合并基准配置
    exp_config: Dict[str, Any] = {
        "common_params": _deep_merge(PRODUCTION_COMMON_PARAMS, common),
        "search_bounds": _deep_merge(PRODUCTION_SEARCH_BOUNDS, search),
    }

    # 4. 透传控制字段
    for key in _CONTROL_KEYS:
        if key in user_config and user_config[key] is not None:
            exp_config[key] = copy.deepcopy(user_config[key])

    return exp_config
