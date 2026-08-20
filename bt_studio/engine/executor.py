#! /usr/bin/env python3

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import networkx as nx

from bt_studio.tune import (
    node_prepare_macro,
    node_extract_feature_monthly,
    node_check_decay_monthly,
    node_prepare_train_data,
    node_tune_monthly,
    node_update_fsm_matrix,
    node_oos_inference_monthly,
)
from bt_studio.utils.paths import io_setup
from bt_studio.utils.months import list_available_months
from bt_studio.pipeline.utils import consume_time

from .parser import PipelineDag

logger = logging.getLogger(__name__)

__all__ = ["EngineResult", "run_dag", "repr_graph", "NODE_REGISTRY"]

NODE_REGISTRY: Dict[str, object] = {
    f.__name__: f
    for f in (
        node_prepare_macro,
        node_extract_feature_monthly,
        node_check_decay_monthly,
        node_prepare_train_data,
        node_tune_monthly,
        node_update_fsm_matrix,
        node_oos_inference_monthly,
    )
}


@dataclass
class EngineResult:
    dag_name: str
    windows_executed: int = 0
    models_produced: List[int] = field(default_factory=list)
    last_model_id: Optional[int] = None


# --------------------------------------------------------------------------- #
# Graph diagnostics
# --------------------------------------------------------------------------- #

def repr_graph(dag: PipelineDag, ran: set, skipped: set,
               failed: Optional[str] = None) -> str:
    lines = [f"graph[{dag.name}]:"]
    for nid in nx.topological_sort(dag.graph):
        if nid == failed:
            mark = "✗ failed here"
        elif nid in ran:
            mark = "✓ ran"
        elif nid in skipped:
            mark = "⊘ skipped (deps unsatisfied)"
        else:
            mark = "⏸ not reached"
        lines.append(f"  {nid:<16} {mark}")
    return "\n".join(lines)


def _validate_registry(dag: PipelineDag) -> None:
    unknown = [n for n, attrs in dag.graph.nodes(data=True)
               if attrs["fn"] not in NODE_REGISTRY]
    if unknown:
        raise ValueError(
            f"DAG {dag.name!r} references unknown node fns at {unknown}. "
            f"Available: {sorted(NODE_REGISTRY)}")


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #

def _window_indexes(dag: PipelineDag, yms: list, common_config: dict) -> list[int]:
    """
      - train_yms = yms[idx - TRAIN_WINDOW : idx]
      - oos_yms   = yms[idx : idx + STEP]
    idx >= TRAIN_WINDOW and idx < len(yms)
    """
    train_window = common_config[dag.window.train_key]
    step = common_config[dag.window.step_key]
    n_months = len(yms)

    if n_months < train_window:
        logger.warning(
            f"[dag] 可用月份数 ({n_months}) 小于训练窗口 ({train_window})，终止执行"
        )
        return []

    # last-trainable
    if dag.window.from_ == "last-trainable":
        return [n_months]

    # Walk-forward
    all_window_splits = list(range(train_window, n_months, step))

    # Resume --- model_id and metadata 
    last_model_id = common_config.get("last_model_id")
    if last_model_id is not None:
        if last_model_id in yms:
            last_idx = yms.index(last_model_id)
            remaining_splits = [idx for idx in all_window_splits if idx > last_idx]
            print(
                f"[dag] 从 last_model_id={last_model_id} (idx={last_idx}) 断点恢复"
                f"剩余 {len(remaining_splits)} 个窗口待执行"
            )
            return remaining_splits
        else:
            print(f"[dag] last_model_id={last_model_id} 不在当前月份列表中 执行全量冷启动")

    return all_window_splits

# --------------------------------------------------------------------------- #
# Dependency evaluation + injection
# --------------------------------------------------------------------------- #

def _resolve_dep_term(term, ran: set, ctx: dict) -> bool:
    """检查单个 Term 的条件是否满足且上游是否已成功运行"""
    if term.cond is not None:
        if term.cond not in ran:
            return False  # 条件依赖的节点未运行
        
        actual_val = bool(ctx.get(term.cond))
        
        # 如果带了 ! (negated=True)，则期望实际值为 False（not actual_val）
        # 如果没带 ! (negated=False)，则期望实际值为 True（actual_val）
        condition_passed = (not actual_val) if term.negated else actual_val
        
        if not condition_passed:
            return False  # 条件未通过，跳过此依赖项

    return term.node in ran


def _deps_satisfied(deps, ran: set, ctx: dict) -> bool:
    """每个 dep slot 逗号分隔的槽位 至少有一个 term 满足条件并已成功运行"""
    for dep in deps:
        if not any(_resolve_dep_term(t, ran, ctx) for t in dep.terms()):
            return False
    return True


def _bind_keys(deps, ran: set, ctx: dict) -> None:
    """
    精确提取上游返回值中的 key 并绑定进 ctx。
    只有当某个依赖项的条件满足且已运行时，才提取其数据。
    """
    for dep in deps:
        for t in dep.terms():
            # 找到当前 slot 中满足条件且已运行的那个 term
            if _resolve_dep_term(t, ran, ctx):
                if t.key:
                    upstream_val = ctx[t.node]
                    
                    # 1. 如果 upstream_val 是列表或基本类型，并且我们在找 key，说明
                    # 可能是直接将这个返回值重命名为 t.key，特别地，如果 DAG 写法是
                    # node>key 而返回值不是对象字典，那么就是简单的别名重命名。
                    if isinstance(upstream_val, dict) and t.key in upstream_val:
                        ctx[t.key] = upstream_val[t.key]
                    elif hasattr(upstream_val, t.key) and not isinstance(upstream_val, (list, str, int, float, bool)):
                        ctx[t.key] = getattr(upstream_val, t.key)
                    else:
                        # Direct alias mapping if the return value is not a dict-like container
                        ctx[t.key] = upstream_val
                
                # 当前 slot 已经成功绑定了命中的 term，跳出内层循环处理下一个 slot
                break


def _invoke(fn, ctx: dict, nid: str):
    sig = inspect.signature(fn)
    kwargs = {}
    missing = []

    for pname, param in sig.parameters.items():
        # 1. skip *args and **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # 2. ctx inject 
        if pname in ctx:
            kwargs[pname] = ctx[pname]
        # 3. default not in ctx eg.epochs=10
        elif param.default is not inspect.Parameter.empty:
            continue
        # 4. especially skip
        elif pname == "config":
            continue
        # 5. missing 
        else:
            missing.append(pname)

    if missing:
        raise TypeError(
            f"Node {nid!r} (function: {fn.__name__!r}) requires missing positional arguments {missing}, "
            f"which were not found in DAG context. (Available in context: {sorted(ctx.keys())})"
        )

    return fn(**kwargs)

# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

@consume_time
def run_dag(dag: PipelineDag, exp_config: dict, ast_recipe=None) -> EngineResult:
    """
        walkfoward Dag
    """
    _validate_registry(dag)
    common_config = exp_config.get("common_config") or exp_config.get("common_params", {})
    io_setup()

    result = EngineResult(dag_name=dag.name)
    order = list(nx.topological_sort(dag.graph))
    deps_map = {n: attrs.get("deps", ()) for n, attrs in dag.graph.nodes(data=True)}

    # 1. calculate macro global 
    sources = [n for n in order if dag.graph.in_degree(n) == 0]
    if not sources or dag.graph.nodes[sources[0]].get("fn") != "node_prepare_macro":
        raise ValueError(
            f"DAG 必须有且仅有一个 source 节点 'macro' (fn='node_prepare_macro'), 实际获得: {sources}"
        )
    macro_id = sources[0]
    global_data = NODE_REGISTRY["node_prepare_macro"](common_config)
    yms = list_available_months(global_data["dret_path"])

    train_window = common_config[dag.window.train_key]
    step = common_config[dag.window.step_key]
    last_model_id = common_config.get("last_model_id")

    # 2. walkforward loop
    for idx in _window_indexes(dag, yms, common_config):
        # yms[n_months - train_window : n_months]
        train_yms = yms[idx - train_window : idx]
        oos_yms = yms[idx : min(idx + step, len(yms))]
        warmup_yms = [train_yms[-1]] if train_yms else []
        prev_oos_yms = train_yms[-step:] if len(train_yms) >= step else train_yms

        # Information Cutoff 
        if not train_yms:
            raise ValueError(f"[{dag.name}] train_yms 为空，无法生成有效的 model_id (idx={idx})")
        
        model_id = train_yms[-1]  

        oos_range_str = f"{oos_yms[0]}-{oos_yms[-1]}" if oos_yms else "None (实盘/最新)"
        print(f"\n{'=' * 75}\n[{dag.name}] Window Split @ idx={idx} | Model Cutoff={model_id}\n"
            f"Train={train_yms[0]}-{train_yms[-1]} | OOS={oos_range_str}\n{'=' * 75}\n")

        # Context
        ctx: dict[str, Any] = {
            "global_data": global_data,
            "dret_path": global_data.get("dret_path"),
            "universe_sids": global_data.get("universe_sids"),
            "common_config": common_config,
            "exp_config": exp_config,
            "ast_recipe": ast_recipe,
            "train_yms": train_yms,
            "oos_yms": oos_yms,
            "warmup_yms": warmup_yms,
            "prev_oos_yms": prev_oos_yms,
            "model_id": model_id,
            "prev_model_id": last_model_id,
        }
        ctx[macro_id] = global_data
        ran: set[str] = {macro_id}
        skipped: set[str] = set()

        # 5. execute
        for nid in order:
            if nid == macro_id: # incase duplicate calculate
                continue

            node_attrs = dag.graph.nodes[nid]
            deps = deps_map[nid]

            if not _deps_satisfied(deps, ran, ctx):
                skipped.add(nid)
                continue

            _bind_keys(deps, ran, ctx)

            role = node_attrs.get("role")
            fn_obj = NODE_REGISTRY[node_attrs["fn"]]
            if role and "ymonths" in inspect.signature(fn_obj).parameters:
                role_months_map = {
                    "train": train_yms,
                    "oos": oos_yms,
                    "warmup": warmup_yms,
                    "prev_oos": prev_oos_yms,
                }
                ctx["ymonths"] = role_months_map.get(role, train_yms)

            try:
                out = _invoke(fn_obj, ctx, nid)
            except Exception as e:
                logger.error(f"Execution failed at node {nid!r} in window {model_id}")
                raise RuntimeError(
                    f"Node {nid!r} failed: {e!r}\n"
                    + repr_graph(dag, ran, skipped, failed=nid)
                ) from e

            ran.add(nid)
            ctx[nid] = out

            if nid in ("tune", "update") and out is not None:
                last_model_id = out
                result.models_produced.append(out)

        result.windows_executed += 1

    result.last_model_id = last_model_id
    return result


import atexit
import ray
import threading

# ensure_ray threadsafe
_RAY_INIT_LOCK = threading.Lock()

# CPU Thrashing
_RUNTIME_ENV = {
    "env_vars": {
        "POLARS_MAX_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    },
}


def ensure_ray(
    num_cpus: Optional[int] = None,
    object_store_memory_bytes: Optional[int] = None,
    include_dashboard: bool = False,
    logging_level: int = logging.WARNING,
) -> None:
    """
    Ray no-op 
    
    Args:
        num_cpus: default os.cpu_count
        object_store_memory_bytes: Plasma object (byte), default 30% RAM
        include_dashboard: Web Dashboard 
        logging_level: int
    """
    import os
    if ray.is_initialized():
        return

    with _RAY_INIT_LOCK:
        # Double-Checked Locking
        if ray.is_initialized():
            return

        cpus = num_cpus or max(1, os.cpu_count() or 1)

        init_kwargs: Dict[str, Any] = {
            "num_cpus": cpus,
            "runtime_env": _RUNTIME_ENV,
            "ignore_reinit_error": True,
            "include_dashboard": include_dashboard,
            "logging_level": logging_level,
        }

        if object_store_memory_bytes is not None:
            init_kwargs["_memory"] = object_store_memory_bytes

        try:
            ray.init(**init_kwargs)
            logger.info(
                f"[Ray] 本地集群初始化成功: num_cpus={cpus}, "
                f"dashboard={include_dashboard}"
            )
        except Exception as e:
            logger.error(f"[Ray] 集群初始化失败: {e}")
            raise


def is_ray_ready() -> bool:
    return ray.is_initialized()


def get_ray_cluster_resources() -> Dict[str, float]:
    if not ray.is_initialized():
        return {}
    return ray.cluster_resources()


def _shutdown() -> None:
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("[Ray] 本地集群已安全关闭。")
    except Exception:
        pass


atexit.register(_shutdown)
