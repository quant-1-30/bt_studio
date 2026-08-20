#! /usr/bin/env python3

from __future__ import annotations

import inspect
import logging
import networkx as nx

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .parser import PipelineDag

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
# Dependency evaluation + injection
# --------------------------------------------------------------------------- #

def _resolve_dep_term(term, ran: set, ctx: dict) -> bool:
    if term.cond is not None:
        if term.cond not in ran:
            return False  
        
        actual_val = bool(ctx.get(term.cond))
        
        condition_passed = (not actual_val) if term.negated else actual_val
        
        if not condition_passed:
            return False  

    return term.node in ran


def _deps_satisfied(deps, ran: set, ctx: dict) -> bool:
    for dep in deps:
        if not any(_resolve_dep_term(t, ran, ctx) for t in dep.terms()):
            return False
    return True


def _bind_keys(deps, ran: set, ctx: dict) -> None:

    for dep in deps:
        for t in dep.terms():
            if _resolve_dep_term(t, ran, ctx):
                if t.key:
                    upstream_val = ctx[t.node]
                    
                    if isinstance(upstream_val, dict) and t.key in upstream_val:
                        ctx[t.key] = upstream_val[t.key]
                    elif hasattr(upstream_val, t.key) and not isinstance(upstream_val, (list, str, int, float, bool)):
                        ctx[t.key] = getattr(upstream_val, t.key)
                    else:
                        # Direct alias mapping if the return value is not a dict-like container
                        ctx[t.key] = upstream_val
                
                break

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
            f"DAG only source node 'macro' (fn='node_prepare_macro'): {sources}"
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
            raise ValueError(f"[{dag.name}] train_yms is null and not generate model_id (idx={idx})")
        
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



# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #

def _window_indexes(dag: PipelineDag, yms: list, common_config: dict) -> list[int]:
    train_window = common_config[dag.window.train_key]
    step = common_config[dag.window.step_key]
    n_months = len(yms)

    if n_months < train_window:
        logger.warning(f"[dag] 可用月份数 ({n_months}) 小于训练窗口 ({train_window}) 终止执行")
        return []

    if dag.window.from_ == "last-trainable":
        return [n_months]

    all_window_splits = list(range(train_window, n_months, step))

    last_model_id = common_config.get("last_model_id")
    if last_model_id is not None:
        if last_model_id in yms:
            last_cutoff_idx = yms.index(last_model_id)
            # fixbug train index is idx-1 and cutoff > last_cutoff_idx
            remaining_splits = [idx for idx in all_window_splits if (idx - 1) > last_cutoff_idx]
            logger.info(
                f"[dag] 从 last_model_id={last_model_id} (cutoff_idx={last_cutoff_idx}) 断点恢复"
                f"剩余 {len(remaining_splits)} 个窗口待执行"
            )
            return remaining_splits
        else:
            logger.warning(f"[dag] last_model_id={last_model_id} 不在当前月份列表中，执行全量冷启动")

    return all_window_splits


# --------------------------------------------------------------------------- #
# Invocation
# --------------------------------------------------------------------------- #

def _invoke(fn, ctx: dict, nid: str):
    sig = inspect.signature(fn)
    kwargs = {}
    missing = []

    for pname, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        if pname in ctx:
            kwargs[pname] = ctx[pname]
        # fixbug 
        elif pname == "config":
            kwargs[pname] = ctx.get("common_config", {})
        elif param.default is not inspect.Parameter.empty:
            continue
        else:
            missing.append(pname)

    if missing:
        raise TypeError(
            f"Node {nid!r} (function: {fn.__name__!r}) requires missing positional arguments {missing}, "
            f"which were not found in DAG context. (Available: {sorted(ctx.keys())})"
        )

    return fn(**kwargs)


# --------------------------------------------------------------------------- 
# Runner
# --------------------------------------------------------------------------- 

@consume_time
def run_dag(dag: PipelineDag, exp_config: dict, ast_recipe=None) -> EngineResult:
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
            f"DAG only source node 'macro' (fn='node_prepare_macro'): {sources}"
        )
    macro_id = sources[0]
    global_data = NODE_REGISTRY["node_prepare_macro"](common_config)
    yms = list_available_months(global_data["dret_path"])

    train_window = common_config[dag.window.train_key]
    step = common_config[dag.window.step_key]
    last_model_id = common_config.get("last_model_id")

    # 2. walkforward loop
    for idx in _window_indexes(dag, yms, common_config):
        train_yms = yms[idx - train_window : idx]
        oos_yms = yms[idx : min(idx + step, len(yms))]
        warmup_yms = [train_yms[-1]] if train_yms else []
        prev_oos_yms = train_yms[-step:] if len(train_yms) >= step else train_yms

        if not train_yms:
            raise ValueError(f"[{dag.name}] train_yms is null and not generate model_id (idx={idx})")
        
        model_id = train_yms[-1]  

        oos_range_str = f"{oos_yms[0]}-{oos_yms[-1]}" if oos_yms else "None (实盘/最新)"
        logger.info(f"\n{'=' * 75}\n[{dag.name}] Window Split @ idx={idx} | Model Cutoff={model_id}\n"
                    f"Train={train_yms[0]}-{train_yms[-1]} | OOS={oos_range_str}\n{'=' * 75}\n")

        # Context (注意: prev_model_id 使用当前窗口外层的 last_model_id)
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

        # 3. execute nodes
        for nid in order:
            if nid == macro_id:
                continue

            node_attrs = dag.graph.nodes[nid]
            deps = deps_map[nid]

            if not _deps_satisfied(deps, ran, ctx):
                skipped.add(nid)
                continue

            _bind_keys(deps, ran, ctx)

            role = node_attrs.get("role")
            fn_name = node_attrs.get("fn")
            fn_obj = NODE_REGISTRY[fn_name]
            
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

            # fixbug
            if (nid in ("tune", "update") or fn_name in ("node_tune_monthly", "node_update_fsm_matrix")) and out is not None:
                last_model_id = out
                result.models_produced.append(out)

        result.windows_executed += 1

    result.last_model_id = last_model_id
    return result
