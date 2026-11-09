#! /usr/bin/env python3
"""DAG execution engine."""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx

from bt_studio.constant import DEFAULT_WARMUP_MONTHS, FEATURE_DIR, TASK_DIR_KEYS, TUNE_BASE_DIR
from bt_studio.default_config import common_params_of, resolve_feature_col, search_spec_of
from bt_studio.schema import EngineResult, PipelineDag
from bt_studio.tune import (
    node_check_decay_monthly,
    node_extract_feature_monthly,
    node_oos_inference_monthly,
    node_prepare_macro,
    node_prepare_train,
    node_tune_monthly,
    node_update_fsm_matrix,
)
from bt_studio.utils.io import consume_time, io_setup, read_json_safe, sha1_hex, slugify
from bt_studio.utils.months import retrieve_available_months
from .ctx import resolve_platform_concurrency

logger = logging.getLogger(__name__)

SignCacheEntry = Tuple[Callable[..., Any], List[str], Set[str]]

__all__ = [
    "EngineResult",
    "run_dag",
    "repr_graph",
    "NODE_REGISTRY",
    "attach_task_dirs",
    "resolve_task_id",
    "window_artifacts_complete",
]

NODE_REGISTRY: Dict[str, Callable[..., Any]] = {
    f.__name__: f
    for f in (
        node_prepare_macro,
        node_extract_feature_monthly,
        node_check_decay_monthly,
        node_prepare_train,
        node_tune_monthly,
        node_update_fsm_matrix,
        node_oos_inference_monthly,
    )
}


def resolve_task_id(
    exp_config: Dict[str, Any],
    ast_recipe: Any = None,
) -> Optional[str]:
    common = common_params_of(exp_config)
    explicit = (
        exp_config.get("task_id")
        or exp_config.get("run_name")
        or common.get("task_id")
        or common.get("run_name")
    )
    if explicit:
        return slugify(explicit)

    feature_col = exp_config.get("feature_col") or common.get("feature_col")
    if not feature_col:
        return None

    artifact_common_keys = (
        "start_date", "end_date", "benchmark", "warmup_months",
        "days_since_ipo", "min_active_ratio", "min_amount_rank",
        "train_window", "oos_step", "T1_rets",
        "decay_minutes", "decay_eval_months", "eps", "exclude_bars",
        "dtw_window_frac", "ranking_window", "topk", "trigger",
        "alternative", "u_pval", "min_excess_ret",
        "min_curve_valid_ratio", "positive_motif_quantile", "max_t1_missing_ratio",
    )
    artifact_common = {k: common.get(k) for k in artifact_common_keys if k in common}

    digest = sha1_hex(
        {
            "feature_col": feature_col,
            "ast_recipe": ast_recipe if ast_recipe is not None else exp_config.get("ast_recipe"),
            "common": artifact_common,
            "search_spec": search_spec_of(exp_config),
        },
        length=8,
    )
    return f"{slugify(feature_col)}-{digest}"


def attach_task_dirs(
    exp_config: Dict[str, Any],
    ast_recipe: Any = None,
    common_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    task_id = resolve_task_id(exp_config, ast_recipe)
    if task_id is None:
        return None

    cc = common_config if common_config is not None else exp_config.setdefault("common_params", {})
    cc.setdefault("_task_id", task_id)
    cc.setdefault("feature_dir", os.path.join(FEATURE_DIR, task_id))
    tune_task_root = os.path.join(TUNE_BASE_DIR, task_id)
    cc.setdefault("tune_base_dir", tune_task_root)
    cc.setdefault("tune_model_dir", os.path.join(tune_task_root, "models"))
    cc.setdefault("tune_score_dir", os.path.join(tune_task_root, "scores"))
    cc.setdefault("tune_collapse_dir", os.path.join(tune_task_root, "collapse"))
    return task_id


def sync_task_dirs_to_exp_config(
    exp_config: Dict[str, Any], common_config: Dict[str, Any]
) -> None:
    exp_config.setdefault("common_params", {}).update(
        {k: common_config[k] for k in TASK_DIR_KEYS if k in common_config}
    )


def window_artifacts_complete(model_id: int, common_config: Dict[str, Any]) -> bool:
    model_dir = common_config.get("tune_model_dir")
    score_dir = common_config.get("tune_score_dir")
    if not model_dir or not score_dir:
        return True
    model_path = os.path.join(model_dir, f"model_{model_id}.pkl")
    score_path = os.path.join(score_dir, f"scores_{model_id}.parquet")
    has_model = os.path.exists(model_path)
    if has_model and not os.path.exists(score_path):
        return False
    if not has_model and os.path.exists(model_path[:-4] + ".manifest.json"):
        return False
    return True


def analyze_signature(fn: Callable[..., Any]) -> SignCacheEntry:
    """Extract required and optional parameter names for a DAG node function."""
    sig = inspect.signature(fn)
    req: List[str] = []
    all_p: Set[str] = set()
    for pname, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        all_p.add(pname)
        if param.default is inspect.Parameter.empty:
            req.append(pname)
    return fn, req, all_p


_SIGNATURE_CACHE: Dict[str, SignCacheEntry] = {
    _name: analyze_signature(_func) for _name, _func in NODE_REGISTRY.items()
}


def _resolve_dep_term(term: Any, ran: Set[str], ctx: Dict[str, Any]) -> bool:
    if term.cond is not None:
        if term.cond not in ran:
            return False
        actual_val = bool(ctx.get(term.cond))
        if ((not actual_val) if term.negated else actual_val) is False:
            return False
    return term.node in ran


def deps_satisfied(deps: Any, ran: Set[str], ctx: Dict[str, Any]) -> bool:
    """Return True if all conjunctive dependencies have at least one satisfied branch."""
    return all(any(_resolve_dep_term(t, ran, ctx) for t in dep.terms()) for dep in deps)


def bind_dep_keys(deps: Any, ran: Set[str], ctx: Dict[str, Any]) -> None:
    """Bind satisfied upstream node outputs into the execution context dictionary."""
    for dep in deps:
        for t in dep.terms():
            if _resolve_dep_term(t, ran, ctx):
                if t.key:
                    upstream_val = ctx[t.node]
                    if isinstance(upstream_val, dict) and t.key in upstream_val:
                        ctx[t.key] = upstream_val[t.key]
                    elif hasattr(upstream_val, t.key) and not isinstance(
                        upstream_val, (list, str, int, float, bool)
                    ):
                        ctx[t.key] = getattr(upstream_val, t.key)
                    else:
                        ctx[t.key] = upstream_val
                break


def repr_graph(
    dag: PipelineDag,
    ran: Set[str],
    skipped: Set[str],
    failed: Optional[str] = None,
) -> str:
    """Format a human-readable execution trace of a PipelineDag."""
    lines = [f"graph[{dag.name}]:"]
    for nid in nx.topological_sort(dag.graph):
        mark = (
            "✗ failed here" if nid == failed
            else "✓ ran" if nid in ran
            else "⊘ skipped (deps unsatisfied)" if nid in skipped
            else "⏸ not reached"
        )
        lines.append(f"  {nid:<16} {mark}")
    return "\n".join(lines)


def resolve_window_indexes(
    dag: PipelineDag, yms: List[int], common_config: Dict[str, Any]
) -> List[int]:
    """Compute walk-forward window split indices with resume watermark filtering."""
    train_window = int(common_config[dag.window.train_key])
    step = int(common_config[dag.window.step_key])
    n_months = len(yms)

    if n_months < train_window:
        raise ValueError(f"[{dag.name}] available months ({n_months}) < train_window={train_window}")

    all_window_splits = (
        [n_months]
        if dag.window.from_ == "last-trainable"
        else list(range(train_window, n_months, step))
    )

    last_model_id = common_config.get("last_model_id")
    if last_model_id is not None:
        last_model_id_int = int(last_model_id)
        if last_model_id_int in yms:
            last_cutoff_idx = yms.index(last_model_id_int)
            return [
                idx for idx in all_window_splits
                if (idx - 1) > last_cutoff_idx
                or not window_artifacts_complete(yms[idx - 1], common_config)
            ]
    return all_window_splits


def _validate_registry(dag: PipelineDag) -> None:
    unknown = [n for n, attrs in dag.graph.nodes(data=True) if attrs.get("fn") not in NODE_REGISTRY]
    if unknown:
        raise ValueError(
            f"DAG {dag.name!r} references unknown node fns at {unknown}. "
            f"Available: {sorted(NODE_REGISTRY)}"
        )


def _invoke(fn_name: str, fn: Callable[..., Any], ctx: Dict[str, Any], nid: str) -> Any:
    cached = _SIGNATURE_CACHE.get(fn_name)
    if cached is None or cached[0] is not fn:
        cached = analyze_signature(fn)
        _SIGNATURE_CACHE[fn_name] = cached
    _, required_params, all_params = cached

    missing = [p for p in required_params if p not in ctx]
    if missing:
        raise TypeError(
            f"Node {nid!r} ({fn_name}) missing required params: {missing}. "
            f"(Available context keys: {sorted(ctx.keys())})"
        )

    kwargs = {k: ctx[k] for k in all_params if k in ctx}
    return fn(**kwargs)


def _execute_window_nodes(
    dag: PipelineDag,
    order: List[str],
    deps_map: Dict[str, Any],
    macro_id: str,
    ctx: Dict[str, Any],
    role_ymonths: Dict[str, List[int]],
    result: EngineResult,
    last_model_id: Any,
) -> Any:
    ran: Set[str] = {macro_id}
    skipped: Set[str] = set()
    model_id = ctx["model_id"]

    for nid in order:
        if nid == macro_id:
            continue

        node_attrs = dag.graph.nodes[nid]
        deps = deps_map[nid]

        if not deps_satisfied(deps, ran, ctx):
            skipped.add(nid)
            continue

        bind_dep_keys(deps, ran, ctx)

        role = node_attrs.get("role")
        fn_name = node_attrs.get("fn")
        fn_obj = NODE_REGISTRY[fn_name]
        cached_sig = _SIGNATURE_CACHE.get(fn_name)
        if cached_sig is None or cached_sig[0] is not fn_obj:
            cached_sig = analyze_signature(fn_obj)
            _SIGNATURE_CACHE[fn_name] = cached_sig
        all_params = cached_sig[2]

        if role and "ymonths" in all_params:
            ctx["ymonths"] = role_ymonths[role]

        try:
            out = _invoke(fn_name, fn_obj, ctx, nid)
        except Exception as e:
            logger.error(f"Execution failed at node {nid!r} in window {model_id}")
            raise RuntimeError(
                f"Node {nid!r} failed: {e!r}\n" + repr_graph(dag, ran, skipped, failed=nid)
            ) from e

        ran.add(nid)
        ctx[nid] = out

        output_key = node_attrs.get("output")
        if output_key and out is not None:
            ctx[output_key] = out[output_key] if (isinstance(out, dict) and output_key in out) else out
            if output_key == "model_id":
                last_model_id = ctx[output_key]
                result.models_produced.append(last_model_id)

    return last_model_id


@consume_time
def run_dag(dag: PipelineDag, exp_config: Dict[str, Any], ast_recipe: Any = None) -> EngineResult:
    _validate_registry(dag)

    common_config = dict(common_params_of(exp_config))
    hw_conc = resolve_platform_concurrency()
    common_config.setdefault("num_workers", hw_conc["num_workers"])
    common_config.setdefault("tick_batch_sids", hw_conc["tick_batch_sids"])
    common_config["feature_col"] = resolve_feature_col(
        common_config, ast_recipe, exp_config.get("feature_col")
    )
    exp_config.setdefault("common_params", {})["feature_col"] = common_config["feature_col"]

    task_id = attach_task_dirs(exp_config, ast_recipe, common_config=common_config)
    if task_id:
        sync_task_dirs_to_exp_config(exp_config, common_config)
        logger.info(f"[{dag.name}] task_id={task_id} -> artifact directories isolated")

    io_setup()

    result = EngineResult(dag_name=dag.name, task_id=task_id)
    order = list(nx.topological_sort(dag.graph))
    deps_map = {n: attrs.get("deps", ()) for n, attrs in dag.graph.nodes(data=True)}

    sources = [n for n in order if dag.graph.in_degree(n) == 0]
    if len(sources) != 1:
        raise ValueError(
            f"DAG must start with a single root source node (in_degree == 0), got: {sources}"
        )

    macro_id = sources[0]
    source_fn_name = dag.graph.nodes[macro_id].get("fn")
    global_data = _invoke(
        source_fn_name, NODE_REGISTRY[source_fn_name], {"common_config": common_config}, macro_id
    )

    raw_yms = retrieve_available_months(global_data["dret_path"])
    yms = sorted(list({int(m) for m in raw_yms}))

    train_window = int(common_config[dag.window.train_key])
    step = int(common_config[dag.window.step_key])
    warmup_n = int(common_config.get("warmup_months", DEFAULT_WARMUP_MONTHS))
    last_model_id = common_config.get("last_model_id")

    for idx in resolve_window_indexes(dag, yms, common_config):
        train_yms = yms[idx - train_window : idx]
        oos_yms = yms[idx : min(idx + step, len(yms))]
        prev_oos_yms = yms[max(0, idx - step) : idx]

        if len(train_yms) < train_window:
            raise ValueError(
                f"[{dag.name}] train slice ({len(train_yms)}) < train_window ({train_window}) at idx={idx}"
            )

        model_id = train_yms[-1]
        ctx: Dict[str, Any] = {
            "global_data": global_data,
            "dret_path": global_data.get("dret_path"),
            "universe_sids": global_data.get("universe_sids"),
            "common_config": common_config,
            "exp_config": exp_config,
            "ast_recipe": ast_recipe,
            "train_yms": train_yms,
            "oos_yms": oos_yms,
            "prev_oos_yms": prev_oos_yms,
            "model_id": model_id,
            "prev_model_id": last_model_id,
            macro_id: global_data,
        }
        role_ymonths = {
            "train": train_yms,
            "oos": oos_yms,
            "prev_oos": prev_oos_yms,
            "warmup": yms[max(0, idx - train_window - warmup_n) : idx - train_window],
        }
        last_model_id = _execute_window_nodes(
            dag, order, deps_map, macro_id, ctx, role_ymonths, result, last_model_id
        )

        collapse_dir = common_config.get("tune_collapse_dir")
        feat_col = common_config.get("feature_col")
        if collapse_dir and feat_col:
            c_path = os.path.join(str(collapse_dir), f"collapse_{model_id}_{feat_col}.json")
            c_data = read_json_safe(c_path)
            if isinstance(c_data, dict):
                result.collapse_report = {
                    "verdict": c_data.get("verdict", "unknown"),
                    "report_path": c_path,
                }

        result.windows_executed += 1

    result.last_model_id = last_model_id
    return result
