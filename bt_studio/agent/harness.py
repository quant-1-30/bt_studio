#! /usr/bin/env python3
"""Two-stage feature evaluation harness: AST compilation -> Stage 1 physical prefilter -> Stage 2 walk-forward grid evaluation."""

from __future__ import annotations

import itertools
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import polars as pl

from bt_studio.compiler.ast import (
    ASTCompilationError,
    ast_fingerprint,
    compile_ast,
    compile_recipe,
    rewrite_ast_col_refs,
)
from bt_studio.constant import (
    BASE_MARKET_COLS,
    DEFAULT_GRID,
    EVAL_CACHE_ROOT,
    EVAL_LOGIC_VERSION,
    FEATURE_DIR,
    MIN_WINDOW_PASS_RATIO,
    PREFILTER_THRESHOLDS,
)
from bt_studio.pipeline.patterns import discover_fsm_pattern
from bt_studio.pipeline.preprocess.panel import build_static_panel, extract_curves_from_panel
from bt_studio.schema import (
    EvalResult,
    FeatureMiningResult,
    PrefilterResult,
    WindowReplayState,
)
from bt_studio.utils.io import (
    atomic_save_parquet,
    build_eval_cache_locator,
    compute_eval_data_fingerprint,
    eval_cache_load,
    eval_cache_save,
    guard_disk_watermark,
)
from bt_studio.utils.math_func import group_combos_by_downsample, resolve_search_hyperparams
from bt_studio.utils.months import month_id_expr, parse_ym_from_path

_evaluate_physical_stats = PrefilterResult.from_stats
aggregate_candidate_result = FeatureMiningResult.from_candidate_windows


# ---------------------------------------------------------------------------
# Stage 1: Physical Prefilter
# ---------------------------------------------------------------------------

def run_stage1_prefilter(
    hf_enhanced: pl.LazyFrame,
    compiled_names: List[str],
    common_config: dict,
    prefilter_thresholds: Optional[dict] = None,
) -> Tuple[Dict[str, PrefilterResult], List[str]]:
    """Run single-pass streaming aggregation for Stage 1 dead-water and null-ratio filtering."""
    if not compiled_names:
        return {}, []

    print(f"[Harness] Stage 1: physical prefilter ({len(compiled_names)} features) via Lazy Agg...")
    # Strictly causal Stage 1 physical prefilter: evaluate std/null_ratio ONLY on the earliest
    # available month (initial window) so future walk-forward months never leak into candidate survival.
    stats_lf = hf_enhanced
    try:
        if "day" in hf_enhanced.collect_schema().names():
            first_ym = (
                hf_enhanced.select(month_id_expr("day").min().alias("_min_ym"))
                .collect(engine="streaming")["_min_ym"][0]
            )
            if first_ym is not None:
                stats_lf = hf_enhanced.filter(month_id_expr("day") == int(first_ym))
    except Exception:
        stats_lf = hf_enhanced

    def _stats_row(names: List[str]) -> dict:
        exprs = [pl.len().alias("_total_len")]
        for n in names:
            exprs += [
                pl.col(n).null_count().alias(f"{n}_null"),
                pl.col(n).std().alias(f"{n}_std"),
            ]
        return stats_lf.select(exprs).collect(engine="streaming").row(0, named=True)

    stage1_map: Dict[str, PrefilterResult] = {}
    try:
        row = _stats_row(compiled_names)
        stage1_map = {
            n: _evaluate_physical_stats(n, row, row["_total_len"], prefilter_thresholds)
            for n in compiled_names
        }
    except Exception as bulk_err:
        print(f"  [Stage1] bulk agg failed ({bulk_err!r}); fallback to candidate isolation")
        for n in compiled_names:
            try:
                r = _stats_row([n])
                stage1_map[n] = _evaluate_physical_stats(n, r, r["_total_len"], prefilter_thresholds)
            except Exception as e:
                stage1_map[n] = PrefilterResult(
                    feature_name=n, passed=False, reason=f"Stage1 runtime: {str(e)[:160]}"
                )

    for n in compiled_names:
        print(f"  {'PASS' if stage1_map[n].passed else 'REJECT'} {stage1_map[n].to_summary()}")

    survivors = [n for n in compiled_names if stage1_map[n].passed]
    print(f"  Stage 1 summary: {len(survivors)}/{len(compiled_names)} passed")
    return stage1_map, survivors


# ---------------------------------------------------------------------------
# Stage 2: Grid Evaluation & Multiprocess Window Workers
# ---------------------------------------------------------------------------

def evaluate_feature(
    panel_lf: pl.LazyFrame,
    feature_col: str,
    common_config: dict,
    cfg: dict,
) -> EvalResult:
    t0 = time.perf_counter()
    default_tune_cfg: dict = {}

    try:
        default_tune_cfg = resolve_search_hyperparams(cfg)
        local_common = dict(common_config, feature_col=feature_col)
        result = discover_fsm_pattern(panel_lf, default_tune_cfg, local_common)
        elapsed = time.perf_counter() - t0

        if result and result.get("status") == "success":
            raw_score = result.get("metrics_score")
            raw_pval = result.get("u_pval")
            return EvalResult(
                feature_name=feature_col,
                status="success",
                score=float(raw_score) if raw_score is not None else -500.0,
                u_pval=float(raw_pval) if raw_pval is not None else 1.0,
                trigger_count=int(result.get("trigger_count") or 0),
                valid_sample_ratio=float(result.get("valid_sample_ratio") or 0.0),
                autocorr=float(result.get("autocorr") or 0.0),
                elapsed_s=elapsed,
                fsm_matrix=result.get("fsm_matrix"),
                learned_motif=result.get("learned_motif"),
                config=default_tune_cfg,
            )

        return EvalResult(
            feature_name=feature_col,
            status="failed",
            reason=str(result.get("reason", "unknown") if result else "empty_result"),
            elapsed_s=elapsed,
            config=default_tune_cfg,
        )
    except Exception as exc:
        return EvalResult(
            feature_name=feature_col,
            status="failed",
            reason=f"Exception: {exc!r}",
            elapsed_s=time.perf_counter() - t0,
            config=default_tune_cfg,
        )


def eval_feature_on_grid(
    window_lf: pl.LazyFrame,
    dret_lf: pl.LazyFrame,
    static_df: pl.DataFrame,
    feat: str,
    common_config: dict,
    combos_by_ds: Dict[int, List[dict]],
) -> EvalResult:
    max_pval_gate = float(common_config.get("stage2_max_pval", 0.1))
    panel_cache: Dict[int, pl.LazyFrame] = {}
    best_fallback: Optional[EvalResult] = None
    last_failed: Optional[EvalResult] = None

    for ds, ds_combos in combos_by_ds.items():
        if ds not in panel_cache:
            local_cc = dict(common_config, feature_col=feat)
            curve_lf = extract_curves_from_panel(window_lf, {"downsample": ds}, local_cc)
            panel_cache[ds] = curve_lf.join(static_df.lazy(), on=["day", "sid"], how="inner")
        panel_lf = panel_cache[ds]
        for combo in ds_combos:
            r = evaluate_feature(panel_lf, feat, common_config, combo)
            if r.is_success:
                if r.u_pval <= max_pval_gate:
                    return r
                if best_fallback is None or r.u_pval < best_fallback.u_pval:
                    best_fallback = r
            else:
                last_failed = r

    return best_fallback or last_failed or EvalResult(feature_name=feat, status="failed", reason="empty_grid")


def _eval_window_worker(
    dret_path: str,
    feat_month_paths: Dict[str, str],
    feature_names: List[str],
    common_config: dict,
    combos: List[dict],
) -> Dict[str, EvalResult]:
    first_path = next(iter(feat_month_paths.values()), None)
    if first_path is None:
        return {
            f: EvalResult(feature_name=f, status="failed", reason="missing_parquet")
            for f in feature_names
        }
    dret_lf = pl.scan_parquet(dret_path)
    static_df = build_static_panel(
        pl.scan_parquet(first_path), dret_lf, common_config, is_train=True
    ).collect(engine="streaming")
    combos_by_ds = group_combos_by_downsample(combos)

    out: Dict[str, EvalResult] = {}
    for feat in feature_names:
        f_path = feat_month_paths.get(feat)
        if not f_path:
            out[feat] = EvalResult(feature_name=feat, status="failed", reason="missing_parquet")
        else:
            out[feat] = eval_feature_on_grid(
                pl.scan_parquet(f_path), dret_lf, static_df, feat, common_config, combos_by_ds
            )
    return out


# ---------------------------------------------------------------------------
# Main Harness Pipeline
# ---------------------------------------------------------------------------

class TwoStageAgentHarness:
    """AST Compilation -> Stage 1 Physical Prefilter -> Stage 2 Walk-Forward Evaluation."""

    def __init__(
        self,
        common_config: dict,
        grid_config: Optional[dict] = None,
        prefilter_thresholds: Optional[dict] = None,
        stage3_callback: Optional[Any] = None,
    ):
        self.common_config = common_config
        self.grid_config = grid_config if grid_config is not None else DEFAULT_GRID
        self.prefilter_thresholds = prefilter_thresholds
        self.stage3_callback = stage3_callback
        self._static_cache: Dict[Tuple[str, int], pl.DataFrame] = {}

    def _compile_asts(
        self,
        candidate_asts: List[Union[dict, list[dict]]],
        hf_lf: pl.LazyFrame,
    ) -> Tuple[pl.LazyFrame, List[dict], List[str]]:
        records: List[dict] = []
        compiled_names: List[str] = []
        seen_names: Set[str] = set()

        for idx, candidate in enumerate(candidate_asts):
            fp = ast_fingerprint(candidate)
            try:
                if isinstance(candidate, list):
                    renames: dict = {}
                    isolated_recipe: list[dict] = []
                    for step in candidate:
                        step_copy = dict(step)
                        orig_name = step["name"]
                        new_name = f"_c{idx}_{orig_name}"
                        renames[orig_name] = new_name
                        step_copy["name"] = new_name
                        step_copy["ast"] = rewrite_ast_col_refs(step["ast"], renames)
                        isolated_recipe.append(step_copy)

                    for e in compile_recipe(isolated_recipe, check_causal=True):
                        hf_lf = hf_lf.with_columns(e)
                    raw_name = candidate[-1]["name"]
                    iso_last = f"_c{idx}_{raw_name}"
                    feat_name = f"f_{fp[:12]}"
                    if feat_name != iso_last:
                        hf_lf = hf_lf.rename({iso_last: feat_name})
                elif isinstance(candidate, dict):
                    expr, raw_name = compile_ast(candidate, check_causal=True)
                    feat_name = f"f_{fp[:12]}"
                    hf_lf = hf_lf.with_columns(expr.alias(feat_name))
                else:
                    raise ASTCompilationError(f"Unsupported candidate type: {type(candidate)}")

                if feat_name in seen_names:
                    records.append({
                        "raw_ast": candidate, "name": feat_name, "display_name": raw_name,
                        "fingerprint": fp, "error": "Duplicate feature name in batch",
                    })
                    continue

                seen_names.add(feat_name)
                compiled_names.append(feat_name)
                records.append({
                    "raw_ast": candidate, "name": feat_name, "display_name": raw_name,
                    "fingerprint": fp, "error": None,
                })
            except Exception as e:
                records.append({
                    "raw_ast": candidate, "name": f"invalid_{idx}", "display_name": f"invalid_{idx}",
                    "fingerprint": fp, "error": str(e),
                })

        return hf_lf, records, compiled_names

    def _persist_features(
        self,
        hf_enhanced: pl.LazyFrame,
        winners: List[str],
        fingerprints: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, List[str]], List[int]]:
        if not winners:
            return {}, []

        os.makedirs(FEATURE_DIR, exist_ok=True)
        guard_disk_watermark(FEATURE_DIR)

        available_schema = hf_enhanced.collect_schema().names()
        required_cols = list(dict.fromkeys(["day"] + list(BASE_MARKET_COLS)))
        base_cols = [c for c in required_cols if c in available_schema]
        cols_to_select = base_cols + [w for w in winners if w not in base_cols]

        full_df = (
            hf_enhanced.select(cols_to_select)
            .with_columns(month_id_expr("day").alias("_month"))
            .collect(engine="streaming")
        )
        if full_df.height == 0:
            return {w: [] for w in winners}, []

        parts_by_m = full_df.partition_by("_month", as_dict=True, include_key=False)
        months = sorted(int(k[0] if isinstance(k, tuple) else k) for k in parts_by_m.keys())
        out: Dict[str, List[str]] = {w: [] for w in winners}

        for m in months:
            m_df = parts_by_m.get((m,), parts_by_m.get(m))
            if m_df is None or m_df.height == 0:
                continue
            for feat in winners:
                fp = (fingerprints or {}).get(feat, "")
                feat_dir = os.path.join(FEATURE_DIR, str(fp)[:12])
                os.makedirs(feat_dir, exist_ok=True)
                p = os.path.join(feat_dir, f"hf_{feat}_{m}.parquet")
                atomic_save_parquet(m_df.select(base_cols + [feat]), p)
                out[feat].append(p)

        for feat in winners:
            print(f"  [Persist] {feat}: {len(out[feat])} monthly parquets generated")
        return out, months

    def _grid_combos(self) -> List[dict]:
        keys = list(self.grid_config.keys())
        return [dict(zip(keys, combo)) for combo in itertools.product(*self.grid_config.values())]

    def _build_eval_cache_locator(self, feature_names: List[str]):
        return build_eval_cache_locator(
            self.common_config,
            self.grid_config,
            feature_names,
            fingerprints=getattr(self, "_last_fingerprints", None),
            eval_data_fp=getattr(self, "_eval_data_fp", None),
            cache_root=EVAL_CACHE_ROOT,
            logic_version=EVAL_LOGIC_VERSION,
        )

    def _map_month_feat_paths(
        self, feature_names: List[str], months: List[int], check_exists: bool = False
    ) -> Dict[int, Dict[str, str]]:
        feat_paths: Dict[str, List[str]] = getattr(self, "_last_feat_paths", {})
        month_path_of: Dict[int, Dict[str, str]] = {m: {} for m in months}
        for feat, paths in feat_paths.items():
            if feat not in feature_names:
                continue
            for p in paths:
                m = parse_ym_from_path(p)
                if m in month_path_of and (not check_exists or os.path.exists(p)):
                    month_path_of[m][feat] = p
        return month_path_of

    def _evaluate_windows_parallel(
        self,
        hf_enhanced: pl.LazyFrame,
        dret_lf: pl.LazyFrame,
        feature_names: List[str],
        months: List[int],
        combos: List[dict],
        eval_procs: int,
    ) -> Dict[str, List[EvalResult]]:
        for _var in ("NUMBA_NUM_THREADS", "POLARS_MAX_THREADS", "OMP_NUM_THREADS"):
            os.environ.setdefault(_var, "1")

        month_path_of = self._map_month_feat_paths(feature_names, months)
        if any(not month_path_of[m] for m in months):
            return self._evaluate_windows_serial(hf_enhanced, dret_lf, feature_names, months, combos)

        n_proc = max(1, min(int(eval_procs), os.cpu_count() or 1, len(months)))
        replay_state = WindowReplayState.create(feature_names, len(months), self.common_config)

        with tempfile.TemporaryDirectory(prefix="stage2_mp_") as tmp_dir:
            dret_path = os.path.join(tmp_dir, "dret.parquet")
            dret_lf.collect(engine="streaming").write_parquet(dret_path)

            worker_cc = dict(self.common_config, max_workers=max(1, (os.cpu_count() or 1) // n_proc))
            _, _cache_path = self._build_eval_cache_locator(feature_names)

            per_month: Dict[int, Dict[str, EvalResult]] = {}
            with ProcessPoolExecutor(max_workers=n_proc) as pool:
                blocks = [months[i : i + n_proc] for i in range(0, len(months), n_proc)]
                month_ord = {m: i for i, m in enumerate(months)}
                for block in blocks:
                    batch_targets = sorted(replay_state.active_features)
                    for m in block:
                        per_month[m] = {}
                        for feat in batch_targets:
                            hit = eval_cache_load(_cache_path(feat, m))
                            if hit is not None:
                                per_month[m][feat] = hit
                    uncached = {
                        m: [f for f in batch_targets if f not in per_month[m]]
                        for m in block
                    }
                    futures = {
                        m: pool.submit(
                            _eval_window_worker,
                            dret_path,
                            {f: month_path_of[m][f] for f in uncached[m]},
                            uncached[m],
                            dict(worker_cc),
                            combos,
                        )
                        for m in block if uncached[m] and month_path_of[m]
                    }
                    for m in block:
                        if m in futures:
                            res_map = futures[m].result()
                            per_month[m].update(res_map)
                            for feat, res in res_map.items():
                                eval_cache_save(res, _cache_path(feat, m))

                    for m in block:
                        replay_state.replay_month(m, month_ord[m], per_month[m])
                    if not replay_state.active_features:
                        break

        return replay_state.pad_remaining_windows()

    def _evaluate_windows(
        self,
        hf_enhanced: pl.LazyFrame,
        dret_lf: pl.LazyFrame,
        feature_names: List[str],
        months: List[int],
    ) -> Dict[str, List[EvalResult]]:
        combos = self._grid_combos()
        if not months or not feature_names:
            return {f: [] for f in feature_names}

        eval_procs = int(self.common_config.get("eval_procs") or 1)
        if eval_procs > 1 and len(months) > 1:
            return self._evaluate_windows_parallel(hf_enhanced, dret_lf, feature_names, months, combos, eval_procs)
        return self._evaluate_windows_serial(hf_enhanced, dret_lf, feature_names, months, combos)

    def _evaluate_windows_serial(
        self,
        hf_enhanced: pl.LazyFrame,
        dret_lf: pl.LazyFrame,
        feature_names: List[str],
        months: List[int],
        combos: List[dict],
    ) -> Dict[str, List[EvalResult]]:
        month_path_of = self._map_month_feat_paths(feature_names, months, check_exists=True)
        with_month = (
            hf_enhanced.with_columns(month_id_expr("day").alias("_month"))
            if hf_enhanced is not None
            else None
        )

        replay_state = WindowReplayState.create(feature_names, len(months), self.common_config)
        combos_by_ds = group_combos_by_downsample(combos)
        data_fp, _cache_path = self._build_eval_cache_locator(feature_names)

        for i, m in enumerate(months):
            eval_targets = [f for f in feature_names if f in replay_state.active_features]
            if not eval_targets:
                break

            persisted_for_m = month_path_of.get(m) or {}
            first_persisted = next(iter(persisted_for_m.values()), None)
            if first_persisted is not None:
                window_lf = pl.scan_parquet(first_persisted)
            elif with_month is not None:
                window_lf = with_month.filter(pl.col("_month") == m).drop("_month")
            else:
                for f in eval_targets:
                    replay_state.results[f].append(EvalResult(feature_name=f, status="failed", reason="missing_parquet"))
                continue

            static_cache_key = (data_fp, m)
            if static_cache_key not in self._static_cache:
                self._static_cache[static_cache_key] = build_static_panel(
                    window_lf, dret_lf, self.common_config, is_train=True
                ).collect(engine="streaming")
            static_df = self._static_cache[static_cache_key]

            def _run_feat_combo(feat: str) -> EvalResult:
                hit = eval_cache_load(_cache_path(feat, m))
                if hit is not None:
                    return hit
                feat_lf = pl.scan_parquet(persisted_for_m[feat]) if feat in persisted_for_m else window_lf
                res = eval_feature_on_grid(feat_lf, dret_lf, static_df, feat, self.common_config, combos_by_ds)
                eval_cache_save(res, _cache_path(feat, m))
                return res

            month_evals = {f: _run_feat_combo(f) for f in eval_targets}
            replay_state.replay_month(m, i, month_evals)

        return replay_state.pad_remaining_windows()

    def run(
        self,
        candidate_asts: List[Union[dict, list[dict]]],
        hf_lf: pl.LazyFrame,
        dret_lf: pl.LazyFrame,
    ) -> List[FeatureMiningResult]:
        t_total = time.perf_counter()
        if not candidate_asts:
            return []

        hf_enhanced, compile_records, compiled_names = self._compile_asts(candidate_asts, hf_lf)
        stage1_map, survivors = run_stage1_prefilter(
            hf_enhanced, compiled_names, self.common_config, self.prefilter_thresholds
        )

        window_months: List[int] = []
        if survivors:
            fingerprints = {rec["name"]: rec["fingerprint"] for rec in compile_records}
            feat_paths, window_months = self._persist_features(hf_enhanced, survivors, fingerprints)
            self._last_feat_paths = feat_paths
            self._last_fingerprints = {
                rec["name"]: rec["fingerprint"] for rec in compile_records if rec.get("fingerprint")
            }
            self._eval_data_fp = compute_eval_data_fingerprint(self.common_config, dret_lf)

        window_results: Dict[str, List[EvalResult]] = {n: [] for n in compiled_names}
        if survivors and window_months:
            window_results.update(self._evaluate_windows(hf_enhanced, dret_lf, survivors, window_months))

        min_ratio = self.common_config.get("min_window_pass_ratio", MIN_WINDOW_PASS_RATIO)
        max_pval_gate = float(self.common_config.get("stage2_max_pval", 0.1))
        all_results = [
            aggregate_candidate_result(rec, stage1_map, window_results, min_ratio, max_pval_gate)
            for rec in compile_records
        ]

        if self.stage3_callback:
            for r in (res for res in all_results if res.stage2_passed):
                try:
                    cb_common = dict(self.common_config)
                    if r.stage2_eval and isinstance(r.stage2_eval.config, dict):
                        cb_common["prior_config"] = {
                            k: r.stage2_eval.config[k]
                            for k in ("downsample", "motif_minutes", "threshold_r")
                            if k in r.stage2_eval.config
                        }
                    r.stage3_triggered = bool(self.stage3_callback(r.feature_name, r.raw_ast, cb_common))
                except Exception as e:
                    print(f"  [Harness] Stage 3 failed for {r.feature_name}: {e}")

        print(f"[Harness] complete in {time.perf_counter() - t_total:.1f}s")
        return all_results
