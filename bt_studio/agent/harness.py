#! /usr/bin/env python3
"""Two-stage feature evaluation harness: AST compilation -> Stage 1 physical prefilter -> Stage 2 walk-forward grid evaluation."""

from __future__ import annotations

import itertools
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

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
    DEFAULT_WARMUP_MONTHS,
    EVAL_CACHE_ROOT,
    EVAL_LOGIC_VERSION,
    FEATURE_DIR,
    MIN_WINDOW_PASS_RATIO,
)
from bt_studio.pipeline.patterns import discover_fsm_pattern
from bt_studio.pipeline.preprocess import prepare_tick
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
from bt_studio.utils.months import (
    calc_warmup_start_date,
    month_bounds,
    month_id_expr,
    parse_ym_from_path,
    resolve_walk_forward_windows,
    window_tag,
)


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
            n: PrefilterResult.from_stats(n, row, row["_total_len"], prefilter_thresholds)
            for n in compiled_names
        }
    except Exception as bulk_err:
        print(f"  [Stage1] bulk agg failed ({bulk_err!r}); fallback to candidate isolation")
        for n in compiled_names:
            try:
                r = _stats_row([n])
                stage1_map[n] = PrefilterResult.from_stats(n, r, r["_total_len"], prefilter_thresholds)
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
    static_df: pl.DataFrame,
    feat: str,
    common_config: dict,
    combos_by_ds: Dict[int, List[dict]],
) -> EvalResult:
    max_pval_gate = float(common_config.get("stage2_max_pval", common_config.get("u_pval", 0.1)))
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
    feat_month_paths: Dict[str, Union[str, List[str]]],
    feature_names: List[str],
    common_config: dict,
    combos: List[dict],
) -> Dict[str, EvalResult]:
    first_path = next(iter(feat_month_paths.values()), None)
    if not first_path:
        return {
            f: EvalResult(feature_name=f, status="failed", reason="missing_parquet")
            for f in feature_names
        }
    dret_lf = pl.scan_parquet(dret_path)
    static_df = build_static_panel(
        pl.scan_parquet(first_path), dret_lf, common_config, is_train=True
    ).collect(engine="streaming")
    combos_by_ds = group_combos_by_downsample(combos)

    return {
        feat: (
            eval_feature_on_grid(pl.scan_parquet(f_path), static_df, feat, common_config, combos_by_ds)
            if (f_path := feat_month_paths.get(feat))
            else EvalResult(feature_name=feat, status="failed", reason="missing_parquet")
        )
        for feat in feature_names
    }


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
        universe_sids: Optional[Dict[int, List[bytes]]] = None,
    ):
        self.common_config = common_config
        self.grid_config = grid_config if grid_config is not None else DEFAULT_GRID
        self.prefilter_thresholds = prefilter_thresholds
        self.stage3_callback = stage3_callback
        self.universe_sids: Dict[int, List[bytes]] = universe_sids or {}
        self._static_cache: Dict[Tuple[str, Union[int, str]], pl.DataFrame] = {}
        self._last_feat_paths: Dict[str, List[str]] = {}
        self._last_fingerprints: Dict[str, str] = {}
        self._last_raw_asts: Dict[str, Any] = {}

    def _compile_asts(
        self,
        candidate_asts: List[Union[dict, list[dict]]],
        hf_lf: pl.LazyFrame,
    ) -> Tuple[pl.LazyFrame, List[dict], List[str]]:
        records: List[dict] = []
        compiled_names: List[str] = []
        seen_names: Set[str] = set()

        for idx, candidate in enumerate(candidate_asts):
            if isinstance(candidate, dict) and "ast" in candidate and "op" not in candidate and "col" not in candidate:
                inner = candidate["ast"]
                candidate = inner if isinstance(inner, list) else ([candidate] if "name" in candidate else inner)
            fp = ast_fingerprint(candidate)
            feat_name = f"f_{fp[:12]}"
            try:
                if isinstance(candidate, list):
                    renames: Dict[str, str] = {}
                    isolated_recipe: list[dict] = []
                    for step in candidate:
                        orig = step["name"]
                        renames[orig] = f"_c{idx}_{orig}"
                        isolated_recipe.append({
                            **step,
                            "name": renames[orig],
                            "ast": rewrite_ast_col_refs(step["ast"], renames),
                        })
                    for e in compile_recipe(isolated_recipe, check_causal=True):
                        hf_lf = hf_lf.with_columns(e)
                    raw_name = candidate[-1]["name"]
                    iso_last = f"_c{idx}_{raw_name}"
                    if feat_name != iso_last:
                        hf_lf = hf_lf.rename({iso_last: feat_name})
                elif isinstance(candidate, dict):
                    expr, raw_name = compile_ast(candidate, check_causal=True)
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

    def _fetch_tick_with_warmup(self, target_ymonths: List[int]) -> pl.LazyFrame:
        """Fetch tick LazyFrame for target_ymonths with causal warmup_months prepended."""
        if not target_ymonths or not self.universe_sids:
            return pl.LazyFrame()
        raw_start_date = min(target_ymonths) * 100 + 1
        _, end_date = month_bounds(max(target_ymonths))
        warmup_months = int(self.common_config.get("warmup_months", DEFAULT_WARMUP_MONTHS))
        fetch_start_date = calc_warmup_start_date(raw_start_date, warmup_months=warmup_months)

        unique_sids = list(dict.fromkeys(
            sid for ym in target_ymonths for sid in self.universe_sids.get(ym, [])
        ))
        if not unique_sids:
            return pl.LazyFrame()

        print(f"  [OnDemand-Tick] Window months {min(target_ymonths)}..{max(target_ymonths)} "
              f"(warmup from {fetch_start_date}, {len(unique_sids)} sids)")
        return prepare_tick(fetch_start_date, end_date, unique_sids)

    def _persist_features(
        self,
        hf_enhanced: pl.LazyFrame,
        winners: List[str],
        fingerprints: Optional[Dict[str, str]] = None,
        target_months: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, List[str]], List[int]]:
        if not winners:
            return {}, []

        os.makedirs(FEATURE_DIR, exist_ok=True)
        guard_disk_watermark(FEATURE_DIR)

        available_schema = hf_enhanced.collect_schema().names()
        required_cols = list(dict.fromkeys(["day"] + list(BASE_MARKET_COLS)))
        base_cols = [c for c in required_cols if c in available_schema]
        cols_to_select = base_cols + [w for w in winners if w not in base_cols]

        lf_to_collect = hf_enhanced.select(cols_to_select).with_columns(month_id_expr("day").alias("_month"))
        if target_months is not None:
            lf_to_collect = lf_to_collect.filter(pl.col("_month").is_in([int(m) for m in target_months]))

        full_df = lf_to_collect.collect(engine="streaming")
        if full_df.height == 0:
            return {w: [] for w in winners}, []

        parts_by_m = full_df.partition_by("_month", as_dict=True, include_key=False)
        months = sorted(int(k[0] if isinstance(k, tuple) else k) for k in parts_by_m.keys())
        out: Dict[str, List[str]] = {w: [] for w in winners}
        fps = fingerprints or {}

        for m in months:
            m_df = parts_by_m.get((m,), parts_by_m.get(m))
            if m_df is None or m_df.height == 0:
                continue
            if m in self.universe_sids and "sid" in m_df.columns:
                valid_sids = self.universe_sids[m]
                if m_df.schema["sid"] == pl.String:
                    valid_sids = [s.decode() if isinstance(s, bytes) else str(s) for s in valid_sids]
                m_df = m_df.filter(pl.col("sid").is_in(valid_sids))
                if m_df.height == 0:
                    continue
            for feat in winners:
                feat_dir = os.path.join(FEATURE_DIR, str(fps.get(feat, ""))[:12])
                os.makedirs(feat_dir, exist_ok=True)
                p = os.path.join(feat_dir, f"hf_{feat}_{m}.parquet")
                atomic_save_parquet(m_df.select(base_cols + [feat]), p)
                out[feat].append(p)

        for feat in winners:
            print(f"  [Persist] {feat}: {len(out[feat])} monthly parquets generated")
        return out, months

    def _ensure_window_parquets(self, feature_names: List[str], win_months: List[int]) -> None:
        """Lazily fetch ticks and compile missing monthly feature parquets for win_months on demand."""
        if not feature_names or not win_months or not self.universe_sids:
            return

        missing_months_set: Set[int] = set()
        feats_needing_compile: List[str] = []

        for feat in feature_names:
            feat_dir = os.path.join(FEATURE_DIR, str(self._last_fingerprints.get(feat, ""))[:12])
            feat_missing = False
            for m in win_months:
                p = os.path.join(feat_dir, f"hf_{feat}_{m}.parquet")
                if os.path.exists(p):
                    if p not in self._last_feat_paths.setdefault(feat, []):
                        self._last_feat_paths[feat].append(p)
                else:
                    missing_months_set.add(int(m))
                    feat_missing = True
            if feat_missing and feat in self._last_raw_asts:
                feats_needing_compile.append(feat)

        if not missing_months_set or not feats_needing_compile:
            return

        missing_months = sorted(missing_months_set)
        raw_lf = self._fetch_tick_with_warmup(missing_months)
        cands = [self._last_raw_asts[f] for f in feats_needing_compile]
        compiled_lf, _, compiled_names = self._compile_asts(cands, raw_lf)
        new_paths, _ = self._persist_features(
            compiled_lf, compiled_names, self._last_fingerprints, target_months=missing_months
        )
        for feat, paths in new_paths.items():
            self._last_feat_paths[feat] = list(dict.fromkeys(self._last_feat_paths.get(feat, []) + paths))

    def _grid_combos(self) -> List[dict]:
        keys = list(self.grid_config.keys())
        return [dict(zip(keys, combo)) for combo in itertools.product(*self.grid_config.values())]

    def _build_eval_cache_locator(self, feature_names: List[str]):
        cfg = dict(self.common_config)
        if getattr(self, "_eval_data_fp", None):
            cfg["_eval_data_fp"] = self._eval_data_fp
        return build_eval_cache_locator(
            cfg,
            self.grid_config,
            feature_names,
            fingerprints=getattr(self, "_last_fingerprints", None),
            cache_ctx=(EVAL_CACHE_ROOT, EVAL_LOGIC_VERSION),
        )

    @staticmethod
    def _normalize_window(w: Union[int, List[int], Tuple[int, ...]]) -> List[int]:
        return [int(x) for x in w] if isinstance(w, (list, tuple)) else [int(w)]

    def _map_month_feat_paths(
        self,
        feature_names: List[str],
        windows: Sequence[Union[int, List[int]]],
        check_exists: bool = False,
    ) -> Dict[Union[int, str], Dict[str, Union[str, List[str]]]]:
        feat_paths: Dict[str, List[str]] = getattr(self, "_last_feat_paths", {})
        by_feat_ym: Dict[str, Dict[int, str]] = {f: {} for f in feature_names}
        for feat, paths in feat_paths.items():
            if feat in by_feat_ym:
                for p in paths:
                    m = parse_ym_from_path(p)
                    if m is not None and (not check_exists or os.path.exists(p)):
                        by_feat_ym[feat][int(m)] = p

        out: Dict[Union[int, str], Dict[str, Union[str, List[str]]]] = {}
        for w in windows:
            w_tag = window_tag(w)
            w_months = self._normalize_window(w)
            out[w_tag] = {
                feat: (ym_map[w_months[0]] if len(w_months) == 1 else [ym_map[m] for m in w_months])
                for feat in feature_names
                if all(m in (ym_map := by_feat_ym.get(feat, {})) for m in w_months)
            }
        return out

    def _prepare_window_eval(
        self,
        batch_targets: List[str],
        w: Union[int, List[int]],
        cache_path_fn: Any,
        on_demand: bool,
    ) -> Tuple[Union[int, str], List[int], Dict[str, EvalResult], List[str]]:
        w_tag = window_tag(w)
        w_months = self._normalize_window(w)
        cached: Dict[str, EvalResult] = {}
        uncached: List[str] = []
        for feat in batch_targets:
            hit = eval_cache_load(cache_path_fn(feat, w_tag))
            if hit is not None:
                cached[feat] = hit
            else:
                uncached.append(feat)
        if uncached and on_demand:
            self._ensure_window_parquets(uncached, w_months)
        return w_tag, w_months, cached, uncached

    def _evaluate_windows_parallel(
        self,
        hf_enhanced: Optional[pl.LazyFrame],
        dret_lf: pl.LazyFrame,
        feature_names: List[str],
        months: List[Union[int, List[int]]],
    ) -> Dict[str, List[EvalResult]]:
        eval_procs = int(self.common_config.get("eval_procs", 1))
        on_demand = bool(self.universe_sids and self._last_raw_asts)
        combos = self._grid_combos()
        month_path_of = self._map_month_feat_paths(feature_names, months)

        if not on_demand and any(not month_path_of[window_tag(w)] for w in months):
            return self._evaluate_windows_serial(hf_enhanced, dret_lf, feature_names, months)

        n_proc = max(1, min(eval_procs, os.cpu_count() or 1, len(months)))
        replay_state = WindowReplayState.create(feature_names, len(months), self.common_config)

        with tempfile.TemporaryDirectory(prefix="stage2_mp_") as tmp_dir:
            dret_path = os.path.join(tmp_dir, "dret.parquet")
            dret_lf.collect(engine="streaming").write_parquet(dret_path)

            worker_cc = dict(self.common_config, max_workers=max(1, (os.cpu_count() or 1) // n_proc))
            _, _cache_path = self._build_eval_cache_locator(feature_names)

            per_win: Dict[Union[int, str], Dict[str, EvalResult]] = {}
            block_step = 1 if on_demand else n_proc
            blocks = [months[i : i + block_step] for i in range(0, len(months), block_step)]
            win_ord = {window_tag(w): i for i, w in enumerate(months)}

            with ProcessPoolExecutor(max_workers=n_proc) as pool:
                for block in blocks:
                    batch_targets = sorted(replay_state.active_features)
                    if not batch_targets:
                        break
                    uncached: Dict[Union[int, str], List[str]] = {}
                    for w in block:
                        w_tag, _, cached, unc = self._prepare_window_eval(batch_targets, w, _cache_path, on_demand)
                        per_win[w_tag] = cached
                        uncached[w_tag] = unc
                    if on_demand:
                        month_path_of = self._map_month_feat_paths(feature_names, months, check_exists=True)

                    futures: List[Tuple[Union[int, str], Any]] = []
                    for w in block:
                        w_tag = window_tag(w)
                        avail = [f for f in uncached[w_tag] if f in month_path_of.get(w_tag, {})]
                        if not avail:
                            continue
                        if on_demand and len(avail) > 1:
                            f_cc = dict(self.common_config, max_workers=max(1, (os.cpu_count() or 1) // min(n_proc, len(avail))))
                            for f in avail:
                                futures.append((
                                    w_tag,
                                    pool.submit(_eval_window_worker, dret_path, {f: month_path_of[w_tag][f]}, [f], f_cc, combos),
                                ))
                        else:
                            w_cc = dict(self.common_config, max_workers=(os.cpu_count() or 1) if on_demand else worker_cc["max_workers"])
                            futures.append((
                                w_tag,
                                pool.submit(
                                    _eval_window_worker,
                                    dret_path,
                                    {f: month_path_of[w_tag][f] for f in avail},
                                    avail,
                                    w_cc,
                                    combos,
                                ),
                            ))
                    for w_tag, fut in futures:
                        res_map = fut.result()
                        per_win[w_tag].update(res_map)
                        for feat, res in res_map.items():
                            eval_cache_save(res, _cache_path(feat, w_tag))
                    for w in block:
                        w_tag = window_tag(w)
                        replay_state.replay_month(w_tag, win_ord[w_tag], per_win[w_tag])
                    if not replay_state.active_features:
                        break

        return replay_state.pad_remaining_windows()

    def _evaluate_windows(
        self,
        hf_enhanced: Optional[pl.LazyFrame],
        dret_lf: pl.LazyFrame,
        feature_names: List[str],
        months: List[Union[int, List[int]]],
    ) -> Dict[str, List[EvalResult]]:
        if not months or not feature_names:
            return {f: [] for f in feature_names}
        if int(self.common_config.get("eval_procs", 1)) > 1 and len(months) > 1:
            return self._evaluate_windows_parallel(hf_enhanced, dret_lf, feature_names, months)
        return self._evaluate_windows_serial(hf_enhanced, dret_lf, feature_names, months)

    def _evaluate_windows_serial(
        self,
        hf_enhanced: Optional[pl.LazyFrame],
        dret_lf: pl.LazyFrame,
        feature_names: List[str],
        months: List[Union[int, List[int]]],
    ) -> Dict[str, List[EvalResult]]:
        on_demand = bool(self.universe_sids and self._last_raw_asts)
        month_path_of = self._map_month_feat_paths(feature_names, months, check_exists=True)
        with_month = (
            hf_enhanced.with_columns(month_id_expr("day").alias("_month"))
            if hf_enhanced is not None
            else None
        )

        replay_state = WindowReplayState.create(feature_names, len(months), self.common_config)
        combos_by_ds = group_combos_by_downsample(self._grid_combos())
        data_fp, _cache_path = self._build_eval_cache_locator(feature_names)

        for i, w in enumerate(months):
            eval_targets = [f for f in feature_names if f in replay_state.active_features]
            if not eval_targets:
                break

            w_tag, w_months, cached_evals, uncached_targets = self._prepare_window_eval(
                eval_targets, w, _cache_path, on_demand
            )
            if uncached_targets and on_demand:
                month_path_of = self._map_month_feat_paths(feature_names, months, check_exists=True)

            persisted_for_w = month_path_of.get(w_tag, {})
            first_persisted = next(iter(persisted_for_w.values()), None)
            if first_persisted is not None:
                window_lf = pl.scan_parquet(first_persisted)
            elif with_month is not None:
                window_lf = with_month.filter(pl.col("_month").is_in(w_months)).drop("_month")
            elif not uncached_targets:
                replay_state.replay_month(w_tag, i, cached_evals)
                continue
            else:
                for f in eval_targets:
                    replay_state.results[f].append(EvalResult(feature_name=f, status="failed", reason="missing_parquet"))
                continue

            static_cache_key = (data_fp, w_tag)
            if uncached_targets and static_cache_key not in self._static_cache:
                self._static_cache[static_cache_key] = build_static_panel(
                    window_lf, dret_lf, self.common_config, is_train=True
                ).collect(engine="streaming")
            static_df = self._static_cache.get(static_cache_key)

            month_evals = dict(cached_evals)
            for feat in uncached_targets:
                feat_lf = pl.scan_parquet(persisted_for_w[feat]) if feat in persisted_for_w else window_lf
                res = eval_feature_on_grid(feat_lf, static_df, feat, self.common_config, combos_by_ds)
                eval_cache_save(res, _cache_path(feat, w_tag))
                month_evals[feat] = res

            replay_state.replay_month(w_tag, i, month_evals)

        return replay_state.pad_remaining_windows()

    def run(
        self,
        candidate_asts: List[Union[dict, list[dict]]],
        hf_lf: Optional[pl.LazyFrame],
        dret_lf: pl.LazyFrame,
    ) -> List[FeatureMiningResult]:
        t_total = time.perf_counter()
        if not candidate_asts:
            return []

        on_demand = hf_lf is None and bool(self.universe_sids)
        if on_demand:
            wf_windows = resolve_walk_forward_windows(list(self.universe_sids.keys()), self.common_config)
            if not wf_windows:
                raise RuntimeError("[Harness] No valid walk-forward windows resolved from universe_sids")
            target_months: Optional[List[int]] = [wf_windows[0][0]]
            base_lf = self._fetch_tick_with_warmup(target_months)
        else:
            if hf_lf is None:
                raise ValueError("[Harness] Either hf_lf or universe_sids must be provided")
            wf_windows = []
            target_months = None
            base_lf = hf_lf

        compiled_lf, compile_records, compiled_names = self._compile_asts(candidate_asts, base_lf)
        hf_enhanced = (
            compiled_lf.filter(month_id_expr("day").is_in(target_months))
            if target_months is not None
            else compiled_lf
        )
        stage1_map, survivors = run_stage1_prefilter(
            hf_enhanced, compiled_names, self.common_config, self.prefilter_thresholds
        )

        windows: List[Union[int, List[int]]] = []
        if survivors:
            self._last_fingerprints = {
                rec["name"]: rec["fingerprint"] for rec in compile_records if rec.get("fingerprint")
            }
            self._last_raw_asts = (
                {rec["name"]: rec["raw_ast"] for rec in compile_records if rec.get("error") is None}
                if on_demand
                else {}
            )
            self._last_feat_paths, window_months = self._persist_features(
                compiled_lf, survivors, self._last_fingerprints, target_months=target_months
            )
            self._eval_data_fp = compute_eval_data_fingerprint(self.common_config, dret_lf)
            if not wf_windows:
                wf_windows = resolve_walk_forward_windows(window_months, self.common_config)
            windows = [w[0] if len(w) == 1 else w for w in wf_windows]

        window_results: Dict[str, List[EvalResult]] = {n: [] for n in compiled_names}
        if survivors and windows:
            window_results.update(self._evaluate_windows(hf_enhanced, dret_lf, survivors, windows))

        min_ratio = self.common_config.get("min_window_pass_ratio", MIN_WINDOW_PASS_RATIO)
        max_pval_gate = float(self.common_config.get("stage2_max_pval", self.common_config.get("u_pval", 0.1)))
        all_results = [
            FeatureMiningResult.from_candidate_windows(rec, stage1_map, window_results, min_ratio, max_pval_gate)
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
