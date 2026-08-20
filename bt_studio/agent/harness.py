#! /usr/bin/env python3

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

from bt_studio.pipeline.preprocess.panel import build_static_panel, extract_curves_from_panel
from bt_studio.pipeline.patterns import discover_fsm_pattern


@dataclass
class EvalResult:
    """Result of a single-point evaluation."""
    feature_name: str
    status: str = "unknown"          # "success" | "failed"
    score: float = -500.0            # metrics_score from discover_fsm_pattern
    u_pval: float = 1.0
    trigger_count: int = 0
    valid_sample_ratio: float = 0.0
    autocorr: float = 0.0
    elapsed_s: float = 0.0
    reason: str = ""
    fsm_matrix: Optional[dict] = None
    learned_motif: Optional[list] = None
    config: dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    def to_summary(self) -> dict:
        """Compact dict for logging (safely handles None/NaN and heavy objects)."""
        def _safe_round(val: Any, decimals: int, fallback: float = 0.0) -> float:
            if val is None or not isinstance(val, (int, float)) or np.isnan(val):
                return fallback
            return round(float(val), decimals)

        return {
            "feature": self.feature_name,
            "status": self.status,
            "score": _safe_round(self.score, 4, -500.0),
            "u_pval": _safe_round(self.u_pval, 6, 1.0),
            "triggers": int(self.trigger_count or 0),
            "vsr": _safe_round(self.valid_sample_ratio, 4, 0.0),
            "autocorr": _safe_round(self.autocorr, 4, 0.0),
            "elapsed_s": _safe_round(self.elapsed_s, 3, 0.0),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_feature(
    hf_lf: pl.LazyFrame,
    dret_lf: pl.LazyFrame,
    feature_col: str,
    common_config: dict,
    static_lf: Optional[Union[pl.LazyFrame, pl.DataFrame]],
    cfg: dict,
) -> EvalResult:
    """
    Parameters
    ----------
    hf_lf: pl.LazyFrame 
        [day, sid, bar_idx, <feature_col>]

    dret_lf: pl.LazyFrame 
        [day, sid, close]

    feature_col: str
        eg 'ofi_ratio'

    common_config: dict

    static_lf: optional [None, pl.LazyFrame]
    
    cfg: optional[None, Dict]
        DEFAULT_FSM_CFG

    Returns
    -------
    EvalResult

    """
    t0 = time.perf_counter()
    default_tune_cfg: dict = {}

    try:
        
        default_tune_cfg = cfg.copy()

        m = int(default_tune_cfg["motif_minutes"] // default_tune_cfg["downsample"])
        threshold_d = float(np.sqrt(2 * m * (1.0 - default_tune_cfg["threshold_r"])))

        default_tune_cfg["m"] = m
        default_tune_cfg["threshold_d"] = threshold_d

        local_common = dict(common_config)
        local_common["feature_col"] = feature_col

        # 4. LazyFrame
        if static_lf is None:
            resolved_static_lf = build_static_panel(hf_lf, dret_lf, local_common, is_train=True)
        elif isinstance(static_lf, pl.DataFrame):
            resolved_static_lf = static_lf.lazy()
        else:
            resolved_static_lf = static_lf


        curve_lf = extract_curves_from_panel(hf_lf, dret_lf, default_tune_cfg, local_common)
        panel_lf = curve_lf.join(resolved_static_lf, on=["day", "sid"], how="inner")

        result = discover_fsm_pattern(panel_lf, default_tune_cfg, local_common)
        elapsed = time.perf_counter() - t0

        if result and result.get("status") == "success":
            return EvalResult(
                feature_name=feature_col,
                status="success",
                score=result.get("metrics_score") if result.get("metrics_score") else -500.0,
                u_pval=result.get("u_pval") if result.get("u_pval") else 1.0,
                trigger_count=int(result.get("trigger_count") or 0),
                valid_sample_ratio=float(result.get("valid_sample_ratio") or 0.0),
                autocorr=float(result.get("autocorr") or 0.0),
                elapsed_s=elapsed,
                fsm_matrix=result.get("fsm_matrix"),
                learned_motif=result.get("learned_motif"),
                config=default_tune_cfg,
            )

        fail_reason = result.get("reason", "unknown") if result else "empty_result"
        return EvalResult(
            feature_name=feature_col,
            status="failed",
            reason=str(fail_reason),
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


#! /usr/bin/env python3

from __future__ import annotations

import os
import sys
import time
import platform
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import polars as pl

from bt_studio.compiler.ast import compile_ast, compile_recipe, ast_fingerprint, ASTCompilationError
# EvalResult, evaluate_feature are in this file now
from bt_studio.pipeline.preprocess.panel import build_static_panel
from bt_studio.constant import FEATURE_DIR, MIN_WINDOW_PASS_RATIO, BASE_MARKET_COLS

# ---------------------------------------------------------------------------
# default config
# ---------------------------------------------------------------------------

DEFAULT_GRID: Dict[str, list] = {
    "downsample": [3, 4, 5],
    "motif_minutes": [30, 45, 60],
    "threshold_r": [0.55, 0.65, 0.75],
}

PREFILTER_THRESHOLDS = {
    "min_std": 1e-6,        
    "max_null_ratio": 0.20,  
}

# ---------------------------------------------------------------------------
# dataclass
# ---------------------------------------------------------------------------

@dataclass
class PrefilterResult:
    """Stage 1"""
    feature_name: str
    passed: bool = False
    std: float = 0.0
    null_ratio: float = 0.0
    reason: str = ""

    def to_summary(self) -> dict:
        return {
            "feature": self.feature_name,
            "passed": self.passed,
            "std": round(self.std, 8),
            "null_ratio": round(self.null_ratio, 4),
            "reason": self.reason,
        }


@dataclass
class FeatureMiningResult:
    """Stage Complete result"""
    feature_name: str
    raw_ast: Any = None                   # AST / Recipe
    ast_fingerprint: str = ""
    stage1_prefilter: Optional[PrefilterResult] = None
    stage2_eval: Optional[EvalResult] = None
    stage2_passed: bool = False
    stage3_triggered: bool = False
    final_score: float = -500.0
    window_pass_ratio: float = 0.0

    @property
    def stage_passed(self) -> str:
        if self.stage3_triggered:
            return "stage3_hpo"
        if self.stage2_passed:
            return "stage2_eval"
        if self.stage1_prefilter and self.stage1_prefilter.passed:
            return "stage1_prefilter"
        return "rejected"

    def to_summary(self) -> dict:
        return {
            "feature": self.feature_name,
            "fingerprint": self.ast_fingerprint,
            "stage": self.stage_passed,
            "score": round(self.final_score, 4),
            "window_pass_ratio": round(self.window_pass_ratio, 4),
            "prefilter": self.stage1_prefilter.to_summary() if self.stage1_prefilter else None,
            "eval": self.stage2_eval.to_summary() if self.stage2_eval else None,
        }


# ---------------------------------------------------------------------------
# Stage 1: physical filter
# ---------------------------------------------------------------------------

def _evaluate_physical_stats(
    feature_col: str,
    stats_dict: dict,
    total_len: int,
    thresholds: Optional[dict] = None,
) -> PrefilterResult:
    cfg = {**PREFILTER_THRESHOLDS, **(thresholds or {})}

    if total_len == 0:
        return PrefilterResult(feature_name=feature_col, reason="Empty DataFrame/Series")

    null_count = stats_dict.get(f"{feature_col}_null")
    if null_count is None:
        null_count = total_len

    std_val = stats_dict.get(f"{feature_col}_std")
    if std_val is None or not isinstance(std_val, (int, float)):
        std_val = 0.0

    null_ratio = float(null_count / total_len)
    non_null_len = total_len - null_count

    if non_null_len < 10:
        return PrefilterResult(
            feature_name=feature_col,
            null_ratio=null_ratio,
            reason="Too few non-null values (< 10)",
        )

    reasons = []
    if std_val < cfg["min_std"]:
        reasons.append(f"std={std_val:.2e} < {cfg['min_std']:.0e} (dead water)")
    if null_ratio > cfg["max_null_ratio"]:
        reasons.append(f"null_ratio={null_ratio:.2%} > {cfg['max_null_ratio']:.0%}")

    return PrefilterResult(
        feature_name=feature_col,
        passed=not reasons,
        std=float(std_val),
        null_ratio=null_ratio,
        reason="; ".join(reasons) if reasons else "OK",
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class TwoStageAgentHarness:
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

    def _compile_asts(
            self,
            candidate_asts: List[dict | list[dict]],
            hf_lf: pl.LazyFrame,
        ) -> tuple[pl.LazyFrame, List[dict], List[str]]:
            """
            ASTs / Recipes
            
            Returns
            -------
            hf_lf : pl.LazyFrame
            records : List[dict]
            compiled_names : List[str]
            """
            records: List[dict] = []
            compiled_names: List[str] = []
            seen_names: set[str] = set()

            for idx, candidate in enumerate(candidate_asts):
                fp = ast_fingerprint(candidate)
                try:
                    if isinstance(candidate, list):
                        exprs = compile_recipe(candidate, check_causal=True)
                        for e in exprs:
                            hf_lf = hf_lf.with_columns(e)
                        base_name = candidate[-1].get("name", f"recipe_feat_{idx}")
                    else:
                        expr, base_name = compile_ast(candidate, check_causal=True)
                        hf_lf = hf_lf.with_columns(expr.alias(base_name))

                    # resovle duplicate name 
                    feat_name = base_name
                    collision = 1
                    while feat_name in seen_names:
                        feat_name = f"{base_name}_{collision}"
                        collision += 1
                        if not isinstance(candidate, list):
                            hf_lf = hf_lf.with_columns(expr.alias(feat_name))

                    seen_names.add(feat_name)
                    compiled_names.append(feat_name)
                    records.append({
                        "index": idx,
                        "raw_ast": candidate,
                        "fingerprint": fp,
                        "name": feat_name,
                        "error": None,
                    })
                except Exception as e:
                    print(f"  [Harness] SKIP AST [{idx}] (compile error): {e}")
                    records.append({
                        "index": idx,
                        "raw_ast": candidate,
                        "fingerprint": fp,
                        "name": f"rejected_{idx}",
                        "error": str(e),
                    })

            return hf_lf, records, compiled_names

    def _persist_features(
        self,
        hf_enhanced: pl.LazyFrame,
        winners: List[str],
    ) -> Dict[str, List[str]]:
        """Stage 2"""
        if not winners:
            return {}

        os.makedirs(FEATURE_DIR, exist_ok=True)

        # 动态探测基础列，避免硬编码 open/close 报错
        available_schema = hf_enhanced.collect_schema().names()
        base_cols = [c for c in BASE_MARKET_COLS if c in available_schema]
        cols_to_select = base_cols + [w for w in winners if w not in base_cols]

        df = (
            hf_enhanced.select(cols_to_select)
            .with_columns((pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).alias("_month"))
            .collect(engine="streaming")
        )

        out: Dict[str, List[str]] = {}
        months = sorted(df["_month"].unique().to_list())
        for feat in winners:
            paths = []
            for m in months:
                p = f"{FEATURE_DIR}/hf_{feat}_{m}.parquet"
                if not os.path.exists(p):
                    part = df.filter(pl.col("_month") == m).select(base_cols + [feat])
                    part.write_parquet(p)
                paths.append(p)
            out[feat] = paths
            print(f"  [Persist] {feat}: {len(paths)} monthly parquets under FEATURE_DIR")
        return out

    def _grid_combos(self) -> List[dict]:
        keys = list(self.grid_config.keys())
        return [dict(zip(keys, combo)) for combo in itertools.product(*self.grid_config.values())]

    def _evaluate_windows(
        self,
        hf_enhanced: pl.LazyFrame,
        dret_lf: pl.LazyFrame,
        feature_names: List[str],
        months: List[int],
    ) -> Dict[str, List[EvalResult]]:
        """Walk-forward  static_panel"""
        combos = self._grid_combos()
        if not months or not feature_names:
            return {f: [] for f in feature_names}

        results: Dict[str, List[EvalResult]] = {f: [] for f in feature_names}

        # OS-aware concurrency check:
        # Numba workqueue is fundamentally not thread-safe and crashes on concurrent accesses.
        # On macOS ARM, pip provides NO threadsafe layer (omp/tbb not available via pure pip).
        # We degrade to serial execution on macOS ARM to avoid zsh: abort.
        is_mac_arm = sys.platform == "darwin" and platform.machine() == "arm64"
        eval_workers = self.common_config.get("eval_workers") or os.cpu_count() or 1
        nw = 1 if is_mac_arm else max(1, int(eval_workers))

        with_month = hf_enhanced.with_columns(
            (pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).alias("_month")
        )

        for i, m in enumerate(months):
            window_lf = with_month.filter(pl.col("_month") == m).drop("_month")
            print(f"  [Stage2] window {i + 1}/{len(months)}: {m}")

            # avoid recalculate lazy graph
            static_df = build_static_panel(
                window_lf, dret_lf, self.common_config, is_train=True
            ).collect(engine="streaming")

            def _run_feat_combo(feat: str) -> EvalResult:
                best: Optional[EvalResult] = None
                for combo in combos:
                    r = evaluate_feature(
                        window_lf,
                        dret_lf,
                        feat,
                        self.common_config,
                        static_lf=static_df,
                        cfg=combo,
                    )
                    if (
                        best is None
                        or (r.is_success and not best.is_success)
                        or (r.is_success == best.is_success and r.score > best.score)
                    ):
                        best = r
                    if r.is_success:
                        break  
                return best or EvalResult(feature_name=feat, status="failed", reason="empty_grid")

            if nw > 1 and len(feature_names) > 1:
                with ThreadPoolExecutor(max_workers=nw) as pool:
                    for feat, res in zip(feature_names, pool.map(_run_feat_combo, feature_names)):
                        results[feat].append(res)
            else:
                for feat in feature_names:
                    results[feat].append(_run_feat_combo(feat))

        return results

    def run(
            self,
            candidate_asts: List[dict | list[dict]],
            hf_lf: pl.LazyFrame,
            dret_lf: pl.LazyFrame,
        ) -> List[FeatureMiningResult]:
            
            t_total = time.perf_counter()

            if not candidate_asts:
                return []

            # ====================================================================
            # 1. compile ast
            # ====================================================================
            print(f"\n[Harness] Compiling {len(candidate_asts)} candidate AST(s)...")
            hf_enhanced, compile_records, compiled_names = self._compile_asts(candidate_asts, hf_lf)
            n_skipped = sum(1 for rec in compile_records if rec["error"] is not None)
            print(f"  Compiled: {len(compiled_names)}, Skipped: {n_skipped}")

            # ====================================================================
            # 2. Stage 1: physical filer
            # ====================================================================
            stage1_map: Dict[str, PrefilterResult] = {}
            survivors: List[str] = []

            if compiled_names:
                print(f"[Harness] Stage 1: physical prefilter ({len(compiled_names)} features) via Lazy Agg...")
                agg_exprs = [pl.len().alias("_total_len")]
                for name in compiled_names:
                    agg_exprs.extend([
                        pl.col(name).null_count().alias(f"{name}_null"),
                        pl.col(name).std().alias(f"{name}_std"),
                    ])

                stats_row = hf_enhanced.select(agg_exprs).collect(engine="streaming").row(0, named=True)
                total_len = stats_row["_total_len"]

                for name in compiled_names:
                    res = _evaluate_physical_stats(name, stats_row, total_len, self.prefilter_thresholds)
                    stage1_map[name] = res
                    print(f"  {'PASS' if res.passed else 'REJECT'} {res.to_summary()}")

                survivors = [n for n in compiled_names if stage1_map[n].passed]
                print(f"  Stage 1 summary: {len(survivors)}/{len(compiled_names)} passed")

            # ====================================================================
            # 3. Stage 2 Walk-forward
            # ====================================================================
            window_results: Dict[str, List[EvalResult]] = {n: [] for n in compiled_names}
            if survivors:
                months = sorted(
                    hf_lf.select(
                        (pl.col("day").dt.year() * 100 + pl.col("day").dt.month()).alias("_m")
                    )
                    .unique()
                    .collect(engine="streaming")
                    .to_series()
                    .to_list()
                )
                print(f"[Harness] Stage 2: walk-forward ({len(survivors)} features across {len(months)} windows)")
                window_results.update(
                    self._evaluate_windows(hf_enhanced, dret_lf, survivors, months)
                )

            # ====================================================================
            # 4. candidate_asts 
            # ====================================================================
            min_ratio = self.common_config.get("min_window_pass_ratio", MIN_WINDOW_PASS_RATIO)
            all_results: List[FeatureMiningResult] = []
            winner_names: List[str] = []

            for rec in compile_records:
                idx = rec["index"]
                ast = rec["raw_ast"]
                fp = rec["fingerprint"]
                feat_name = rec["name"]
                compile_err = rec["error"]

                if compile_err is not None:
                    all_results.append(
                        FeatureMiningResult(
                            feature_name=feat_name,
                            raw_ast=ast,
                            ast_fingerprint=fp,
                            stage1_prefilter=PrefilterResult(
                                feature_name=feat_name,
                                passed=False,
                                reason=f"Compile: {compile_err[:100]}",
                            ),
                        )
                    )
                    continue

                p_res = stage1_map.get(feat_name)
                wins = window_results.get(feat_name, [])
                valid_wins = [r for r in wins if r and r.is_success]
                
                n_pass = len(valid_wins)
                ratio = n_pass / len(wins) if wins else 0.0
                passed = bool(wins) and (ratio >= min_ratio)
                best_eval = max(valid_wins, key=lambda r: r.score) if valid_wins else None

                if passed:
                    winner_names.append(feat_name)

                all_results.append(
                    FeatureMiningResult(
                        feature_name=feat_name,
                        raw_ast=ast,
                        ast_fingerprint=fp,
                        stage1_prefilter=p_res,
                        stage2_eval=best_eval,
                        stage2_passed=passed,
                        final_score=best_eval.score if best_eval else -500.0,
                        window_pass_ratio=ratio,
                    )
                )
                print(f"  [Agg] {feat_name}: {n_pass}/{len(wins)} windows ({ratio:.0%}) -> {'PASS' if passed else 'FAIL'}")

            # ====================================================================
            # 5. Stage 4: dump
            # ====================================================================
            persisted_paths = self._persist_features(hf_enhanced, winner_names)

            # ====================================================================
            # 6. Stage 5: HPO 
            # ====================================================================
            if self.stage3_callback:
                winners = [r for r in all_results if r.stage2_passed]
                if winners:
                    print(f"[Harness] Stage 3: triggering HPO for {len(winners)} winner(s)")
                for r in winners:
                    try:
                        cb_kwargs = {"persisted_paths": persisted_paths.get(r.feature_name, [])}
                        r.stage3_triggered = bool(
                            self.stage3_callback(
                                r.feature_name,
                                r.raw_ast,
                                self.common_config,
                                **cb_kwargs,
                            )
                        )
                    except TypeError:
                        r.stage3_triggered = bool(
                            self.stage3_callback(
                                r.feature_name,
                                r.raw_ast,
                                self.common_config,
                            )
                        )
                    except Exception as e:
                        print(f"  [Harness] Stage 3 failed for {r.feature_name}: {e}")

            print(f"[Harness] complete in {time.perf_counter() - t_total:.1f}s")
            return all_results
