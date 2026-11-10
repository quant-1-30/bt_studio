#!/usr/bin/env python3
"""Unified data schema and state contracts across bt_studio."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from bt_studio.constant import MIN_WINDOW_PASS_RATIO, PREFILTER_THRESHOLDS


# ============================================================================
# Pipeline Schemas (Metrics)
# ============================================================================

@dataclass(frozen=True)
class HPOScoreInputs:
    """Structured inputs for HPO objective score calculation."""
    u_pval: float
    trigger_count: int
    cond_rets: np.ndarray
    uncond_rets: np.ndarray
    cond_momentum: np.ndarray
    tune_config: dict
    common_config: dict


# ============================================================================
# Agent Evaluation & Flywheel Schemas
# ============================================================================

@dataclass
class PrefilterResult:
    """Stage 1 physical prefilter result for a single candidate feature."""
    feature_name: str
    passed: bool = False
    std: float = 0.0
    null_ratio: float = 1.0
    reason: str = ""

    def to_summary(self) -> dict:
        return {
            "feature": self.feature_name,
            "passed": self.passed,
            "std": round(self.std, 8),
            "null_ratio": round(self.null_ratio, 4),
            "reason": self.reason,
        }

    @classmethod
    def from_stats(
        cls,
        feature_col: str,
        stats_dict: dict,
        total_len: int,
        thresholds: Optional[dict] = None,
    ) -> "PrefilterResult":
        cfg = {**PREFILTER_THRESHOLDS, **(thresholds or {})}
        if total_len == 0:
            return cls(feature_name=feature_col, reason="Empty DataFrame/Series")

        null_count = stats_dict.get(f"{feature_col}_null")
        if null_count is None:
            null_count = total_len

        std_val = stats_dict.get(f"{feature_col}_std")
        if std_val is None or not isinstance(std_val, (int, float)):
            std_val = 0.0

        null_ratio = float(null_count / total_len)
        if (total_len - null_count) < 10:
            return cls(
                feature_name=feature_col,
                null_ratio=null_ratio,
                reason="Too few non-null values (< 10)",
            )

        reasons = []
        if np.isnan(std_val) or std_val < cfg["min_std"]:
            reasons.append(f"std={std_val:.2e} < {cfg['min_std']:.0e} (dead water or nan)")
        if null_ratio > cfg["max_null_ratio"]:
            reasons.append(f"null_ratio={null_ratio:.2%} > {cfg['max_null_ratio']:.0%}")

        return cls(
            feature_name=feature_col,
            passed=not reasons,
            std=float(std_val) if not np.isnan(std_val) else 0.0,
            null_ratio=null_ratio,
            reason="; ".join(reasons) if reasons else "OK",
        )


@dataclass
class EvalResult:
    """Stage 2 single-window evaluation result."""
    feature_name: str
    status: str = "failed"
    score: float = -500.0
    u_pval: float = 1.0
    trigger_count: int = 0
    valid_sample_ratio: float = 0.0
    autocorr: float = 0.0
    elapsed_s: float = 0.0
    reason: str = ""
    fsm_matrix: Optional[dict] = field(default=None, repr=False)
    learned_motif: Optional[list] = field(default=None, repr=False)
    config: Optional[dict] = field(default=None, repr=False)

    @property
    def is_success(self) -> bool:
        return (
            self.status == "success"
            and np.isfinite(self.score)
            and self.score > -500.0
            and np.isfinite(self.u_pval)
            and 0.0 <= self.u_pval <= 1.0
        )

    def to_summary(self) -> dict:
        return {
            "feature": self.feature_name,
            "status": self.status,
            "score": round(self.score, 4),
            "u_pval": round(self.u_pval, 4),
            "triggers": self.trigger_count,
            "valid_ratio": round(self.valid_sample_ratio, 4),
            "autocorr": round(self.autocorr, 4),
            "elapsed_s": round(self.elapsed_s, 2),
            "reason": self.reason,
        }


@dataclass
class FeatureMiningResult:
    """End-to-end two-stage mining summary for a single candidate feature."""
    feature_name: str
    display_name: str = ""
    raw_ast: Any = None
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
            return "stage2_passed"
        if self.stage1_prefilter and self.stage1_prefilter.passed:
            return "stage1_prefilter"
        return "rejected"

    def to_summary(self) -> dict:
        return {
            "feature": self.feature_name,
            "display_name": self.display_name or self.feature_name,
            "fingerprint": self.ast_fingerprint,
            "stage": self.stage_passed,
            "score": round(self.final_score, 4),
            "window_pass_ratio": round(self.window_pass_ratio, 4),
            "prefilter": self.stage1_prefilter.to_summary() if self.stage1_prefilter else None,
            "eval": self.stage2_eval.to_summary() if self.stage2_eval else None,
        }

    @classmethod
    def from_candidate_windows(
        cls,
        rec: dict,
        stage1_map: Dict[str, PrefilterResult],
        window_results: Dict[str, List[EvalResult]],
        min_ratio: float,
        max_pval_gate: float,
    ) -> "FeatureMiningResult":
        feat_name = rec["name"]
        compile_err = rec["error"]
        if compile_err is not None:
            return cls(
                feature_name=feat_name,
                raw_ast=rec["raw_ast"],
                ast_fingerprint=rec["fingerprint"],
                stage1_prefilter=PrefilterResult(
                    feature_name=feat_name, passed=False, reason=f"Compile: {compile_err[:100]}"
                ),
            )

        wins = window_results.get(feat_name, [])
        valid_wins = [r for r in wins if r and r.is_success and r.u_pval <= max_pval_gate]
        n_pass = len(valid_wins)
        ratio = n_pass / len(wins) if wins else 0.0
        passed = bool(wins) and (ratio >= min_ratio)
        best_eval = max(valid_wins, key=lambda r: r.score) if valid_wins else None

        representative_eval = best_eval
        if not passed and wins:
            real_fails = [
                r for r in wins
                if r and not (r.is_success and r.u_pval <= max_pval_gate)
                and r.reason != "early_stop_mathematically_doomed"
            ]
            if real_fails:
                rep = real_fails[0]
                if rep.is_success and rep.u_pval > max_pval_gate and not rep.reason:
                    rep = replace(rep, reason=f"pval gate failed (u_pval={rep.u_pval:.4f} > {max_pval_gate})")
                representative_eval = rep
            elif wins[0] is not None:
                representative_eval = wins[0]

        print(f"  [Agg] {feat_name}: {n_pass}/{len(wins)} windows ({ratio:.0%}) -> {'PASS' if passed else 'FAIL'}")
        return cls(
            feature_name=feat_name,
            display_name=rec.get("display_name", feat_name),
            raw_ast=rec["raw_ast"],
            ast_fingerprint=rec["fingerprint"],
            stage1_prefilter=stage1_map.get(feat_name),
            stage2_eval=representative_eval,
            stage2_passed=passed,
            final_score=best_eval.score if best_eval else -500.0,
            window_pass_ratio=ratio,
        )


@dataclass
class WindowReplayState:
    """Mutable state tracker across Walk-Forward Stage 2 evaluation windows."""
    feature_names: List[str]
    total_windows: int
    min_ratio: float
    max_pval_gate: float
    max_allowed_failures: int
    results: Dict[str, List[EvalResult]] = field(default_factory=dict)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    active_features: Set[str] = field(default_factory=set)

    @classmethod
    def create(
        cls,
        feature_names: List[str],
        total_windows: int,
        common_config: Dict[str, Any],
    ) -> "WindowReplayState":
        min_ratio = float(common_config.get("min_window_pass_ratio", MIN_WINDOW_PASS_RATIO))
        max_pval_gate = float(common_config.get("u_pval", 0.1))
        max_allowed_failures = int(total_windows * (1.0 - min_ratio + 1e-9))
        return cls(
            feature_names=list(feature_names),
            total_windows=total_windows,
            min_ratio=min_ratio,
            max_pval_gate=max_pval_gate,
            max_allowed_failures=max_allowed_failures,
            results={f: [] for f in feature_names},
            failure_counts={f: 0 for f in feature_names},
            active_features=set(feature_names),
        )

    def replay_month(
        self,
        m: int,
        month_idx: int,
        month_res: Dict[str, EvalResult],
    ) -> None:
        """Record a single window's evaluation results and prune mathematically doomed features."""
        eval_targets = [f for f in self.feature_names if f in self.active_features]
        print(f"  [Stage2] window {month_idx + 1}/{self.total_windows}: {m} (回放, {len(eval_targets)} 个活跃特征)")
        for feat in eval_targets:
            res = month_res.get(feat) or EvalResult(
                feature_name=feat, status="failed", reason="missing_window_result"
            )
            self.results[feat].append(res)
            if not (res.is_success and res.u_pval <= self.max_pval_gate):
                if res.is_success:
                    print(f"    [Pval-Gate] {feat} u_pval={res.u_pval:.3f} > {self.max_pval_gate:.1f}, 该窗口不计通过")
                self.failure_counts[feat] += 1
                if self.failure_counts[feat] > self.max_allowed_failures:
                    self.active_features.discard(feat)
                    print(
                        f"    [Early-Stop] {feat} 在第 {month_idx + 1}/{self.total_windows} 窗口累计失败 "
                        f"{self.failure_counts[feat]} 次，无法达成 {self.min_ratio:.0%} 通过率，剪枝淘汰"
                    )

        for f in self.feature_names:
            if f not in self.active_features and len(self.results[f]) < (month_idx + 1):
                self.results[f].append(
                    EvalResult(feature_name=f, status="failed", reason="early_stop_mathematically_doomed")
                )

    def pad_remaining_windows(self) -> Dict[str, List[EvalResult]]:
        for f in self.feature_names:
            while len(self.results[f]) < self.total_windows:
                self.results[f].append(
                    EvalResult(
                        feature_name=f,
                        status="failed",
                        reason="early_stop_mathematically_doomed",
                    )
                )
        return self.results


# ============================================================================
# Engine & Task Manager Schemas
# ============================================================================

@dataclass(frozen=True)
class DepTerm:
    """Single dependency term in a DAG edge specification."""
    node: str
    key: Optional[str] = None
    cond: Optional[str] = None
    negated: bool = False


@dataclass(frozen=True)
class Dep:
    """Disjunctive dependency specification for a DAG node."""
    alternatives: Tuple[DepTerm, ...]

    def terms(self) -> Tuple[DepTerm, ...]:
        return self.alternatives


@dataclass(frozen=True)
class DagWindow:
    """Walk-forward window configuration for a PipelineDag."""
    from_: str = "first"
    step_key: str = "oos_step"
    train_key: str = "train_window"


@dataclass(frozen=True)
class PipelineDag:
    """Parsed and validated pipeline DAG specification."""
    name: str
    graph: Any
    window: DagWindow
    source_path: str

    @property
    def nodes(self):
        return self.graph.nodes


@dataclass
class EngineResult:
    """Execution summary returned by executor.run_dag."""
    dag_name: str
    windows_executed: int = 0
    models_produced: List[int] = field(default_factory=list)
    last_model_id: Optional[int] = None
    task_id: Optional[str] = None
    collapse_report: Optional[Dict[str, Any]] = None


class TaskStatus(Enum):
    """Lifecycle states for asynchronous DAG tasks."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)


@dataclass
class TaskResult:
    """Runtime status and result record for a managed DAG task."""
    task_id: str
    status: TaskStatus
    feature_col: str = ""
    dag: str = ""
    result: Any = None
    error: Optional[str] = None
    progress: float = 0.0
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "feature_col": self.feature_col,
            "dag": self.dag,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }
