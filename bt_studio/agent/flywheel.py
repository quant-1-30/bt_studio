#! /usr/bin/env python3
"""RL-driven Alpha feature mining flywheel, replay buffer, and persistent run store."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from bt_studio.compiler.ast import ast_fingerprint
from bt_studio.constant import FEATURE_DIR, LLM_RUN_DIR
from bt_studio.utils.io import (
    atomic_save_json,
    exclusive_file_lock,
    get_git_sha as _git_sha,
    read_json_safe,
    sha1_hex,
)

from .harness import FeatureMiningResult, TwoStageAgentHarness
from .prompt import build_rl_context_prompt, build_system_prompt
from .vendors import get_llm_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Bounded experience replay buffer (Top-K score winners + FIFO failure pool)."""

    def __init__(self, max_success: int = 10, max_failed: int = 10):
        self.max_success = max_success
        self.max_failed = max_failed
        self.successful_history: List[Dict[str, Any]] = []
        self.failed_history: List[Dict[str, Any]] = []
        self._success_fps: Dict[str, float] = {}
        self._failed_fps: Set[str] = set()

    @staticmethod
    def _extract_fp(ast_obj: Any, fallback_key: str = "") -> str:
        if not ast_obj:
            return fallback_key
        try:
            return ast_fingerprint(ast_obj)
        except Exception:
            return sha1_hex(ast_obj, length=12)

    @classmethod
    def _failure_fp(cls, item: Dict[str, Any]) -> str:
        if item.get("fp"):
            return str(item["fp"])
        ast_fp = cls._extract_fp(item.get("ast"), "")
        reason = str(item.get("reason", ""))
        return f"{ast_fp}_{reason[:30]}" if ast_fp else reason

    @staticmethod
    def shape_reward(raw_score: float, window_pass_ratio: float) -> float:
        """Compute bounded monotonic RL reward in [0, 1] combining window pass ratio (60%) and tanh score (40%)."""
        safe_raw = float(raw_score) if np.isfinite(raw_score) else 0.0
        score_unit = 0.5 * (float(np.tanh(safe_raw / 10.0)) + 1.0)
        return round(0.6 * float(window_pass_ratio) + 0.4 * score_unit, 6)

    def add_success(self, item: Dict[str, Any]) -> None:
        ast_obj = item.get("ast")
        fp = self._extract_fp(ast_obj)
        raw_score = item.get("score")
        score = (
            float(raw_score)
            if isinstance(raw_score, (int, float)) and np.isfinite(raw_score)
            else -500.0
        )
        item["score"] = score

        if fp and fp in self._success_fps:
            if score <= self._success_fps[fp]:
                return
            self.successful_history = [
                x for x in self.successful_history if self._extract_fp(x.get("ast")) != fp
            ]

        self.successful_history.append(item)
        if fp:
            self._success_fps[fp] = score

        self.successful_history.sort(key=lambda x: float(x.get("score", -9999.0)), reverse=True)
        while len(self.successful_history) > self.max_success:
            removed = self.successful_history.pop()
            rem_fp = self._extract_fp(removed.get("ast"))
            self._success_fps.pop(rem_fp, None)

    def add_failure(self, item: Dict[str, Any]) -> None:
        fp = self._failure_fp(item)
        if fp and fp in self._failed_fps:
            return
        item["fp"] = fp
        self.failed_history.insert(0, item)
        if fp:
            self._failed_fps.add(fp)

        while len(self.failed_history) > self.max_failed:
            evicted = self.failed_history.pop()
            self._failed_fps.discard(self._failure_fp(evicted))

    def restore(self, rb: Dict[str, Any]) -> Set[str]:
        """Restore successful and failed history from persisted state dict and return seen candidate fingerprints."""
        seen_fps: Set[str] = set()
        self.successful_history = list(rb.get("successful_history", []))
        self.failed_history = list(rb.get("failed_history", []))
        self._success_fps.clear()
        self._failed_fps.clear()
        for it in self.successful_history:
            fp = self._extract_fp(it.get("ast"), it.get("name", ""))
            self._success_fps[fp] = float(it.get("score", 0.0))
            if fp:
                seen_fps.add(fp)
        for it in self.failed_history:
            ast_fp = self._extract_fp(it.get("ast", {}), "")
            if ast_fp:
                seen_fps.add(ast_fp)
            fp = self._failure_fp(it)
            if fp:
                self._failed_fps.add(fp)
        return seen_fps


# ---------------------------------------------------------------------------
# Flywheel Run Store & Viable Feature Manifest
# ---------------------------------------------------------------------------

def load_viable_features(feature_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    base = feature_dir or FEATURE_DIR
    manifest = read_json_safe(os.path.join(base, "agent", "viable_features.json"), default={}) or {}
    return list(manifest.get("features", []))


class FlywheelRunStore:
    """Atomic persistence manager for state.json, pending_step.json, step_*.json, and viable_features.json."""

    def __init__(self, run_dir: str, config_fingerprint: str = ""):
        self.run_dir = run_dir
        self.config_fingerprint = config_fingerprint

    @property
    def state_path(self) -> str:
        return os.path.join(self.run_dir, "state.json")

    @property
    def pending_path(self) -> str:
        return os.path.join(self.run_dir, "pending_step.json")

    def save_state(self, step_counter: int, target: str, buffer: ReplayBuffer) -> None:
        atomic_save_json(
            {
                "version": 1,
                "config_fingerprint": self.config_fingerprint,
                "step_counter": step_counter,
                "updated_at": datetime.now().isoformat(),
                "target": target,
                "replay_buffer": {
                    "successful_history": buffer.successful_history,
                    "failed_history": buffer.failed_history,
                },
            },
            self.state_path,
        )

    def load_state(self) -> Dict[str, Any]:
        st = read_json_safe(self.state_path, default={})
        return st if isinstance(st, dict) else {}

    def save_pending(self, step_index: int, hyp_id: str, reasoning: str, candidates: Any) -> None:
        atomic_save_json(
            {
                "step_index": step_index,
                "config_fingerprint": self.config_fingerprint,
                "hypothesis_id": hyp_id,
                "economic_reasoning": reasoning,
                "candidates": candidates,
            },
            self.pending_path,
        )

    def take_pending(self, expected_step: int) -> Optional[Dict[str, Any]]:
        pend = read_json_safe(self.pending_path)
        if (
            isinstance(pend, dict)
            and int(pend.get("step_index", -1)) == expected_step
            and pend.get("config_fingerprint") == self.config_fingerprint
        ):
            return pend
        return None

    def clear_pending(self) -> None:
        try:
            if os.path.exists(self.pending_path):
                os.remove(self.pending_path)
        except OSError:
            pass

    def persist_step(
        self,
        step_index: int,
        results: List[FeatureMiningResult],
        buffer: ReplayBuffer,
        step_meta: Dict[str, Any],
    ) -> None:
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_hyp = step_meta.get("hypothesis_id", "unknown")
            safe_hyp = re.sub(r"[^\w\-]", "_", str(raw_hyp))
            path = os.path.join(self.run_dir, f"step_{ts}_{safe_hyp}_s{step_index}.json")

            rec = {
                "step_index": step_index,
                "timestamp": ts,
                "git_sha": _git_sha(),
                "config_fingerprint": self.config_fingerprint,
                "timing": step_meta.get("timing", {"llm_s": 0.0, "eval_s": 0.0, "step_s": 0.0}),
                "llm_usage": step_meta.get("llm_usage"),
                "target": step_meta.get("target", ""),
                "hypothesis_id": raw_hyp,
                "economic_reasoning": step_meta.get("economic_reasoning", ""),
                "candidates": step_meta.get("sub_features", step_meta.get("candidates", [])),
                "features": [r.to_summary() for r in results],
                "replay_buffer": {
                    "successful_history": buffer.successful_history,
                    "failed_history": buffer.failed_history,
                },
            }
            atomic_save_json(rec, path)
            print(f"💾 [Flywheel] Artifact Cache: {path}")
        except Exception as e:
            logger.error(f"[Flywheel] Artifact Error: {e}")

    @staticmethod
    def export_viable_features(results: List[FeatureMiningResult]) -> None:
        try:
            viable = []
            for r in (res for res in results if res.stage2_passed):
                fp = r.ast_fingerprint or (ast_fingerprint(r.raw_ast) if r.raw_ast else "")
                cfg = r.stage2_eval.config if (r.stage2_eval and isinstance(r.stage2_eval.config, dict)) else {}
                viable.append({
                    "feature_id": r.feature_name,
                    "feature_name": r.feature_name,
                    "display_name": r.display_name or r.feature_name,
                    "ast_recipe": r.raw_ast,
                    "fingerprint": fp,
                    "final_score": float(r.final_score),
                    "window_pass_ratio": float(r.window_pass_ratio),
                    "prior_config": {k: cfg[k] for k in ("downsample", "motif_minutes", "threshold_r") if k in cfg},
                    "parquet_dir": os.path.join(FEATURE_DIR, str(fp)[:12]),
                })

            manifest_path = os.path.join(FEATURE_DIR, "agent", "viable_features.json")
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            with exclusive_file_lock(manifest_path + ".lock"):
                existing = {item.get("fingerprint", ""): item for item in load_viable_features()}
                for item in viable:
                    existing[item["fingerprint"]] = item
                atomic_save_json(
                    {
                        "updated_at": datetime.now().isoformat(),
                        "n_features": len(existing),
                        "features": sorted(existing.values(), key=lambda x: -x["final_score"]),
                    },
                    manifest_path,
                )
            if viable:
                print(f"📦 [Flywheel] Viable features manifest -> {manifest_path} ({len(existing)} total)")
        except Exception as e:
            logger.error(f"[Flywheel] viable features export failed: {e}")


# ---------------------------------------------------------------------------
# RL Feature Flywheel
# ---------------------------------------------------------------------------

class RLFeatureFlywheel:
    """RL Feature Mining Flywheel."""

    def __init__(
        self,
        harness: TwoStageAgentHarness,
        vendor: Optional[str] = None,
        run_dir: Optional[str] = None,
        config_fingerprint: str = "",
        warm_start: bool = False,
    ):
        self.harness = harness
        self.llm_vendor = get_llm_provider(vendor)

        cc = getattr(harness, "common_config", None) or {}
        self.max_depth = int(cc.get("max_depth", 4))
        self._min_exploration_steps = int(cc.get("min_exploration_steps", 4))

        self.buffer = ReplayBuffer()
        self._step_counter = 0
        self._seen_candidate_fps: Set[str] = set()
        self._duplicate_count: int = 0
        self._config_fingerprint = config_fingerprint

        state_obj = read_json_safe(os.path.join(run_dir, "state.json")) if run_dir else None
        if isinstance(state_obj, dict) and state_obj.get("config_fingerprint") == config_fingerprint:
            self._run_dir = str(run_dir)
            self.store = FlywheelRunStore(self._run_dir, config_fingerprint)
            self._load_state(state_obj)
            print(f"[Flywheel] RESUME {run_dir} (从 step {self._step_counter + 1} 续跑, "
                  f"buffer: {len(self.buffer.successful_history)}s/{len(self.buffer.failed_history)}f)")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fp8 = (config_fingerprint or "nofp")[:8]
            self._run_dir = os.path.join(LLM_RUN_DIR, f"{ts}_{fp8}")
            os.makedirs(self._run_dir, exist_ok=True)
            self.store = FlywheelRunStore(self._run_dir, config_fingerprint)
            if warm_start or os.environ.get("AGENT_WARM_START") == "1":
                self._warm_start_from_viable()
            if run_dir and state_obj is not None:
                print(f"[Flywheel] not Match and new run: {self._run_dir}")

    def _warm_start_from_viable(self) -> None:
        for item in load_viable_features()[: self.buffer.max_success]:
            ast_obj = item.get("ast_recipe")
            if not ast_obj:
                continue
            fp = str(item.get("fingerprint", ReplayBuffer._extract_fp(ast_obj)))
            if fp:
                self._seen_candidate_fps.add(fp)
            raw_sc = float(item.get("final_score", 0.0))
            w_ratio = float(item.get("window_pass_ratio", 0.6))
            self.buffer.add_success({
                "name": item.get("display_name", item.get("feature_name", fp[:8])),
                "feature_id": item.get("feature_id", item.get("feature_name", "")),
                "score": ReplayBuffer.shape_reward(raw_sc, w_ratio),
                "raw_score": raw_sc,
                "reasoning": "Prior validated winner from viable_features.json",
                "ast": ast_obj,
            })

    def _state_path(self) -> str:
        return self.store.state_path

    def _pending_path(self) -> str:
        return self.store.pending_path

    def _save_state(self, target_description: str) -> None:
        self.store.save_state(self._step_counter, target_description, self.buffer)

    def _load_state(self, st: Optional[Dict[str, Any]] = None) -> None:
        st = st if isinstance(st, dict) else self.store.load_state()
        self._step_counter = int(st.get("step_counter", 0))
        self._seen_candidate_fps.update(self.buffer.restore(st.get("replay_buffer", {})))

    def _parse_llm_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text or not raw_text.strip():
            return None
        snippets = [b.strip() for b in reversed(re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL))]
        first_brace, last_brace = raw_text.find("{"), raw_text.rfind("}")
        if 0 <= first_brace < last_brace:
            snippets.append(raw_text[first_brace : last_brace + 1])

        for snippet in snippets:
            for candidate_str in (snippet, re.sub(r",\s*([\]}])", r"\1", snippet).strip()):
                try:
                    return json.loads(candidate_str)
                except json.JSONDecodeError:
                    continue
        logger.error(f"[Flywheel] Not effective JSON: {raw_text[:120]!r}")
        return None

    @staticmethod
    def _failure_code(r: FeatureMiningResult) -> str:
        if r.stage1_prefilter is None:
            return "COMPILE_OR_GUARD"
        if not r.stage1_prefilter.passed:
            reason = (r.stage1_prefilter.reason or "").lower()
            return "STAGE1_NULL" if "null" in reason else ("STAGE1_DEAD" if ("std" in reason or "dead" in reason) else "STAGE1_RUNTIME")
        if r.stage2_eval is None:
            return "STAGE2_NO_EVAL"
        reason = (r.stage2_eval.reason or "").lower()
        return "STAGE2_TRIGGER" if "trigger" in reason else ("STAGE2_ECON" if ("excess" in reason or "economic" in reason) else "STAGE2_PVAL")

    @classmethod
    def _failure_reason(cls, r: FeatureMiningResult) -> str:
        if r.stage1_prefilter is None:
            return "Rejected by compile / causal / depth / window guard"
        if not r.stage1_prefilter.passed:
            return f"Stage 1 prefilter: {r.stage1_prefilter.reason}"
        if r.stage2_eval is None:
            return "Stage 2 produced no evaluation result"
        return (
            f"{cls._failure_code(r)}: pass ratio {r.window_pass_ratio:.0%} "
            f"({r.stage2_eval.reason or 'statistical tests not passed'})"
        )

    def _acquire_step_candidates(
        self, target_description: str
    ) -> Optional[Tuple[str, str, Any, float, Optional[Dict[str, Any]]]]:
        pending = self.store.take_pending(self._step_counter)
        if pending is not None:
            cands = pending["candidates"]
            hyp_id = pending.get("hypothesis_id", f"hyp_{self._step_counter}")
            reasoning = pending.get("economic_reasoning", "")
            print(f"♻️ [Flywheel] Step {self._step_counter} restore candidate ({hyp_id}, {len(cands)}) — skip LLM")
            return hyp_id, reasoning, cands, 0.0, pending

        system_prompt = build_system_prompt(self.max_depth)
        user_prompt = build_rl_context_prompt(
            target_description, self.buffer.successful_history, self.buffer.failed_history
        )
        t_llm0 = time.time()
        try:
            raw_output = self.llm_vendor.generate(user_prompt, system_prompt)
            llm_s = round(time.time() - t_llm0, 1)
        except Exception as e:
            logger.error(f"[Flywheel] LLM Error: {e}")
            raise

        agent_data = self._parse_llm_json(raw_output)
        if not agent_data:
            self.buffer.add_failure({"reason": "Invalid JSON format generated by LLM", "ast": {"error_raw": raw_output[:150]}})
            self._save_state(target_description)
            return None

        reasoning = agent_data.get("economic_reasoning", "")
        raw_cands = agent_data.get("sub_features", agent_data.get("candidates", []))
        if isinstance(raw_cands, dict) and ("op" in raw_cands or "col" in raw_cands or "ast" in raw_cands):
            raw_cands = [raw_cands]
        hyp_id = agent_data.get("hypothesis_id", f"hyp_{self._step_counter}")
        self.store.save_pending(self._step_counter, hyp_id, reasoning, raw_cands)
        return hyp_id, reasoning, raw_cands, llm_s, None

    def step(self, target_description: str, hf_lf: Any, dret_lf: Any) -> List[FeatureMiningResult]:
        self._step_counter += 1
        print("\n" + "=" * 80)
        print(f"🚀 [Flywheel] Step {self._step_counter} Start Itering...")
        t_step0 = time.time()

        acquired = self._acquire_step_candidates(target_description)
        if acquired is None:
            return []
        hyp_id, reasoning, candidate_asts, llm_s, pending = acquired

        n_candidates = len(candidate_asts) if isinstance(candidate_asts, list) else 0
        print(f"💡 [Flywheel] [{hyp_id}]: {n_candidates}")
        if not candidate_asts or not isinstance(candidate_asts, list):
            self.buffer.add_failure({"reason": "Empty or non-list sub_features", "ast": {"hypothesis_id": hyp_id}})
            self._save_state(target_description)
            return []

        novel_candidates = []
        for cand in candidate_asts:
            if not cand:
                novel_candidates.append(cand)
                continue
            cand_fp = ReplayBuffer._extract_fp(cand)
            if cand_fp and cand_fp in self._seen_candidate_fps:
                self._duplicate_count += 1
                print(f"  [Flywheel] SKIP duplicate AST (fp={cand_fp}, 累计重复={self._duplicate_count}) -> 免评跳过")
            else:
                if cand_fp:
                    self._seen_candidate_fps.add(cand_fp)
                novel_candidates.append(cand)

        if not novel_candidates:
            self._save_state(target_description)
            return []

        t_eval0 = time.time()
        try:
            results = self.harness.run(novel_candidates, hf_lf, dret_lf)
            eval_s = round(time.time() - t_eval0, 1)
        except Exception as e:
            traceback.print_exc()
            for ast in candidate_asts:
                self.buffer.add_failure({"reason": f"Harness runtime crash: {e}", "ast": ast})
            self._save_state(target_description)
            return []

        self._record_step_feedback(results, reasoning)

        step_payload = dict(
            pending if pending is not None else {
                "hypothesis_id": hyp_id,
                "economic_reasoning": reasoning,
                "candidates": candidate_asts,
            },
            target=target_description,
            llm_usage=getattr(self.llm_vendor, "last_usage", None),
            timing={"llm_s": llm_s, "eval_s": eval_s, "step_s": round(time.time() - t_step0, 1)},
        )
        self.store.persist_step(self._step_counter, results, self.buffer, step_payload)
        self.store.export_viable_features(results)
        self.store.clear_pending()
        self._save_state(target_description)
        return results

    def _record_step_feedback(self, results: List[FeatureMiningResult], reasoning: str) -> None:
        for r in results:
            target_ast = r.raw_ast
            if not target_ast:
                self.buffer.add_failure({"code": "EMPTY_AST", "reason": "Empty candidate AST payload", "ast": target_ast})
                continue
            if r.stage2_passed:
                raw_sc = float(r.final_score) if np.isfinite(r.final_score) else 0.0
                self.buffer.add_success({
                    "name": r.display_name or r.feature_name,
                    "feature_id": r.feature_name,
                    "score": ReplayBuffer.shape_reward(raw_sc, float(r.window_pass_ratio)),
                    "raw_score": raw_sc,
                    "reasoning": str(reasoning)[:600],
                    "ast": target_ast,
                })
            else:
                self.buffer.add_failure({
                    "name": r.display_name or r.feature_name,
                    "code": self._failure_code(r),
                    "reason": self._failure_reason(r)[:240],
                    "window_pass_ratio": float(r.window_pass_ratio),
                    "ast": target_ast,
                })

    def run(
        self,
        target_description: str,
        hf_lf: Any,
        dret_lf: Any,
        n_steps: int = 3,
        min_new_success: int = 1,
    ) -> List[FeatureMiningResult]:
        all_results: List[FeatureMiningResult] = []
        barren_rounds = 0
        remaining_steps = max(0, int(n_steps) - int(self._step_counter))
        required_new = max(1, int(min_new_success or 1))
        min_explore = max(2, int(self._min_exploration_steps))

        for _ in range(remaining_steps):
            try:
                results = self.step(target_description, hf_lf, dret_lf)
            except Exception as e:
                print(f"\n[Flywheel] Infra Error and Shutdown (already finished {self._step_counter - 1} step): {e}")
                break
            all_results.extend(results)

            if sum(1 for r in results if r.stage2_passed) >= required_new:
                barren_rounds = 0
            else:
                barren_rounds += 1
                if barren_rounds >= 2 and self._step_counter >= min_explore:
                    print(f"[Flywheel] {barren_rounds} no new feature, early stop (step {self._step_counter}/{n_steps})")
                    break

        return all_results
