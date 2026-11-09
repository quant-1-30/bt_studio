#! /usr/bin/env python3

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
import numpy as np

from datetime import datetime
from typing import Any, Dict, List, Optional, Set


from bt_studio.compiler.ast import ast_fingerprint
from bt_studio.constant import LLM_RUN_DIR, FEATURE_DIR
from bt_studio.utils.io import (
    atomic_save_json,
    exclusive_file_lock,
    get_git_sha as _git_sha,
    json_default,
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
    """Replay Buffer

    - success: score Top-K 
    - failure: O(1) FIFO 
    """

    def __init__(self, max_success: int = 10, max_failed: int = 10):
        self.max_success = max_success
        self.max_failed = max_failed
        self.successful_history: List[Dict[str, Any]] = []
        self.failed_history: List[Dict[str, Any]] = []
        self._success_fps: Dict[str, float] = {}  # fp -> best_score
        self._failed_fps: Set[str] = set()

    @staticmethod
    def _extract_fp(ast_obj: Any, fallback_key: str = "") -> str:
        if not ast_obj:
            return fallback_key
        try:
            return ast_fingerprint(ast_obj)
        except Exception:
            return sha1_hex(ast_obj, length=12)

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
        if raw_score is None or not isinstance(raw_score, (int, float)) or np.isnan(raw_score):
            score = -500.0
        else:
            score = float(raw_score)
        item["score"] = score

        if fp and fp in self._success_fps:
            if score <= self._success_fps[fp]:
                return
            self.successful_history = [
                x for x in self.successful_history
                if self._extract_fp(x.get("ast")) != fp
            ]

        self.successful_history.append(item)
        if fp:
            self._success_fps[fp] = score

        self.successful_history.sort(
            key=lambda x: (
                float(x.get("score", -9999.0))
                if isinstance(x.get("score"), (int, float)) and not np.isnan(x.get("score"))
                else -9999.0
            ),
            reverse=True,
        )

        while len(self.successful_history) > self.max_success:
            removed = self.successful_history.pop()
            rem_fp = self._extract_fp(removed.get("ast"))
            if rem_fp and rem_fp in self._success_fps:
                self._success_fps.pop(rem_fp, None)

    def add_failure(self, item: Dict[str, Any]) -> None:
        ast_obj = item.get("ast")
        reason = str(item.get("reason", ""))
        ast_fp = self._extract_fp(ast_obj)
        fp = f"{ast_fp}_{reason[:30]}" if ast_fp else reason

        if fp and fp in self._failed_fps:
            return

        item["fp"] = fp  # resume and add_failure 
        self.failed_history.insert(0, item)
        if fp:
            self._failed_fps.add(fp)

        while len(self.failed_history) > self.max_failed:
            evicted = self.failed_history.pop()
            evicted_fp = evicted.get("fp")
            if not evicted_fp:
                e_ast_fp = self._extract_fp(evicted.get("ast"))
                e_reason = str(evicted.get("reason", ""))
                evicted_fp = f"{e_ast_fp}_{e_reason[:30]}" if e_ast_fp else e_reason
            self._failed_fps.discard(evicted_fp)

    def restore(self, rb: Dict[str, Any]) -> Set[str]:
        """Restore successful and failed history from persisted state dict and return seen candidate fingerprints."""
        seen_fps: Set[str] = set()
        self.successful_history = list(rb.get("successful_history", []))
        self.failed_history = list(rb.get("failed_history", []))
        self._success_fps.clear()
        self._failed_fps.clear()
        for it in self.successful_history:
            fp = self._extract_fp(it.get("ast"), it.get("name", ""))
            self._success_fps[fp] = float(it.get("score") or 0.0)
            if fp:
                seen_fps.add(fp)
        for it in self.failed_history:
            ast_fp = self._extract_fp(it.get("ast", {}), "")
            if ast_fp:
                seen_fps.add(ast_fp)
            reason = str(it.get("reason", ""))
            fp = it.get("fp") or (f"{ast_fp}_{reason[:30]}" if ast_fp else reason)
            if fp:
                self._failed_fps.add(fp)
        return seen_fps


# ---------------------------------------------------------------------------
# Flywheel Run Store & Viable Feature Manifest
# ---------------------------------------------------------------------------

def load_viable_features(feature_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """viable_features.json"""
    base = feature_dir or FEATURE_DIR
    manifest = read_json_safe(os.path.join(base, "agent", "viable_features.json"), default={}) or {}
    return list(manifest.get("features", []) or [])


class FlywheelRunStore:
    """Flywheel Run state.json、pending_step.json、step_*.json 与 viable_features.json """

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
                "step_counter": step_counter,
                "config_fingerprint": self.config_fingerprint,
                "target": target,
                "replay_buffer": {
                    "successful_history": buffer.successful_history,
                    "failed_history": buffer.failed_history,
                },
            },
            self.state_path,
        )

    def load_state(self) -> Dict[str, Any]:
        st = read_json_safe(self.state_path)
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
        if not isinstance(pend, dict):
            return None
        if int(pend.get("step_index", -1)) != expected_step:
            return None
        if pend.get("config_fingerprint") != self.config_fingerprint:
            return None
        return pend

    def clear_pending(self) -> None:
        try:
            if os.path.exists(self.pending_path):
                os.remove(self.pending_path)
        except OSError:
            pass

    def persist_step(
        self,
        step_index: int,
        agent_data: Dict[str, Any],
        results: List[FeatureMiningResult],
        target: str,
        buffer: ReplayBuffer,
        llm_usage: Any = None,
        timing: Optional[Dict[str, float]] = None,
    ) -> None:
        try:
            os.makedirs(self.run_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_hyp = agent_data.get("hypothesis_id", "unknown")
            safe_hyp = re.sub(r"[^\w\-]", "_", str(raw_hyp))
            step_id = f"step_{ts}_{safe_hyp}_s{step_index}"

            rec = {
                "step_index": step_index,
                "timestamp": ts,
                "git_sha": _git_sha(),
                "config_fingerprint": self.config_fingerprint,
                "timing": timing or {"llm_s": 0.0, "eval_s": 0.0, "step_s": 0.0},
                "llm_usage": llm_usage,
                "target": target,
                "hypothesis_id": raw_hyp,
                "economic_reasoning": agent_data.get("economic_reasoning", ""),
                "candidates": agent_data.get("sub_features", agent_data.get("candidates", [])),
                "features": [r.to_summary() for r in results],
                "replay_buffer": {
                    "successful_history": buffer.successful_history,
                    "failed_history": buffer.failed_history,
                },
            }
            rec_safe = json.loads(json.dumps(rec, default=json_default))
            path = os.path.join(self.run_dir, f"{step_id}.json")
            atomic_save_json(rec_safe, path)
            print(f"💾 [Flywheel] Artifact Cache: {path}")
        except Exception as e:
            logger.error(f"[Flywheel] Artifact Error: {e}")

    @staticmethod
    def export_viable_features(results: List[FeatureMiningResult]) -> None:
        try:
            viable = []
            for r in results:
                if not r.stage2_passed:
                    continue
                fp = r.ast_fingerprint or (ast_fingerprint(r.raw_ast) if r.raw_ast else "")
                prior_cfg = {}
                if r.stage2_eval and isinstance(r.stage2_eval.config, dict):
                    prior_cfg = {
                        k: r.stage2_eval.config[k]
                        for k in ("downsample", "motif_minutes", "threshold_r")
                        if k in r.stage2_eval.config
                    }
                viable.append({
                    "feature_id": r.feature_name,
                    "feature_name": r.feature_name,
                    "display_name": r.display_name or r.feature_name,
                    "ast_recipe": r.raw_ast,
                    "fingerprint": fp,
                    "final_score": float(r.final_score),
                    "window_pass_ratio": float(r.window_pass_ratio),
                    "prior_config": prior_cfg,
                    "parquet_dir": os.path.join(FEATURE_DIR, str(fp)[:12]),
                })

            manifest_path = os.path.join(FEATURE_DIR, "agent", "viable_features.json")
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            with exclusive_file_lock(manifest_path + ".lock"):
                existing: Dict[str, Dict] = {
                    item.get("fingerprint", ""): item for item in load_viable_features()
                }
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
    """RL Feature Mining Flywheel"""

    def __init__(
        self,
        harness: TwoStageAgentHarness,
        vendor: Optional[str] = None,
        max_depth: int = 4,
        min_exploration_steps: int = 4,
        run_dir: Optional[str] = None,
        config_fingerprint: str = "",
        warm_start: bool = False,
    ):
        self.harness = harness
        self.llm_vendor = get_llm_provider(vendor)
        self.max_depth = max_depth
        self._min_exploration_steps = min_exploration_steps
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
        """viable_features.json inject Top Prior seed"""
        for item in load_viable_features()[: self.buffer.max_success]:
            ast_obj = item.get("ast_recipe")
            if not ast_obj:
                continue
            fp = str(item.get("fingerprint") or ReplayBuffer._extract_fp(ast_obj))
            if fp:
                self._seen_candidate_fps.add(fp)
            raw_sc = float(item.get("final_score", 0.0) or 0.0)
            w_ratio = float(item.get("window_pass_ratio", 0.6) or 0.6)
            self.buffer.add_success({
                "name": item.get("display_name") or item.get("feature_name") or fp[:8],
                "feature_id": item.get("feature_id") or item.get("feature_name") or "",
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

    def _clean_json_str(self, text: str) -> str:
        return re.sub(r",\s*([\]}])", r"\1", text).strip()

    def _parse_llm_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text or not raw_text.strip():
            return None

        snippets = [b.strip() for b in reversed(re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL))]
        first_brace, last_brace = raw_text.find("{"), raw_text.rfind("}")
        if 0 <= first_brace < last_brace:
            snippets.append(raw_text[first_brace : last_brace + 1])

        for snippet in snippets:
            for candidate_str in (snippet, self._clean_json_str(snippet)):
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
            if "null" in reason:
                return "STAGE1_NULL"
            if "std" in reason or "dead" in reason:
                return "STAGE1_DEAD"
            return "STAGE1_RUNTIME"
        if r.stage2_eval is None:
            return "STAGE2_NO_EVAL"
        reason = (r.stage2_eval.reason or "").lower()
        if "trigger" in reason:
            return "STAGE2_TRIGGER"
        if "excess" in reason or "economic" in reason:
            return "STAGE2_ECON"
        return "STAGE2_PVAL"

    @classmethod
    def _failure_reason(cls, r: FeatureMiningResult) -> str:
        code = cls._failure_code(r)
        if r.stage1_prefilter is None:
            return "Rejected by compile / causal / depth / window guard"
        if not r.stage1_prefilter.passed:
            return f"Stage 1 prefilter: {r.stage1_prefilter.reason}"
        if r.stage2_eval is None:
            return "Stage 2 produced no evaluation result"
        return (
            f"{code}: pass ratio {r.window_pass_ratio:.0%} "
            f"({r.stage2_eval.reason or 'statistical tests not passed'})"
        )

    def step(self, target_description: str, hf_lf: Any, dret_lf: Any) -> List[FeatureMiningResult]:
        self._step_counter += 1
        print("\n" + "=" * 80)
        print(f"🚀 [Flywheel] Step {self._step_counter} Start Itering...")
        _t_step0 = time.time()

        _llm_s = 0.0
        store = self.store
        pending = store.take_pending(self._step_counter)
        if pending is not None:
            candidate_asts = pending["candidates"]
            hyp_id = pending.get("hypothesis_id", f"hyp_{self._step_counter}")
            reasoning = pending.get("economic_reasoning", "")
            print(f"♻️ [Flywheel] Step {self._step_counter} restore candidate ({hyp_id}, "
                  f"{len(candidate_asts)} ) — skip LLM")
        else:
            system_prompt = build_system_prompt(self.max_depth)
            user_prompt = build_rl_context_prompt(
                target_description,
                self.buffer.successful_history,
                self.buffer.failed_history,
            )

            _t_llm0 = time.time()
            try:
                raw_output = self.llm_vendor.generate(user_prompt, system_prompt)
                _llm_s = round(time.time() - _t_llm0, 1)
            except Exception as e:
                logger.error(f"[Flywheel] LLM Error: {e}")
                raise
            agent_data = self._parse_llm_json(raw_output)
            if not agent_data:
                self.buffer.add_failure({
                    "reason": "Invalid JSON format generated by LLM",
                    "ast": {"error_raw": raw_output[:150]},
                })
                self._save_state(target_description)
                return []

            reasoning = agent_data.get("economic_reasoning", "")
            candidate_asts = agent_data.get("sub_features", [])
            hyp_id = agent_data.get("hypothesis_id", f"hyp_{self._step_counter}")

            # [resume] 
            store.save_pending(self._step_counter, hyp_id, reasoning, candidate_asts)

        n_candidates = len(candidate_asts) if isinstance(candidate_asts, list) else 0
        print(f"💡 [Flywheel] [{hyp_id}]: {n_candidates}")

        if not candidate_asts or not isinstance(candidate_asts, list):
            self.buffer.add_failure({
                "reason": "Empty or non-list sub_features",
                "ast": {"hypothesis_id": hyp_id},
            })
            self._save_state(target_description)
            return []

        # DUPLICATE_AST avoid inject failed_history to occupany max_failed 
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

        # test suit for two_stage validation
        _eval_s = 0.0
        _t_eval0 = time.time()
        try:
            results = self.harness.run(novel_candidates, hf_lf, dret_lf)
            _eval_s = round(time.time() - _t_eval0, 1)
        except Exception as e:
            traceback.print_exc()
            for ast in candidate_asts:
                self.buffer.add_failure({
                    "reason": f"Harness runtime crash: {e}",
                    "ast": ast,
                })
            self._save_state(target_description)
            return []

        self._record_step_feedback(results, reasoning)

        step_payload = pending if pending is not None else {
            "hypothesis_id": hyp_id, "economic_reasoning": reasoning,
            "candidates": candidate_asts,
        }
        store.persist_step(
            self._step_counter, step_payload, results, target_description, self.buffer,
            llm_usage=getattr(self.llm_vendor, "last_usage", None),
            timing={"llm_s": _llm_s, "eval_s": _eval_s, "step_s": round(time.time() - _t_step0, 1)},
        )
        store.export_viable_features(results)

        # [resume] 
        store.clear_pending()
        self._save_state(target_description)

        return results

    def _record_step_feedback(self, results: List[FeatureMiningResult], reasoning: str) -> None:
        """Harness result ---> RL ReplayBuffer"""
        for r in results:
            target_ast = r.raw_ast
            if not target_ast:
                self.buffer.add_failure({
                    "code": "EMPTY_AST",
                    "reason": "Empty candidate AST payload",
                    "ast": target_ast,
                })
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
        """prompt → LLM → compile → Stage1/2 metrics → RL"""
        all_results: List[FeatureMiningResult] = []
        barren_rounds = 0
        remaining_steps = max(0, int(n_steps) - int(self._step_counter))
        required_new = max(1, int(min_new_success or 1))

        for _ in range(remaining_steps):
            try:
                results = self.step(target_description, hf_lf, dret_lf)
            except Exception as e:
                print(f"\n[Flywheel] Infra Error and Shutdwon (already finished {self._step_counter - 1} step): {e}")
                break
            all_results.extend(results)

            step_success_count = sum(1 for r in results if r.stage2_passed)
            if step_success_count >= required_new:
                barren_rounds = 0
            else:
                barren_rounds += 1
                min_explore = max(2, int(self._min_exploration_steps))
                if barren_rounds >= 2 and self._step_counter >= min_explore:
                    print(f"[Flywheel] {barren_rounds} no new feature, early stop (step {self._step_counter}/{n_steps})")
                    break

        return all_results