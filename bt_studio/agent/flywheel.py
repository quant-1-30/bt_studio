from __future__ import annotations

import json
import os
import re
import traceback
import numpy as np
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .vendors import get_llm_provider
from .prompt import build_system_prompt, build_rl_context_prompt
from .harness import TwoStageAgentHarness, FeatureMiningResult
from bt_studio.compiler.ast import ast_fingerprint
from bt_studio.constant import LLM_RUN_DIR
from bt_studio.utils.io import atomic_save_json


class ReplayBuffer:
    """
    - success:  score desc Top-K and support fingerprint
    - failure:  fingerprint to unqiue
    """

    def __init__(self, max_success: int = 10, max_failed: int = 10):
        self.max_success = max_success
        self.max_failed = max_failed
        self.successful_history: List[Dict[str, Any]] = []
        self.failed_history: List[Dict[str, Any]] = []
        self._success_fps: Dict[str, float] = {}  # fp -> best_score

    def add_success(self, item: Dict[str, Any]) -> None:
        ast_obj = item.get("ast")
        fp = ast_fingerprint(ast_obj) if ast_obj else ""

        score = item.get("score")
        if score is None or not isinstance(score, (int, float)) or np.isnan(score):
            score = -500.0
            item["score"] = score

        # fixbug
        if fp and fp in self._success_fps:
            if score <= self._success_fps[fp]:
                return
            
            self.successful_history = [
                x for x in self.successful_history 
                if ast_fingerprint(x.get("ast")) != fp
            ]

        self.successful_history.append(item)
        if fp:
            self._success_fps[fp] = score

        # fixbug avoid Nan
        self.successful_history.sort(
            key=lambda x: (
                x.get("score") 
                if isinstance(x.get("score"), (int, float)) and not np.isnan(x.get("score"))
                else -9999.0
            ),
            reverse=True,
        )

        if len(self.successful_history) > self.max_success:
            removed = self.successful_history.pop()
            rem_fp = ast_fingerprint(removed.get("ast"))
            self._success_fps.pop(rem_fp, None)

    def add_failure(self, item: Dict[str, Any]) -> None: # FIFO
        
        ast_obj = item.get("ast")
        fp = ast_fingerprint(ast_obj) if ast_obj else str(item.get("reason", ""))

        if self.failed_history:
            prev_ast = self.failed_history[0].get("ast")
            prev_fp = ast_fingerprint(prev_ast) if prev_ast else str(self.failed_history[0].get("reason", ""))
            if fp and fp == prev_fp:
                return

        self.failed_history.insert(0, item)
        del self.failed_history[self.max_failed:]


class RLFeatureFlywheel:
    """
        LLM Hyperthesis -> Harness -> ReplayBuffer RL -> Artifact Persist
    """

    def __init__(
        self,
        # llm_call_fn: Callable[[str, str], str],
        harness: TwoStageAgentHarness,
        vendor: str=None,
        max_depth: int = 4,
    ):
        # self.llm_call_fn = llm_call_fn
        self.harness = harness
        self.llm_vendor = get_llm_provider(vendor) 
        self.max_depth = max_depth
        self.buffer = ReplayBuffer()
        self._step_counter = 0

    def _clean_json_str(self, text: str) -> str:
        """LLM JSON Format bug"""
        text = re.sub(r",\s*([\]}])", r"\1", text)
        return text.strip()

    def _parse_llm_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text or not raw_text.strip():
            return None

        # 1. retrieve markdown
        code_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
        for block in reversed(code_blocks): 
            cleaned = self._clean_json_str(block)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            snippet = self._clean_json_str(raw_text[first_brace : last_brace + 1])
            try:
                return json.loads(snippet)
            except json.JSONDecodeError as e:
                print(f"[Flywheel] JSON 括号裁剪解析失败: {e} | 片段: {snippet[:100]!r}")

        print(f"[Flywheel] 无法从 LLM 输出中解析有效 JSON | 原始片段: {raw_text[:120]!r}")
        return None

    @staticmethod
    def _failure_reason(r: FeatureMiningResult) -> str:
        if r.stage1_prefilter is None:
            return "Rejected by compile / causal / depth guard"
        if not r.stage1_prefilter.passed:
            return f"Stage 1 prefilter: {r.stage1_prefilter.reason}"
        if r.stage2_eval is None:
            return "Stage 2 produced no eval result"
        return (
            f"Stage 2 walk-forward pass ratio {r.window_pass_ratio:.0%} below threshold "
            f"({r.stage2_eval.reason or 'statistical tests not passed'})"
        )

    def step(self, target_description: str, hf_lf: Any, dret_lf: Any) -> List[FeatureMiningResult]:
        self._step_counter += 1
        print("\n" + "=" * 80)
        print(f"[Flywheel] Step {self._step_counter} 开始迭代...")

        system_prompt = build_system_prompt(self.max_depth)
        user_prompt = build_rl_context_prompt(
            target_description,
            self.buffer.successful_history,
            self.buffer.failed_history,
        )

        try:
            # raw_output = self.llm_call_fn(system_prompt, user_prompt)
            raw_output = self.llm_vendor.generate(user_prompt, system_prompt)
        except Exception as e:
            print(f"[Flywheel] LLM 调用异常: {e}")
            return []

        agent_data = self._parse_llm_json(raw_output)
        if not agent_data:
            self.buffer.add_failure({
                "reason": "Invalid JSON format generated by LLM",
                "ast": {"error_raw": raw_output[:150]},
            })
            return []

        reasoning = agent_data.get("economic_reasoning", "")
        candidate_asts = agent_data.get("sub_features", [])
        hyp_id = agent_data.get("hypothesis_id", f"hyp_{self._step_counter}")

        print(f"[Flywheel] [{hyp_id}] 提出候选特征 AST 数量: {len(candidate_asts) if isinstance(candidate_asts, list) else 0}")

        if not candidate_asts or not isinstance(candidate_asts, list):
            self.buffer.add_failure({
                "reason": "Empty or non-list sub_features",
                "ast": {"hypothesis_id": hyp_id},
            })
            return []

        # Harness Pipeline
        try:
            results = self.harness.run(candidate_asts, hf_lf, dret_lf)
        except Exception as e:
            traceback.print_exc()
            for ast in candidate_asts:
                self.buffer.add_failure({
                    "reason": f"Harness runtime crash: {e}",
                    "ast": ast,
                })
            return []

        # RL
        for r in results:
            target_ast = r.raw_ast
            if not target_ast:
                continue

            if r.stage2_passed:
                self.buffer.add_success({
                    "score": r.final_score,
                    "reasoning": reasoning,
                    "ast": target_ast,
                })
            else:
                self.buffer.add_failure({
                    "reason": self._failure_reason(r),
                    "ast": target_ast,
                })

        # Artifact
        self._persist_step(agent_data, results, target_description)

        return results

    def _persist_step(
        self,
        agent_data: Dict[str, Any],
        results: List[FeatureMiningResult],
        target: str,
    ) -> None:
        try:
            os.makedirs(LLM_RUN_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_hyp = agent_data.get("hypothesis_id", "unknown")
            
            # fixbug
            safe_hyp = re.sub(r"[^\w\-]", "_", str(raw_hyp))
            step_id = f"step_{ts}_{safe_hyp}_s{self._step_counter}"

            rec = {
                "step_index": self._step_counter,
                "timestamp": ts,
                "target": target,
                "hypothesis_id": raw_hyp,
                "economic_reasoning": agent_data.get("economic_reasoning", ""),
                "candidates": agent_data.get("sub_features", []),
                "features": [r.to_summary() for r in results],
                "replay_buffer": {
                    "n_success": len(self.buffer.successful_history),
                    "n_failure": len(self.buffer.failed_history),
                },
            }

            path = os.path.join(LLM_RUN_DIR, f"{step_id}.json")
            atomic_save_json(rec, path)
            print(f"[Flywheel] Artifact 保存至: {path}")
        except Exception as e:
            print(f"[Flywheel] Artifact 保存异常: {e}")