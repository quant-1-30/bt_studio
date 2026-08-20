#! /usr/bin/env python3
from __future__ import annotations

import glob
import json
import logging
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from bt_studio.constant import LLM_PENDING_DIR, TUNE_COLLAPSE_DIR, TUNE_MODEL_DIR
from bt_studio.default_config import build_exp_config
from bt_studio.engine import parse_dag, run_dag
from bt_studio.engine.ctx import ensure_ray

logger = logging.getLogger(__name__)

_INTAKE_INTERVAL = float(os.environ.get("BT_STUDIO_LLM_INTAKE_INTERVAL", "5"))


class TaskStatus(Enum):
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


class AsyncTaskManager:
    def __init__(
        self,
        llm_intake_dir: Optional[str] = None,
        max_queued_tasks: int = 50,    
        max_history_tasks: int = 200,  
    ):
        self._tasks: Dict[str, TaskResult] = {}
        self._max_queued_tasks = max_queued_tasks
        self._max_history_tasks = max_history_tasks
        self._lock = threading.Lock()
        self._queue: queue.Queue[Tuple[str, str, Dict[str, Any], Any]] = queue.Queue()
        self._running = True

        ensure_ray(num_cpus=12)

        self._worker = threading.Thread(
            target=self._queue_loop, daemon=True, name="dag-serial-worker"
        )
        self._worker.start()

        # LLM Monitor
        if llm_intake_dir is None and os.environ.get("BT_STUDIO_LLM_INTAKE") == "1":
            llm_intake_dir = LLM_PENDING_DIR
        self._intake_dir = llm_intake_dir
        if self._intake_dir:
            threading.Thread(target=self._intake_loop, daemon=True, name="llm-intake").start()

    # ------------------------------------------------------------------ #
    # Load Shedding
    # ------------------------------------------------------------------ #
    def submit_dag_task(self, dag_ref: str, config: Dict[str, Any]) -> str:
        exp_config = build_exp_config(config)
        ast_recipe = exp_config.get("ast_recipe")
        feature_col = exp_config.get("feature_col")
        if feature_col is None:
            raise ValueError("submit_dag_task: feature_col must be provided in config")

        dag = parse_dag(dag_ref)
        task_id = f"dag_{uuid.uuid4().hex[:8]}"

        with self._lock:
            # 1. Shedding
            queued_task_ids = [
                tid for tid, t in self._tasks.items() if t.status == TaskStatus.QUEUED
            ]
            if len(queued_task_ids) >= self._max_queued_tasks:
                
                victim_id = queued_task_ids[0] # oldest
                victim_task = self._tasks[victim_id]
                
                victim_task.status = TaskStatus.CANCELLED
                victim_task.message = f"因队列积压达到上限 ({self._max_queued_tasks})，被新任务挤出淘汰"
                victim_task.timestamp = datetime.now()
                logger.warning(
                    f"[Load Shedding] 队列拥堵，已主动淘汰最旧排队任务: {victim_id} (Feature={victim_task.feature_col})"
                )

            # 2. recycle record
            terminal_ids = [
                tid for tid, t in self._tasks.items() if t.status.is_terminal
            ]
            if len(terminal_ids) >= self._max_history_tasks:
                del self._tasks[terminal_ids[0]]

            # 3. register 
            self._tasks[task_id] = TaskResult(
                task_id=task_id,
                status=TaskStatus.QUEUED,
                feature_col=feature_col,
                dag=dag.name,
                message=f"排队中: DAG={dag.name} 等待 Ray 独占槽位...",
            )

        self._queue.put((task_id, dag_ref, exp_config, ast_recipe))
        logger.info(
            f"任务已入队: {task_id} (DAG={dag.name}, Feature={feature_col}), "
            f"当前有效排队深度={len([t for t in self._tasks.values() if t.status == TaskStatus.QUEUED])}"
        )
        return task_id

    # ------------------------------------------------------------------ #
    # Worker Loop
    # ------------------------------------------------------------------ #
    def _queue_loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            task_id, dag_ref, exp_config, ast_recipe = item
            try:
                with self._lock:
                    current_task = self._tasks.get(task_id)

                    if not current_task or current_task.status == TaskStatus.CANCELLED:
                        logger.info(f"任务 {task_id} 已处于取消/淘汰状态 Worker 直接跳过")
                        continue

                self._execute_pipeline(task_id, dag_ref, exp_config, ast_recipe)
            except Exception as e:
                logger.error(f"任务 {task_id} 执行崩溃: {e}")
                self._update(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=f"{e}\n{traceback.format_exc()}",
                    message="任务执行异常崩溃",
                )
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    # LLM Intake
    # ------------------------------------------------------------------ #
    def _intake_loop(self) -> None:
        while self._running:
            try:
                self._drain_intake_once()
            except Exception as e:
                logger.error(f"Intake Scan 异常: {e}\n{traceback.format_exc()}")
            time.sleep(_INTAKE_INTERVAL)

    def _drain_intake_once(self) -> int:
        if not self._intake_dir or not os.path.isdir(self._intake_dir):
            return 0

        enqueued = 0
        for path in sorted(glob.glob(os.path.join(self._intake_dir, "*.json"))):
            #
            if path.endswith(".consumed") or path.endswith(".invalid"):
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    task = json.load(f)
                
                dag_ref = task.pop("dag", "wfo_hpo")
                self.submit_dag_task(dag_ref, task)
                
                os.replace(path, path + ".consumed")
                enqueued += 1

            except Exception as e:
                # os.rename atomic ops
                logger.warning(f"Intake 文件损坏或语法非法 {path}: {e}")
                try:
                    os.replace(path, path + ".invalid")
                except OSError:
                    pass

        return enqueued

    # ------------------------------------------------------------------ 
    # Pipeline Execute 
    # ------------------------------------------------------------------ 

    def _execute_pipeline(
        self, task_id: str, dag_ref: str, exp_config: Dict[str, Any], ast_recipe: Any
        ) -> None:
        dag = parse_dag(dag_ref)
        feature_col = exp_config.get("feature_col")
        exec_start_time = time.time()

        self._update(
            task_id,
            status=TaskStatus.RUNNING,
            progress=0.05,
            message=f"DAG [{dag.name}] 执行中...",
        )

        engine_result = run_dag(dag, exp_config, ast_recipe)

        collapse = None
        if os.path.isdir(TUNE_COLLAPSE_DIR):
            matched_reports = []
            for rpath in glob.glob(os.path.join(TUNE_COLLAPSE_DIR, f"*{feature_col}*.json")):
                try:
                    mtime = os.path.getmtime(rpath)
                    if mtime >= exec_start_time - 2.0:
                        matched_reports.append((mtime, rpath))
                except OSError:
                    continue

            if matched_reports:
                matched_reports.sort(key=lambda x: x[0])
                latest_report_path = matched_reports[-1][1]
                try:
                    with open(latest_report_path, "r", encoding="utf-8") as f:
                        rep = json.load(f)
                    collapse = {
                        "verdict": rep.get("verdict", "unknown"),
                        "report_path": latest_report_path,
                    }
                except Exception as e:
                    logger.warning(f"读取坍塌报告失败: {latest_report_path}: {e}")

        self._update(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=1.0,
            message=f"DAG [{dag.name}] 执行完成 (Feature={feature_col})",
            result={
                "feature_col": feature_col,
                "dag": dag.name,
                "model_id": engine_result.last_model_id,
                "models_produced": engine_result.models_produced,
                "model_dir": TUNE_MODEL_DIR,
                "collapse": collapse,
            },
        )

    # ------------------------------------------------------------------ 
    # State Management
    # ------------------------------------------------------------------ 
    def _update(self, task_id: str, **kwargs) -> None:
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                task.timestamp = datetime.now()

    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TaskResult]:
        with self._lock:
            return self._tasks.copy()

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if task.status == TaskStatus.QUEUED:
                    task.status = TaskStatus.CANCELLED
                    task.message = "任务已在排队中被取消"
                    task.timestamp = datetime.now()
                    return True
                elif task.status == TaskStatus.RUNNING:
                    logger.warning(f"任务 {task_id} 正在运行中，无法即时中断")
                    return False
            return False

    def get_task_count(self) -> Dict[str, int]:
        with self._lock:
            counts = {
                "total": 0,
                "queued": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
            }
            for t in self._tasks.values():
                counts["total"] += 1
                counts[t.status.value] += 1
            return counts

    def shutdown(self, wait: bool = True) -> None:
        self._running = False
        if wait:
            self._queue.join()


# ---------------------------------------------------------------------- #
# Singleton
# ---------------------------------------------------------------------- #
_task_manager_instance: Optional[AsyncTaskManager] = None
_singleton_lock = threading.Lock()


def get_task_manager() -> AsyncTaskManager:
    global _task_manager_instance
    if _task_manager_instance is None:
        with _singleton_lock:
            if _task_manager_instance is None:
                _task_manager_instance = AsyncTaskManager()
    return _task_manager_instance
