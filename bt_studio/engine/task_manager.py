#! /usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import time
import uuid
import queue
import threading
import logging
import traceback
from typing import Any, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from bt_studio.constant import TUNE_MODEL_DIR, TUNE_COLLAPSE_DIR, LLM_PENDING_DIR
from bt_studio.default_config import build_exp_config
from bt_studio.engine import parse_dag, run_dag
from bt_studio.engine.executor import ensure_ray

logger = logging.getLogger(__name__)

_INTAKE_INTERVAL = float(os.environ.get("BT_STUDIO_LLM_INTAKE_INTERVAL", "5"))


class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    """
        FIFO  + signal Worker  => Ray Occupy execute
        Intake thread listen LLM-Agent Deploy task
    """

    def __init__(self, llm_intake_dir: Optional[str] = None, max_concurrency: int = 100):
        self._tasks: Dict[str, TaskResult] = {}
        self._max_concurrency = max_concurrency
        self._lock = threading.Lock()
        self._queue: queue.Queue[Tuple[str, str, Dict[str, Any], Any]] = queue.Queue()

        # Orchestrator
        ensure_ray(num_cpus=12)

        # MainLoop Queue Worker
        self._worker = threading.Thread(
            target=self._queue_loop, daemon=True, name="dag-serial-worker"
        )
        self._worker.start()

        # LLM Scan threading 
        if llm_intake_dir is None and os.environ.get("BT_STUDIO_LLM_INTAKE") == "1":
            llm_intake_dir = LLM_PENDING_DIR

        self._intake_dir = llm_intake_dir

        self._intake_thread: Optional[threading.Thread] = None
        if self._intake_dir:
            self._intake_thread = threading.Thread(
                target=self._intake_loop, daemon=True, name="llm-intake"
            )
            self._intake_thread.start()
            logger.info(f"LLM Intake 监听已启用: {self._intake_dir}")

    # ------------------------------------------------------------------ #
    # submit Dag task
    # ------------------------------------------------------------------ #
    def submit_dag_task(self, dag_ref: str, config: Dict[str, Any]) -> str:
        exp_config = build_exp_config(config)
        ast_recipe = exp_config.get("ast_recipe")
        feature_col = exp_config.get("feature_col")
        if feature_col is None:
            raise ValueError("submit_dag_task: feature_col must be provided in config")
        
        # Fail-Fast
        dag = parse_dag(dag_ref)

        task_id = f"dag_{uuid.uuid4().hex[:8]}"
        with self._lock:
            if len(self._tasks) >= self._max_concurrency:
                oldest_key = next(iter(self._tasks))
                del self._tasks[oldest_key]

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
            f"当前排队深度={self._queue.qsize()}"
        )
        return task_id

    # ------------------------------------------------------------------ #
    # LLM Consumer
    # ------------------------------------------------------------------ #
    def _intake_loop(self) -> None:
        while True:
            try:
                self._drain_intake_once()
            except Exception as e:
                logger.error(f"Intake Scan: {e}\n{traceback.format_exc()}")
            time.sleep(_INTAKE_INTERVAL)

    def _drain_intake_once(self) -> int:
        if not self._intake_dir or not os.path.isdir(self._intake_dir):
            return 0

        enqueued = 0
        for path in sorted(glob.glob(os.path.join(self._intake_dir, "*.json"))):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    task = json.load(f)
                dag_ref = task.pop("dag", "wfo_hpo")
                self.submit_dag_task(dag_ref, task)
                os.replace(path, path + ".consumed")
                enqueued += 1
            except Exception as e:
                logger.warning(f"Intake Parse {path}: {e}")
                try:
                    os.replace(path, path + ".invalid")
                except OSError:
                    pass
        return enqueued

    # ------------------------------------------------------------------ #
    # Worker 
    # ------------------------------------------------------------------ #
    def _queue_loop(self) -> None:
        while True:
            task_id, dag_ref, exp_config, ast_recipe = self._queue.get()
            try:
                with self._lock:
                    current_task = self._tasks.get(task_id)
                    if current_task and current_task.status == TaskStatus.CANCELLED:
                        logger.info(f" taks {task_id} skip due to cancelled")
                        continue

                self._execute_pipeline(task_id, dag_ref, exp_config, ast_recipe)
            except Exception as e:
                logger.error(f"execute {task_id} crash {e}")
                self._update(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=f"{e}\n{traceback.format_exc()}",
                    message="任务执行异常崩溃",
                )
            finally:
                self._queue.task_done()

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
            for rpath in glob.glob(os.path.join(TUNE_COLLAPSE_DIR, f"*_{feature_col}.json")):
                mtime = os.path.getmtime(rpath)
                
                if mtime >= exec_start_time - 2.0:
                    matched_reports.append((mtime, rpath))

            if matched_reports:
                matched_reports.sort(key=lambda x: x[0])
                latest_report_path = matched_reports[-1][1]
                with open(latest_report_path, "r", encoding="utf-8") as f:
                    rep = json.load(f)
                collapse = {
                    "verdict": rep.get("verdict", "unknown"),
                    "report_path": latest_report_path,
                }

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

    # ------------------------------------------------------------------ #
    # state management
    # ------------------------------------------------------------------ #
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
                    logger.warning(f"任务 {task_id} 正在运行中，无法即时中断。")
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
                if t.status == TaskStatus.QUEUED:
                    counts["queued"] += 1
                elif t.status == TaskStatus.RUNNING:
                    counts["running"] += 1
                elif t.status == TaskStatus.COMPLETED:
                    counts["completed"] += 1
                elif t.status == TaskStatus.FAILED:
                    counts["failed"] += 1
                elif t.status == TaskStatus.CANCELLED:
                    counts["cancelled"] += 1
            return counts


# ---------------------------------------------------------------------- #
# singleton
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