#! /usr/bin/env python3

from __future__ import annotations

import glob
import logging
import os
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from bt_studio.compiler.ast import ast_fingerprint, compile_ast, compile_recipe
from bt_studio.constant import LLM_PENDING_DIR, TUNE_MODEL_DIR
from bt_studio.default_config import build_exp_config, resolve_feature_col
from bt_studio.engine import parse_dag, run_dag
from bt_studio.engine.ctx import ensure_ray
from bt_studio.schema import TaskResult, TaskStatus
from bt_studio.utils.io import read_json_safe
from .parser import list_dags

logger = logging.getLogger(__name__)

_INTAKE_INTERVAL = float(os.environ.get("BT_STUDIO_LLM_INTAKE_INTERVAL", "5"))


def _is_safe_dag_ref(dag_ref: Any) -> bool:
    """Validate that dag_ref is a whitelisted basename within DAGS_DIR."""
    if not isinstance(dag_ref, str) or not dag_ref:
        return False
    if os.path.isabs(dag_ref) or "/" in dag_ref or "\\" in dag_ref:
        return False
    name = dag_ref[:-4] if dag_ref.endswith(".xml") else dag_ref
    return name in set(list_dags())


class TaskRegistry:

    def __init__(self, max_queued: int, max_history: int):
        self._tasks: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()
        self._max_queued = max_queued
        self._max_history = max_history

    # -- 读写 ------------------------------------------------------------- #

    def add(self, task: TaskResult) -> int:
        with self._lock:
            self._shed_oldest_queued_locked()
            self._gc_terminal_locked()
            self._tasks[task.task_id] = task
            return sum(1 for t in self._tasks.values() if t.status == TaskStatus.QUEUED)

    def update(self, task_id: str, **kwargs) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.timestamp = datetime.now()

    def get(self, task_id: str) -> Optional[TaskResult]:
        with self._lock:
            return self._tasks.get(task_id)

    def all(self) -> Dict[str, TaskResult]:
        with self._lock:
            return self._tasks.copy()

    def counts(self) -> Dict[str, int]:
        with self._lock:
            out = {"total": 0, "queued": 0, "running": 0,
                   "completed": 0, "failed": 0, "cancelled": 0}
            for t in self._tasks.values():
                out["total"] += 1
                out[t.status.value] += 1
            return out

    def cancel_queued(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status == TaskStatus.QUEUED:
                task.status = TaskStatus.CANCELLED
                task.message = "任务已在排队中被手动取消"
                task.timestamp = datetime.now()
                return True
            return False

    def try_start_running(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.QUEUED:
                return False
            task.status = TaskStatus.RUNNING
            task.timestamp = datetime.now()
            return True


    def _shed_oldest_queued_locked(self) -> None:
        """Load Shedding) """
        queued = [tid for tid, t in self._tasks.items() if t.status == TaskStatus.QUEUED]
        if len(queued) < self._max_queued:
            return
        victim = self._tasks[queued[0]]
        victim.status = TaskStatus.CANCELLED
        victim.message = f"由于队列积压达到上限 ({self._max_queued})，被新任务挤出淘汰"
        victim.timestamp = datetime.now()
        logger.warning(f"[Load Shedding] 队列满载，淘汰排队任务: "
                       f"{victim.task_id} (Feature={victim.feature_col})")

    def _gc_terminal_locked(self) -> None:
        terminal = [(tid, t.timestamp) for tid, t in self._tasks.items()
                    if t.status.is_terminal]
        if len(terminal) >= self._max_history:
            terminal.sort(key=lambda x: x[1])
            del self._tasks[terminal[0][0]]


class AsyncTaskManager:

    def __init__(
        self,
        llm_intake_dir: Optional[str] = None,
        max_queued_tasks: int = 50,
        max_history_tasks: int = 200,
        ray_cpus: Optional[int] = None,
    ):
        self._registry = TaskRegistry(max_queued_tasks, max_history_tasks)
        self._queue: queue.Queue[Tuple[str, str, Dict[str, Any], Any]] = queue.Queue()
        self._running = True

        ensure_ray(num_cpus=ray_cpus)

        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="dag-serial-worker"
        )
        self._worker.start()

        if llm_intake_dir is None and os.environ.get("BT_STUDIO_LLM_INTAKE") == "1":
            llm_intake_dir = LLM_PENDING_DIR
        self._intake_dir = llm_intake_dir
        if self._intake_dir:
            self._intake_thread = threading.Thread(
                target=self._intake_loop, daemon=True, name="llm-intake"
            )
            self._intake_thread.start()

    # ------------------------------------------------------------------ #
    # Task Submission 
    # ------------------------------------------------------------------ #

    def submit_dag_task(self, dag_ref: str, config: Dict[str, Any]) -> str:
        if not _is_safe_dag_ref(dag_ref):
            raise ValueError(
                f"非法 DAG 引用 {dag_ref!r}: DAGS_DIR basename "
                f"({sorted(list_dags())}), 拒绝绝对路径/路径分隔符"
            )
        exp_config = build_exp_config(config)
        ast_recipe = exp_config.get("ast_recipe")

        try:
            feature_col = resolve_feature_col(
                exp_config.get("common_params"),
                ast_recipe,
                exp_config.get("feature_col"),
            )
        except ValueError as err:
            raise ValueError(
                "submit_dag_task: feature_col 必须在配置中显式声明 (顶层或 common_params 内)"
            ) from err

        # [契约预检] 若附带 ast_recipe, 入队前静态验证 AST/Recipe 可编译且满足因果律;
        # 若 feature_col 采用 f_{fp12} 约定, 服务端重算指纹核验一致性, 拦截损坏或篡改负载。
        if ast_recipe is not None:
            try:
                if isinstance(ast_recipe, list):
                    compile_recipe(ast_recipe, check_causal=True)
                elif isinstance(ast_recipe, dict):
                    compile_ast(ast_recipe, check_causal=True)
                else:
                    raise ValueError(f"非法 ast_recipe 类型: {type(ast_recipe).__name__}")
            except Exception as err:
                raise ValueError(f"submit_dag_task: ast_recipe 静态编译校验失败: {err}") from err

            expected_fp = ast_fingerprint(ast_recipe)
            if feature_col.startswith("f_") and len(feature_col) == 14 and feature_col != f"f_{expected_fp}":
                raise ValueError(
                    f"submit_dag_task: feature_col ({feature_col}) 与服务端重算指纹 (f_{expected_fp}) 不匹配"
                )

        dag = parse_dag(dag_ref)
        task_id = f"dag_{uuid.uuid4().hex[:8]}"

        depth = self._registry.add(TaskResult(
            task_id=task_id,
            status=TaskStatus.QUEUED,
            feature_col=feature_col,
            dag=dag.name,
            message=f"排队中: DAG={dag.name} 等待调度槽位...",
        ))
        self._queue.put((task_id, dag_ref, exp_config, ast_recipe))
        logger.info(f"任务已提交: {task_id} (DAG={dag.name}, Feature={feature_col}), "
                    f"当前排队深度={depth}")
        return task_id

    # ------------------------------------------------------------------ #
    # Worker Execution Loop
    # ------------------------------------------------------------------ #

    def _worker_loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            task_id, dag_ref, exp_config, ast_recipe = item
            try:
                if not self._registry.try_start_running(task_id):
                    logger.info(f"任务 {task_id} 已取消或淘汰，跳过执行")
                    continue

                self._execute_pipeline(task_id, dag_ref, exp_config, ast_recipe)
            except Exception as e:
                logger.error(f"任务 {task_id} 运行崩溃: {e}\n{traceback.format_exc()}")
                self._registry.update(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=f"{e}\n{traceback.format_exc()}",
                    message="任务执行异常崩溃",
                )
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    # LLM Intake Pipeline
    # ------------------------------------------------------------------ #

    def _intake_loop(self) -> None:
        while self._running:
            try:
                self._drain_intake_once()
            except Exception as e:
                logger.error(f"Intake 轮询异常: {e}\n{traceback.format_exc()}")
            time.sleep(_INTAKE_INTERVAL)

    def _drain_intake_once(self) -> int:
        if not self._intake_dir or not os.path.isdir(self._intake_dir):
            return 0

        enqueued = 0
        for path in sorted(glob.glob(os.path.join(self._intake_dir, "*.json"))):
            consumed_path = path + ".consumed"
            try:
                os.replace(path, consumed_path)   # rename-first 原子认领
            except OSError:
                continue

            try:
                task = read_json_safe(consumed_path)
                if not isinstance(task, dict):
                    raise ValueError("Invalid JSON object in pending file")

                task.pop("created_at", None)
                task.pop("source", None)
                task.pop("feature_id", None)
                task.pop("display_name", None)
                dag_ref = task.pop("dag", "wfo_hpo")

                self.submit_dag_task(dag_ref, task)
                enqueued += 1
                # 保留 .consumed 标记: 审计痕迹 + 重启后不重复入队
            except Exception as e:
                logger.warning(f"Intake 文件损坏或解析失败 {path}: {e}")
                try:
                    os.replace(consumed_path, path + ".invalid")
                except OSError:
                    pass

        return enqueued

    # ------------------------------------------------------------------ #
    # Pipeline Execution Core
    # ------------------------------------------------------------------ #

    def _execute_pipeline(
        self, task_id: str, dag_ref: str, exp_config: Dict[str, Any], ast_recipe: Any
    ) -> None:
        dag = parse_dag(dag_ref)
        common_params = exp_config.get("common_params") or {}
        feature_col = exp_config.get("feature_col") or common_params.get("feature_col")

        self._registry.update(task_id, progress=0.05,
                              message=f"DAG [{dag.name}] 执行中...")

        engine_result = run_dag(dag, exp_config, ast_recipe)

        self._registry.update(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=1.0,
            message=f"DAG [{dag.name}] 执行完成 (Feature={feature_col})",
            result={
                "task_id": engine_result.task_id or task_id,
                "feature_col": feature_col,
                "dag": dag.name,
                "model_id": engine_result.last_model_id,
                "models_produced": engine_result.models_produced,
                "model_dir": common_params.get("tune_model_dir", TUNE_MODEL_DIR),
                "collapse": getattr(engine_result, "collapse_report", None),
            },
        )

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def registry(self) -> TaskRegistry:
        return self._registry

    def snapshot(self, task_id: Optional[str] = None) -> Any:
        if task_id is not None:
            return self._registry.get(task_id)
        return {"tasks": self._registry.all(), "counts": self._registry.counts()}

    def shutdown(self, wait: bool = True) -> None:
        if wait:
            self._queue.join()
            self._running = False
        else:
            self._running = False
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if isinstance(item, tuple) and item:
                        self._registry.cancel_queued(str(item[0]))
                    self._queue.task_done()
                except (queue.Empty, ValueError):
                    break

        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception as e:
            logger.warning(f"Shutdown Ray failed: {e}")


# ---------------------------------------------------------------------- #
# Singleton Factory
# ---------------------------------------------------------------------- #

_task_manager_instance: Optional[AsyncTaskManager] = None
_singleton_lock = threading.Lock()


def get_task_manager(**kwargs: Any) -> AsyncTaskManager:
    global _task_manager_instance
    if _task_manager_instance is None:
        with _singleton_lock:
            if _task_manager_instance is None:
                _task_manager_instance = AsyncTaskManager(**kwargs)
    return _task_manager_instance
