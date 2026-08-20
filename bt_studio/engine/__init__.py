"""DAG engine (networkx-based): XML-declared pipelines parsed & executed.

- parser: minimal XML (id/fn/role/deps) → nx.DiGraph
- executor:   topological run, signature injection, fail-fast with
              repr_graph diagnostics; return-value model_id collection
- task_manager: queue and manage DAG execution jobs
"""

from .parser import (
    Dep, DepTerm, DagWindow, PipelineDag, DagValidationError,
    parse_dag,
)
from bt_studio.utils.paths import list_dags
from .executor import EngineResult, run_dag, repr_graph, NODE_REGISTRY, ensure_ray
from .task_manager import AsyncTaskManager, TaskResult, TaskStatus, get_task_manager


__all__ = [
    "Dep", "DepTerm", "DagWindow", "PipelineDag", "DagValidationError",
    "parse_dag", "list_dags",
    "EngineResult", "run_dag", "repr_graph", "NODE_REGISTRY",
    "ensure_ray",
    "AsyncTaskManager", "TaskResult", "TaskStatus", "get_task_manager",
]
