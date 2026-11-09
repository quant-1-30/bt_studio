#! /usr/bin/env python3

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import networkx as nx

from typing import Dict, List, Optional, Tuple

from bt_studio.constant import DAGS_DIR
from bt_studio.schema import DagWindow, Dep, DepTerm, PipelineDag


_ROLES = {"train", "oos", "prev_oos", "warmup"}
_WINDOW_FROM_VALUES = {"first", "last-trainable"}
_CTX_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# node_id [> ctx_key] [?|?! cond_node]
_TERM_RE = re.compile(
    r"^(?P<node>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:>(?P<key>[A-Za-z_][A-Za-z0-9_]*))?"
    r"(?:\?(?P<neg>!)?(?P<cond>[A-Za-z_][A-Za-z0-9_]*))?$"
)


class DagValidationError(Exception):
    """Raised when a DAG XML file is malformed or fails semantic validation."""


def _resolve_path(ref: str) -> str:
    if ref.endswith(".xml"):
        if os.path.isabs(ref) or os.path.exists(ref):
            return ref
        return os.path.join(DAGS_DIR, ref)
    return os.path.join(DAGS_DIR, f"{ref}.xml")


def _parse_deps(raw: str, path: str, nid: str) -> Tuple[Dep, ...]:
    deps: List[Dep] = []
    for slot in raw.split(","):
        slot = slot.strip()
        if not slot:
            raise DagValidationError(f"{path}: node {nid}: 存在空的依赖槽位")
        alts: List[DepTerm] = []
        for part in slot.split("|"):
            part = part.strip()
            m = _TERM_RE.match(part)
            if not m:
                raise DagValidationError(f"{path}: node {nid}: 非法依赖语法 {part!r}")
            alts.append(
                DepTerm(
                    node=m.group("node"),
                    key=m.group("key"),
                    cond=m.group("cond"),
                    negated=bool(m.group("neg")),
                )
            )
        deps.append(Dep(alternatives=tuple(alts)))
    return tuple(deps)


def list_dags() -> list[str]:
    if not os.path.isdir(DAGS_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(DAGS_DIR)
                  if f.endswith(".xml"))


def parse_dag(ref: str) -> PipelineDag:
    path = _resolve_path(ref)
    if not os.path.exists(path):
        raise DagValidationError(f"DAG 文件不存在: {ref} (解析路径: {path})")

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise DagValidationError(f"{path}: XML 解析异常: {e}") from e

    dag_name = root.get("name")
    if root.tag != "pipeline" or not dag_name:
        raise DagValidationError(f"{path}: 根节点必须为带有有效 name 属性的 <pipeline>")

    window = DagWindow()
    w_el = root.find("window")
    if w_el is not None:
        from_ = w_el.get("from", "first")
        if from_ not in _WINDOW_FROM_VALUES:
            raise DagValidationError(
                f"{path}: window from={from_!r} 非法 (允许值: {sorted(_WINDOW_FROM_VALUES)})"
            )
        window = DagWindow(
            from_=from_,
            step_key=w_el.get("step-key", "oos_step"),
            train_key=w_el.get("train-window-key", "train_window"),
        )

    g = nx.MultiDiGraph()
    deps_by_node: Dict[str, Tuple[Dep, ...]] = {}

    for el in root.findall("node"):
        nid, fn = el.get("id"), el.get("fn")
        if not nid or not fn:
            raise DagValidationError(f"{path}: <node> 必须具备 id 与 fn 属性")
        if nid in g:
            raise DagValidationError(f"{path}: 发现重复节点 id: {nid!r}")

        role = el.get("role")
        if role is not None and role not in _ROLES:
            raise DagValidationError(f"{path}: node {nid}: role={role!r} 不合法")

        output = el.get("output")
        if output is not None and not _CTX_KEY_RE.match(output):
            raise DagValidationError(
                f"{path}: node {nid}: output={output!r} 必须为合法的上下文标识符"
            )

        raw_deps = (el.get("deps") or "").strip()
        node_deps = _parse_deps(raw_deps, path, nid) if raw_deps else ()

        g.add_node(nid, fn=fn, role=role, output=output, deps=node_deps)
        deps_by_node[nid] = node_deps

    for nid, deps in deps_by_node.items():
        for dep in deps:
            for t in dep.terms():
                if t.node == nid or t.cond == nid:
                    raise DagValidationError(
                        f"{path}: 节点 {nid!r} 不允许依赖自身 (Self-dependency forbidden)"
                    )

                if t.node not in g:
                    raise DagValidationError(
                        f"{path}: node {nid}: 依赖了未定义的节点 {t.node!r}"
                    )

                if t.cond:
                    if t.cond not in g:
                        raise DagValidationError(
                            f"{path}: node {nid}: 条件门控依赖了未定义的节点 {t.cond!r}"
                        )
                    ctrl_key = f"ctrl_{t.cond}"
                    if nid not in g[t.cond] or ctrl_key not in g[t.cond][nid]:
                        g.add_edge(
                            t.cond,
                            nid,
                            key=ctrl_key,
                            edge_type="control_cond",
                        )

                neg_flag = "not_" if t.negated else ""
                data_key = f"data_{t.node}_{t.key or 'all'}_{neg_flag}{t.cond or 'direct'}"
                g.add_edge(
                    t.node,
                    nid,
                    key=data_key,
                    edge_type="data",
                    key_alias=t.key,
                    cond=t.cond,
                    negated=t.negated,
                )

    if not nx.is_directed_acyclic_graph(g):
        cycle = list(nx.find_cycle(g, orientation="original"))
        raise DagValidationError(f"{path}: 检测到静态依赖闭环: {cycle}")

    return PipelineDag(
        name=str(dag_name),
        graph=g,
        window=window,
        source_path=path,
    )


__all__ = [
    "DagValidationError",
    "DepTerm",
    "Dep",
    "DagWindow",
    "PipelineDag",
    "parse_dag",
    "list_dags",
]