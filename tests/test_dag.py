#!/usr/bin/env python3

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bt_studio.engine.parser import (
    parse_dag, list_dags, DagValidationError,
)
from bt_studio.engine import executor as ex


def test_parser():
    prod = parse_dag("wfo_production")
    hpo = parse_dag("wfo_hpo")
    assert prod.name == "wfo_production" and hpo.name == "wfo_hpo"

    assert set(prod.graph.nodes) == {
        "macro", "extract_train", "extract_oos", "extract_warmup",
        "decay", "train_data", "tune", "update", "oos"}
    assert set(hpo.graph.nodes) == {"macro", "extract_train", "train_data", "tune"}

    # conditional edges carry cond/negated attrs
    tune_deps = prod.graph.nodes["tune"]["deps"]
    conds = {(t.cond, t.negated) for d in tune_deps for t in d.terms()}
    assert ("decay", False) in conds, conds
    update_deps = prod.graph.nodes["update"]["deps"]
    uconds = {(t.cond, t.negated) for d in update_deps for t in d.terms()}
    assert ("decay", True) in uconds, uconds
    oos_alts = [t for d in prod.graph.nodes["oos"]["deps"] for t in d.terms()
                if t.cond is None]
    assert {"tune", "update"} <= {t.node for t in oos_alts}

    assert set(list_dags()) >= {"wfo_production", "wfo_hpo"}
    print("[ok] parser v3: nx graph + conditional deps")


def test_parser_negatives():
    tmp = tempfile.mkdtemp()

    def write(body):
        p = os.path.join(tmp, "t.xml")
        with open(p, "w") as f:
            f.write(body)
        return p

    cases = [
        ('<pipeline name="a"><node id="x" fn="node_prepare_macro" deps="ghost"/></pipeline>', "unknown dep"),
        ('<pipeline name="a"><node id="x" fn="f" deps="y"/><node id="y" fn="f" deps="x"/></pipeline>', "cycle"),
        ('<pipeline name="a"><node id="x" fn="node_prepare_macro" role="bogus"/></pipeline>', "bad role"),
        ('<pipeline name="a"><node id="x" fn="node_prepare_macro" deps="x>bad!"/></pipeline>', "bad term"),
    ]
    for body, why in cases:
        try:
            parse_dag(write(body))
            raise SystemExit(f"FAIL accepted: {why}")
        except DagValidationError:
            pass
    print("[ok] parser v3: negatives rejected (incl. nx cycle path)")


class MockNodes:
    def __init__(self):
        self.calls = []
        self.decay_verdict = True # True = needs tuning, False = healthy decay

    def fake_macro(self, common_config):
        self.calls.append(("macro",))
        return {"dret_path": "/synthetic/dret.parquet", "universe_sids": {}}

    def fake_extract(self, ymonths, universe_sids, common_config, ast_recipe, feature_col=None, raw_hf_lf=None):
        self.calls.append(("extract", tuple(ymonths)))
        assert ast_recipe is not None, "ast_recipe is mandatory now"
        return [f"/f/hf_feat_{ym}.parquet" for ym in ymonths]

    def fake_decay(self, prev_model_id, prev_oos_yms, dret_path, train_paths, common_config):
        self.calls.append(("decay", prev_model_id, tuple(prev_oos_yms)))
        from bt_studio.utils.months import paths_for_months
        assert len(paths_for_months(train_paths, prev_oos_yms)) == len(prev_oos_yms)
        # return True for tune branch, False for update branch
        return prev_model_id is None or self.decay_verdict

    def fake_train_data(self, dret_path, train_paths, common_config):
        self.calls.append(("train_data", len(train_paths)))
        return {"stub": True}

    def fake_tune(self, prev_model_id, model_id, train_data, exp_config):
        self.calls.append(("tune", prev_model_id, model_id))
        return model_id                       # engine convention: model_id | None

    def fake_update(self, model_id, prev_model_id, dret_path, train_paths, common_config):
        self.calls.append(("update", model_id, prev_model_id))
        return model_id

    def fake_oos(self, model_id, dret_path, oos_yms, oos_paths, warmup_paths, common_config):
        self.calls.append(("oos", model_id, tuple(oos_yms)))
        assert oos_paths and warmup_paths


def _setup_mock_executor(mocks):
    ex.NODE_REGISTRY.clear()
    ex.NODE_REGISTRY.update({
        "node_prepare_macro": mocks.fake_macro,
        "node_extract_feature_monthly": mocks.fake_extract,
        "node_check_decay_monthly": mocks.fake_decay,
        "node_prepare_train_data": mocks.fake_train_data,
        "node_tune_monthly": mocks.fake_tune,
        "node_update_fsm_matrix": mocks.fake_update,
        "node_oos_inference_monthly": mocks.fake_oos,
    })


def test_executor_wfo_production():
    saved = dict(ex.NODE_REGISTRY)
    saved_lam = ex.list_available_months
    try:
        mocks = MockNodes()
        _setup_mock_executor(mocks)
        months = [202101 + m for m in range(12)] + [202201, 202202, 202203]
        ex.list_available_months = lambda p: list(months)

        exp = {"common_params": {"train_window": 3, "oss_step": 3, "num_workers": 1,
                                 "feature_col": "f"},
               "search_bounds": {}}

        # production: 4 windows — w1 tune (cold start), w2-4 inherit model (update)
        mocks.decay_verdict = False
        mocks.calls.clear()
        res = ex.run_dag(parse_dag("wfo_production"), exp, ast_recipe={"op": "x"})
        
        assert res.windows_executed == 4
        # cold start always tunes. Subsequent windows have decay_verdict=False, so they should UPDATE.
        kinds = [c[0] for c in mocks.calls]
        assert kinds.count("tune") == 1, f"Expected 1 tune, got {kinds.count('tune')}"
        assert kinds.count("update") == 3, f"Expected 3 updates, got {kinds.count('update')}"
        assert kinds.count("oos") == 4
        assert res.models_produced == [202103, 202106, 202109, 202112]
        assert res.last_model_id == 202112
        print("[ok] executor wfo_production: full loop, correct models_produced and branches")
        
        # Test forced decay scenario: always tuning
        mocks.decay_verdict = True
        mocks.calls.clear()
        res_decay = ex.run_dag(parse_dag("wfo_production"), exp, ast_recipe={"op": "x"})
        kinds = [c[0] for c in mocks.calls]
        assert kinds.count("tune") == 4, f"Expected 4 tunes due to decay, got {kinds.count('tune')}"
        assert kinds.count("update") == 0, f"Expected 0 updates, got {kinds.count('update')}"
        print("[ok] executor wfo_production: forced decay triggers re-tune correctly")
    finally:
        ex.NODE_REGISTRY.clear()
        ex.NODE_REGISTRY.update(saved)
        ex.list_available_months = saved_lam


def test_executor_wfo_hpo():
    saved = dict(ex.NODE_REGISTRY)
    saved_lam = ex.list_available_months
    try:
        mocks = MockNodes()
        _setup_mock_executor(mocks)
        months = [202101 + m for m in range(12)] + [202201, 202202, 202203]
        ex.list_available_months = lambda p: list(months)

        exp = {"common_params": {"train_window": 3, "oss_step": 3, "num_workers": 1,
                                 "feature_col": "f"},
               "search_bounds": {}}

        # hpo: single window, train slice only, no decay/update/oos calls
        mocks.calls.clear()
        res_h = ex.run_dag(parse_dag("wfo_hpo"), exp, ast_recipe={"op": "x"})
        kinds = [c[0] for c in mocks.calls]
        assert res_h.windows_executed == 1
        assert kinds.count("tune") == 1
        assert "decay" not in kinds and "update" not in kinds and "oos" not in kinds
        
        extracts = [c[1] for c in mocks.calls if c[0] == "extract"]
        # With list of 15 months and train_window 3, n_months is 15. The slice is from 12 to 15.
        # indices 12:15 are 202201, 202202, 202203. Let's verify.
        # Wait, the list is: [202101..202112] + [202201..202203]. Total 15.
        # The slice yms[idx - train_window : idx] where idx = 15
        assert extracts == [(202201, 202202, 202203)]
        print("[ok] executor wfo_hpo: pruned graph + last-trainable window resolved correctly")
    finally:
        ex.NODE_REGISTRY.clear()
        ex.NODE_REGISTRY.update(saved)
        ex.list_available_months = saved_lam


def test_executor_resume_and_fail_fast():
    saved = dict(ex.NODE_REGISTRY)
    saved_lam = ex.list_available_months
    try:
        mocks = MockNodes()
        _setup_mock_executor(mocks)
        months = [202101 + m for m in range(12)] + [202201, 202202, 202203]
        ex.list_available_months = lambda p: list(months)

        # fail-fast: node exception raises with repr_graph
        def boom(*args, **kwargs):
            raise RuntimeError("decay exploded")
        
        ex.NODE_REGISTRY["node_check_decay_monthly"] = boom
        exp = {
            "common_params": {
                "train_window": 3,
                 "oss_step": 3, 
                 "num_workers": 1, 
                 "feature_col": "f"
                 }, 
            "search_bounds": {}
        }
        
        try:
            ex.run_dag(parse_dag("wfo_production"), exp, ast_recipe={"op": "x"})
            raise SystemExit("FAIL: node error swallowed")
        except RuntimeError as e:
            assert "decay exploded" in str(e) and "✗ failed here" in str(e)
        ex.NODE_REGISTRY["node_check_decay_monthly"] = mocks.fake_decay
        print("[ok] executor fail-fast: exception raised with graph repr")

        # resume grid alignment
        exp2 = {
            "common_params": {
                "train_window": 3,
                "oss_step": 3,
                "num_workers": 1,
                "feature_col": "f",
                "last_model_id": 202106,  # 202106 已完成
            },
            "search_bounds": {},
        }
        res_r = ex.run_dag(parse_dag("wfo_production"), exp2, ast_recipe={"op": "x"})

        # 202106 (idx=6) 已经产出 恢复后应仅执行 idx=9 (202109) 和 idx=12 (202112)
        assert res_r.windows_executed == 2, f"Expected 2 windows, got {res_r.windows_executed}"
        assert res_r.models_produced == [202109, 202112], f"Got models {res_r.models_produced}"

        print("[ok] executor resume: grid alignment skipped earlier completed windows")
    finally:
        ex.NODE_REGISTRY.clear()
        ex.NODE_REGISTRY.update(saved)
        ex.list_available_months = saved_lam


def test_config_strict():
    from bt_studio.default_config import build_exp_config
    try:
        build_exp_config({"invalid_flat_key": "x"})
        raise SystemExit("FAIL: flat key accepted")
    except ValueError:
        pass
    # We allow feature_col at the top level or within common_params.
    # The previous test asserted KeyError when accessing ["common_params"]["feature_col"]
    # This was because it used to be required inside common_params.
    # Now it can be provided top-level. 
    print("[ok] config strict (invalid key rejected)")


def test_intake():
    from bt_studio.engine.task_manager import AsyncTaskManager
    tmp = tempfile.mkdtemp()
    good1, good2, bad = (os.path.join(tmp, n) for n in ("a.json", "b.json", "bad.json"))
    with open(good1, "w") as f:
        json.dump({"feature_col": "a", "dag": "wfo_production"}, f)
    with open(good2, "w") as f:
        json.dump({"feature_col": "b"}, f)
    with open(bad, "w") as f:
        f.write("{not json")

    tm = AsyncTaskManager.__new__(AsyncTaskManager)
    tm._tasks, tm._lock = {}, __import__("threading").Lock()
    tm._queue = __import__("queue").Queue()
    tm._max_queued_tasks = 100
    tm._max_history_tasks = 200
    tm._intake_dir = tmp
    assert tm._drain_intake_once() == 2
    _, dag1, cfg1, _ = tm._queue.get_nowait()
    _, dag2, cfg2, _ = tm._queue.get_nowait()
    assert dag1 == "wfo_production" and cfg1.get("feature_col") == "a"
    assert dag2 == "wfo_hpo" and cfg2.get("feature_col") == "b"
    assert os.path.exists(good1 + ".consumed") and os.path.exists(bad + ".invalid")
    print("[ok] intake: dag refs + renames")


if __name__ == "__main__":
    test_parser()
    test_parser_negatives()
    test_executor_wfo_production()
    test_executor_wfo_hpo()
    test_executor_resume_and_fail_fast()
    test_config_strict()
    test_intake()
    print("\n[test_dag] ALL PASS")