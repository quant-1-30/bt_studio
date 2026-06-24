from bt_studio.pipeline.dags.fsm import get_macro

def test_get_macro_logic():
    test_config = {"run_params": {...}}
    # 使用 .fn 访问原始函数，不触发 Prefect 引擎
    result = get_macro.fn(test_config)
    assert "dret_path" in result
