"""bt_studio.utils — 共享工具层。

模块布局:
    io          : 统一文件 IO (read_json_safe / atomic_save_* / exclusive_file_lock / slugify / sha1_hex / evict_cache_lru)
    months      : 月份序列、PIT 日历、特征/模型路径工具
    math_func   : 纯数学 (折现权重、DTW 阈值)
    manifest    : bt_core handoff model manifest 与模型验签加载 (safe_load_model_ckpt)
    diagnostics : HPO 参数空间塌陷检测与报告 (collapse / hpo_check)
"""
