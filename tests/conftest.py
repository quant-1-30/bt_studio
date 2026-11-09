#!/usr/bin/env python3
"""pytest 全局夹具与确定性环境。

必须在任何测试模块 (进而任何 polars/numba import) 之前执行:
conftest.py 由 pytest 在收集阶段最先加载。

[HPO 门控确定性] 两类逐进程随机源曾使同一 trial 配置的分数漂移 (20.1~21.9),
并被塌陷门控 (1.5xIQR) 边界放大为离散翻转:
  1. 浮点归约并行求和顺序 (polars 流式 collect / numba/stumpy / BLAS) ——
     单线程化 driver 与 ray worker 双侧后消除 (见下方 setdefault);
  2. set 迭代序受 PYTHONHASHSEED 逐进程随机化 → tick concat 顺序 →
     特征 parquet 行序 → prepare_curves 按行序 vstack 曲线 → stumpy 输入
     排列变化。已在源头修复 (tune.py/测试 mock 改 dict.fromkeys 保序去重,
     pattern.py 在 collect 后 sort(["day","sid"]) 规范行序),
     新代码约定见 docs/AGENTS.md §5。
修复后 trial 表跨进程逐位一致。
"""

import os

for _var in ("NUMBA_NUM_THREADS", "POLARS_MAX_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
