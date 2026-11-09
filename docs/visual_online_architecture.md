# `bt_studio` 在线实时可视化架构设计（A / B / C 多通道与多屏联动方案）

本文档定义 `bt_studio` 从离线静态分析升级为**在线实时可视化工作台（Real-Time Quant Desk）**的完整架构设计、通道协议、跨屏联动机制与代码实现契约。

---

## 一、 设计目标与核心痛点解决

在量化研究与实盘交易中，单一屏幕或单一图表库无法同时兼顾三种截然不同的时间尺度与数据形态：
1. **研究与超参尺度（秒~分钟级低频更新）**：RL 飞轮迭代进度、Stage 2 幸存特征池、Ray Tune 参数空间 3D 地形图与塌陷诊断；
2. **个股微观结构尺度（毫秒~秒级高频序列）**：单只标的 1 分钟 / Tick 级 OHLCV K 线、成交量、AST 特征曲线（`lag_0`）、DTW 命中子序列窗口高亮（Motif Overlay）以及实盘/回测买卖点标记；
3. **全市场横截面尺度（5000+ 标的并发扫描）**：全市场实时 `FSMPredictor` 触发排行榜（`sid, distance, macro_state, expected_return, fsm_score`）与 `FSMSizer` 风险平价目标权重矩阵。

为此，`bt_studio` 采用 **“一核三通道 · 单屏切换与多屏联动（Single Hub, Multi-Channel & Multi-Monitor）”** 架构。

---

## 二、 整体拓扑架构（Single Hub, Multi-Channel）

```text
                    ┌────────────────────────────────────────────────────────┐
                    │           bt_studio 统一后端网关 (plugins/api_server)    │
                    │   FastAPI + MonitorIndex (SQLite WAL) + _WsHub + ZMQ   │
                    └────────┬───────────────────┬───────────────────┬───────┘
                             │                   │                   │
          [Channel A: 投研流] │ [Channel B: K线流] │ [Channel C: 截面流]│
          /api/v2/overview   │ /api/v2/focus     │ /api/v2/screener  │
          /api/v2/features   │ /ws/v2 (focus)    │ /ws/v2 (changed)  │
                             ▼                   ▼                   ▼
                    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                    │   Screen A      │ │   Screen B      │ │   Screen C      │
                    │ 投研与超参诊断台 │◄┼─►个股微观K线形态│◄┼─►全市场截面雷达 │
                    │ Flywheel + HPO  │ │ Bokeh / TV Chart│ │ Trigger Screener│
                    └─────────────────┘ └─────────────────┘ └─────────────────┘
                             ▲                   ▲                   ▲
                             └───────────────────┴───────────────────┘
                           跨屏焦点联动总线 (desk_focus.json + WebSocket 扇出)
                 在 Screen C 点击任意触发股票行 -> Screen B / Screen A 自动同步切换
```

---

## 三、 A / B / C 三大场景通道规范

### 1. Screen A：投研飞轮与超参空间诊断台（Research & HPO Landscape）
- **定位**：监控 LLM 特征挖掘飞轮实时进展、Stage 2 幸存特征库以及 Ray Tune 超参数空间的平坦度/尖刺塌陷诊断。
- **后端接口**：
  - `GET /api/v2/overview`：当日与全量 Token 消耗、活跃飞轮数、特征池规模、Raw PIT 月包覆盖率与系统告警；
  - `GET /api/v2/features`：按 `final_score` 排序的可行特征清单（含 `window_pass_ratio` 与人工评审状态）；
  - `visual/collapse.py::plot_collapse_dashboard`：2D 等高线 + 3D 响应曲面 + 归一化香农熵/组间方差比诊断面板。
- **刷新策略**：`@st.fragment(run_every="3s")` 局部增量轮询，提交 HPO 目录分析时通过 `st.form` 批处理防抖。

### 2. Screen B：个股微观结构 K 线与 DTW 形态复盘台（Microstructure K-Line & Motif）
- **定位**：展示当前焦点股票（`focus.sid`）的 OHLCV K 线、成交量柱、`ind_*` 因子子图与 `Portfolio` / `drawDown` 绩效子图。
- **技术实现**：
  - 现阶段内置：`bt_studio/visual/bkh.py::Plot`（Bokeh 垂直堆叠多子图 + 共享 `x_range` 缩放 + `CustomJS` 跨子图红线十字光标同步）；
  - 高频实盘扩展插件：可无缝挂载 **TradingView `lightweight-charts`**（HTML5 Canvas 硬件加速，通过 `series.update(bar)` 支持每秒数十笔 Tick 级增量刷新与 `setMarkers` 买卖点标注）。
- **联动响应**：自动订阅 `MonitorIndex.get_focus()["sid"]`，当 Screen C 切换焦点标的时自动更新当前展示标的。

### 3. Screen C：全市场横截面实时触发雷达（Cross-Sectional Market Screener）
- **定位**：全市场扫描最新窗口（或指定 `model_id`）下所有触发 DTW 距离阈值（`distance <= threshold_d`）的标的池，按 `fsm_score` 实时降序排列。
- **后端接口**：
  - `GET /api/v2/screener?model_id={YYYYMM}&limit=200`（由 `MonitorIndex.screener()` 实现）：直接读取 `scores_{model_id}.parquet`，输出 `day, sid, distance, macro_state, expected_return, fsm_score`；
  - 大规模流式扩展插件：可挂载 **FINOS `perspective-python`**（WebAssembly C++ 透视引擎，直接消费 Polars `df.to_arrow()` 二进制流）。
- **点选联动（Row-Click Broadcast）**：
  - Screen C 表格启用 `st.dataframe(..., on_select="rerun", selection_mode="single-row")`；
  - 点击任意股票行即自动调用 `idx.set_focus(sid=clicked_sid, model_id=focus_model)` 并通过 `/api/v2/focus` / `_WsHub.broadcast` 通知所有屏幕。

---

## 四、 部署与多屏使用指南

### 1. 启动服务
```bash
# 1. 启动在线可视化工作台 (默认 http://127.0.0.1:8501)
poetry run python scripts/run_streamlit.py

# 2. (可选) 启动独立 FastAPI + WebSocket 多端推送网关 (默认 http://127.0.0.1:8800)
poetry run python -m bt_studio.plugins.api_server.main
```

### 2. 单屏模式 vs 多显示器多屏联动模式

顶部导航栏 `st.segmented_control` 已通过 `bind="query-params"` 双向绑定至 URL 查询参数 `?view_mode=...`：

| 使用模式 | 操作方式 | 适用硬件场景 |
| :--- | :--- | :--- |
| **单屏综合工作台** | 打开 `http://localhost:8501`（默认 `All-in-one desk`），或点击顶部切换栏在 A / B / C 间切换 | 单台笔记本或单显示器投研 |
| **三屏物理显示器联动** | - **显示器 1（左屏）**：打开 `http://localhost:8501/?view_mode=Screen+C:+Market+screener`<br>- **显示器 2（中屏）**：打开 `http://localhost:8501/?view_mode=Screen+B:+K-line+%26+motif`<br>- **显示器 3（右屏）**：打开 `http://localhost:8501/?view_mode=Screen+A:+Research+%26+HPO`<br>在三个窗口右上角开启 **Live stream (3s)** 开关 | 多屏量化交易台（在左屏点击触发股票，中屏与右屏 3 秒内自动同步焦点） |
