# 运行手册 —— A 形态研究生产（人工在环）

> 边界：挖掘/RL 自动运行；人工审阅面板批注；**tune 由人工显式发起**（intake 不常开）。

## 1. 部署

> 原生系统守护服务模板已内置于 `deploy/` 目录（Linux 用 `deploy/systemd/*.service`，macOS 用 `deploy/launchd/*.plist`），详细安装命令见 [`deploy/README.md`](../deploy/README.md)。

```bash
# 快速初始化跨平台确定性环境与 Tailscale 绑定
source scripts/start_env.sh

# 1) 监控面板 (token 必配; BT_STUDIO_ENV=prod 下未配直接拒绝启动)
BT_STUDIO_ENV=prod BT_STUDIO_API_TOKEN='<secret>' \
  poetry run python -m uvicorn bt_studio.plugins.api_server.main:app \
  --host "${BT_STUDIO_BIND_HOST:-127.0.0.1}" --port 8800 --workers 1

# 2) 挖掘主进程 (prod 模式必须显式三参, 并发度根据 macOS/Linux 核数自动适配)
BT_STUDIO_ENV=prod FROM_DATE=20220101 TO_DATE=20230630 N_SIDS=100 \
AGENT_STEPS=6 AGENT_STAGE3=off RESUME=auto \
LLM_DAILY_TOKEN_BUDGET=500000 \
  poetry run python scripts/run_agent.py
```

**旋钮**：`FROM_DATE/TO_DATE/N_SIDS`（prod 必填，缺省拒绝）｜`AGENT_STEPS`（默认 3）｜`EVAL_PROCS`（默认 8）｜`AGENT_STAGE3=off`（A 形态保持 off）｜`RESUME=auto`（指纹匹配续跑）｜`LLM_DAILY_TOKEN_BUDGET`（调用前熔断）｜`TICK_BATCH_SIDS`（默认 8）。

## 2. 日常操作（面板）

- `GET /api/v2/overview`（需 `Authorization: Bearer <token>`）：今日/累计 token、runs、viable、alerts
- run 详情：每步 timing（llm_s/eval_s）、git_sha、succ/fail 历史
- 候选审阅：`GET /api/v2/features` → `POST /api/v2/review`（`reviewed/rejected/promote_candidate`，唯一写路径，落 `result/artifacts/review/`）
- WS：`/ws/v2?token=...&last_seq=N`（断线重放）

## 3. 告警处置 (支持 `BT_STUDIO_ALERT_WEBHOOK` 自动推送)

| 告警 | 处置 |
|---|---|
| `token_budget_80` | 停发新 run；查 `/api/v2/usage/daily` 明细；必要时调预算 |
| `stage2_zero_success_streak` | 连续 ≥2 轮 0 幸存 → 审 prompt/门控，勿继续烧 token |
| `repeat_fingerprint` | 同指纹跨 run 重探 → 开启 `AGENT_WARM_START=1`（默认已开启，从 `viable_features.json` 注入历史先验并拦截同指纹重复评估） |
| `forbidden_pending_tune` | `AGENT_STAGE3=off` 下出现 pending/tune 产物 → 检查是否误开 intake |
| `pending_invalid_quarantined` | `pending/` 下出现 `.invalid` 隔离文件 → 检查对应文件是否因 AST 因果校验或指纹不匹配被拒 |
| `disk_usage_high` / `disk_usage_critical` | 磁盘水位 ≥85% 告警，≥95% 自动按 LRU 驱逐 `result/cache/evals/` 旧缓存 |
| `index_lag` | 索引落后磁盘 >60s → 系统已自动触发 `scan_once()` 自愈重扫 |
| `raw_pit_coverage_low` | 查 md-server / 日历 |

## 4. 断点续跑 / 崩溃恢复

- 进程级崩溃（OOM/断电）：同配置直接重启 —— `RESUME=auto` 命中 `state.json` 续跑；`pending_step.json` 存在则该步不重烧 LLM
- 配置变更（门控/标的池/窗口）：指纹不匹配 → 自动新开 run（旧断点按设计作废）
- 429/配额：flywheel 打印"基础设施异常, 终止飞轮"后退出 —— 等配额窗口重置再拉起（token 账本自动续累计）

## 5. 缓存治理

| 目录 | 语义 | 清理 |
|---|---|---|
| `result/artifacts/` | **产物层**（审计/备份对象）: runs/features/review/usage/pending/tune | 备份即整目录 |
| `result/cache/bars/` | 分钟 PIT v2 **月包**（写时验证/增量合并/邻置 meta） | `align_skeleton` 语义变更时 bump `RAW_PIT_VERSION` |
| `result/cache/evals/v2/` | 窗口评估结果（键不含门控 → 调阈值重放免费） | 评分语义变更时 bump `EVAL_LOGIC_VERSION` |
| `result/cache/monitor_index.sqlite` | 面板索引（可重建） | 删除后面板全量重扫 |

> 布局 v2 (2026-10-13): 旧 `result/{features,llm,tune,fsm,cache}` 已归档至
> 仓库根 `result_legacy_*`（只读回查，不再写入）；新布局两层语义
> `artifacts/`(产物) + `cache/`(可再生)。特征目录键 = 纯指纹 fp12。

## 6. 升级到 B 形态（自动 tune）的前置

1. 真实规模 tune 全链演练（挖→spool→intake→wfo_hpo→模型落盘）
2. 塌陷阈值在真实特征上的校准
3. ~~pending schema 校验 + 指纹服务端重算~~ ✅ **已完成**（`submit_dag_task` 已内置 `check_causal=True` 静态编译预检与 `f_{fp12}` 服务端 `ast_fingerprint` 重算校验，且模型 `.pkl` 加载已启用 `pkl_sha256` 签名校验）

前两项演练完成前 `BT_STUDIO_LLM_INTAKE` 保持 unset。

## 附：历史 run 的 succ/fail 显示修复

2026-10-13 前的旧 step artifact 使用计数键（`n_success/n_failure`），新 indexer 读 history 列表 → 旧 run 面板显示 0/0。一次性修复：

```bash
$PY - <<'EOF'
import json, glob
for p in glob.glob("result/llm/runs/*/step_*.json"):
    d = json.load(open(p))
    rb = d.get("replay_buffer", {})
    if "n_success" in rb and "successful_history" not in rb:
        rb["successful_history"], rb["failed_history"] = [], []
        json.dump(d, open(p, "w"), ensure_ascii=False, default=str)
EOF
rm result/cache/monitor_index.sqlite*   # 触发重索引
```
