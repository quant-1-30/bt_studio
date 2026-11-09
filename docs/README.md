# docs/ 索引

> 权威分工:**仓库根 `README.md`** = 项目概览;**`AGENTS.md`** = 操作指南(环境/测试/机制/约定/安全/性能);
> **`architecture.md`** = 架构专题(LLM 挖掘环路 / 可视化栈 / DAG 引擎决策);**`runbook.md`** = 生产运行手册。

| 文档 | 内容 |
|------|------|
| [AGENTS.md](AGENTS.md) | 操作指南: 环境、测试、仓库结构、关键机制、开发约定、安全、性能与资源 |
| [architecture.md](architecture.md) | 架构专题: LLM 特征挖掘环路(四层架构/ADR/pending 契约/生产化契约)、可视化栈(塌陷检测/Bokeh/Streamlit)、DAG 引擎设计决策 |
| [visual_online_architecture.md](visual_online_architecture.md) | 在线实时可视化专题: A/B/C 单核三通道架构、跨屏焦点联动总线与单屏切换/多屏部署指南 |
| [refactoring_audit_summary.md](refactoring_audit_summary.md) | 全栈重构审计总结: 因果后复权、Numba DTW 编译核、奥卡姆剃刀去参数化与模块高内聚收敛 |
| [runbook.md](runbook.md) | A 形态研究生产运行手册: 部署、面板操作、告警处置、断点恢复、缓存治理 |

## 维护规则

1. 现行文档随代码同步更新;过时内容直接修订或删除,不再另行归档。
2. 路径必须真实存在:新增代码引用前 `grep` 验证。
3. `AGENTS.md` 的测试命令、目录结构、入口脚本与仓库根 `README.md` 保持一致。
