# Pipeline DAGs (XML) — v2: 显式依赖边

pipeline 结构的**单一声明处**。v2 用显式 `deps` 依赖边取代了 v1 的
gate/skip/branch 控制流属性——图即依赖，控制流即边的条件。

## Schema

```xml
<pipeline name="wfo_production">
  <window from="first|last-trainable"
          train-window-key="train_window" step-key="oss_step"/>
  <node id="decay" fn="node_check_decay_monthly" role="prev_oos"
        deps="macro>global_data, extract_train>train_paths" output="gate"/>
</pipeline>
```

### `<window>`
| 属性 | 含义 |
|---|---|
| `from` | `first`=全部滚动窗（支持 `last_model_id` 网格对齐续跑）；`last-trainable`=仅最后一个可训练窗 |
| `train-window-key` / `step-key` | 从 common_config 取窗长/步长的键名 |

### `<node>`
| 属性 | 含义 |
|---|---|
| `id` | 节点标识；输出存入 ctx[id] |
| `fn` | tune.py 节点函数名（executor 按签名参数名从 ctx 注入实参，无硬编码分发） |
| `role` | 窗口切片映射 `train/oos/warmup/prev_oos` → `ymonths`（纯数据声明，非控制流） |
| `deps` | 逗号分隔依赖项（见下） |
| `output` | 返回值额外存入 ctx 的键名（如 decay → `gate`） |

### `deps` 依赖项语法
| 语法 | 语义 |
|---|---|
| `node` | 朴素依赖：上游已运行 |
| `node>key` | 依赖 + 把上游输出绑定为 ctx 键 `key`（按参数名注入） |
| `…?gate` | 条件依赖：仅当 ctx.gate 为 True 时需要，否则本节点跳过 |
| `…?!gate` | 反向条件依赖：仅当 ctx.gate 为 False |
| `a\|b` | 或依赖：任一备选已运行即满足（如 oos 依赖 `tune\|update`） |

条件不满足或上游被跳过时，本节点**级联跳过**——分支/剪枝都是同一规则。

## 标准图

| 文件 | 语义 |
|---|---|
| `wfo_production.xml` | 生产 WFO：decay→gate；`?gate`→train_data/tune（重训），`?!gate`→update（继承 FSM），oos 依赖 `tune\|update` |
| `wfo_hpo.xml` | 单窗口强制 HPO：**精简图**——不声明 decay/update/oos 及 extract_oos/warmup，未声明的节点天然不运行、不拉取任何数据（取代 v1 的 skip + 消费者剪枝） |

## 引擎语义（bt_studio/engine）
- `dag_parser`：解析 + 校验（未知依赖/环/非法 role 编译期失败）
- `executor`：Kahn 拓扑执行；per-window ctx 注入（签名驱动）；条件依赖评估；Ray 生命周期不归 executor（由 orchestrator 经 `ray_ctx.ensure_ray` 持有）