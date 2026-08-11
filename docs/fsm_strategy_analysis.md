# FSM 策略分析文档

> 量化研究中的有限状态机（FSM）策略深度解析，涵盖统计学陷阱、工程优化与行业评估

---

## 目录

1. [四大关键优化维度](#一四大关键优化维度)
2. [均衡采样的深度解析](#二均衡采样的深度解析)
3. [行业客观评估](#三行业客观评估)
4. [实战问答与最佳实践](#四实战问答与最佳实践)
5. [多维度 vs 单维度分析](#五多维度-vs-单维度分析)
6. [HPO 诊断指标解析](#六hpo-诊断指标解析)
7. [HPO致命Bug与数理逆反](#七hpo致命bug与数理逆反)
8. [优化建议总结](#八优化建议总结)

---

## 一、四大关键优化维度

### 1.1 统计学暗雷：极小样本的"独角兽陷阱"与胖尾效应

#### 痛点1：触发次数阈值过低

**问题分析**：
- 在全市场250,000个日频观测样本中，一个形态仅触发5次是毫无统计学意义的"独角兽/孤本"
- 即使 P-value < 0.01，超额收益高达5%，实盘中可能一年都遇不到一次
- 极容易是某只妖股的偶然数据拟合

**优化方案**：
```python
# 提高硬性统计学底线（大样本定理的底线是30）
n_triggers = len(cond_rets)
if n_triggers < 30 or np.std(cond_rets) < 1e-8:
    return {"status": "failed", "reason": f"Triggers ({n_triggers}) < 30", "metrics_score": -9999.0}
```

#### 痛点2：均值超额极易被A股"胖尾"欺骗

**问题场景**：
- 30次触发中，29次亏损-1%，1次连续涨停赚40%
- `np.mean` 会算出正的超额收益，Optuna认为是好因子
- 实盘中极大概率倾家荡产

**优化方案**：
```python
def calculate_hpo_score(u_pval, trigger_count, cond_rets, uncond_rets, tune_config, common_config):
    # 使用中位数 (Median) 计算超额，过滤极值妖股的欺骗
    excess_ret = np.median(cond_rets) - np.median(uncond_rets)
    
    # 计算胜率 (Win Rate: 收益 > 0 的比例)
    win_rate = np.mean(cond_rets > 0)
    
    # 胜率低于50%在A股Long-Only极度危险
    if win_rate < 0.50:
        return -9999.0
    
    # ... 后续计算逻辑
```

### 1.2 宏观状态噪音：Rolling Quantile消除一日游Regime

#### 问题分析

当前使用单日OFI均值判定宏观状态存在严重缺陷：
- 单日资金净流入具有极强的反转噪音
- 昨天暴跌资金流出（State=0），今天可能直接修复反弹
- 状态矩阵剧烈抖动，失去指导意义

#### 优化方案：使用20日滚动分位数

```python
daily_macro_lf = (
    panel_df.lazy()
    .select(["day", "sid", pl.col(macro_col).list.sum().alias("sid_ofi_sum")])
    .group_by("day")
    .agg(pl.col("sid_ofi_sum").mean().alias("daily_ofi_mean"))
    .sort("day") 
    # 核心修复：使用20日滚动分位数，彻底消灭未来函数
    .with_columns([
        pl.col("daily_ofi_mean")
          .rolling_quantile(quantile=0.33, window_size=20, min_periods=5)
          .alias("p33"),
        pl.col("daily_ofi_mean")
          .rolling_quantile(quantile=0.67, window_size=20, min_periods=5)
          .alias("p67")
    ])
    .with_columns(
        pl.when(pl.col("daily_ofi_mean") <= pl.col("p33")).then(0)
        .when(pl.col("daily_ofi_mean") <= pl.col("p67")).then(1)
        .otherwise(2)
        .cast(pl.Int32)
        .alias("macro_state")
    )
    .with_columns(pl.col("macro_state").shift(1)) # 防未来数据核心
    .drop(["p33", "p67", "daily_ofi_mean"])
    .drop_nulls()
)
```

**关键原则**：
- 不需要叠加rolling_mean（会造成双重平滑延迟）
- rolling_quantile本身就是动态自适应的稳定阈值
- 保持信号最强时效性

### 1.3 交易执行脱节：14:50截断的"滑点盲区"

#### 问题分析

**执行时间不匹配**：
- 特征提取在14:50截断（设置NaN墙）
- 收益率标签使用T+1收盘价（15:00）
- 14:50-15:00尾盘10分钟是量化抢筹最凶阶段
- 实际买入价格可能偏离收盘价1%-2%

#### 优化方案：精准对齐执行价格

```python
# 在 build_fsm_panel 中，提取 14:50 的截面价格作为基准
entry_price_lf = (
    all_feat_lf
    .filter(pl.col("bar_idx") == 230)  # 14:50对应的bar_idx
    .select(["day", "sid", pl.col("close").alias("entry_price")])
)

daily_ret_lf = (
    daily_lf.join(entry_price_lf, on=["day", "sid"], how="left")
    .sort(["sid", "day"])
    # 真实交易收益 = 明天收盘价 / 今天14:50的入场价 - 1
    .with_columns([
        (pl.col("close").shift(-1).over("sid") / pl.col("entry_price") - 1.0).alias("fwd_ret_1"),
        (pl.col("close").shift(-2).over("sid") / pl.col("entry_price") - 1.0).alias("fwd_ret_2"),
    ])
    # 如果找不到14:50的价格，退化为使用当日收盘价
    .with_columns(
        pl.col("fwd_ret_1").fill_null((pl.col("close").shift(-1).over("sid") / pl.col("close") - 1.0))
    )
)
```

### 1.4 MStump二次采样：均衡采样打破信息茧房

#### 问题分析

**当前做法的风险**：
- 100%选择波动最剧烈的Top N股票
- 模型只学习"极端妖股"或"高贝塔股"的量价形态
- 普通股票根本触发不了

#### 优化方案：50%妖股 + 50%普适股

```python
# 计算每只股票的最大突变跳跃度
mutation_scores = np.nansum(np.nanmax(np.abs(np.diff(curves_md, axis=2)), axis=2), axis=1)

# 均衡采样：一半取波动极大的妖股，另一半等距抽样普通股
half_size = sample_size // 2

# 1. Top 50%：最活跃的
top_active_idx = np.argsort(mutation_scores)[-half_size:]

# 2. 剩余 50%：在全体股票中随机/等距抽样
remaining_idx = np.setdiff1d(np.arange(N), top_active_idx)
if len(remaining_idx) > (sample_size - half_size):
    random_idx = np.random.choice(remaining_idx, size=(sample_size - half_size), replace=False)
else:
    random_idx = remaining_idx
    
# 合并索引
final_sample_idx = np.concatenate([top_active_idx, random_idx])
sampled_curves = curves_md[final_sample_idx]
```

---

## 二、均衡采样的深度解析

### 2.1 为什么不100%选择最高分股票？

#### 底层逻辑1：Z-Score无法抹平的"拓扑结构撕裂"

- **高波动股票**：Z-Score标准化后仍是"尖锐刺刀、近乎直角的折线"
- **普通股票**：Z-Score标准化后是"平滑、连续的波浪曲线"
- **DTW距离**：无论如何缩放，尖锐刺刀与平滑波浪的距离不可能小于threshold_d

#### 底层逻辑2：违反"IID"导致的协变量偏移

- 训练集：100%极端噪音分布
- 测试集：10%极端 + 90%正常分布
- 结果：严重的协变量偏移，训练集高分→OOS/实盘崩塌

#### 底层逻辑3：微观结构噪音的伪因子陷阱

得分最高的股票往往不是"资金意图"，而是"市场微观缺陷"：
- 小盘股买卖价差过大
- 流动性断层造成的无意义跳跃
- 学到的是"流动性差导致的盘口空气墙形态"

### 2.2 50%+50%的数学正则化本质

**50%突变股（侦察兵）**：
- 提供强烈信号源（低搜索壁垒）
- 把可能的博弈形态"揪"出来

**50%随机股（定海神针）**：
- 提供普适性约束（高泛化壁垒）
- 强制妥协，寻找通用物理意义的博弈形态

**逼迫效应**：
- Stumpy必须找到"既在最活跃股票里有鲜明表达，又在普通股票里能温和复现"的形态

---

## 三、行业客观评估

### 3.1 核心结论

**定性**：这套架构在数学逻辑和工程回测上完全自洽，但属于"高维护成本、薄利且具有高稀疏性"的特异型策略。

**定位建议**：不适合作为独立撑起大容量资金的"母基金级策略"，但极佳的"交易信号过滤器"或"日内执行算法增强模块"。

### 3.2 行业客观优势

| 优势 | 说明 |
|------|------|
| 白盒可解释性 | 能明确指出"当OFI出现特定双峰流入，且大盘低波时，T+1跑赢截面概率68%" |
| 捕捉非线性博弈 | 传统线性多因子模型对日内"博弈形态"完全失效，本系统捕捉时序非线性几何特征 |
| 滚动进化自适应 | WFO和衰减熔断是应对"概念漂移"的标准做法，保证模型安全 |

### 3.3 业内实盘挑战

#### 挑战1：交易摩擦与A股T+1的"绞肉机效应"

- 微观结构形态产生的Alpha边际通常只有15-30bps
- A股交易成本（印花税、佣金、滑点、冲击成本）轻易达到20-25bps
- 微薄Alpha极易被交易手续费吃光

#### 挑战2：信息离散化的重大损失

- 将连续收益率通过Rank强制划分为4个离散状态
- 损失60%以上的信息熵
- 业内更倾向于高斯过程（GP）或KDE连续化刻画

#### 挑战3：信号的极度稀疏性

- 严苛的threshold_r和-BIC惩罚导致很多月份只有3-5次完美Motif触发
- 持仓极度不稳定，资金大部分时间空转
- 拉低年化复利收益率

### 3.4 项目可行性进化建议

#### 建议1：退化为多因子策略的强力过滤器

**玩法**：
- 传统年化Sharpe=1.5的多因子池推荐10只股票
- FSM引擎作为终审裁判扫描日内OFI/Vol曲线
- 发现"庄家尾盘拉高出货"等恶劣Motif，一票否决

**效果**：让母因子的Sharpe瞬间飙升，避开FSM信号稀疏的死穴

#### 建议2：缩短target期限到日内

如果有T+0渠道或期货：
- 将FSM状态转移期限设置为"未来15分钟、30分钟、1小时"
- 在手续费极低、反应极迅速的日内T+0战场，OFI形态挖掘才是真正常胜将军

---

## 四、实战问答与最佳实践

### 4.1 停牌率active_ratio：0.90 vs 0.95？

**行业标准值：0.90（90%）是A股中低频策略的黄金分割点**

| 阈值 | 允许停牌天数 | 评价 |
|------|-------------|------|
| 0.95 | 1天/月 | 极其苛刻，股票池容量萎缩15%-20% |
| 0.90 | 2天/月 | 刚好容纳正常临时停牌，容量和代表性最佳 |
| <0.85 | 3-4天/月 | .shift(1)会拼接停牌前后，微观结构记忆性漂移 |

### 4.2 rolling_mean vs rolling_quantile？

**结论：完全不需要rolling_mean！叠加会造成"双重平滑延迟"**

理由：
1. `daily_ofi_mean`已经是全市场股票的OFI均值，横截面均值本身就把噪音滤掉了
2. 叠加rolling_mean会导致严重的"相位滞后"
3. `rolling_quantile`本身具有过滤属性

### 4.3 过滤条件是否冗余？

#### volume > 0 & high > low 的必要性

**防止"执行幻觉"**：
- 不过滤：停牌日close向前填充，daily_curve是水平直线
- Stumpy可能识别为"高分Motif"
- 实盘买入那天股票停牌，根本买不进去

#### active_ratio >= 0.90 的必要性

**防止"时空折叠"**：
- 过滤掉停牌日会在DataFrame中物理消失
- 10月1日和10月25日之间隔了20天
- Polars的`.shift(1)`会把两天拼成"连续2天"
- 大盘从3000跌到2500，宏观环境天翻地覆，匹配纯垃圾

### 4.4 致命的未来函数修复

#### 问题代码

```python
# 🚨 致命未来函数！
.with_columns([
    pl.col("daily_ofi_mean").quantile(1/3).alias("p33"),
    pl.col("daily_ofi_mean").quantile(2/3).alias("p67")
])
```

**问题**：拿整年数据计算分位数，1月份的判定"知道"了10月份的大牛市

#### 修复代码

```python
# 💡 只允许往回看20个交易日
.with_columns([
    pl.col("daily_ofi_mean")
      .rolling_quantile(quantile=0.33, window_size=20, min_periods=5)
      .alias("p33"),
    pl.col("daily_ofi_mean")
      .rolling_quantile(quantile=0.67, window_size=20, min_periods=5)
      .alias("p67")
])
```

---

## 五、多维度 vs 单维度分析

### 5.1 物理本质对比

| 维度 | 多维(mstump) | 单维(stump) |
|------|-------------|------------|
| 物理本质 | D维空间中完全重合的轨迹 | 降维到低维流形 |
| 优势 | 捕捉耦合相位差、发现子维度形态 | 距离测度稳定、抗噪强、计算快 |
| 致命劣势 | 维度诅咒、度量失效、极易过拟合 | 投影函数设计错误会信息丢失 |

### 5.2 业内量化实战准则

**"能降维坚决降维，绝不在原始距离空间中做高维DTW"**

顶级私募的做法：
1. 绝不把毫无量纲关系的原始特征直接扔进mstump
2. 使用多维时D绝对不超过2，且必须严苛正交化
3. 90%以上使用经过精妙数学投影后的1D综合特征

### 5.3 高级投影方法

#### 1. 物理学域驱：构建无量纲常量

```python
# 微观流动性冲击指数
Impact_Curve_t = OFI_t / (Volatility_t + ε)
```

#### 2. 相空间极坐标投影

```python
# 模长曲线（能量强度）
r_t = sqrt(Z(OFI)_t^2 + Z(Vol)_t^2)

# 相位曲线（博弈结构）
θ_t = arctan2(Z(Vol)_t, Z(OFI)_t)
```

#### 3. 时序自编码器

```python
# 将(N, D, L)的多维张量输入Encoder
# 在Bottleneck Layer强制压缩为(N, 1, L)的一维隐状态曲线
```

---

## 六、HPO 诊断指标解析

### 6.1 KDE Entropy（核密度估计微分熵）

#### 用途

精准识别贝叶斯优化器是否发生"过早收敛/参数空间塌陷"

#### 数学原理

```python
# 1. 高斯核密度估计
p(x) = KDE(x)

# 2. 微分熵
H(X) = -∫ p(x) ln p(x) dx

# 3. 标准化
kde_entropy = exp(H(X) - H_max)
```

#### 判读标准

| 取值范围 | 分布形态 | 优化器状态 | 建议 |
|---------|---------|-----------|------|
| 0.70-1.00 | 均匀/宽平 | 充分探索 | 样本代表性极强 |
| 0.30-0.70 | 单峰/双峰 | 正常收敛 | 正常寻优过程 |
| <0.15 | 极陡针尖 | 空间塌陷 | 增大n_startup_trials到30-50 |

### 6.2 spread_ratio（探索范围比）

#### 公式
```
连续参数: spread_ratio = (max(x_sampled) - min(x_sampled)) / (U_bounds - L_bounds)
离散参数: spread_ratio = 实际采样种类数 / 预设搜索空间总种类数
```

#### 诊断
- <0.40：搜索空间设定过大，或外围60%区域被放弃
- 建议：下一次HPO缩窄搜索边界

### 6.3 landscape_variance_ratio（组间方差比）

#### 公式
```
landscape_variance_ratio = SS_between / SS_total
```

#### 诊断
- >0.25：核心敏感参数，必须精细调优
- <0.05：废参数，可固定为常数降维

### 6.4 worst_drop_ratio（邻域最坏落差）

#### 公式
```
worst_drop_ratio = (z_best - min(z_neighbor)) / IQR_global
```

#### 判读标准
- ≤1.0×IQR：平缓安全高原，准予部署
- >1.5×IQR：孤立过拟合刺峰，拒绝部署

---

## 七、HPO致命Bug与数理逆反

> **核心发现**：经过对4个核心函数的像素级审计，抓出导致系统"疯狂选择过拟合刺峰"的3个致命代码Bug和1个数理逆反陷阱。

### 7.1 Bug总览

| Bug | 位置 | 问题 | 后果 |
|-----|------|------|------|
| Bug 1 | `_get_bounds` | 类型判定`isinstance(val, float)`对int返回False | 离散参数邻域检测全面瘫痪 |
| Bug 2 | `score_std`分母 | 被-3500极值撑爆 | 刺峰检测器形同虚设 |
| Bug 3 | fANOVA异常捕获 | `return True`盲目放行 | 异常样本轻松绕过安全网 |
| 数理逆反 | `calculate_hpo_score` | `-k*log(n)`惩罚大样本 | 偏向"触发次数极少"的过拟合模型 |

### 7.2 Bug 1：_get_bounds类型判定崩坏

#### 问题代码

```python
def _get_bounds(val):
    # 💥 致命 Bug！
    return (val - 0.05, val + 0.05) if isinstance(val, float) else (val, val)
```

#### 逻辑崩溃推演

1. `downsample`（3, 4, 5）和`motif_minutes`（30, 45, 60）在Python中都是`int`
2. 当`top1_name = downsample`，最佳值`t1_val = 4`时：
   - `isinstance(4, float)` → `False`
   - `_get_bounds(4)` → `(4, 4)`
3. `is_between(4, 4)`只筛选出参数=4的点，根本没检查3和5
4. **后果**：如果downsample=4是孤立刺峰，而3和5得分极差，代码因为只检查4自身，判定"邻域完美"！

#### 修复方案：按范围百分比动态计算

```python
def _get_relative_bounds(pname, val, valid_df):
    """💡 按参数总范围的15%动态计算，彻底解决int/float类型Bug"""
    col_vals = valid_df[f"config/{pname}"].to_numpy()
    p_range = col_vals.max() - col_vals.min()
    delta = p_range * 0.15 if p_range > 0 else 0.5
    return (val - delta, val + delta)
```

### 7.3 Bug 2：score_std膨胀导致标尺失效

#### 问题代码

```python
if nearby_mean < best_score - 1.5 * score_std:
    return False
```

#### 数理崩溃推演

1. `df_results`包含所有Trial得分，少数失败Trial拿到-3500或-500
2. 全局`score_std`被拉到巨大数值（如600）
3. `1.5 * score_std = 900`
4. 判定变成：`nearby_mean < best_score - 900`
5. **后果**：即使邻域跌落200分（严重刺峰），系统也觉得"跌得不够"

#### 修复方案：使用IQR稳健标尺

```python
# 💡 采用稳健标尺IQR，彻底规避极值撑爆分母
scores = valid_df["metrics_score"].to_numpy()
q75, q25 = np.percentile(scores, [75, 25])
iqr_scale = max(q75 - q25, 1e-4)

# 检查邻域内最坏邻居
if nearby_trials.height > 0:
    worst_neighbor = nearby_trials["metrics_score"].min()
    # 跌幅超过1.5倍IQR → 判定为孤立刺峰
    if (best_score - worst_neighbor) > 1.5 * iqr_scale:
        return False
```

### 7.4 Bug 3：fANOVA异常后的盲目放行

#### 问题代码

```python
try:
    fanova = FanovaImportanceEvaluator()
    importances = ...
except Exception as e:
    return True  # 💥 算不出来就直接判定为"通过"！
```

#### 后果

- fANOVA因数据矩阵奇异或重复点算不出来
- 最需要被检测的异常样本反而轻松绕过安全网
- 直接`return True`让刺峰模型通过

#### 修复方案

```python
try:
    fanova = FanovaImportanceEvaluator()
    importances = optuna.importance.get_param_importances(study, evaluator=fanova)
except Exception as e:
    print(f"❌ fANOVA计算失败: {e}")
    return False  # 💡 计算失败时严谨拒绝！
```

### 7.5 数理逆反：打分公式惩罚大样本

#### 问题代码

```python
# 💥 数理逻辑倒扣！
n = max(2, len(cond_rets))
neg_bic = 2 * ln_L - k * np.log(n)
```

#### 数理悖论

| 模型类型 | 触发次数n | ln(n)惩罚 | 得分影响 |
|---------|----------|-----------|---------|
| 平缓高原模型 | 1000 | 6.9 | 极重扣分 ❌ |
| 孤立刺峰模型 | 30 | 3.4 | 轻微扣分 ⚠️ |

**问题本质**：`-k*log(n)`随着样本量增加而猛烈扣分，导致打分公式天然偏向"触发次数极少"的过拟合模型！

#### 修复方案：使用sqrt(n)风险缩放

```python
# 💡 消除对大样本量n的log(n)倒扣，改用k/√n
n = max(2, len(cond_rets))
score = ln_L - (k / np.sqrt(n))
```

**数学原理**：
- `k/√n`是标准误的标准形式，代表"复杂度按样本量的置信度缩放"
- 当n大时，`k/√n`变小 → 鼓励大样本、平缓的通用模式
- 当n小时，`k/√n`变大 → 惩罚小样本、偶然的孤立刺峰

### 7.6 完整修复代码

#### validate_parameter_plateau_fanova（完整版）

```python
def validate_parameter_plateau_fanova(df_results: pl.DataFrame, best_config: dict, best_score: float) -> bool:
    """fANOVA 高原评估 (防刺峰/防虚假通过加固版)"""
    param_cols = [c for c in df_results.columns if c.startswith("config/")]
    
    # 1. 物理清洗掉熔断的极低分，保证方差/IQR计算纯净
    valid_df = df_results.filter(pl.col("metrics_score") > -9990.0).drop_nulls(subset=param_cols + ["metrics_score"])
    
    if valid_df.height < 20: 
        print("⚠️ 有效Trial不足20个，无法进行稳健高原拟合，拒绝该模型")
        return False

    # 2. 构建Optuna分布
    distributions = {}
    for col in param_cols:
        param_name = col.replace("config/", "")
        min_val, max_val = valid_df[col].min(), valid_df[col].max()
        if min_val == max_val:
            min_val = min_val - 1e-5 if min_val != 0 else -1e-5
            max_val = max_val + 1e-5 if max_val != 0 else 1e-5
            
        dtype = valid_df[col].dtype
        if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            distributions[param_name] = optuna.distributions.IntDistribution(int(min_val), int(max_val))
        elif dtype in [pl.Categorical, pl.String]:
            choices = valid_df[col].unique().to_list()
            distributions[param_name] = optuna.distributions.CategoricalDistribution(choices)
        else:
            distributions[param_name] = optuna.distributions.FloatDistribution(float(min_val), float(max_val))

    study = optuna.create_study(direction="maximize")
    for row in valid_df.iter_rows(named=True):
        trial = optuna.trial.create_trial(
            params={k.replace("config/", ""): v for k, v in row.items() if k.startswith("config/")},
            distributions=distributions,  
            value=row["metrics_score"]
        )
        study.add_trial(trial) 
        
    # 3. 计算fANOVA重要性
    try:
        fanova = FanovaImportanceEvaluator()
        importances = optuna.importance.get_param_importances(study, evaluator=fanova)
    except Exception as e:
        print(f"❌ fANOVA计算失败: {e}")
        return False  # 计算失败→严谨拒绝

    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_imp = sorted_params[0]
    top2_name, top2_imp = sorted_params[1]
    
    # 4. 修正邻域边界计算：按参数总范围的15%动态计算
    def _get_relative_bounds(pname, val):
        col_vals = valid_df[f"config/{pname}"].to_numpy()
        p_range = col_vals.max() - col_vals.min()
        delta = p_range * 0.15 if p_range > 0 else 0.5
        return (val - delta, val + delta)
        
    b1_min, b1_max = _get_relative_bounds(top1_name, best_config[top1_name])
    b2_min, b2_max = _get_relative_bounds(top2_name, best_config[top2_name])
    
    # 5. 筛选邻域内的点（排除最佳点自身）
    nearby_trials = valid_df.filter(
        (pl.col(f"config/{top1_name}").is_between(b1_min, b1_max)) &
        (pl.col(f"config/{top2_name}").is_between(b2_min, b2_max)) &
        (pl.col("metrics_score") < best_score - 1e-5)
    )
    
    # 6. 采用IQR稳健标尺
    scores = valid_df["metrics_score"].to_numpy()
    q75, q25 = np.percentile(scores, [75, 25])
    iqr_scale = max(q75 - q25, 1e-4)
    
    # 7. 判定孤立刺峰
    if nearby_trials.height > 0:
        worst_neighbor = nearby_trials["metrics_score"].min()
        if (best_score - worst_neighbor) > 1.5 * iqr_scale:
            print(f"❌ fANOVA孤立刺峰拦截！最高分:{best_score:.2f}, 邻域最差分:{worst_neighbor:.2f}")
            return False
            
    print(f"✅ 通过平缓高原检验！fANOVA核心因子:{top1_name}({top1_imp:.1%}), {top2_name}({top2_imp:.1%})")
    return True 
```

#### calculate_hpo_score（完整版）

```python
def calculate_hpo_score(
    u_pval: float, 
    trigger_count: int, 
    cond_rets: np.ndarray, 
    uncond_rets: np.ndarray,
    cond_z_gaps: np.ndarray, 
    cond_intra: np.ndarray,
    tune_config: dict,
    common_config: dict
) -> float:
    
    # 1. 实盘开盘门控模拟
    gap_z_lower = common_config.get("gap_z_lower", -2.0)
    gap_z_upper = common_config.get("gap_z_upper", 3.0)
    is_executed = (cond_z_gaps >= gap_z_lower) & (cond_z_gaps <= gap_z_upper)
    execution_rate = np.mean(is_executed) if len(is_executed) > 0 else 0.0

    realized_rets = np.where(is_executed, cond_rets, 0.0)
    
    # 2. 计算超额收益与胜率（使用中位数抗极值）
    excess_ret = np.median(realized_rets) - np.median(uncond_rets) 
    win_rate = np.mean(realized_rets > 0)
    intra_win_rate = np.mean(cond_intra > 0)
    eps = common_config.get("eps", 1e-4)
    
    # 3. 计算excess_factor
    alternative = common_config["alternative"]
    if alternative == "greater":
        excess_factor = max(excess_ret, eps) 
    elif alternative == "less":
        excess_factor = max(-excess_ret, eps)
    else: 
        excess_factor = max(abs(excess_ret), eps)
        
    # 4. 安全处理
    safe_u_pval = max(u_pval, 1e-10)
    safe_win_rate = max(win_rate, eps)
    safe_intra_rate = max(intra_win_rate, eps)
    safe_exec_rate = max(execution_rate, eps)
    
    # 5. 对数似然（包含执行率与胜率）
    ln_L = np.log(excess_factor) + np.log(safe_win_rate) + np.log(safe_intra_rate) + np.log(safe_exec_rate) - np.log(safe_u_pval)
    
    # 6. 计算复杂度k
    targets = common_config.get("T1_rets", {})
    m_mins = float(tune_config["motif_minutes"])
    dtw_frac = float(common_config["dtw_window_frac"])
    max_offset = max(targets.values()) if targets else 240
    
    k = (max_offset / 60.0) + (dtw_frac * 10.0) + (m_mins / 30.0)

    # 7. 修正评分公式：消灭对大样本量n的log(n)倒扣
    n = max(2, len(cond_rets))
    score = ln_L - (k / np.sqrt(n))
    
    return float(score)
```

### 7.7 修复效果总结

| 修复项 | 修复前 | 修复后 |
|--------|--------|--------|
| 离散参数邻域检查 | 瘫痪（只检查自身值） | 恢复（±15%范围邻域） |
| 刺峰检测标尺 | score_std（易撑爆） | IQR（稳健） |
| fANOVA异常处理 | 盲目放行True | 严谨拒绝False |
| 样本量惩罚 | -k·log(n)（惩罚大样本） | k/√n（鼓励大样本） |

**最终效果**：
- fANOVA刺峰检测器恢复战斗力
- 打分公式开始偏向"大样本平缓模型"
- 触发次数多、泛化能力强的平缓模型将获得更高得分

---

## 八、优化建议总结

### 7.1 统计学层面

| 优化项 | 修复前 | 修复后 |
|--------|--------|--------|
| 触发次数阈值 | 5次 | 30次 |
| 收益计算 | np.mean | np.median |
| 胜率要求 | 无 | ≥50% |
| P-value处理 | 直接-9999 | 0.15软漏斗+0.05硬拦截 |

### 7.2 宏观状态层面

| 优化项 | 修复前 | 修复后 |
|--------|--------|--------|
| 分位数计算 | 全局quantile | rolling_quantile(20日) |
| 平滑处理 | 叠加rolling_mean | 不需要，直接用原始值 |
| 停牌率阈值 | 不确定 | 0.90（2天/月） |

### 7.3 交易执行层面

| 优化项 | 修复前 | 修复后 |
|--------|--------|--------|
| 入场价格 | 当日收盘价 | 14:50 close价格 |
| 收益率计算 | shift(-1)/close(T)-1 | shift(-1)/entry_price-1 |
| 持有期限 | T+1日15:00 | 可考虑缩短到日内 |

### 7.4 采样策略层面

| 优化项 | 修复前 | 修复后 |
|--------|--------|--------|
| 采样方式 | 100%最高分 | 50%最高分+50%随机 |
| 维度选择 | 直接高维DTW | 优先1D投影 |

---

## 附录：核心代码模板

### A. validate_parameter_plateau_fanova（完整版）

```python
def validate_parameter_plateau_fanova(df_results: pl.DataFrame, best_config: dict, best_score: float) -> bool:
    """fANOVA 高原评估 (防刺峰/防虚假通过加固版)"""
    param_cols = [c for c in df_results.columns if c.startswith("config/")]
    
    # 1. 物理清洗掉熔断的极低分，保证方差/IQR计算纯净
    valid_df = df_results.filter(pl.col("metrics_score") > -9990.0).drop_nulls(subset=param_cols + ["metrics_score"])
    
    if valid_df.height < 20: 
        print("⚠️ 有效Trial不足20个，无法进行稳健高原拟合，拒绝该模型")
        return False

    # 2. 构建Optuna分布
    distributions = _build_distributions(valid_df, param_cols)

    study = optuna.create_study(direction="maximize")
    for row in valid_df.iter_rows(named=True):
        trial = optuna.trial.create_trial(
            params={k.replace("config/", ""): v for k, v in row.items() if k.startswith("config/")},
            distributions=distributions,  
            value=row["metrics_score"]
        )
        study.add_trial(trial) 
        
    # 3. 计算fANOVA重要性
    try:
        fanova = FanovaImportanceEvaluator()
        importances = optuna.importance.get_param_importances(study, evaluator=fanova)
    except Exception as e:
        print(f"❌ fANOVA计算失败: {e}")
        return False  # 计算失败→严谨拒绝

    sorted_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    top1_name, top1_imp = sorted_params[0]
    top2_name, top2_imp = sorted_params[1]
    
    # 4. 修正邻域边界计算：按参数总范围的15%动态计算
    def _get_relative_bounds(pname, val):
        col_vals = valid_df[f"config/{pname}"].to_numpy()
        p_range = col_vals.max() - col_vals.min()
        delta = p_range * 0.15 if p_range > 0 else 0.5
        return (val - delta, val + delta)
        
    b1_min, b1_max = _get_relative_bounds(top1_name, best_config[top1_name])
    b2_min, b2_max = _get_relative_bounds(top2_name, best_config[top2_name])
    
    # 5. 筛选邻域内的点（排除最佳点自身）
    nearby_trials = valid_df.filter(
        (pl.col(f"config/{top1_name}").is_between(b1_min, b1_max)) &
        (pl.col(f"config/{top2_name}").is_between(b2_min, b2_max)) &
        (pl.col("metrics_score") < best_score - 1e-5)
    )
    
    # 6. 采用IQR稳健标尺
    scores = valid_df["metrics_score"].to_numpy()
    q75, q25 = np.percentile(scores, [75, 25])
    iqr_scale = max(q75 - q25, 1e-4)
    
    # 7. 判定孤立刺峰
    if nearby_trials.height > 0:
        worst_neighbor = nearby_trials["metrics_score"].min()
        if (best_score - worst_neighbor) > 1.5 * iqr_scale:
            print(f"❌ fANOVA孤立刺峰拦截！最高分:{best_score:.2f}, 邻域最差分:{worst_neighbor:.2f}")
            return False
            
    print(f"✅ 通过平缓高原检验！fANOVA核心因子:{top1_name}({top1_imp:.1%}), {top2_name}({top2_imp:.1%})")
    return True 
```

### B. calculate_hpo_score（完整版）

```python
def calculate_hpo_score(
    u_pval: float, 
    trigger_count: int, 
    cond_rets: np.ndarray, 
    uncond_rets: np.ndarray,
    cond_z_gaps: np.ndarray, 
    cond_intra: np.ndarray,
    tune_config: dict,
    common_config: dict
) -> float:
    
    # 1. 实盘开盘门控模拟
    gap_z_lower = common_config.get("gap_z_lower", -2.0)
    gap_z_upper = common_config.get("gap_z_upper", 3.0)
    is_executed = (cond_z_gaps >= gap_z_lower) & (cond_z_gaps <= gap_z_upper)
    execution_rate = np.mean(is_executed) if len(is_executed) > 0 else 0.0

    realized_rets = np.where(is_executed, cond_rets, 0.0)
    
    # 2. 计算超额收益与胜率（使用中位数抗极值）
    excess_ret = np.median(realized_rets) - np.median(uncond_rets) 
    win_rate = np.mean(realized_rets > 0)
    intra_win_rate = np.mean(cond_intra > 0)
    eps = common_config.get("eps", 1e-4)
    
    # 3. 计算excess_factor
    alternative = common_config["alternative"]
    if alternative == "greater":
        excess_factor = max(excess_ret, eps) 
    elif alternative == "less":
        excess_factor = max(-excess_ret, eps)
    else: 
        excess_factor = max(abs(excess_ret), eps)
        
    # 4. 安全处理
    safe_u_pval = max(u_pval, 1e-10)
    safe_win_rate = max(win_rate, eps)
    safe_intra_rate = max(intra_win_rate, eps)
    safe_exec_rate = max(execution_rate, eps)
    
    # 5. 对数似然（包含执行率与胜率）
    ln_L = np.log(excess_factor) + np.log(safe_win_rate) + np.log(safe_intra_rate) + np.log(safe_exec_rate) - np.log(safe_u_pval)
    
    # 6. 计算复杂度k
    targets = common_config.get("T1_rets", {})
    m_mins = float(tune_config["motif_minutes"])
    dtw_frac = float(common_config["dtw_window_frac"])
    max_offset = max(targets.values()) if targets else 240
    
    k = (max_offset / 60.0) + (dtw_frac * 10.0) + (m_mins / 30.0)

    # 7. 修正评分公式：消灭对大样本量n的log(n)倒扣
    n = max(2, len(cond_rets))
    score = ln_L - (k / np.sqrt(n))
    
    return float(score)
```

### B. 终审法庭（node_tune_monthly结尾）

```python
best_trial_result = results.get_best_result("metrics_score", "max")
best_pval = best_trial_result.metrics.get("u_pval", 1.0)

# 后台终审法庭：不管Optuna怎么给0.08的模型打分，只要没突破0.05，统统枪毙
if best_pval > 0.05 or best_score <= -9990.0:
    print(f"⚠️ [Failed] {model_id} P-val ({best_pval:.4f}) 未达到5%实盘要求，抛弃该模型。")
    return False
```

---

## 结论

完成这些维度的打磨后，model.ckpt将发生质的飞跃：

1. **只关注**触发>30次、胜率>50%且中位数稳健的普适性规律
2. **状态转移**基于20日平滑的宏观基调，告别情绪反复拉扯
3. **收益计算**严丝合缝地对齐14:50这一交易执行节点
4. **采样覆盖**整个市场的不同风格

这是量化工程中从"跑通"走向"实盘盈利"的最后一步深水区！

---

> **最后建议**：将FSM定位为"多因子策略的强力过滤器"而非独立选股策略，这是在A股T+1约束下最明智的战术选择
