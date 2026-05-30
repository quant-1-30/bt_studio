# 监控协议 opentelemetry-api / opentelemetry-sdk / opentelemetry-distro

 chromedriver / chromium/ 
 | 浏览器     | driver       |
| ------- | ------------ |
| Firefox | geckodriver  |
| Chrome  | chromedriver |
| Edge    | msedgedriver |


(experiments-py3.11) hengxinliu@hengxindeMacBook-Pro tests % poetry cache list
PyPI
_default_cache
devpi
tuna
(experiments-py3.11) hengxinliu@hengxindeMacBook-Pro tests % poetry cache clear devpi --all

brew services start redis
caffeinate -i -s python finetune.py


# Dag rpc处理 + extract_Feature + ray tune ----> dag

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError(f"Script failed: {result.stderr}")
return result.stdout

@dag(
    dag_id="fsm_wfo_pipeline_v2",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["quant", "wfo"],
    
    # 🌟 补充 1：限制当前 DAG 同时在跑的实例数量。
    # 防止多个人同时手动狂点触发，导致多个 WFO 任务并跑把 GPU/Ray 显存顶爆。
    max_active_runs=1,
    
    # 🌟 补充 2：整个 DAG 的超时死锁控制。
    # 比如 WFO 优化由于参数空间太大在 Ray 里卡死了，3小时内不完事系统自动强行报错，释放算力
    dagrun_timeout=timedelta(hours=3),
    
    # 🌟 补充 3：默认的任务级别配置（会继承给该 DAG 下的所有 Task）
    default_args={
        "retries": 2,                  # 任何一个 Task 失败了，自动重试 2 次
        "retry_delay": timedelta(seconds=60), # 重试间隔 60 秒
        "owner": "hengxinliu"          # 负责人标签
    }
)

<!-- schedule=None
含义：调度策略（频次）

常见配置

None：不自动运行，只靠外部触发（非常适合你的量化参数调优 wfo 流水线）

"@daily" 或 "0 0 * * *"：每天凌晨运行一次（适合收盘后日线级别因子挖掘）

"@hourly"：每小时运行一次 -->

    # num_rows = filter_df.height  
    # num_samples = max(1, int(num_rows * frac))
    # random_idx = np.random.choice(num_rows, size=num_samples, replace=False)
    # sample_df = filter_df[random_idx]  
    
    # samples = sample_df["sid"].cast(pl.Binary).to_list()


<!-- def generate_quarter(start_date: int, end_date: int, overlap_days: int = 15):
    start_dt = datetime.strptime(str(start_date), "%Y%m%d")
    end_dt = datetime.strptime(str(end_date), "%Y%m%d")
    chunks =[]
    curr_start = start_dt
    
    while curr_start <= end_dt:
        curr_end = curr_start + relativedelta(months=3) - timedelta(days=1)
        if curr_end > end_dt: curr_end = end_dt
            
        req_start = curr_start - timedelta(days=overlap_days)
        chunks.append({
            "req_start": int(req_start.strftime("%Y%m%d")),
            "end_date": int(curr_end.strftime("%Y%m%d")),
            "valid_start": int(curr_start.strftime("%Y%m%d"))
        })
        curr_start = curr_end + timedelta(days=1)
    return chunks -->

放弃在 Cerebro 中添加多个策略，而是创建一个**主策略 (MetaStrategy)**，将 StrategyA 和 StrategyB 降级为单纯的**信号生成器 (Signal Generators)**。由 MetaStrategy 统筹 Kelly 权重和统一下单

### 第二步：使用 `multiple_outputs=True`（Airflow 最佳实践）
当一个 Task 返回字典时，强烈建议加上 `multiple_outputs=True`。这样你就可以在函数外部直接使用字典的 `[key]` 进行传参，Airflow 会在底层自动解构并解析为**真实的字符串**。
