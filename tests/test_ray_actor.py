"""验证 reuse_actors=True 能否让 stumpy 在 worker 内只编译一次"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import ray
from ray import tune

ray.init(num_cpus=4, ignore_reinit_error=True)

# 全局计数器，用 ray actor 跟踪 worker pid
@ray.remote
class Counter:
    def __init__(self):
        self.pid_times = {}
    def record(self, pid, elapsed):
        self.pid_times.setdefault(pid, []).append(elapsed)
        return len(self.pid_times[pid])
    def get_stats(self):
        return dict(self.pid_times)

counter = Counter.remote()

def trainable_with_stumpy(config):
    """每个 trial 调一次 stumpy.stump，记录耗时"""
    import os, time, numpy as np, stumpy
    pid = os.getpid()
    
    arr = np.random.randn(8000)
    t0 = time.time()
    stumpy.stump(arr, m=10)  # 首次编译~9s, 后续~0.1s
    elapsed = time.time() - t0
    
    call_num = ray.get(counter.record.remote(pid, elapsed))
    print(f"[PID {pid}] trial call#{call_num}  stumpy={elapsed:.3f}s", flush=True)
    
    tune.report({"score": float(config.get("a", 0))})

print("=" * 60)
print("测试: reuse_actors=True, 8 trials, 4 CPU")
print("预期: 只有前4个trial(每个pid第一次)慢~9s, 后续都快~0.1s")
print("=" * 60)

wrapped = tune.with_resources(trainable_with_stumpy, resources={"cpu": 1})

tuner = tune.Tuner(
    wrapped,
    param_space={"a": tune.uniform(0, 1)},
    tune_config=tune.TuneConfig(
        metric="score",
        mode="max",
        num_samples=8,
        max_concurrent_trials=4,
        reuse_actors=True,   # <<< 关键
    ),
)

t0 = time.time()
results = tuner.fit()
total = time.time() - t0

stats = ray.get(counter.get_stats.remote())
print(f"\n{'='*60}")
print(f"总耗时: {total:.1f}s")
print(f"如果每次都编译(8×9s/4并发): ~18s")
print(f"如果只编译4次(reuse_actors生效): ~9s")
print(f"\n按 PID 分组的 stumpy 调用耗时:")
for pid, times in stats.items():
    print(f"  PID {pid}: {[f'{t:.2f}s' for t in times]}")
    if len(times) > 1:
        print(f"    → 第1次编译={times[0]:.2f}s, 后续平均={sum(times[1:])/len(times[1:]):.2f}s ✅ 复用成功")
ray.shutdown()
