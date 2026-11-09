#!/usr/bin/env bash
# bt_studio 跨平台 (macOS / Linux 高核工作站) 生产运行环境动态初始化脚本
# 用法: source scripts/start_env.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 1. 确定性与并发控制: 限制单进程底层 C/Rust 线程数为 1, 并行度统一由 Ray / ProcessPool 托管
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# 2. 平台识别 (Darwin vs Linux) 与硬件资源动态适配
OS_NAME="$(uname -s)"
if [ "${OS_NAME}" = "Darwin" ]; then
    CPU_CORES="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"
    RAM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 17179869184)"
    RAM_GB=$(( RAM_BYTES / 1073741824 ))
    ulimit -n 65536 2>/dev/null || true

    export EVAL_PROCS="${EVAL_PROCS:-$(( CPU_CORES > 16 ? 8 : (CPU_CORES / 2 > 1 ? CPU_CORES / 2 : 2) ))}"
    export HPO_NUM_WORKERS="${HPO_NUM_WORKERS:-$(( CPU_CORES > 14 ? 12 : (CPU_CORES - 2 > 1 ? CPU_CORES - 2 : 2) ))}"
    export TICK_BATCH_SIDS="${TICK_BATCH_SIDS:-$(( RAM_GB >= 64 ? 16 : 8 ))}"
    export RAY_OBJECT_STORE_GB="${RAY_OBJECT_STORE_GB:-$(( RAM_GB / 5 > 16 ? 16 : (RAM_GB / 5 > 2 ? RAM_GB / 5 : 2) ))}"
    export STREAMLIT_HEADLESS="${STREAMLIT_HEADLESS:-false}"
else
    # Linux (如 Threadripper 9995WX 64C / 512G~1T)
    CPU_CORES="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 64)"
    RAM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 67108864)"
    RAM_GB=$(( RAM_KB / 1048576 ))
    # 64 核高并发扫描 Parquet / Ray Worker 必须调高文件描述符上限
    ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true

    if [ "${CPU_CORES}" -ge 32 ]; then
        export EVAL_PROCS="${EVAL_PROCS:-$(( CPU_CORES / 3 > 24 ? 24 : CPU_CORES / 3 ))}"
        export HPO_NUM_WORKERS="${HPO_NUM_WORKERS:-$(( CPU_CORES * 3 / 4 ))}"
    else
        export EVAL_PROCS="${EVAL_PROCS:-$(( CPU_CORES / 2 > 1 ? CPU_CORES / 2 : 2 ))}"
        export HPO_NUM_WORKERS="${HPO_NUM_WORKERS:-$(( CPU_CORES * 3 / 4 > 1 ? CPU_CORES * 3 / 4 : 2 ))}"
    fi
    export TICK_BATCH_SIDS="${TICK_BATCH_SIDS:-$(( RAM_GB >= 128 ? 128 : 32 ))}"
    export RAY_OBJECT_STORE_GB="${RAY_OBJECT_STORE_GB:-$(( RAM_GB >= 256 ? 64 : (RAM_GB / 4 > 4 ? RAM_GB / 4 : 4) ))}"
    export STREAMLIT_HEADLESS="${STREAMLIT_HEADLESS:-true}"
    export RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-127.0.0.1}"
fi

# 3. Tailscale 虚拟组网自动探测 (安全绑定内网 IP, 零公网暴露)
if command -v tailscale >/dev/null 2>&1; then
    TS_IP="$(tailscale ip -4 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
    if [ -n "${TS_IP}" ]; then
        export TAILSCALE_IP="${TS_IP}"
        export BT_STUDIO_BIND_HOST="${BT_STUDIO_BIND_HOST:-${TS_IP}}"
        export BT_STUDIO_CORS_ORIGINS="${BT_STUDIO_CORS_ORIGINS:-http://localhost,http://127.0.0.1,http://${TS_IP}:8800,http://${TS_IP}:8501}"
    fi
fi
export BT_STUDIO_BIND_HOST="${BT_STUDIO_BIND_HOST:-127.0.0.1}"

echo "[bt_studio] Platform=${OS_NAME} | Cores=${CPU_CORES} | RAM=${RAM_GB}GB"
echo "  -> EVAL_PROCS=${EVAL_PROCS} | HPO_NUM_WORKERS=${HPO_NUM_WORKERS} | TICK_BATCH_SIDS=${TICK_BATCH_SIDS} | RAY_STORE=${RAY_OBJECT_STORE_GB}GB"
echo "  -> Bind Host=${BT_STUDIO_BIND_HOST} (API :8800 / Streamlit :8501)"
