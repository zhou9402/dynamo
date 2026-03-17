#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch all disaggregated diffusion services (etcd + workers + orchestrator)
# and optionally send a test request.
#
# Each stage supports multiple workers: use ';' to separate workers in GPU
# specs.  Each worker is an independent top-level process; Dynamo discovers
# them via etcd and the orchestrator round-robins requests automatically.
#
# Usage:
#   ./run_all.sh                    # launch all services (1 worker/stage)
#   ./run_all.sh --test             # launch + send a test request
#   ./run_all.sh --test --quick     # launch + quick smoke test (9 frames, 3 steps)
#
#   # Multi-worker (8 GPU):
#   GPU_ENC="0;4" GPU_DEN="1,2;5,6" GPU_VAE="3;7" ./run_all.sh --test --quick
#
# Environment variables:
#   MODEL_PATH    HuggingFace model (default: hunyuanvideo-community/HunyuanVideo)
#   GPU_ENC       GPU(s) for encoder  (default: 0; use "0;4" for 2 workers)
#   GPU_DEN       GPU(s) for denoiser (default: 1,2; use "1,2;5,6" for 2 TP=2 workers)
#   GPU_VAE       GPU(s) for VAE      (default: 3; use "3;7" for 2 workers)
#   PORT          HTTP port (default: 8080)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS_DIR="$SCRIPT_DIR/workers"
ORCH_DIR="$SCRIPT_DIR/orchestrator"
LOG_DIR="/tmp/disagg_logs"

GPU_ENC="${GPU_ENC:-0}"
GPU_DEN="${GPU_DEN:-1,2}"
GPU_VAE="${GPU_VAE:-3}"
PORT="${PORT:-8080}"

DO_TEST=false
QUICK=false
for arg in "$@"; do
    case "$arg" in
        --test) DO_TEST=true ;;
        --quick) QUICK=true ;;
    esac
done

mkdir -p "$LOG_DIR"
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down all services..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # Kill child processes (sglang schedulers)
    for pid in "${PIDS[@]}"; do
        pkill -P "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All services stopped."
}
trap cleanup EXIT INT TERM

echo "=========================================="
echo "  Disaggregated Diffusion — Launch All"
echo "=========================================="
echo "  Encoder:     GPU $GPU_ENC"
echo "  Denoiser:    GPU $GPU_DEN"
echo "  VAE:         GPU $GPU_VAE"
echo "  HTTP port:   $PORT"
echo "  Logs:        $LOG_DIR/"
echo "=========================================="

# 1. etcd
if ! pgrep -x etcd > /dev/null 2>&1; then
    echo "[1/5] Starting etcd..."
    etcd --data-dir /tmp/etcd_disagg \
         --listen-client-urls http://0.0.0.0:2379 \
         --advertise-client-urls http://127.0.0.1:2379 \
         > "$LOG_DIR/etcd.log" 2>&1 &
    PIDS+=($!)
    sleep 2
else
    echo "[1/5] etcd already running, skipping."
fi

# 2. Encoder Worker(s)
STEP=2
IFS=';' read -ra ENC_GPUS <<< "$GPU_ENC"
for i in "${!ENC_GPUS[@]}"; do
    port=$((15600 + i * 10))
    echo "[$STEP] Starting Encoder Worker $i (GPU ${ENC_GPUS[$i]}, port $port)..."
    CUDA_VISIBLE_DEVICES="${ENC_GPUS[$i]}" SCHEDULER_PORT=$port \
        python "$WORKERS_DIR/encoder_worker.py" > "$LOG_DIR/encoder_$i.log" 2>&1 &
    PIDS+=($!)
    STEP=$((STEP + 1))
done

# 3. Denoiser Worker(s)
IFS=';' read -ra DEN_GPUS <<< "$GPU_DEN"
for i in "${!DEN_GPUS[@]}"; do
    port=$((15700 + i * 10))
    echo "[$STEP] Starting Denoiser Worker $i (GPU ${DEN_GPUS[$i]}, port $port)..."
    CUDA_VISIBLE_DEVICES="${DEN_GPUS[$i]}" SCHEDULER_PORT=$port \
        python "$WORKERS_DIR/denoiser_worker.py" > "$LOG_DIR/denoiser_$i.log" 2>&1 &
    PIDS+=($!)
    STEP=$((STEP + 1))
done

# 4. VAE Worker(s)
IFS=';' read -ra VAE_GPUS <<< "$GPU_VAE"
for i in "${!VAE_GPUS[@]}"; do
    port=$((15800 + i * 10))
    echo "[$STEP] Starting VAE Worker $i (GPU ${VAE_GPUS[$i]}, port $port)..."
    CUDA_VISIBLE_DEVICES="${VAE_GPUS[$i]}" SCHEDULER_PORT=$port \
        python "$WORKERS_DIR/vae_worker.py" > "$LOG_DIR/vae_$i.log" 2>&1 &
    PIDS+=($!)
    STEP=$((STEP + 1))
done

N_WORKERS=$(( ${#ENC_GPUS[@]} + ${#DEN_GPUS[@]} + ${#VAE_GPUS[@]} ))
echo ""
echo "Launched $N_WORKERS worker(s): ${#ENC_GPUS[@]} encoder, ${#DEN_GPUS[@]} denoiser, ${#VAE_GPUS[@]} vae"

# 5. Orchestrator
echo "[$STEP] Starting Orchestrator (port $PORT)..."
PORT="$PORT" python "$ORCH_DIR/run_disagg.py" \
    > "$LOG_DIR/orchestrator.log" 2>&1 &
PIDS+=($!)

echo ""
echo "All services launching. Waiting for workers to be ready..."
echo "  tail -f $LOG_DIR/encoder_*.log   # monitor encoder(s)"
echo "  tail -f $LOG_DIR/denoiser_*.log  # monitor denoiser(s)"
echo "  tail -f $LOG_DIR/vae_*.log       # monitor vae(s)"
echo "  tail -f $LOG_DIR/orchestrator.log"
echo ""

# Wait for orchestrator HTTP to be ready
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Orchestrator ready at http://localhost:$PORT"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: Orchestrator not ready after 120s. Check logs in $LOG_DIR/"
        exit 1
    fi
    sleep 1
done

# Test request
if [ "$DO_TEST" = true ]; then
    echo ""
    echo "Sending test request..."
    if [ "$QUICK" = true ]; then
        curl -s -X POST "http://localhost:$PORT/v1/videos/generations" \
            -H "Content-Type: application/json" \
            -d '{"prompt": "A cat walking on green grass", "num_frames": 9, "num_inference_steps": 3}' | python -m json.tool
    else
        curl -s -X POST "http://localhost:$PORT/v1/videos/generations" \
            -H "Content-Type: application/json" \
            -d '{"prompt": "A golden retriever running on a sunny beach with waves crashing in the background"}' | python -m json.tool
    fi
fi

echo ""
echo "Services running. Press Ctrl+C to stop all."
wait
