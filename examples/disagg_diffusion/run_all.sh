#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch all disaggregated diffusion services (etcd + 3 workers + orchestrator)
# and optionally send a test request.
#
# Usage:
#   ./run_all.sh                    # launch all services
#   ./run_all.sh --test             # launch + send a test request
#   ./run_all.sh --test --quick     # launch + quick smoke test (9 frames, 3 steps)
#
# Environment variables:
#   MODEL_PATH    HuggingFace model (default: hunyuanvideo-community/HunyuanVideo)
#   GPU_ENC       GPU for encoder (default: 0)
#   GPU_DEN       GPUs for denoiser (default: 1,2)
#   GPU_VAE       GPU for VAE (default: 3)
#   PORT          HTTP port (default: 8080)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKERS_DIR="$SCRIPT_DIR/phase1_workers"
ORCH_DIR="$SCRIPT_DIR/phase2_orchestrator"
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

# 2. Encoder Worker
echo "[2/5] Starting Encoder Worker (GPU $GPU_ENC)..."
CUDA_VISIBLE_DEVICES="$GPU_ENC" python "$WORKERS_DIR/encoder_worker.py" \
    > "$LOG_DIR/encoder.log" 2>&1 &
PIDS+=($!)

# 3. Denoiser Worker
echo "[3/5] Starting Denoiser Worker (GPU $GPU_DEN)..."
CUDA_VISIBLE_DEVICES="$GPU_DEN" python "$WORKERS_DIR/denoiser_worker.py" \
    > "$LOG_DIR/denoiser.log" 2>&1 &
PIDS+=($!)

# 4. VAE Worker
echo "[4/5] Starting VAE Worker (GPU $GPU_VAE)..."
CUDA_VISIBLE_DEVICES="$GPU_VAE" python "$WORKERS_DIR/vae_worker.py" \
    > "$LOG_DIR/vae.log" 2>&1 &
PIDS+=($!)

# 5. Orchestrator
echo "[5/5] Starting Orchestrator (port $PORT)..."
PORT="$PORT" python "$ORCH_DIR/run_disagg.py" \
    > "$LOG_DIR/orchestrator.log" 2>&1 &
PIDS+=($!)

echo ""
echo "All services launching. Waiting for workers to be ready..."
echo "  tail -f $LOG_DIR/encoder.log    # monitor encoder"
echo "  tail -f $LOG_DIR/denoiser.log   # monitor denoiser"
echo "  tail -f $LOG_DIR/vae.log        # monitor vae"
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
