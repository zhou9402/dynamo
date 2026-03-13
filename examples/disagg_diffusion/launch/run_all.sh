#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated Diffusion POC — all-in-one launcher (Wan video model)
#
# Starts three stage workers (Encoder, Denoiser, VAE) as background
# processes, waits for them to be ready, then runs the orchestrator.
#
# Requirements:
#   - 3 GPUs (or 1 large GPU >48 GB — set SINGLE_GPU=1)
#   - diffusers, transformers, torch, dynamo runtime, uvloop, ftfy, imageio
#
# Usage:
#   bash launch/run_all.sh [MODEL] [PROMPT]
#
# Examples:
#   bash launch/run_all.sh
#   bash launch/run_all.sh Wan-AI/Wan2.2-TI2V-5B-Diffusers "A sunset over mountains"
#   SINGLE_GPU=1 bash launch/run_all.sh   # all stages on GPU 0

set -euo pipefail

export MODEL_PATH="${1:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
export PROMPT="${2:-A cat walking slowly on green grass}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/lustre/raplab/client/menyu/workspace/hub}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKERS_DIR="${SCRIPT_DIR}/phase1_workers"
ORCH_DIR="${SCRIPT_DIR}/phase2_orchestrator"

# If dynamo.sglang components are available, add them to PYTHONPATH.
# Handlers fall back to local shims when the components package is absent.
COMPONENTS_SRC="${SCRIPT_DIR}/../../components/src"
if [[ -d "${COMPONENTS_SRC}" ]]; then
    export PYTHONPATH="${COMPONENTS_SRC}:${PYTHONPATH:-}"
fi

echo "============================================"
echo "  Disaggregated Diffusion POC (Wan)"
echo "  Model:  ${MODEL_PATH}"
echo "  Prompt: ${PROMPT}"
echo "============================================"
echo ""

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down workers …"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    # Clean stale endpoint registrations from etcd
    etcdctl del --prefix "disagg_diffusion/" >/dev/null 2>&1 || true
    echo "Done."
}
trap cleanup EXIT

# Clean stale registrations from previous runs
etcdctl del --prefix "disagg_diffusion/" >/dev/null 2>&1 || true

if [[ "${SINGLE_GPU:-0}" == "1" ]]; then
    GPU_ENC=0; GPU_DEN=0; GPU_VAE=0
else
    GPU_ENC="${GPU_ENC:-0}"; GPU_DEN="${GPU_DEN:-1}"; GPU_VAE="${GPU_VAE:-2}"
fi

# --- Start workers --------------------------------------------------------

echo "[1/3] Starting Encoder Worker (GPU ${GPU_ENC}) …"
CUDA_VISIBLE_DEVICES=${GPU_ENC} python "${WORKERS_DIR}/encoder_worker.py" \
    2>&1 | sed 's/^/  [encoder]  /' &
PIDS+=($!)

echo "[2/3] Starting Denoiser Worker (GPU ${GPU_DEN}) …"
CUDA_VISIBLE_DEVICES=${GPU_DEN} python "${WORKERS_DIR}/denoiser_worker.py" \
    2>&1 | sed 's/^/  [denoiser] /' &
PIDS+=($!)

echo "[3/3] Starting VAE Worker (GPU ${GPU_VAE}) …"
CUDA_VISIBLE_DEVICES=${GPU_VAE} python "${WORKERS_DIR}/vae_worker.py" \
    2>&1 | sed 's/^/  [vae]      /' &
PIDS+=($!)

# Wait for model loading.  In production use Dynamo health checks.
WAIT_SECS="${WAIT_SECS:-120}"
echo ""
echo "Waiting ${WAIT_SECS}s for workers to load models …"
sleep "${WAIT_SECS}"

# --- Run orchestrator -----------------------------------------------------

PORT="${PORT:-8080}"
export PORT

# Kill any stale orchestrator on the same port from a previous run.
# fuser/lsof may not be installed in all containers — guard each attempt.
set +e
if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" >/dev/null 2>&1
elif command -v lsof >/dev/null 2>&1; then
    _STALE_PID="$(lsof -t -i:"${PORT}" 2>/dev/null)"
    if [ -n "${_STALE_PID:-}" ]; then
        kill -9 ${_STALE_PID} 2>/dev/null
    fi
fi
set -e
sleep 1

echo ""
echo "Running orchestrator (port ${PORT}) …"
echo ""

export OUTPUT="/tmp/disagg_output.mp4"
python "${ORCH_DIR}/run_disagg.py"
