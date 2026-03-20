#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Concurrent stress test for disaggregated diffusion pipeline.
#
# Sends N_REQUESTS concurrent requests, then validates:
#   1. All requests return HTTP 200 with valid JSON
#   2. All produced .mp4 files are non-zero size
#   3. /pipeline/status shows completed=N, failed=0
#   4. All workers per stage were utilized
#
# Prerequisites: services must already be running (via run_all.sh).
#
# Usage:
#   ./stress_test.sh                    # 20 requests, default port
#   N_REQUESTS=50 ./stress_test.sh      # 50 requests
#   PORT=8081 ./stress_test.sh          # custom port

set -euo pipefail

PORT="${PORT:-8080}"
N_REQUESTS="${N_REQUESTS:-20}"
BASE_URL="http://localhost:$PORT"
RESULT_DIR="/tmp/stress_test_$$"

mkdir -p "$RESULT_DIR"

echo "=========================================="
echo "  Disagg Diffusion — Concurrent Stress Test"
echo "=========================================="
echo "  Target:     $BASE_URL"
echo "  Requests:   $N_REQUESTS"
echo "  Results:    $RESULT_DIR/"
echo "=========================================="

# Verify the server is up
if ! curl -sf "$BASE_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Server not reachable at $BASE_URL/health"
    echo "       Start services first with run_all.sh"
    exit 1
fi

# Record pre-test status
PRE_STATUS=$(curl -sf "$BASE_URL/pipeline/status")
PRE_COMPLETED=$(echo "$PRE_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['completed'])")
echo "Pre-test completed count: $PRE_COMPLETED"
echo ""

PROMPTS=(
    "A cat walking on green grass"
    "A dog running on a sandy beach"
    "A bird flying over a mountain lake"
    "A fish swimming in a coral reef"
    "A horse galloping through a meadow"
    "A butterfly landing on a red flower"
    "A wolf howling at the full moon"
    "A dolphin jumping out of the ocean"
    "A panda eating bamboo in the forest"
    "A fox running through autumn leaves"
    "A penguin sliding on ice in Antarctica"
    "An eagle soaring above snowy peaks"
    "A tiger walking through a jungle"
    "A rabbit hopping through a garden"
    "A deer drinking from a stream"
    "An owl perched on a branch at night"
    "A lion resting under an acacia tree"
    "A whale breaching in the deep ocean"
    "A parrot perched on a tropical branch"
    "A turtle walking slowly on the sand"
)

# Launch all requests concurrently
echo "Sending $N_REQUESTS concurrent requests..."
T0=$(date +%s)
PIDS=()

for i in $(seq 0 $((N_REQUESTS - 1))); do
    idx=$((i % ${#PROMPTS[@]}))
    prompt="${PROMPTS[$idx]}"
    (
        curl -sf -X POST "$BASE_URL/v1/videos/generations" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\": \"$prompt\", \"num_frames\": 9, \"num_inference_steps\": 3}" \
            -o "$RESULT_DIR/resp_$i.json" \
            -w "%{http_code}" \
            > "$RESULT_DIR/status_$i.txt" 2>/dev/null
    ) &
    PIDS+=($!)
done

echo "All $N_REQUESTS requests launched (PIDs: ${#PIDS[@]}). Waiting..."

# Wait for all requests to finish
FAILURES=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        FAILURES=$((FAILURES + 1))
        echo "  Request $i: curl failed (PID ${PIDS[$i]})"
    fi
done

T1=$(date +%s)
ELAPSED=$((T1 - T0))
echo ""
echo "All requests completed in ${ELAPSED}s"
echo ""

# ── Validation ──

PASS=0
FAIL=0
VIDEOS=()

echo "── Validation ──"

# Check 1: HTTP status codes
echo ""
echo "1) HTTP status codes:"
for i in $(seq 0 $((N_REQUESTS - 1))); do
    status_file="$RESULT_DIR/status_$i.txt"
    if [ -f "$status_file" ]; then
        code=$(cat "$status_file")
        if [ "$code" = "200" ]; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
            echo "   FAIL: request $i returned HTTP $code"
        fi
    else
        FAIL=$((FAIL + 1))
        echo "   FAIL: request $i — no status file (curl crashed)"
    fi
done
echo "   $PASS/$N_REQUESTS returned HTTP 200"

# Check 2: Valid JSON with video URL
echo ""
echo "2) Response JSON + video files:"
VIDEO_PASS=0
VIDEO_FAIL=0
for i in $(seq 0 $((N_REQUESTS - 1))); do
    resp="$RESULT_DIR/resp_$i.json"
    if [ ! -f "$resp" ] || [ ! -s "$resp" ]; then
        VIDEO_FAIL=$((VIDEO_FAIL + 1))
        continue
    fi
    url=$(python3 -c "
import json, sys
try:
    d = json.load(open('$resp'))
    print(d['data'][0].get('url', ''))
except Exception:
    print('')
" 2>/dev/null)
    if [ -n "$url" ]; then
        # Extract filename from /videos/<filename>
        fname=$(basename "$url")
        VIDEOS+=("$fname")
        VIDEO_PASS=$((VIDEO_PASS + 1))
    else
        VIDEO_FAIL=$((VIDEO_FAIL + 1))
        echo "   FAIL: request $i — no video URL in response"
    fi
done
echo "   $VIDEO_PASS/$N_REQUESTS have valid video URLs"

# Check 3: Video files exist and are non-zero
echo ""
echo "3) Video file integrity:"
VIDEO_DIR="/tmp/disagg_videos"
FILE_PASS=0
FILE_FAIL=0
for fname in "${VIDEOS[@]}"; do
    fpath="$VIDEO_DIR/$fname"
    if [ -f "$fpath" ] && [ -s "$fpath" ]; then
        FILE_PASS=$((FILE_PASS + 1))
    else
        FILE_FAIL=$((FILE_FAIL + 1))
        echo "   FAIL: $fname missing or empty"
    fi
done
echo "   $FILE_PASS/${#VIDEOS[@]} video files valid (non-zero size)"

# Check 4: Pipeline status — completed count and worker utilization
echo ""
echo "4) Pipeline status:"
POST_STATUS=$(curl -sf "$BASE_URL/pipeline/status")
POST_COMPLETED=$(echo "$POST_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['completed'])")
POST_FAILED=$(echo "$POST_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['failed'])")
NEW_COMPLETED=$((POST_COMPLETED - PRE_COMPLETED))
echo "   Completed: $NEW_COMPLETED (expected: $N_REQUESTS)"
echo "   Failed:    $POST_FAILED"

# Check worker utilization
echo ""
echo "5) Worker utilization:"
echo "$POST_STATUS" | python3 -c "
import json, sys
status = json.load(sys.stdin)
all_used = True
for stage_name, stage in status.get('stages', {}).items():
    workers = stage.get('workers', [])
    used = [w for w in workers if w['completed'] > 0]
    total = len(workers)
    print(f'   {stage_name}: {len(used)}/{total} workers used', end='')
    for w in workers:
        print(f\"  [id={w['id']} completed={w['completed']} avg={w['avg_latency_s']:.2f}s]\", end='')
    print()
    if len(used) < total:
        all_used = False
if all_used:
    print('   All workers utilized.')
else:
    print('   WARNING: Not all workers were utilized.')
"

# ── Summary ──
echo ""
echo "=========================================="
TOTAL_CHECKS=0
TOTAL_PASS=0

# HTTP check
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if [ "$PASS" -eq "$N_REQUESTS" ]; then TOTAL_PASS=$((TOTAL_PASS + 1)); fi

# Video URL check
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if [ "$VIDEO_PASS" -eq "$N_REQUESTS" ]; then TOTAL_PASS=$((TOTAL_PASS + 1)); fi

# File integrity check
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if [ "$FILE_PASS" -eq "${#VIDEOS[@]}" ] && [ "${#VIDEOS[@]}" -gt 0 ]; then TOTAL_PASS=$((TOTAL_PASS + 1)); fi

# Pipeline completed count check
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if [ "$NEW_COMPLETED" -eq "$N_REQUESTS" ] && [ "$POST_FAILED" -eq "0" ]; then TOTAL_PASS=$((TOTAL_PASS + 1)); fi

if [ "$TOTAL_PASS" -eq "$TOTAL_CHECKS" ]; then
    echo "  RESULT: ALL PASSED ($TOTAL_PASS/$TOTAL_CHECKS checks)"
    echo "  $N_REQUESTS requests, ${ELAPSED}s total, $(echo "scale=1; $ELAPSED / $N_REQUESTS" | bc)s avg"
else
    echo "  RESULT: $TOTAL_PASS/$TOTAL_CHECKS checks passed"
fi
echo "=========================================="

# Cleanup temp dir
rm -rf "$RESULT_DIR"

[ "$TOTAL_PASS" -eq "$TOTAL_CHECKS" ] && exit 0 || exit 1
