#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end disaggregated diffusion pipeline with SGLang backend.

Launches Encoder, Denoiser (TP=2), and VAE as separate SGLang scheduler
processes on different GPUs, then runs the full pipeline:
    Encoder -> Denoiser -> VAE

Each stage can have multiple workers (a "pool").  The orchestrator
round-robins requests across workers in each pool independently.
Use ``;`` in GPU_ENC / GPU_DEN / GPU_VAE to separate workers.

Measures per-stage timing.  Supports concurrent requests for benchmarking.

GPU assignment (single-worker, 4 GPU):
    Encoder :  GPU 0   (1 GPU)
    Denoiser:  GPU 1,2 (TP=2)
    VAE     :  GPU 3   (1 GPU)

GPU assignment (multi-worker, 8 GPU):
    Encoder :  GPU 0, 4       (2 workers × 1 GPU)
    Denoiser:  GPU 1,2 | 5,6  (2 workers × TP=2)
    VAE     :  GPU 3, 7       (2 workers × 1 GPU)

Usage:
    # Single request, single worker per stage (4 GPU)
    python run_e2e_sglang.py

    # Multi-worker pools (8 GPU)
    GPU_ENC="0;4" GPU_DEN="1,2;5,6" GPU_VAE="3;7" python run_e2e_sglang.py

    # Benchmark: 4 requests, 2 concurrent
    NUM_REQUESTS=4 CONCURRENCY=2 python run_e2e_sglang.py

Environment variables:
    MODEL_PATH      Model to use (default: hunyuanvideo-community/HunyuanVideo)
    PROMPT          Text prompt (default: A cat walking on green grass)
    GPU_ENC         GPU(s) for encoder  (default: 0; use "0;4" for 2 workers)
    GPU_DEN         GPU(s) for denoiser (default: 1,2; use "1,2;5,6" for 2 TP=2 workers)
    GPU_VAE         GPU(s) for VAE      (default: 3; use "3;7" for 2 workers)
    TP_SIZE         Tensor parallelism for denoiser (default: auto from GPU_DEN)
    NUM_REQUESTS    Number of pipeline runs (default: 1)
    CONCURRENCY     Max concurrent pipelines (default: 1)
    NUM_FRAMES      Number of video frames (default: 61)
    NUM_STEPS       Denoising steps (default: 50)
    HEIGHT          Frame height (default: 544)
    WIDTH           Frame width  (default: 960)
    GUIDANCE        Guidance scale (default: 1.0; >1.0 enables CFG)
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import statistics
import sys
import time
from typing import List

# --- ensure workers dir is on sys.path so subprocesses find sglang_utils ---
WORKERS_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKERS_DIR not in sys.path:
    sys.path.insert(0, WORKERS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("e2e")

# ── Configuration ────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "hunyuanvideo-community/HunyuanVideo")
PROMPT = os.environ.get("PROMPT", "A cat walking on green grass")
GPU_ENC = os.environ.get("GPU_ENC", "0")
GPU_DEN = os.environ.get("GPU_DEN", "1,2")
GPU_VAE = os.environ.get("GPU_VAE", "3")
TP_SIZE = int(os.environ.get("TP_SIZE", str(len(GPU_DEN.split(";")[0].split(",")))))
NUM_REQUESTS = int(os.environ.get("NUM_REQUESTS", "1"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "61"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "50"))
HEIGHT = int(os.environ.get("HEIGHT", "544"))
WIDTH = int(os.environ.get("WIDTH", "960"))
GUIDANCE = float(os.environ.get("GUIDANCE", "1.0"))
SEED = int(os.environ.get("SEED", "42"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/disagg_e2e")


from sglang_utils import (  # noqa: E402
    StageClient,
    StageWorkerPool,
    patch_hunyuan_config,
    detect_encoder_modules,
    save_video,
)


# ── Stage launchers ─────────────────────────────────────────────────────

def _launch_stage(
    stage_name: str,
    cuda_devices: str,
    required_modules: List[str],
    custom_stages_fn,
    tp_size: int = 1,
    scheduler_port: int = 15600,
):
    """Launch a partial scheduler for one pipeline stage.

    Returns (processes, server_args).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    num_gpus = len(cuda_devices.split(","))

    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs, set_global_server_args,
    )
    from partial_gpu_worker import launch_partial_server

    server_args = ServerArgs.from_kwargs(
        model_path=MODEL_PATH,
        num_gpus=num_gpus,
        tp_size=tp_size,
        scheduler_port=scheduler_port,
    )
    set_global_server_args(server_args)

    logger.info(
        "Launching %s: CUDA_VISIBLE_DEVICES=%s  num_gpus=%d  tp=%d  port=%d",
        stage_name, cuda_devices, num_gpus, tp_size, server_args.scheduler_port,
    )

    processes = launch_partial_server(
        server_args,
        required_modules=required_modules,
        custom_stages_fn=custom_stages_fn,
    )

    logger.info("%s ready (%d processes)", stage_name, len(processes))
    return processes, server_args


def terminate_processes(processes, name=""):
    for p in processes:
        p.terminate()
    for p in processes:
        p.join(timeout=10)
    logger.info("Terminated %s processes", name)


def _launch_stage_pool(
    stage_name: str,
    gpu_spec: str,
    required_modules: List[str],
    custom_stages_fn,
    tp_size: int = 1,
    base_port: int = 15600,
) -> tuple:
    """Launch N workers for one stage, return (all_processes, pool).

    *gpu_spec* uses ``;`` to separate workers and ``,`` for TP GPUs within
    a worker.  E.g. ``"1,2;5,6"`` means 2 workers each with TP=2.
    Single-worker specs (no ``;``) are backward compatible.
    """
    worker_gpu_lists = gpu_spec.split(";")
    all_procs = []
    clients = []
    for i, cuda_devices in enumerate(worker_gpu_lists):
        port = base_port + i * 10
        procs, server_args = _launch_stage(
            f"{stage_name}[{i}]", cuda_devices.strip(),
            required_modules, custom_stages_fn,
            tp_size=tp_size, scheduler_port=port,
        )
        all_procs.append(procs)
        clients.append(StageClient(server_args.scheduler_endpoint(),
                                   f"{stage_name.lower()}_{i}"))
    pool = StageWorkerPool(clients, stage_name)
    logger.info("%s pool: %d worker(s)", stage_name, pool.num_workers)
    return all_procs, pool


def _format_gpu_spec(gpu_spec: str, tp_size: int = 1) -> str:
    """Format a GPU spec for the logging header.

    Returns e.g. ``"GPU 0, 4 (2 workers)"`` or ``"GPU 1,2 | 5,6 (2 workers, TP=2)"``.
    """
    workers = gpu_spec.split(";")
    n = len(workers)
    sep = " | " if tp_size > 1 else ", "
    gpus = sep.join(w.strip() for w in workers)
    if n > 1:
        tp_info = f", TP={tp_size}" if tp_size > 1 else ""
        return f"GPU {gpus} ({n} workers{tp_info})"
    tp_info = f" (TP={tp_size})" if tp_size > 1 else ""
    return f"GPU {gpus}{tp_info}"


# ── Pipeline execution ──────────────────────────────────────────────────

async def run_single_pipeline(
    req_id: int,
    encoder_pool: StageWorkerPool,
    denoiser_pool: StageWorkerPool,
    vae_pool: StageWorkerPool,
    seed: int,
    save_output: bool = False,
) -> dict:
    """Run one Encoder -> Denoiser -> VAE pipeline, return timing dict.

    Each pool.forward() round-robins across workers in the pool.
    """
    import torch
    from sglang_utils import build_req, inject_tensors_to_req

    timings = {"req_id": req_id}
    negative_prompt = "bad quality" if GUIDANCE > 1.0 else ""
    req_kwargs = dict(
        prompt=PROMPT, negative_prompt=negative_prompt,
        height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE, seed=seed,
    )
    t_pipeline = time.monotonic()

    # ── Encoder ──────────────────────────────────────────────────────
    t0 = time.monotonic()
    enc_output = await encoder_pool.forward([build_req(**req_kwargs)])
    timings["encoder_s"] = time.monotonic() - t0
    if enc_output.error:
        raise RuntimeError(f"Encoder error: {enc_output.error}")
    enc_result = enc_output.output
    # enc_result is either {"_nixl_transfer_meta": {...}} (NIXL) or
    # {"prompt_embeds": tensor, ...} (ZMQ fallback)
    nixl_mode = "_nixl_transfer_meta" in enc_result
    logger.info(
        "req %d | Encoder done %.2fs — transfer: %s",
        req_id, timings["encoder_s"],
        "NIXL RDMA" if nixl_mode else f"ZMQ (keys: {list(enc_result.keys())})",
    )

    # ── Denoiser ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    den_req = build_req(**req_kwargs)
    den_req.do_classifier_free_guidance = (GUIDANCE > 1.0)
    if nixl_mode:
        # Pass NIXL metadata — NixlReceiveStage will RDMA-pull the tensors
        den_req._nixl_transfer_meta = enc_result["_nixl_transfer_meta"]
    else:
        # ZMQ fallback — tensors already in enc_result
        inject_tensors_to_req(den_req, enc_result)
    den_output = await denoiser_pool.forward([den_req])
    timings["denoiser_s"] = time.monotonic() - t0
    if den_output.error:
        raise RuntimeError(f"Denoiser error: {den_output.error}")
    den_result = den_output.output
    nixl_mode_den = "_nixl_transfer_meta" in den_result
    logger.info(
        "req %d | Denoiser done %.2fs — transfer: %s",
        req_id, timings["denoiser_s"],
        "NIXL RDMA" if nixl_mode_den else "ZMQ",
    )

    # ── VAE ──────────────────────────────────────────────────────────
    t0 = time.monotonic()
    vae_req = build_req(prompt="", height=HEIGHT, width=WIDTH,
                        num_frames=NUM_FRAMES, num_inference_steps=NUM_STEPS,
                        guidance_scale=0.0, seed=seed)
    if nixl_mode_den:
        vae_req._nixl_transfer_meta = den_result["_nixl_transfer_meta"]
    else:
        vae_req.latents = den_result["latents"].cpu()
    vae_output = await vae_pool.forward([vae_req])
    timings["vae_s"] = time.monotonic() - t0
    if vae_output.error:
        raise RuntimeError(f"VAE error: {vae_output.error}")

    timings["total_s"] = time.monotonic() - t_pipeline
    logger.info(
        "req %d | VAE done %.2fs — total pipeline: %.2fs",
        req_id, timings["vae_s"], timings["total_s"],
    )

    # Save output as mp4
    if save_output and vae_output.output is not None:
        _save_video(vae_output.output, req_id)

    return timings


def _save_video(frames_tensor, req_id: int):
    """Save decoded video tensor [B,C,T,H,W] as mp4."""
    try:
        out_path = os.path.join(OUTPUT_DIR, f"output_{req_id}.mp4")
        filepath, n_frames = save_video(frames_tensor, out_path)
        logger.info("req %d | Saved %d frames to %s", req_id, n_frames, filepath)
    except Exception as e:
        logger.warning("req %d | Could not save video: %s", req_id, e)


def print_timing_report(all_timings: list, wall_elapsed: float):
    """Print per-stage timing statistics."""
    stages = ["encoder_s", "denoiser_s", "vae_s", "total_s"]
    n = len(all_timings)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Timing Report  (%d requests, concurrency=%d)", n, CONCURRENCY)
    logger.info("=" * 72)

    for t in all_timings:
        logger.info(
            "  req %2d | enc=%6.2fs  den=%6.2fs  vae=%6.2fs  total=%6.2fs",
            t["req_id"], t["encoder_s"], t["denoiser_s"], t["vae_s"], t["total_s"],
        )

    if n > 1:
        logger.info("-" * 72)
        for stage in stages:
            vals = [t[stage] for t in all_timings]
            mean = statistics.mean(vals)
            med = statistics.median(vals)
            mn, mx = min(vals), max(vals)
            std = statistics.stdev(vals) if n >= 2 else 0.0
            logger.info(
                "  %-10s mean=%6.2fs  median=%6.2fs  min=%6.2fs  max=%6.2fs  std=%5.2fs",
                stage, mean, med, mn, mx, std,
            )

    logger.info("-" * 72)
    throughput = n / wall_elapsed if wall_elapsed > 0 else 0
    logger.info("  Wall time: %.2fs | Throughput: %.2f req/s", wall_elapsed, throughput)
    logger.info("=" * 72)


# ── Main ────────────────────────────────────────────────────────────────

async def main():
    from partial_gpu_worker import build_encoder_stages, build_denoiser_stages, build_vae_stages

    # Apply patches once before any SGLang config is created
    patch_hunyuan_config()

    logger.info("=" * 72)
    logger.info("  Disaggregated Diffusion E2E — SGLang Backend")
    logger.info("  Model:    %s", MODEL_PATH)
    logger.info("  Prompt:   %s", PROMPT)
    logger.info("  Encoder:  %s", _format_gpu_spec(GPU_ENC))
    logger.info("  Denoiser: %s", _format_gpu_spec(GPU_DEN, TP_SIZE))
    logger.info("  VAE:      %s", _format_gpu_spec(GPU_VAE))
    logger.info("  Requests: %d  Concurrency: %d", NUM_REQUESTS, CONCURRENCY)
    logger.info("  Frames: %d  Steps: %d  Size: %dx%d  Guidance: %.1f",
                NUM_FRAMES, NUM_STEPS, WIDTH, HEIGHT, GUIDANCE)
    logger.info("=" * 72)

    enc_all_procs = den_all_procs = vae_all_procs = None
    enc_pool = den_pool = vae_pool = None

    try:
        # ── Launch all 3 stage pools ─────────────────────────────────
        t_launch = time.monotonic()

        enc_all_procs, enc_pool = _launch_stage_pool(
            "Encoder", GPU_ENC,
            required_modules=detect_encoder_modules(MODEL_PATH),
            custom_stages_fn=build_encoder_stages,
            tp_size=1,
            base_port=15600,
        )

        den_all_procs, den_pool = _launch_stage_pool(
            "Denoiser", GPU_DEN,
            required_modules=["transformer", "scheduler"],
            custom_stages_fn=build_denoiser_stages,
            tp_size=TP_SIZE,
            base_port=15700,
        )

        vae_all_procs, vae_pool = _launch_stage_pool(
            "VAE", GPU_VAE,
            required_modules=["vae", "scheduler"],
            custom_stages_fn=build_vae_stages,
            tp_size=1,
            base_port=15800,
        )

        logger.info("All stages launched in %.1fs", time.monotonic() - t_launch)

        # ── Warmup ───────────────────────────────────────────────────
        # Send one warmup request per worker (round-robin) so every
        # worker's NIXL connector + UCX transport is fully initialised
        # before concurrent requests hit them.
        num_warmup = max(
            enc_pool.num_workers, den_pool.num_workers, vae_pool.num_workers,
        )
        logger.info("Warmup: %d sequential request(s) …", num_warmup)
        for wi in range(num_warmup):
            warmup = await run_single_pipeline(
                -(wi + 1), enc_pool, den_pool, vae_pool, SEED,
                save_output=False,
            )
            logger.info(
                "  warmup %d/%d — enc=%.2fs den=%.2fs vae=%.2fs total=%.2fs",
                wi + 1, num_warmup,
                warmup["encoder_s"], warmup["denoiser_s"],
                warmup["vae_s"], warmup["total_s"],
            )
        logger.info("Warmup done")

        # ── Run pipeline(s) ──────────────────────────────────────────
        if NUM_REQUESTS <= 1:
            t_wall = time.monotonic()
            timings = await run_single_pipeline(
                0, enc_pool, den_pool, vae_pool, SEED, save_output=True,
            )
            wall_elapsed = time.monotonic() - t_wall
            print_timing_report([timings], wall_elapsed)
        else:
            logger.info("Firing %d requests (concurrency=%d) …", NUM_REQUESTS, CONCURRENCY)
            sem = asyncio.Semaphore(CONCURRENCY)

            async def _run_one(i):
                async with sem:
                    return await run_single_pipeline(
                        i, enc_pool, den_pool, vae_pool,
                        SEED + i, save_output=(i == 0),
                    )

            t_wall = time.monotonic()
            tasks = [asyncio.create_task(_run_one(i)) for i in range(NUM_REQUESTS)]
            all_timings = list(await asyncio.gather(*tasks))
            wall_elapsed = time.monotonic() - t_wall
            print_timing_report(all_timings, wall_elapsed)

    finally:
        for pool in [enc_pool, den_pool, vae_pool]:
            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    pass
        for all_procs, name in [
            (enc_all_procs, "encoder"),
            (den_all_procs, "denoiser"),
            (vae_all_procs, "vae"),
        ]:
            if all_procs is not None:
                for i, procs in enumerate(all_procs):
                    terminate_processes(procs, f"{name}[{i}]")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    asyncio.run(main())
