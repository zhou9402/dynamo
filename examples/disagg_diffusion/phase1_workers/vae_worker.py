#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion — VAE Worker (Dynamo RPC)

Wraps an SGLang Scheduler subprocess running NixlReceiveStage + DecodingStage.
Receives latents via NIXL RDMA, decodes to video frames, saves as mp4.

Process architecture::

    Dynamo Worker Process (this file)
    |-- @dynamo_worker
    |   |-- serve_endpoint("generate")  <-- Dynamo RPC from orchestrator
    |   |   +-- handle_generate()
    |   |       +-- StageClient.forward()  <-- ZMQ to local Scheduler
    |   +-- serve_endpoint("health")
    |
    +-- SGLang Scheduler subprocess (spawned by launch_partial_server)
        +-- PartialGPUWorker
            |-- NixlReceiveStage  <-- RDMA-pull latents from denoiser
            +-- DecodingStage     <-- VAE decode to video frames
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import sys
import uuid

import numpy as np
import uvloop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamo.runtime import DistributedRuntime, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "hunyuanvideo-community/HunyuanVideo")
SCHEDULER_PORT = int(os.environ.get("SCHEDULER_PORT", "15800"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/disagg_videos")


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    from run_e2e_sglang import (
        _patch_hunyuan_config_task_type,
        StageClient,
    )
    from partial_gpu_worker import build_vae_stages, launch_partial_server
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs, set_global_server_args,
    )
    from sglang_utils import build_req

    _patch_hunyuan_config_task_type()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    server_args = ServerArgs.from_kwargs(
        model_path=MODEL_PATH,
        num_gpus=1,
        tp_size=1,
        scheduler_port=SCHEDULER_PORT,
    )
    set_global_server_args(server_args)

    logger.info("Launching VAE Scheduler: port=%d", SCHEDULER_PORT)
    processes = launch_partial_server(
        server_args,
        required_modules=["vae", "scheduler"],
        custom_stages_fn=build_vae_stages,
    )

    # Connect ZMQ client to local Scheduler
    client = StageClient(server_args.scheduler_endpoint, "vae")

    # ── Dynamo RPC handlers ──────────────────────────────────────────

    async def handle_generate(request, context):
        try:
            if isinstance(request, str):
                request = json.loads(request)

            req = build_req(
                prompt="",
                height=request.get("height", 544),
                width=request.get("width", 960),
                num_frames=request.get("num_frames", 61),
                num_inference_steps=1,
                guidance_scale=0.0,
                seed=request.get("seed", 42),
            )

            # Pass NIXL metadata for NixlReceiveStage to RDMA-pull latents
            transfer_meta = request.get("transfer_meta", {})
            if transfer_meta:
                req._nixl_transfer_meta = transfer_meta

            output = await client.forward([req])
            if output.error:
                yield {"error": str(output.error), "video_path": "", "num_frames": 0}
                return

            # Extract frames and save video
            frames_tensor = output.output
            request_id = request.get("request_id") or str(uuid.uuid4())[:8]

            loop = asyncio.get_event_loop()
            filename, n_frames = await loop.run_in_executor(
                None, _save_video_frames, frames_tensor, request_id,
            )
            logger.info("Decoded — %d frames -> %s", n_frames, filename)
            yield {"video_path": filename, "num_frames": n_frames}

        except Exception as e:
            logger.error("VAE generate failed: %s", e, exc_info=True)
            yield {"error": str(e), "video_path": "", "num_frames": 0}

    async def handle_health(request, context):
        yield {"status": "ok", "stage": "vae", "model": MODEL_PATH}

    # ── Serve Dynamo endpoints ───────────────────────────────────────

    ns = runtime.namespace("disagg_diffusion")
    gen_ep = ns.component("vae").endpoint("generate")
    health_ep = ns.component("vae").endpoint("health")

    logger.info("Serving: disagg_diffusion.vae.generate + health")
    try:
        await asyncio.gather(
            gen_ep.serve_endpoint(handle_generate),
            health_ep.serve_endpoint(handle_health),
        )
    finally:
        client.close()
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=10)


def _save_video_frames(frames_tensor, request_id: str) -> tuple:
    """Save decoded frames as mp4. Returns (filename, num_frames)."""
    import torch

    if isinstance(frames_tensor, dict):
        # OutputBatch may return dict — extract the video tensor
        for v in frames_tensor.values():
            if hasattr(v, "shape"):
                frames_tensor = v
                break

    if hasattr(frames_tensor, "cpu"):
        frames_tensor = frames_tensor.cpu().float().numpy()

    # [B, C, T, H, W] -> [T, H, W, C]
    frames = (frames_tensor[0].transpose(1, 2, 3, 0) * 255).clip(0, 255).astype(np.uint8)

    filename = f"{request_id}.mp4"
    filepath = os.path.join(OUTPUT_DIR, filename)

    try:
        import imageio
        imageio.mimwrite(filepath, frames, fps=24, codec="libx264")
    except Exception as e:
        logger.warning("mp4 export failed (%s), saving first frame as PNG", e)
        from PIL import Image
        img = Image.fromarray(frames[0])
        filepath = filepath.replace(".mp4", ".png")
        filename = filename.replace(".mp4", ".png")
        img.save(filepath)

    return filename, len(frames)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    mp.set_start_method("spawn", force=True)
    uvloop.install()
    asyncio.run(worker())
