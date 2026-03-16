#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion — Denoiser Worker (Dynamo RPC)

Wraps SGLang Scheduler subprocess(es) running NixlReceiveStage + Denoising +
NixlSendStage. Supports TP via launch_partial_server(tp_size=N).

Process architecture::

    Dynamo Worker Process (this file)
    |-- @dynamo_worker
    |   |-- serve_endpoint("generate")  <-- Dynamo RPC from orchestrator
    |   |   +-- handle_generate()
    |   |       +-- StageClient.forward()  <-- ZMQ to local Scheduler
    |   +-- serve_endpoint("health")
    |
    +-- SGLang Scheduler subprocess(es) (spawned by launch_partial_server)
        +-- PartialGPUWorker (TP=N)
            |-- NixlReceiveStage  <-- RDMA-pull embeddings from encoder
            |-- LatentPreparationStage
            |-- TimestepPreparationStage
            |-- DenoisingStage
            +-- NixlSendStage  <-- register latents as NIXL-readable
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import sys

import uvloop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamo.runtime import DistributedRuntime, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "hunyuanvideo-community/HunyuanVideo")
SCHEDULER_PORT = int(os.environ.get("SCHEDULER_PORT", "15700"))


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    from sglang_utils import StageClient, patch_hunyuan_config, build_req
    from partial_gpu_worker import build_denoiser_stages, launch_partial_server
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs, set_global_server_args,
    )

    patch_hunyuan_config()

    # Auto-detect GPU count from CUDA_VISIBLE_DEVICES
    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    tp_size = int(os.environ.get("TP_SIZE", str(num_gpus)))

    server_args = ServerArgs.from_kwargs(
        model_path=MODEL_PATH,
        num_gpus=num_gpus,
        tp_size=tp_size,
        scheduler_port=SCHEDULER_PORT,
    )
    set_global_server_args(server_args)

    logger.info(
        "Launching denoiser Scheduler: num_gpus=%d, tp=%d, port=%d",
        num_gpus, tp_size, SCHEDULER_PORT,
    )
    processes = launch_partial_server(
        server_args,
        required_modules=["transformer", "scheduler"],
        custom_stages_fn=build_denoiser_stages,
    )

    # Connect ZMQ client to local Scheduler
    client = StageClient(server_args.scheduler_endpoint, "denoiser")

    # ── Dynamo RPC handlers ──────────────────────────────────────────

    async def handle_generate(request, context):
        try:
            if isinstance(request, str):
                request = json.loads(request)

            req = build_req(
                prompt="(embeddings via NIXL)",
                negative_prompt="",
                height=request.get("height", 544),
                width=request.get("width", 960),
                num_frames=request.get("num_frames", 61),
                num_inference_steps=request.get("num_inference_steps", 50),
                guidance_scale=request.get("guidance_scale", 1.0),
                seed=request.get("seed", 42),
            )
            req.do_classifier_free_guidance = (req.guidance_scale > 1.0)

            # Pass NIXL metadata for NixlReceiveStage to RDMA-pull embeddings
            transfer_meta = request.get("transfer_meta", {})
            if transfer_meta:
                req._nixl_transfer_meta = transfer_meta

            output = await client.forward([req])
            if output.error:
                yield {"error": str(output.error), "transfer_meta": {}, "shape": []}
                return

            result = output.output
            transfer_meta_out = result.get("_nixl_transfer_meta", {})
            logger.info("Denoised — NIXL latent metadata ready")
            yield {"transfer_meta": transfer_meta_out, "shape": []}

        except Exception as e:
            logger.error("Denoiser generate failed: %s", e, exc_info=True)
            yield {"error": str(e), "transfer_meta": {}, "shape": []}

    async def handle_health(request, context):
        yield {
            "status": "ok", "stage": "denoiser",
            "model": MODEL_PATH, "tp_size": tp_size,
        }

    # ── Serve Dynamo endpoints ───────────────────────────────────────

    ns = runtime.namespace("disagg_diffusion")
    gen_ep = ns.component("denoiser").endpoint("generate")
    health_ep = ns.component("denoiser").endpoint("health")

    logger.info("Serving: disagg_diffusion.denoiser.generate + health")
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    mp.set_start_method("spawn", force=True)
    uvloop.install()
    asyncio.run(worker())
