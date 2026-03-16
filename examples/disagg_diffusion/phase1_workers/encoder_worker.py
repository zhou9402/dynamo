#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion — Encoder Worker (Dynamo RPC)

Wraps an SGLang Scheduler subprocess running TextEncodingStage + NixlSendStage.
Exposes a Dynamo RPC endpoint. The Scheduler handles model loading, GPU
execution, and NIXL tensor registration. This worker bridges Dynamo RPC <-> ZMQ.

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
            |-- TextEncodingStage
            +-- NixlSendStage  <-- registers embeddings as NIXL-readable
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
SCHEDULER_PORT = int(os.environ.get("SCHEDULER_PORT", "15600"))


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    from run_e2e_sglang import (
        _patch_hunyuan_config_task_type,
        _detect_encoder_modules,
        StageClient,
    )
    from partial_gpu_worker import build_encoder_stages, launch_partial_server
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs, set_global_server_args,
    )
    from sglang_utils import build_req

    _patch_hunyuan_config_task_type()

    # Launch SGLang Scheduler subprocess with text encoder stages
    enc_modules = _detect_encoder_modules(MODEL_PATH)
    server_args = ServerArgs.from_kwargs(
        model_path=MODEL_PATH,
        num_gpus=1,
        tp_size=1,
        scheduler_port=SCHEDULER_PORT,
    )
    set_global_server_args(server_args)

    logger.info(
        "Launching encoder Scheduler: modules=%s, port=%d",
        enc_modules, SCHEDULER_PORT,
    )
    processes = launch_partial_server(
        server_args,
        required_modules=enc_modules,
        custom_stages_fn=build_encoder_stages,
    )

    # Connect ZMQ client to local Scheduler
    client = StageClient(server_args.scheduler_endpoint, "encoder")

    # ── Dynamo RPC handlers ──────────────────────────────────────────

    async def handle_generate(request, context):
        try:
            if isinstance(request, str):
                request = json.loads(request)

            req = build_req(
                prompt=request.get("prompt", ""),
                negative_prompt=request.get("negative_prompt", ""),
                guidance_scale=request.get("guidance_scale", 1.0),
            )

            output = await client.forward([req])
            if output.error:
                yield {"error": str(output.error), "transfer_meta": {}, "shapes": {}}
                return

            result = output.output
            transfer_meta = result.get("_nixl_transfer_meta", {})
            logger.info("Encoded prompt — NIXL metadata ready")
            yield {"transfer_meta": transfer_meta, "shapes": {}}

        except Exception as e:
            logger.error("Encoder generate failed: %s", e, exc_info=True)
            yield {"error": str(e), "transfer_meta": {}, "shapes": {}}

    async def handle_health(request, context):
        yield {"status": "ok", "stage": "encoder", "model": MODEL_PATH}

    # ── Serve Dynamo endpoints ───────────────────────────────────────

    ns = runtime.namespace("disagg_diffusion")
    gen_ep = ns.component("encoder").endpoint("generate")
    health_ep = ns.component("encoder").endpoint("health")

    logger.info("Serving: disagg_diffusion.encoder.generate + health")
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
