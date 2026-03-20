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
    from sglang_utils import launch_stage_server, detect_encoder_modules, build_req
    from partial_gpu_worker import build_encoder_stages

    enc_modules = detect_encoder_modules(MODEL_PATH)
    logger.info("Launching encoder Scheduler: modules=%s, port=%d", enc_modules, SCHEDULER_PORT)
    processes, client, server_args = launch_stage_server(
        MODEL_PATH, enc_modules, build_encoder_stages,
        SCHEDULER_PORT, client_name="encoder",
    )

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
            if transfer_meta:
                logger.info("Encoded prompt — NIXL metadata ready")
                yield {"transfer_meta": transfer_meta, "shapes": {}}
            else:
                # ZMQ fallback: forward raw tensors (when NIXL is disabled)
                import torch, base64, io
                tensor_data = {}
                for k, v in result.items():
                    if isinstance(v, torch.Tensor):
                        buf = io.BytesIO()
                        torch.save(v.cpu(), buf)
                        tensor_data[k] = base64.b64encode(buf.getvalue()).decode()
                    elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                        tensor_data[k] = []
                        for t in v:
                            buf = io.BytesIO()
                            torch.save(t.cpu(), buf)
                            tensor_data[k].append(base64.b64encode(buf.getvalue()).decode())
                logger.info("Encoded prompt — ZMQ fallback (%d tensor fields)", len(tensor_data))
                yield {"transfer_meta": {}, "tensor_data": tensor_data, "shapes": {}}

        except Exception as e:
            logger.error("Encoder generate failed: %s", e, exc_info=True)
            yield {"error": str(e), "transfer_meta": {}, "shapes": {}}

    async def handle_health(request, context):
        yield {"status": "ok", "stage": "encoder", "model": MODEL_PATH}

    # ── Serve Dynamo endpoints ───────────────────────────────────────

    gen_ep = runtime.endpoint("disagg_diffusion.encoder.generate")
    health_ep = runtime.endpoint("disagg_diffusion.encoder.health")

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
