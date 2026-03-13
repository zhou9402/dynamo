#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion — VAE Worker

Loads the VAE via SGLang DecodingStage and wires it to Dynamo endpoints.
"""

import asyncio
import logging
import os
import sys

import torch
import uvloop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamo.runtime import DistributedRuntime, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
DEVICE = os.environ.get("DEVICE", "cuda")


async def build_vae_engine():
    """Load VAE pipeline and return (StageEngine, Config)."""
    from sglang_utils import (
        build_server_args, build_partial_pipeline,
        build_config, get_component_backend,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
        DecodingStage,
    )
    from stage_engine import StageEngine

    logger.info("Loading VAE from %s (SGLang PipelineStage)", MODEL_PATH)
    server_args = build_server_args(MODEL_PATH)
    pipeline = build_partial_pipeline(
        server_args,
        required_modules=["vae", "scheduler"],
    )

    vae = pipeline.get_module("vae")
    logger.info("vae backend: %s", get_component_backend(vae))

    stage = DecodingStage(vae=vae, pipeline=pipeline)

    engine = StageEngine(stages=[stage], server_args=server_args)
    await engine.start()

    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    z_dim = server_args.pipeline_config.vae_config.arch_config.z_dim
    warmup_latents = torch.randn(1, z_dim, 3, 60, 104, device="cuda")
    warmup_req = Req(latents=warmup_latents, suppress_logs=True)
    warmup_req.set_as_warmup(1)
    await engine.warmup(warmup_req)

    config = build_config(server_args)
    logger.info("VAE ready — VRAM: %.0f MB", torch.cuda.memory_allocated() / 1e6)
    return engine, config


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    engine, config = await build_vae_engine()

    from vae_handler import VAEStageHandler
    handler = VAEStageHandler(engine=engine, config=config)

    ns = runtime.namespace("disagg_diffusion")
    gen_ep = ns.component("vae").endpoint("generate")
    health_ep = ns.component("vae").endpoint("health")

    logger.info("Serving: disagg_diffusion.vae.generate + health")
    try:
        await asyncio.gather(
            gen_ep.serve_endpoint(handler.generate),
            health_ep.serve_endpoint(handler.health),
        )
    finally:
        handler.cleanup()
        await engine.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    uvloop.install()
    asyncio.run(worker())
