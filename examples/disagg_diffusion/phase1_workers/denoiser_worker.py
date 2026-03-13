#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion — Denoiser Worker

Thin launcher: builds the StageEngine (3 stages), creates a
DenoiserStageHandler (inheriting BaseGenerativeHandler), and wires it
to Dynamo endpoints.
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


async def build_denoiser_engine():
    """Load transformer pipeline and return (StageEngine, Config)."""
    from sglang_utils import (
        build_server_args, build_partial_pipeline,
        build_config, get_component_backend,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
        LatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
        TimestepPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )
    from stage_engine import StageEngine

    logger.info("Loading transformer from %s (SGLang PipelineStage)", MODEL_PATH)
    server_args = build_server_args(MODEL_PATH)
    pipeline = build_partial_pipeline(
        server_args,
        required_modules=["transformer", "scheduler"],
    )

    transformer = pipeline.get_module("transformer")
    scheduler = pipeline.get_module("scheduler")
    logger.info("transformer backend: %s", get_component_backend(transformer))

    latent_prep = LatentPreparationStage(scheduler=scheduler, transformer=transformer)
    timestep_prep = TimestepPreparationStage(scheduler=scheduler)
    denoising = DenoisingStage(transformer=transformer, scheduler=scheduler)

    engine = StageEngine(
        stages=[latent_prep, timestep_prep, denoising],
        server_args=server_args,
    )
    await engine.start()

    from sglang_utils import build_req
    z_dim = server_args.pipeline_config.vae_config.arch_config.z_dim
    warmup_req = build_req(
        prompt="warmup", negative_prompt="warmup",
        height=480, width=832, num_frames=17, num_inference_steps=1,
    )
    warmup_req.prompt_embeds = [torch.zeros(1, 512, 4096, device="cuda")]
    warmup_req.negative_prompt_embeds = [torch.zeros(1, 512, 4096, device="cuda")]
    warmup_req.do_classifier_free_guidance = True
    warmup_req.set_as_warmup(1)
    await engine.warmup(warmup_req)

    config = build_config(server_args)
    logger.info("Denoiser ready — VRAM: %.0f MB", torch.cuda.memory_allocated() / 1e6)
    return engine, config


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    engine, config = await build_denoiser_engine()

    from denoiser_handler import DenoiserStageHandler
    handler = DenoiserStageHandler(engine=engine, config=config)

    ns = runtime.namespace("disagg_diffusion")
    gen_ep = ns.component("denoiser").endpoint("generate")
    health_ep = ns.component("denoiser").endpoint("health")

    logger.info("Serving: disagg_diffusion.denoiser.generate + health")
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
