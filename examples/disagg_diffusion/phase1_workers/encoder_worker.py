#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion — Encoder Worker

Thin launcher: builds the StageEngine, creates an EncoderStageHandler
(inheriting BaseGenerativeHandler), and wires it to Dynamo endpoints.
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


async def build_encoder_engine():
    """Load text encoder pipeline and return (StageEngine, Config)."""
    from sglang_utils import (
        build_server_args, build_partial_pipeline,
        build_config, get_component_backend,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
        TextEncodingStage,
    )
    from stage_engine import StageEngine

    logger.info("Loading text encoder from %s (SGLang PipelineStage)", MODEL_PATH)
    server_args = build_server_args(MODEL_PATH)
    pipeline = build_partial_pipeline(
        server_args,
        required_modules=["text_encoder", "tokenizer", "scheduler"],
    )

    text_encoder = pipeline.get_module("text_encoder")
    tokenizer = pipeline.get_module("tokenizer")
    logger.info("text_encoder backend: %s", get_component_backend(text_encoder))

    stage = TextEncodingStage(
        text_encoders=[text_encoder],
        tokenizers=[tokenizer],
    )

    engine = StageEngine(stages=[stage], server_args=server_args)
    await engine.start()

    from sglang_utils import build_req
    warmup_req = build_req(prompt="warmup", negative_prompt="warmup", num_inference_steps=1)
    warmup_req.set_as_warmup(1)
    await engine.warmup(warmup_req)

    config = build_config(server_args)
    logger.info("Encoder ready — VRAM: %.0f MB", torch.cuda.memory_allocated() / 1e6)
    return engine, config


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    engine, config = await build_encoder_engine()

    from encoder_handler import EncoderStageHandler
    handler = EncoderStageHandler(engine=engine, config=config)

    ns = runtime.namespace("disagg_diffusion")
    gen_ep = ns.component("encoder").endpoint("generate")
    health_ep = ns.component("encoder").endpoint("health")

    logger.info("Serving: disagg_diffusion.encoder.generate + health")
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
