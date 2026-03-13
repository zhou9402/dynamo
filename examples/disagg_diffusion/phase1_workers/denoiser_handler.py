# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Denoiser stage handler for disaggregated diffusion.

Wraps a ``StageEngine`` running LatentPreparation + TimestepPreparation +
Denoising stages.  Receives embeddings from encoder via NIXL, runs
denoising, registers latents as NIXL-readable for the VAE.
"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, Optional

import torch

from handler_base import BaseGenerativeHandler
from protocol import (
    DenoiserRequest, DenoiserResponse,
    NixlTensorReceiver, NixlTensorSender,
)
from stage_engine import StageEngine

logger = logging.getLogger(__name__)

DEVICE = os.environ.get("DEVICE", "cuda")


class DenoiserStageHandler(BaseGenerativeHandler):
    """Handler for the denoising stage of disaggregated diffusion."""

    def __init__(self, engine: StageEngine, config, publisher=None):
        super().__init__(config, publisher)
        self.engine = engine
        self.receiver = NixlTensorReceiver()
        self.sender = NixlTensorSender()

    def cleanup(self) -> None:
        torch.cuda.empty_cache()
        logger.info("DenoiserStageHandler cleanup complete")
        super().cleanup()

    async def generate(
        self, request: Dict[str, Any], context: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        trace_header = self._get_trace_header(context)
        if trace_header:
            logger.debug("Denoiser request with trace: %s", trace_header)

        try:
            if isinstance(request, str):
                request = json.loads(request)
            req_data = DenoiserRequest(**request)
            logger.info(
                "Denoising %dx%d, %d frames, %d steps",
                req_data.width, req_data.height,
                req_data.num_frames, req_data.num_inference_steps,
            )

            embeddings, _ = await self.receiver.pull(
                req_data.transfer_meta, DEVICE
            )

            from sglang_utils import build_req, inject_tensors_to_req

            req = build_req(
                prompt="(embeddings provided via NIXL)",
                negative_prompt=None,
                height=req_data.height,
                width=req_data.width,
                num_frames=req_data.num_frames,
                num_inference_steps=req_data.num_inference_steps,
                guidance_scale=req_data.guidance_scale,
                seed=req_data.seed,
                device=DEVICE,
            )

            inject_tensors_to_req(req, embeddings)
            if req.prompt_embeds:
                req.do_classifier_free_guidance = bool(
                    req_data.guidance_scale > 1.0
                    and req.negative_prompt_embeds
                    and len(req.negative_prompt_embeds) > 0
                )

            req = await self.engine.async_generate(req)

            latents = req.latents
            transfer_meta = await self.sender.prepare({"latents": latents})

            logger.info("Denoised — latents %s", list(latents.shape))
            yield DenoiserResponse(
                transfer_meta=transfer_meta, shape=list(latents.shape)
            ).model_dump()

        except Exception as e:
            logger.error("Denoiser generate failed: %s", e, exc_info=True)
            yield {"error": str(e), "transfer_meta": {}, "shape": []}

    async def health(
        self, request: Dict[str, Any], context: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield self.engine.health()
