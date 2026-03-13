# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder stage handler for disaggregated diffusion.

Wraps a ``StageEngine`` running ``TextEncodingStage``.  Produces text
embeddings and registers them as NIXL-readable for the denoiser.
"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, Optional

import torch

from handler_base import BaseGenerativeHandler
from protocol import (
    EncoderRequest, EncoderResponse,
    NixlTensorSender,
)
from stage_engine import StageEngine

logger = logging.getLogger(__name__)

DEVICE = os.environ.get("DEVICE", "cuda")


class EncoderStageHandler(BaseGenerativeHandler):
    """Handler for the text-encoding stage of disaggregated diffusion."""

    def __init__(self, engine: StageEngine, config, publisher=None):
        super().__init__(config, publisher)
        self.engine = engine
        self.sender = NixlTensorSender()

    def cleanup(self) -> None:
        torch.cuda.empty_cache()
        logger.info("EncoderStageHandler cleanup complete")
        super().cleanup()

    async def generate(
        self, request: Dict[str, Any], context: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        trace_header = self._get_trace_header(context)
        if trace_header:
            logger.debug("Encoder request with trace: %s", trace_header)

        try:
            if isinstance(request, str):
                request = json.loads(request)
            req_data = EncoderRequest(**request)
            logger.info("Encoding prompt: %.80s…", req_data.prompt)

            from sglang_utils import build_req, extract_tensors_from_req

            req = build_req(
                prompt=req_data.prompt,
                negative_prompt=req_data.negative_prompt,
                guidance_scale=req_data.guidance_scale,
                device=DEVICE,
            )

            req = await self.engine.async_generate(req)

            payload = extract_tensors_from_req(
                req, ["prompt_embeds", "negative_prompt_embeds"]
            )
            transfer_meta = await self.sender.prepare(payload)

            shapes = {k: list(v.shape) for k, v in payload.items()}
            logger.info("Encoded — shapes %s", shapes)

            yield EncoderResponse(
                transfer_meta=transfer_meta, shapes=shapes
            ).model_dump()

        except Exception as e:
            logger.error("Encoder generate failed: %s", e, exc_info=True)
            yield {"error": str(e), "transfer_meta": {}, "shapes": {}}

    async def health(
        self, request: Dict[str, Any], context: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield self.engine.health()
