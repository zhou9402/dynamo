# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VAE decoding stage handler for disaggregated diffusion."""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator, Dict

import numpy as np
import torch

from handler_base import BaseGenerativeHandler
from protocol import (
    VAEDecodeRequest, VAEDecodeResponse,
    NixlTensorReceiver,
)

logger = logging.getLogger(__name__)

DEVICE = os.environ.get("DEVICE", "cuda")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/disagg_videos")


class VAEStageHandler(BaseGenerativeHandler):
    """Handler for the VAE decoding stage using SGLang DecodingStage."""

    def __init__(self, engine, config, publisher=None):
        super().__init__(config, publisher)
        self.engine = engine
        self.receiver = NixlTensorReceiver()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def cleanup(self) -> None:
        torch.cuda.empty_cache()
        logger.info("VAEStageHandler cleanup complete")
        super().cleanup()

    async def generate(
        self, request: Dict[str, Any], context: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        trace_header = self._get_trace_header(context)
        if trace_header:
            logger.debug("VAE request with trace: %s", trace_header)

        try:
            if isinstance(request, str):
                request = json.loads(request)
            req_data = VAEDecodeRequest(**request)
            logger.info("[SGLang] Decoding latents via NIXL …")

            tensors, _ = await self.receiver.pull(
                req_data.transfer_meta, DEVICE
            )
            latents = tensors["latents"]

            from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
            req = Req(latents=latents, suppress_logs=True)
            output_batch = await self.engine.async_generate(req)

            loop = asyncio.get_event_loop()
            frames = await loop.run_in_executor(
                None, _extract_frames_sglang, output_batch
            )

            request_id = req_data.request_id or str(uuid.uuid4())[:8]
            filename = f"{request_id}.mp4"
            filepath = os.path.join(OUTPUT_DIR, filename)
            await loop.run_in_executor(None, _save_video, frames, filepath)

            logger.info("[SGLang] Decoded — %d frames → %s", len(frames), filepath)
            yield VAEDecodeResponse(
                video_path=filename, num_frames=len(frames)
            ).model_dump()

        except Exception as e:
            logger.error("VAE generate failed: %s", e, exc_info=True)
            yield {"error": str(e), "video_path": "", "num_frames": 0}

    async def health(
        self, request: Dict[str, Any], context: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield self.engine.health()


def _extract_frames_sglang(req_or_batch) -> np.ndarray:
    """Extract decoded frames from SGLang OutputBatch as float32 [0,1] numpy array."""
    output = getattr(req_or_batch, "output", None)
    if output is None:
        output = getattr(req_or_batch, "latents", None)
    if output is None:
        raise RuntimeError("No output tensor found in engine result")
    # output[0] is [C, F, H, W] in [0,1]; must return float32 because
    # export_to_video unconditionally does (frame*255).astype(uint8).
    video = output[0].permute(1, 2, 3, 0)
    return video.float().cpu().numpy()


def _save_video(frames, filepath: str):
    """Save frames as mp4. Falls back to PNG if ffmpeg is unavailable."""
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, filepath, fps=16)
    except Exception as e:
        logger.warning("mp4 export failed (%s), saving first frame as PNG", e)
        from PIL import Image
        frame = frames[0]
        if isinstance(frame, np.ndarray) and frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        png_path = filepath.replace(".mp4", ".png")
        img.save(png_path)
