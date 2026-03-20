# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for disaggregated diffusion stages.

Uses NIXL RDMA for GPU-direct tensor transfer between stage workers.
Only small metadata (shapes, dtypes, NIXL descriptor) travels over Dynamo RPC.
Tensor transfer is handled by NixlSendStage/NixlReceiveStage in
partial_gpu_worker.py and nixl_transfer.py.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage 1: Encoder
# ---------------------------------------------------------------------------

class EncoderRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    guidance_scale: float = 1.0


class EncoderResponse(BaseModel):
    transfer_meta: Dict[str, Any] = Field(default_factory=dict)
    shapes: Dict[str, List[int]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 2: Denoiser
# ---------------------------------------------------------------------------

class DenoiserRequest(BaseModel):
    transfer_meta: Dict[str, Any]
    tensor_data: Dict[str, Any] = Field(default_factory=dict)
    height: int = 544
    width: int = 960
    num_frames: int = 61
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    seed: int = 42


class DenoiserResponse(BaseModel):
    transfer_meta: Dict[str, Any] = Field(default_factory=dict)
    shape: List[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 3: VAE Decoder
# ---------------------------------------------------------------------------

class VAEDecodeRequest(BaseModel):
    transfer_meta: Dict[str, Any]
    tensor_data: Dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""


class VAEDecodeResponse(BaseModel):
    video_path: str = ""
    num_frames: int = 0


# ---------------------------------------------------------------------------
# End-to-end (orchestrator convenience)
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    height: int = 544
    width: int = 960
    num_frames: int = 61
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Health (per-stage)
# ---------------------------------------------------------------------------

class HealthRequest(BaseModel):
    pass


class HealthResponse(BaseModel):
    status: str = "ok"
    stage: str = ""
    model: str = ""
