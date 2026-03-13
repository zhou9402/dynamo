# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol types for disaggregated diffusion stages.

Uses NIXL RDMA for GPU-direct tensor transfer between stage workers.
Only small metadata (shapes, dtypes, NIXL descriptor) travels over Dynamo RPC.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel

import dynamo.nixl_connect as nixl_connect

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NIXL persistent connector
# ---------------------------------------------------------------------------
# UCX endpoints are pooled at the process level. When a Connection (NIXL
# agent) is destroyed, its UCX resources are returned to the pool in a
# stale state, which corrupts subsequent connections.
#
# A PersistentConnector reuses a single Connection for all operations,
# keeping the UCX transport alive across requests. Remote._release() is
# left intact so old remote-agent references are properly cleaned up
# (remove_remote_agent only drops metadata, not the UCX transport, when
# the local agent is persistent).

class _PersistentConnector(nixl_connect.Connector):
    """Connector that reuses a single NIXL Connection (agent) for all ops."""

    def __init__(self):
        super().__init__()
        self._persistent_connection = None

    async def _create_connection(self) -> nixl_connect.Connection:
        if self._persistent_connection is None:
            self._persistent_connection = nixl_connect.Connection(self, 1)
            await self._persistent_connection.initialize()
        return self._persistent_connection


# ---------------------------------------------------------------------------
# NIXL sender / receiver
# ---------------------------------------------------------------------------

class NixlTensorSender:
    """Registers tensors as NIXL-readable. Produces metadata for the receiver.

    The readable is kept alive as a background task so the generator
    can complete immediately after yielding metadata (avoiding deadlock
    with the orchestrator's stream drain).
    """

    def __init__(self):
        self._connector = _PersistentConnector()
        self._pending_tasks: list = []

    async def prepare(
        self,
        tensors: Dict[str, torch.Tensor],
        extra: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Register tensors and return metadata_dict.

        The NIXL readable is kept alive in a background task.
        The caller should yield the metadata_dict in its response and
        then return — no need to await anything.
        """
        import asyncio

        # Clean up completed tasks to avoid unbounded growth.
        self._pending_tasks = [t for t in self._pending_tasks if not t.done()]

        flat = torch.cat([t.contiguous().view(-1) for t in tensors.values()])
        descriptor = nixl_connect.Descriptor(flat)
        readable = await self._connector.create_readable(descriptor)
        raw_meta = readable.metadata()

        meta_dict = {
            "tensor_keys": list(tensors.keys()),
            "shapes": {k: list(t.shape) for k, t in tensors.items()},
            "dtypes": {k: str(t.dtype).removeprefix("torch.") for k, t in tensors.items()},
            "nixl_metadata": raw_meta.model_dump() if hasattr(raw_meta, "model_dump") else raw_meta,
            "extra": extra or {},
        }

        async def _keep_alive():
            """Background task that keeps readable/descriptor/flat alive."""
            try:
                await readable.wait_for_completion()
            except Exception as e:
                logger.warning("NIXL readable wait failed: %s", e)

        task = asyncio.ensure_future(_keep_alive())
        self._pending_tasks.append(task)
        return meta_dict


class NixlTensorReceiver:
    """Pulls tensors from a remote sender via NIXL RDMA."""

    def __init__(self):
        self._connector = _PersistentConnector()

    async def pull(self, meta: dict, device: str = "cuda") -> tuple:
        """Pull tensors described by meta. Returns (tensors_dict, extra_dict).

        Allocates the receive buffer directly on the target device so that
        NIXL RDMA writes land on GPU memory without an extra CPU→GPU copy.
        """
        total_bytes = 0
        specs = []
        for key in meta["tensor_keys"]:
            shape = meta["shapes"][key]
            dtype = getattr(torch, meta["dtypes"][key])
            numel = 1
            for s in shape:
                numel *= s
            size = numel * dtype.itemsize
            specs.append((key, shape, dtype, size))
            total_bytes += size

        # Allocate directly on the target device for GPU-direct RDMA.
        flat = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        descriptor = nixl_connect.Descriptor(flat)

        rdma_meta = nixl_connect.RdmaMetadata.model_validate(meta["nixl_metadata"])
        read_op = await self._connector.begin_read(rdma_meta, descriptor)
        await read_op.wait_for_completion()

        result = {}
        offset = 0
        for key, shape, dtype, size in specs:
            result[key] = flat[offset : offset + size].view(dtype=dtype).reshape(shape)
            offset += size

        return result, meta.get("extra", {})


# ---------------------------------------------------------------------------
# Stage 1: Encoder
# ---------------------------------------------------------------------------

class EncoderRequest(BaseModel):
    prompt: str
    negative_prompt: str = "Blurry, low quality, distorted"
    guidance_scale: float = 5.0


class EncoderResponse(BaseModel):
    transfer_meta: Dict[str, Any]
    shapes: Dict[str, List[int]] = {}


# ---------------------------------------------------------------------------
# Stage 2: Denoiser
# ---------------------------------------------------------------------------

class DenoiserRequest(BaseModel):
    transfer_meta: Dict[str, Any]
    height: int = 480
    width: int = 832
    num_frames: int = 17
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    seed: int = 42


class DenoiserResponse(BaseModel):
    transfer_meta: Dict[str, Any]
    shape: List[int] = []


# ---------------------------------------------------------------------------
# Stage 3: VAE Decoder
# ---------------------------------------------------------------------------

class VAEDecodeRequest(BaseModel):
    transfer_meta: Dict[str, Any]
    request_id: str = ""


class VAEDecodeResponse(BaseModel):
    video_path: str = ""
    num_frames: int = 0


# ---------------------------------------------------------------------------
# End-to-end (orchestrator convenience)
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "Blurry, low quality, distorted"
    height: int = 480
    width: int = 832
    num_frames: int = 17
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    seed: int = 42


# ---------------------------------------------------------------------------
# Health (per-stage engine metrics)
# ---------------------------------------------------------------------------

class HealthRequest(BaseModel):
    pass


class HealthResponse(BaseModel):
    running: bool = False
    queue_depth: int = 0
    max_queue_depth: int = 0
    active_requests: int = 0
    total_processed: int = 0
    avg_latency_s: float = 0.0
    last_latency_s: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
