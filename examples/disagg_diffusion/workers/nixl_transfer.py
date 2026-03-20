# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL RDMA tensor transfer for disaggregated diffusion stages.

Provides GPU-direct tensor transfer between stage workers. Only small
metadata (shapes, dtypes, NIXL descriptor ~1.5 KB) travels over the ZMQ
control plane; actual tensor data (embeddings, latents) transfers
GPU->GPU via NIXL RDMA.

Usage inside PipelineStage.forward() (synchronous context)::

    sender = NixlTensorSender()
    meta = sender.send({"latents": tensor})   # registers & returns metadata
    # ... pass meta via ZMQ ...

    receiver = NixlTensorReceiver()
    tensors = receiver.recv(meta, device="cuda")  # RDMA pull
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Dict

import torch

logger = logging.getLogger(__name__)

try:
    import dynamo.nixl_connect as nixl_connect
    NIXL_AVAILABLE = not os.environ.get("DISABLE_NIXL", "").lower() in ("1", "true", "yes")
    if not NIXL_AVAILABLE:
        logger.info("NIXL disabled via DISABLE_NIXL env var")
except ImportError:
    NIXL_AVAILABLE = False
    logger.info("NIXL not available — falling back to ZMQ tensor transfer")


class _PersistentConnector:
    """Lazily-initialized NIXL Connector singleton per process."""

    _instance = None

    @classmethod
    async def get(cls):
        if cls._instance is None:
            cls._instance = nixl_connect.Connector()
            await cls._instance.initialize()
        return cls._instance


class NixlTensorSender:
    """Register GPU tensors as NIXL-readable. Returns metadata for the receiver.

    Buffers are held until the receiver completes the RDMA pull (detected
    via synchronous ``readable.status`` polling) or a configurable timeout
    expires.  Previous implementation scheduled a ``_keep_alive`` background
    task via ``asyncio.ensure_future``, but the event loop only runs during
    ``run_until_complete`` and stops immediately after — so the background
    task never executed, leaking GPU memory.
    """

    BUFFER_TIMEOUT_S = float(os.environ.get("NIXL_BUFFER_TIMEOUT_S", "120"))

    def __init__(self):
        # Each entry: (readable, flat_buffer_ref, creation_timestamp)
        self._pending: list[tuple[object, torch.Tensor, float]] = []

    def send(self, tensors: Dict[str, torch.Tensor]) -> dict:
        """Register tensors and return metadata dict (synchronous wrapper)."""
        self._sweep()  # release completed / timed-out buffers first
        return asyncio.get_event_loop().run_until_complete(self._async_send(tensors))

    def _sweep(self):
        """Poll pending readables: release completed or timed-out buffers."""
        now = time.monotonic()
        still_pending = []
        for readable, flat, created_at in self._pending:
            try:
                status = readable.status  # synchronous — calls update_notifs()
            except Exception:
                # If status check fails, treat as completed to avoid leak
                logger.debug("NIXL readable status check failed, releasing buffer")
                continue
            if hasattr(status, "name") and status.name == "COMPLETE":
                logger.debug("NIXL readable completed, releasing buffer")
            elif str(status) == "OperationStatus.COMPLETE":
                logger.debug("NIXL readable completed, releasing buffer")
            elif now - created_at > self.BUFFER_TIMEOUT_S:
                logger.warning(
                    "NIXL readable timed out after %.0fs, force-releasing buffer",
                    now - created_at,
                )
            else:
                still_pending.append((readable, flat, created_at))
        self._pending = still_pending

    async def _async_send(self, tensors: Dict[str, torch.Tensor]) -> dict:
        connector = await _PersistentConnector.get()

        # Flatten all tensors into a single contiguous buffer
        flat = torch.cat([t.contiguous().view(-1) for t in tensors.values()])
        descriptor = nixl_connect.Descriptor(flat)
        readable = await connector.create_readable(descriptor)
        raw_meta = readable.metadata()

        meta = {
            "tensor_keys": list(tensors.keys()),
            "shapes": {k: list(t.shape) for k, t in tensors.items()},
            "dtypes": {k: str(t.dtype).removeprefix("torch.") for k, t in tensors.items()},
            "nixl_metadata": raw_meta.model_dump() if hasattr(raw_meta, "model_dump") else raw_meta,
        }

        # Hold (readable, flat_buffer, timestamp) — prevents GC until sweep releases
        self._pending.append((readable, flat, time.monotonic()))
        return meta


class NixlTensorReceiver:
    """Pull tensors from a remote sender via NIXL RDMA."""

    def recv(self, meta: dict, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Pull tensors described by metadata. Returns {name: tensor}."""
        return asyncio.get_event_loop().run_until_complete(self._async_recv(meta, device))

    async def _async_recv(self, meta: dict, device: str) -> Dict[str, torch.Tensor]:
        connector = await _PersistentConnector.get()

        # Calculate total size and per-tensor specs
        specs = []
        total_bytes = 0
        for key in meta["tensor_keys"]:
            shape = meta["shapes"][key]
            dtype = getattr(torch, meta["dtypes"][key])
            numel = 1
            for s in shape:
                numel *= s
            size = numel * dtype.itemsize
            specs.append((key, shape, dtype, size))
            total_bytes += size

        # Allocate receive buffer directly on target device (GPU-direct)
        flat = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        descriptor = nixl_connect.Descriptor(flat)

        rdma_meta = nixl_connect.RdmaMetadata.model_validate(meta["nixl_metadata"])
        read_op = await connector.begin_read(rdma_meta, descriptor)
        await read_op.wait_for_completion()

        # Slice the flat buffer into individual tensors
        result = {}
        offset = 0
        for key, shape, dtype, size in specs:
            result[key] = flat[offset:offset + size].view(dtype=dtype).reshape(shape)
            offset += size

        return result
