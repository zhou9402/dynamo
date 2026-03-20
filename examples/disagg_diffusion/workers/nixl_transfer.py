# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL RDMA tensor transfer for disaggregated diffusion stages.

Provides GPU-direct tensor transfer between stage workers. Only small
metadata (shapes, dtypes, NIXL descriptor ~1.5 KB) travels over the ZMQ
control plane; actual tensor data (embeddings, latents) transfers
GPU->GPU via NIXL RDMA.

Follows the PersistentConnector pattern from embedding_transfer.py:
- Each Sender/Receiver owns its PersistentConnector (one Connection/agent).
- Remote._release is nooped to keep agent pairs alive.
- The sender returns a ``readable_op`` handle that the caller must hold
  until the receiver completes the RDMA pull.

Usage inside PipelineStage.forward() (synchronous context)::

    sender = NixlTensorSender()       # creates agent eagerly
    meta, readable_op = sender.send({"latents": tensor})
    # ... pass meta via ZMQ, hold readable_op until COMPLETE ...

    receiver = NixlTensorReceiver()   # creates agent eagerly
    tensors = receiver.recv(meta, device="cuda")  # RDMA pull
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    import dynamo.nixl_connect as nixl_connect

    NIXL_AVAILABLE = not os.environ.get("DISABLE_NIXL", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if not NIXL_AVAILABLE:
        logger.info("NIXL disabled via DISABLE_NIXL env var")
except ImportError:
    NIXL_AVAILABLE = False
    logger.info("NIXL not available — falling back to ZMQ tensor transfer")


# ---------------------------------------------------------------------------
# PersistentConnector + Remote._release noop
# ---------------------------------------------------------------------------
# Exact pattern from components/src/dynamo/common/multimodal/embedding_transfer.py

if NIXL_AVAILABLE:

    class PersistentConnector(nixl_connect.Connector):
        """Connector that reuses a single Connection for all operations."""

        def __init__(self):
            super().__init__()
            self._connection = None

        async def _create_connection(self) -> nixl_connect.Connection:
            if self._connection is None:
                self._connection = nixl_connect.Connection(self, 1)
                await self._connection.initialize()
            return self._connection

    # NOTE: We do NOT noop Remote._release here. Our recv() is synchronous
    # (awaits wait_for_completion before returning), so the transfer is
    # always complete before Remote is GC'd. Keeping the remote agent
    # registered across requests causes NIXL_ERR_NOT_ALLOWED on the
    # second add_remote_agent() call.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_event_loop = None

def _run_coro(coro):
    """Run a coroutine from synchronous context (sglang scheduler thread)."""
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
    return _event_loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# NixlTensorSender
# ---------------------------------------------------------------------------


class NixlTensorSender:
    """Register GPU tensors as NIXL-readable. Returns (metadata, readable_op).

    Each instance owns a PersistentConnector whose Connection (nixl_agent)
    is created eagerly in ``__init__`` so UCX is fully initialized before
    any transfer occurs.

    The caller **must** hold ``readable_op`` until the receiver completes
    the RDMA pull.
    """

    def __init__(self):
        self.connector = PersistentConnector()
        # Eagerly create the Connection / nixl_agent so UCX is ready
        _run_coro(self.connector._create_connection())

    def send(self, tensors: Dict[str, torch.Tensor]) -> Tuple[dict, object]:
        """Register tensors and return (metadata_dict, readable_op)."""
        return _run_coro(self._async_send(tensors))

    async def _async_send(
        self, tensors: Dict[str, torch.Tensor]
    ) -> Tuple[dict, object]:
        import time as _time
        t0 = _time.monotonic()

        # Flatten all tensors into a single contiguous buffer
        vals = list(tensors.values())
        if len(vals) == 1:
            flat = vals[0].contiguous().view(-1)
        else:
            flat = torch.cat([t.contiguous().view(-1) for t in vals])
        t1 = _time.monotonic()

        descriptor = nixl_connect.Descriptor(flat)
        t2 = _time.monotonic()

        readable = await self.connector.create_readable(descriptor)
        t3 = _time.monotonic()

        raw_meta = readable.metadata()
        t4 = _time.monotonic()

        logger.info(
            "NIXL send breakdown: flatten=%.1fms descriptor=%.1fms "
            "create_readable=%.1fms metadata=%.1fms total=%.1fms "
            "(buf=%s %.2fMB)",
            (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000,
            (t4-t0)*1000, list(flat.shape), flat.nbytes/1e6,
        )

        meta = {
            "tensor_keys": list(tensors.keys()),
            "shapes": {k: list(t.shape) for k, t in tensors.items()},
            "dtypes": {
                k: str(t.dtype).removeprefix("torch.") for k, t in tensors.items()
            },
            "nixl_metadata": raw_meta.model_dump()
            if hasattr(raw_meta, "model_dump")
            else raw_meta,
        }

        # Return both — caller holds readable to prevent GC
        return meta, readable


# ---------------------------------------------------------------------------
# NixlTensorReceiver
# ---------------------------------------------------------------------------


class NixlTensorReceiver:
    """Pull tensors from a remote sender via NIXL RDMA.

    Each instance owns a PersistentConnector whose Connection (nixl_agent)
    is created eagerly in ``__init__``.
    """

    def __init__(self):
        self.connector = PersistentConnector()
        # Eagerly create the Connection / nixl_agent so UCX is ready
        _run_coro(self.connector._create_connection())

    def recv(self, meta: dict, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Pull tensors described by metadata. Returns {name: tensor}."""
        return _run_coro(self._async_recv(meta, device))

    async def _async_recv(self, meta: dict, device: str) -> Dict[str, torch.Tensor]:
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
        read_op = await self.connector.begin_read(rdma_meta, descriptor)
        await read_op.wait_for_completion()

        # Slice the flat buffer into individual tensors
        result = {}
        offset = 0
        for key, shape, dtype, size in specs:
            result[key] = flat[offset : offset + size].view(dtype=dtype).reshape(shape)
            offset += size

        return result
