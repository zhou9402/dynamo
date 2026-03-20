#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage worker pool manager with busy/idle tracking and targeted dispatch.

Each pipeline stage (encoder, denoiser, vae) gets its own WorkerManager
that tracks which Dynamo worker instances are idle or busy and dispatches
requests to specific workers via ``client.direct()``.  This replaces the
previous ``asyncio.Semaphore``-based flow control with explicit state that
is queryable via ``/pipeline/status``.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    """Runtime state for a single Dynamo worker instance."""

    worker_id: int
    status: Literal["idle", "busy"] = "idle"
    current_request_id: Optional[str] = None
    completed_count: int = 0
    total_latency_s: float = 0.0


class WorkerManager:
    """Manages the worker pool for one pipeline stage.

    Provides acquire/dispatch semantics: callers first ``acquire_worker()``
    (blocks until one is idle), then ``dispatch()`` to send the request to
    that specific worker.  On completion (or failure) the worker is
    automatically returned to the idle pool.
    """

    def __init__(self, stage_name: str, client: Any, worker_ids: List[int]):
        self._stage_name = stage_name
        self._client = client
        self._workers: Dict[int, WorkerState] = {
            wid: WorkerState(worker_id=wid) for wid in worker_ids
        }
        self._idle_queue: asyncio.Queue[int] = asyncio.Queue()
        for wid in worker_ids:
            self._idle_queue.put_nowait(wid)

        # Observability counters
        self._total_completed = 0
        self._total_failed = 0
        self._waiting_count = 0  # requests blocked waiting for an idle worker

    @property
    def stage_name(self) -> str:
        return self._stage_name

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    async def acquire_worker(self) -> int:
        """Block until a worker becomes idle. Returns worker_id."""
        self._waiting_count += 1
        try:
            wid = await self._idle_queue.get()
        finally:
            self._waiting_count -= 1
        return wid

    async def dispatch(
        self, worker_id: int, request_id: str, request_json: str
    ) -> Tuple[dict, float]:
        """Send request to a specific worker via ``client.direct()`` and track state.

        Returns (result_dict, elapsed_seconds).
        """
        ws = self._workers[worker_id]
        ws.status = "busy"
        ws.current_request_id = request_id
        t0 = time.monotonic()
        try:
            result = await self._call_direct(worker_id, request_json)
            elapsed = time.monotonic() - t0
            ws.completed_count += 1
            ws.total_latency_s += elapsed
            self._total_completed += 1
            return result, elapsed
        except Exception:
            self._total_failed += 1
            raise
        finally:
            ws.status = "idle"
            ws.current_request_id = None
            self._idle_queue.put_nowait(worker_id)

    async def _call_direct(self, worker_id: int, request_json: str) -> dict:
        """Issue a Dynamo RPC to a specific worker instance."""
        result = None
        stream = await self._client.direct(request_json, worker_id)
        async for chunk in stream:
            data = chunk.data() if hasattr(chunk, "data") else chunk
            if isinstance(data, str):
                data = json.loads(data)
            result = data
        if result is None:
            raise RuntimeError(
                f"Empty response from {self._stage_name} worker {worker_id}"
            )
        return result

    def status(self) -> dict:
        """Snapshot of per-worker state for ``/pipeline/status``."""
        workers = []
        for ws in self._workers.values():
            avg = (
                round(ws.total_latency_s / ws.completed_count, 3)
                if ws.completed_count
                else 0
            )
            workers.append(
                {
                    "id": ws.worker_id,
                    "status": ws.status,
                    "request_id": ws.current_request_id,
                    "completed": ws.completed_count,
                    "avg_latency_s": avg,
                }
            )
        return {
            "stage": self._stage_name,
            "workers": workers,
            "queue_depth": self._waiting_count,
            "completed": self._total_completed,
            "failed": self._total_failed,
        }
