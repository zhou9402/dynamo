# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage diffusion engine with request queue, health reporting, and dual
sync/async interfaces.

Each disaggregated diffusion stage (Encoder, Denoiser, VAE) runs its own
``StageEngine`` instance wrapping SGLang ``PipelineStage`` objects.

Interface naming is aligned with SGLang:
  - ``async_generate(req)`` — enqueue and await
  - ``generate(req)``       — synchronous blocking call
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List

import torch

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 5


@dataclass
class EngineStats:
    """Mutable statistics tracked by the engine."""

    active_requests: int = 0
    total_processed: int = 0
    total_latency_s: float = 0.0
    last_latency_s: float = 0.0
    warmup_latency_s: float = 0.0
    consecutive_errors: int = 0


class StageEngine:
    """Lightweight per-stage engine wrapping SGLang ``PipelineStage`` instances.

    Provides: request queue with back-pressure, background GPU processing loop,
    health metrics, warmup, and lifecycle management (start / shutdown).
    """

    def __init__(
        self,
        stages: List[Any],
        server_args: Any,
        max_queue_depth: int = 16,
        use_profiler: bool = True,
    ):
        self.stages = stages
        self.server_args = server_args
        self._max_queue_depth = max_queue_depth
        self._use_profiler = use_profiler
        self._queue: asyncio.Queue | None = None
        self._running = False
        self._warmed_up = False
        self._loop_task: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage-gpu")
        self._stats = EngineStats()

    async def start(self) -> None:
        """Start the background processing loop (must be called within an event loop)."""
        if self._running:
            return
        self._queue = asyncio.Queue(maxsize=self._max_queue_depth)
        self._running = True
        self._loop_task = asyncio.create_task(self._process_loop())
        logger.info(
            "StageEngine started (stages=%d, max_queue=%d, profiler=%s)",
            len(self.stages),
            self._max_queue_depth,
            self._use_profiler,
        )

    async def warmup(self, warmup_req: Any) -> None:
        """Run a warmup request to trigger JIT compilation / CUDA graphs.

        Should be called after ``start()`` with a representative Req
        (correct shape/dtype but dummy content).  The result is discarded.
        """
        if self._warmed_up:
            return
        logger.info("StageEngine warmup starting …")
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._forward, warmup_req)
            torch.cuda.synchronize()
        except Exception as e:
            logger.warning("Warmup failed (non-fatal): %s", e)
        elapsed = time.monotonic() - t0
        self._stats.warmup_latency_s = elapsed
        self._warmed_up = True
        logger.info("StageEngine warmup done in %.2fs", elapsed)

    async def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        self._executor.shutdown(wait=False)
        logger.info(
            "StageEngine shut down (total_processed=%d)",
            self._stats.total_processed,
        )

    async def async_generate(self, req: Any) -> Any:
        """Enqueue *req* and await the result."""
        if self._queue is None or not self._running:
            raise RuntimeError("StageEngine is not running; call start() first")

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        try:
            self._queue.put_nowait((req, future))
        except asyncio.QueueFull:
            raise asyncio.QueueFull(
                f"StageEngine queue is full ({self._max_queue_depth}); "
                "apply back-pressure or increase max_queue_depth"
            )
        return await future

    def generate(self, req: Any) -> Any:
        """Synchronous generate — blocks until the result is ready."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return self._forward(req)

        coro = self.async_generate(req)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def health(self) -> dict:
        """Return a snapshot of engine health metrics."""
        queue_depth = self._queue.qsize() if self._queue else 0
        return {
            "running": self._running,
            "warmed_up": self._warmed_up,
            "queue_depth": queue_depth,
            "max_queue_depth": self._max_queue_depth,
            "active_requests": self._stats.active_requests,
            "total_processed": self._stats.total_processed,
            "avg_latency_s": round(
                self._stats.total_latency_s / max(self._stats.total_processed, 1), 4
            ),
            "last_latency_s": round(self._stats.last_latency_s, 4),
            "warmup_latency_s": round(self._stats.warmup_latency_s, 4),
            "consecutive_errors": self._stats.consecutive_errors,
            "gpu_memory_allocated_mb": round(
                torch.cuda.memory_allocated() / 1e6, 1
            ),
            "gpu_memory_reserved_mb": round(
                torch.cuda.memory_reserved() / 1e6, 1
            ),
        }

    def _forward(self, req: Any) -> Any:
        """Run all stages sequentially on the GPU (blocking).

        Stages that return None mutate *req* in-place; stages that return
        a new object (e.g. DecodingStage → OutputBatch) replace *req*.

        When ``use_profiler=True``, calls ``stage(req, server_args)``
        (i.e. ``PipelineStage.__call__``) which wraps ``forward()`` with
        input/output verification and SGLang's ``StageProfiler``.
        """
        result = req
        for stage in self.stages:
            if self._use_profiler:
                ret = stage(result, self.server_args)
            else:
                ret = stage.forward(result, self.server_args)
            if ret is not None:
                result = ret
        return result

    async def _process_loop(self) -> None:
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                req, future = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            self._stats.active_requests += 1
            t0 = time.monotonic()

            try:
                result = await loop.run_in_executor(self._executor, self._forward, req)
                if not future.done():
                    future.set_result(result)
                self._stats.consecutive_errors = 0
            except Exception as exc:
                self._stats.consecutive_errors += 1
                logger.error(
                    "StageEngine forward failed (%d/%d): %s",
                    self._stats.consecutive_errors,
                    MAX_CONSECUTIVE_ERRORS,
                    exc,
                    exc_info=True,
                )
                if not future.done():
                    future.set_exception(exc)
                if self._stats.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.critical(
                        "StageEngine reached %d consecutive errors, stopping",
                        MAX_CONSECUTIVE_ERRORS,
                    )
                    self._running = False
                    break
            finally:
                elapsed = time.monotonic() - t0
                self._stats.active_requests -= 1
                self._stats.total_processed += 1
                self._stats.total_latency_s += elapsed
                self._stats.last_latency_s = elapsed
