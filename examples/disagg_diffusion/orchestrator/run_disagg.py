#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated Diffusion Orchestrator — HTTP Server with Pipeline Parallelism

Persistent server that accepts video generation requests and chains
Encoder → Denoiser → VAE via Dynamo RPC + NIXL RDMA.

Pipeline parallelism: multiple requests can be in different stages
simultaneously. Per-stage WorkerManagers track busy/idle state for
each GPU worker and dispatch requests to specific instances via
``client.direct()``, enabling backpressure-aware scheduling and
per-worker observability through ``/pipeline/status``.

Each stage worker wraps an SGLang Scheduler subprocess via
launch_partial_server(), supporting TP for the denoiser and NIXL RDMA
for GPU-direct tensor transfer between stages.

Usage:
    python run_disagg.py [--port 8080]

API:
    POST /v1/videos/generations
    GET  /health
    GET  /health/stages
    GET  /pipeline/status
    GET  /videos/<filename>
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from typing import Dict

import uvloop

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workers"))

from protocol import (  # noqa: E402
    DenoiserRequest, EncoderRequest, VAEDecodeRequest,
    HealthRequest,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker  # noqa: E402
from worker_manager import WorkerManager  # noqa: E402

logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST", "0.0.0.0")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/disagg_videos")
MAX_PIPELINE_DEPTH = int(os.environ.get("MAX_PIPELINE_DEPTH", "4"))
STAGE_DISPATCH_RETRIES = int(os.environ.get("STAGE_DISPATCH_RETRIES", "2"))


async def query_stage_health(health_client, stage_name: str) -> dict:
    """Query a stage's StageEngine health via its Dynamo health endpoint."""
    try:
        health_req = HealthRequest()
        stream = await health_client.generate(health_req.model_dump_json())
        result = None
        async for chunk in stream:
            data = chunk.data() if hasattr(chunk, "data") else chunk
            if isinstance(data, str):
                data = json.loads(data)
            result = data
        return {"stage": stage_name, **result} if result else {"stage": stage_name, "error": "empty response"}
    except Exception as e:
        return {"stage": stage_name, "error": str(e)}


class PipelineTracker:
    """Tracks requests flowing through the 3-stage pipeline."""

    STAGES = ("encoder", "denoiser", "vae")

    def __init__(self):
        self._active: Dict[str, str] = {}
        self._completed = 0
        self._failed = 0
        self._stage_times: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def enter(self, request_id: str, stage: str):
        async with self._lock:
            self._active[request_id] = stage

    async def leave(self, request_id: str, stage: str, elapsed: float):
        async with self._lock:
            self._stage_times[stage].append(elapsed)
            if request_id in self._active and self._active[request_id] == stage:
                if stage == "vae":
                    del self._active[request_id]

    async def mark_done(self, request_id: str):
        async with self._lock:
            self._active.pop(request_id, None)
            self._completed += 1

    async def mark_failed(self, request_id: str):
        async with self._lock:
            self._active.pop(request_id, None)
            self._failed += 1

    async def status(self) -> dict:
        async with self._lock:
            per_stage = defaultdict(list)
            for rid, stage in self._active.items():
                per_stage[stage].append(rid)
            avg_times = {}
            for stage in self.STAGES:
                times = self._stage_times[stage]
                avg_times[stage] = round(sum(times) / len(times), 3) if times else 0
            return {
                "active_requests": dict(per_stage),
                "active_count": len(self._active),
                "completed": self._completed,
                "failed": self._failed,
                "avg_stage_seconds": avg_times,
            }


@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    encoder_client = await runtime.endpoint("disagg_diffusion.encoder.generate").client()
    denoiser_client = await runtime.endpoint("disagg_diffusion.denoiser.generate").client()
    vae_client = await runtime.endpoint("disagg_diffusion.vae.generate").client()

    encoder_health_client = await runtime.endpoint("disagg_diffusion.encoder.health").client()
    denoiser_health_client = await runtime.endpoint("disagg_diffusion.denoiser.health").client()
    vae_health_client = await runtime.endpoint("disagg_diffusion.vae.health").client()

    health_clients = {
        "encoder": encoder_health_client,
        "denoiser": denoiser_health_client,
        "vae": vae_health_client,
    }

    logger.info("Waiting for stage workers …")
    await encoder_client.wait_for_instances()
    await denoiser_client.wait_for_instances()
    await vae_client.wait_for_instances()

    # Wait for additional workers that may still be registering.
    # wait_for_instances() returns after the first instance; model loading
    # times vary, so poll until the count stabilizes or timeout.
    WORKER_SETTLE_S = int(os.environ.get("WORKER_SETTLE_S", "30"))
    if WORKER_SETTLE_S > 0:
        import time as _time
        deadline = _time.monotonic() + WORKER_SETTLE_S
        prev_count = 0
        while _time.monotonic() < deadline:
            cur = len(denoiser_client.instance_ids())
            if cur > prev_count:
                prev_count = cur
                logger.info("Discovered %d denoiser(s) so far, waiting for more…", cur)
            await asyncio.sleep(2)
        logger.info("Worker settle period done (%ds)", WORKER_SETTLE_S)

    # Discover registered worker instances per stage
    enc_ids = encoder_client.instance_ids()
    den_ids = denoiser_client.instance_ids()
    vae_ids = vae_client.instance_ids()
    logger.info(
        "Workers: encoder=%d %s, denoiser=%d %s, vae=%d %s",
        len(enc_ids), enc_ids, len(den_ids), den_ids, len(vae_ids), vae_ids,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    managers: Dict[str, WorkerManager] = {
        "encoder": WorkerManager("encoder", encoder_client, enc_ids),
        "denoiser": WorkerManager("denoiser", denoiser_client, den_ids),
        "vae": WorkerManager("vae", vae_client, vae_ids),
    }
    n_workers_total = sum(m.worker_count for m in managers.values())
    pipeline_depth = MAX_PIPELINE_DEPTH if MAX_PIPELINE_DEPTH > 0 else n_workers_total
    admission = asyncio.Semaphore(pipeline_depth)
    tracker = PipelineTracker()

    async def dispatch_with_retry(
        mgr: WorkerManager, request_id: str, request_json: str,
    ) -> tuple:
        """Dispatch to a stage worker; on failure retry on a different worker."""
        for attempt in range(STAGE_DISPATCH_RETRIES + 1):
            wid = await mgr.acquire_worker()
            try:
                result, elapsed = await mgr.dispatch(wid, request_id, request_json)
                if "error" in result and result["error"]:
                    raise RuntimeError(result["error"])
                return result, elapsed, wid
            except Exception as e:
                logger.warning(
                    "[%s] %s worker %d failed (attempt %d/%d): %s",
                    request_id, mgr.stage_name, wid, attempt + 1,
                    STAGE_DISPATCH_RETRIES + 1, e,
                )
                if attempt >= STAGE_DISPATCH_RETRIES:
                    raise
        raise RuntimeError("unreachable")

    async def handle_generate(request: dict) -> dict:
        request_id = str(uuid.uuid4())[:8]
        seed = request.get("seed") or int(time.time()) % 1000000
        timings: Dict[str, float] = {}

        async with admission:
            try:
                # Stage 1: Encoder
                enc_req = EncoderRequest(
                    prompt=request["prompt"],
                    negative_prompt=request.get("negative_prompt", ""),
                    guidance_scale=request.get("guidance_scale", 1.0),
                )
                await tracker.enter(request_id, "encoder")
                enc_resp, timings["encoder_s"], enc_wid = await dispatch_with_retry(
                    managers["encoder"], request_id, enc_req.model_dump_json(),
                )
                await tracker.leave(request_id, "encoder", timings["encoder_s"])
                logger.info("[%s] Encoder (worker %d): %.2fs", request_id, enc_wid, timings["encoder_s"])

                # Stage 2: Denoiser
                den_req = DenoiserRequest(
                    transfer_meta=enc_resp.get("transfer_meta", {}),
                    tensor_data=enc_resp.get("tensor_data", {}),
                    height=request.get("height", 544),
                    width=request.get("width", 960),
                    num_frames=request.get("num_frames", 61),
                    num_inference_steps=request.get("num_inference_steps", 50),
                    guidance_scale=request.get("guidance_scale", 1.0),
                    seed=seed,
                )
                await tracker.enter(request_id, "denoiser")
                den_resp, timings["denoiser_s"], den_wid = await dispatch_with_retry(
                    managers["denoiser"], request_id, den_req.model_dump_json(),
                )
                await tracker.leave(request_id, "denoiser", timings["denoiser_s"])
                logger.info("[%s] Denoiser (worker %d): %.2fs", request_id, den_wid, timings["denoiser_s"])

                # Stage 3: VAE
                vae_req = VAEDecodeRequest(
                    transfer_meta=den_resp.get("transfer_meta", {}),
                    tensor_data=den_resp.get("tensor_data", {}),
                    request_id=request_id,
                )
                await tracker.enter(request_id, "vae")
                vae_resp, timings["vae_s"], vae_wid = await dispatch_with_retry(
                    managers["vae"], request_id, vae_req.model_dump_json(),
                )
                await tracker.leave(request_id, "vae", timings["vae_s"])
                logger.info("[%s] VAE (worker %d): %.2fs", request_id, vae_wid, timings["vae_s"])
            except Exception:
                await tracker.mark_failed(request_id)
                raise

        timings["total_s"] = round(sum(timings.values()), 3)
        await tracker.mark_done(request_id)
        logger.info("[%s] Total: %.2fs", request_id, timings["total_s"])

        filename = vae_resp["video_path"]
        resp_format = request.get("response_format", "url")

        if resp_format == "url":
            data = [{"url": f"/videos/{filename}"}]
        else:
            import base64
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "rb") as f:
                data = [{"b64_json": base64.b64encode(f.read()).decode("ascii")}]

        return {
            "id": f"video-{request_id}",
            "created": int(time.time()),
            "data": data,
            "timings": timings,
        }

    from aiohttp import web

    async def handle_post(http_request: web.Request) -> web.Response:
        try:
            body = await http_request.json()
            if "prompt" not in body:
                return web.json_response({"error": "missing 'prompt' field"}, status=400)
            result = await handle_generate(body)
            return web.json_response(result)
        except Exception as e:
            logger.error("Request failed: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(http_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def handle_stages_health(http_request: web.Request) -> web.Response:
        results = await asyncio.gather(
            *[
                query_stage_health(client, name)
                for name, client in health_clients.items()
            ]
        )
        return web.json_response({"stages": list(results)})

    async def handle_pipeline_status(http_request: web.Request) -> web.Response:
        pipeline = await tracker.status()
        stages = {name: mgr.status() for name, mgr in managers.items()}
        pipeline["stages"] = stages
        pipeline["pipeline_depth"] = {
            "active": pipeline["active_count"],
            "max": pipeline_depth,
        }
        return web.json_response(pipeline)

    async def handle_video(http_request: web.Request) -> web.Response:
        filename = http_request.match_info["filename"]
        if "/" in filename or "\\" in filename or ".." in filename:
            return web.json_response({"error": "invalid filename"}, status=400)
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(filepath, headers={"Content-Type": "video/mp4"})

    async def handle_index(http_request: web.Request) -> web.Response:
        files = sorted(
            [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp4")],
            key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)),
            reverse=True,
        ) if os.path.isdir(OUTPUT_DIR) else []
        latest = files[0] if files else None
        video_tag = (
            f'<video src="/videos/{latest}" controls autoplay loop '
            f'style="max-width:100%;max-height:80vh"></video>'
            if latest else "<p>No videos generated yet.</p>"
        )
        history = "".join(
            f'<li><a href="/videos/{f}">{f}</a></li>' for f in files[:20]
        )
        html = (
            "<!DOCTYPE html><html><head><title>Disagg Diffusion</title>"
            "<style>body{font-family:sans-serif;margin:2em;background:#111;color:#eee}"
            "a{color:#4af}video{border-radius:8px}</style></head><body>"
            f"<h2>Latest Video</h2>{video_tag}"
            f"<h3>History</h3><ul>{history}</ul></body></html>"
        )
        return web.Response(text=html, content_type="text/html")

    http_app = web.Application()
    http_app.router.add_get("/", handle_index)
    http_app.router.add_post("/v1/videos/generations", handle_post)
    http_app.router.add_get("/health", handle_health)
    http_app.router.add_get("/health/stages", handle_stages_health)
    http_app.router.add_get("/pipeline/status", handle_pipeline_status)
    http_app.router.add_get("/videos/{filename}", handle_video)

    runner = web.AppRunner(http_app)
    await runner.setup()

    bound_port = PORT
    for attempt in range(10):
        try:
            site = web.TCPSite(runner, HOST, bound_port, reuse_address=True)
            await site.start()
            break
        except OSError as e:
            if e.errno == 98 and attempt < 9:
                logger.warning("Port %d in use, trying %d", bound_port, bound_port + 1)
                bound_port += 1
            else:
                raise

    logger.info("Server listening on http://%s:%d (pipeline depth=%d)", HOST, bound_port, pipeline_depth)
    logger.info("  GET  /                        <- latest video preview")
    logger.info("  POST /v1/videos/generations")
    logger.info("  GET  /health")
    logger.info("  GET  /health/stages")
    logger.info("  GET  /pipeline/status")
    logger.info("  GET  /videos/<id>.mp4")

    await asyncio.Event().wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    uvloop.install()
    asyncio.run(worker())
