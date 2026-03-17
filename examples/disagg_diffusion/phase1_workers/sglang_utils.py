# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang PipelineStage utilities for disaggregated diffusion workers.

Provides helpers to load partial pipelines (only the modules each worker
needs), launch stage servers, and convert between Dynamo protocol types
and SGLang's Req dataclass.

Also contains shared utilities (StageClient, model detection, compatibility
patches) used by both Dynamo workers and the standalone E2E script.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def build_partial_pipeline(
    server_args,
    required_modules: List[str],
):
    """Load a pipeline with only *required_modules* populated.

    Auto-detects pipeline class from model_index.json, suppresses automatic
    stage creation, and syncs all component configs (even unloaded ones).
    """
    from sglang.multimodal_gen.runtime.pipelines import get_model_info

    model_info = get_model_info(server_args.model_path)
    base_pipeline_cls = model_info.pipeline_cls

    # Build a partial pipeline class that:
    #  1. Suppresses automatic stage creation
    #  2. Skips LoRA init (which assumes 'transformer' always exists)
    def _noop_create_stages(self, server_args):
        return None

    def _safe_init(self, **kwargs):
        # Call ComposedPipelineBase.__init__ directly, skipping LoRAPipeline
        # which tries to access self.modules['transformer']
        from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
            ComposedPipelineBase,
        )
        ComposedPipelineBase.__init__(self, **kwargs)

    partial_cls = type(
        f"_Partial{base_pipeline_cls.__name__}",
        (base_pipeline_cls,),
        {
            "create_pipeline_stages": _noop_create_stages,
            "__init__": _safe_init,
        },
    )

    pipeline = partial_cls(
        model_path=server_args.model_path,
        server_args=server_args,
        required_config_modules=required_modules,
    )

    _sync_all_component_configs(server_args, pipeline)

    return pipeline


def _sync_all_component_configs(server_args, pipeline):
    """Read config.json for every component in model_index.json and update
    the corresponding arch_config in ``server_args.pipeline_config``, ensuring
    correct parameters even for components whose weights are not loaded.
    """
    import json

    CONFIG_ATTR_MAP = {
        "vae": ("vae_config", "update_model_arch"),
        "video_vae": ("vae_config", "update_model_arch"),
        "transformer": ("dit_config", "update_model_arch"),
        "video_dit": ("dit_config", "update_model_arch"),
        "audio_dit": ("audio_dit_config", "update_model_arch"),
        "audio_vae": ("audio_vae_config", "update_model_arch"),
    }

    # pipeline.model_path is resolved to a local path by _load_config()
    # (may differ from server_args.model_path if the original was a hub ID).
    model_path = pipeline.model_path
    model_index_path = os.path.join(model_path, "model_index.json")
    if not os.path.isfile(model_index_path):
        return

    with open(model_index_path, "r") as f:
        model_index = json.load(f)

    pipeline_config = server_args.pipeline_config
    for component_name, mapping in CONFIG_ATTR_MAP.items():
        config_attr, update_method_name = mapping
        cfg = getattr(pipeline_config, config_attr, None)
        if cfg is None:
            continue

        if component_name not in model_index:
            continue

        config_json_path = os.path.join(model_path, component_name, "config.json")
        if not os.path.isfile(config_json_path):
            continue

        with open(config_json_path, "r") as f:
            hf_config = json.load(f)

        hf_config.pop("_class_name", None)
        hf_config.pop("_diffusers_version", None)

        update_fn = getattr(cfg, update_method_name, None)
        if update_fn is not None:
            update_fn(hf_config)
            logger.info("Synced %s config from %s (e.g. z_dim=%s)",
                        config_attr,
                        config_json_path,
                        getattr(getattr(cfg, "arch_config", cfg), "z_dim", "N/A"))

        if hasattr(cfg, "post_init"):
            cfg.post_init()


def get_component_backend(module) -> str:
    """Return a human-readable string indicating which backend loaded *module*."""
    mod = type(module).__module__ or ""
    cls = type(module).__qualname__
    if mod.startswith("sglang."):
        return f"sglang-optimized ({cls})"
    if mod.startswith("diffusers."):
        return f"native-diffusers ({cls})"
    if mod.startswith("transformers."):
        return f"native-transformers ({cls})"
    return f"unknown ({mod}.{cls})"


def build_req(
    prompt: str,
    negative_prompt: Optional[str] = "",
    height: int = 544,
    width: int = 960,
    num_frames: int = 61,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    seed: int = 42,
    device: str = "cuda",
    **extra_fields,
) -> "Req":
    """Construct a minimal SGLang ``Req`` for running pipeline stages."""
    from sglang.multimodal_gen.runtime.pipelines.schedule_batch import Req
    from sglang.multimodal_gen.configs.sample.base import DataType

    req = Req(
        data_type=DataType.VIDEO,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        do_classifier_free_guidance=(guidance_scale > 1.0),
    )

    for k, v in extra_fields.items():
        setattr(req, k, v)

    return req


def inject_tensors_to_req(
    req,
    tensors: Dict[str, object],
    list_fields: Optional[List[str]] = None,
):
    """Inject received tensors back into a ``Req``.

    Values may be bare tensors (single-encoder outputs) or lists of tensors
    (dual-encoder outputs like HunyuanVideo's Llama + CLIP embeddings).
    ``list_fields`` specifies Req attributes that expect a list value;
    bare tensors are auto-wrapped in a single-element list.
    """
    list_fields = set(list_fields or [
        "prompt_embeds",
        "negative_prompt_embeds",
        "pooled_embeds",
        "neg_pooled_embeds",
        "image_embeds",
    ])
    for key, value in tensors.items():
        if key in list_fields:
            if isinstance(value, list):
                # Multi-encoder: already a list of tensors
                setattr(req, key, value)
            else:
                # Single encoder: wrap in list
                setattr(req, key, [value])
        else:
            setattr(req, key, value)
    return req


# ═══════════════════════════════════════════════════════════════════════
# Shared utilities — used by Dynamo workers and the standalone E2E script
# ═══════════════════════════════════════════════════════════════════════


class StageClient:
    """Async ZMQ REQ client that talks to a SGLang Scheduler subprocess."""

    def __init__(self, endpoint: str, name: str = ""):
        import zmq.asyncio
        self._name = name
        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(endpoint)
        self._lock = asyncio.Lock()
        logger.info("StageClient(%s) connected to %s", name, endpoint)

    async def forward(self, reqs):
        """Send request(s) and receive response."""
        async with self._lock:
            await self._sock.send_pyobj(reqs)
            return await self._sock.recv_pyobj()

    def close(self):
        self._sock.close()
        self._ctx.term()


class StageWorkerPool:
    """Pool of StageClients for one stage with round-robin dispatch."""

    def __init__(self, clients: List[StageClient], name: str = ""):
        self._clients = clients
        self._name = name
        self._counter = 0

    @property
    def num_workers(self) -> int:
        return len(self._clients)

    async def forward(self, reqs):
        """Round-robin dispatch to next available worker."""
        idx = self._counter % len(self._clients)
        self._counter += 1
        return await self._clients[idx].forward(reqs)

    def close(self):
        for c in self._clients:
            c.close()


def patch_hunyuan_config():
    """HunyuanConfig inherits ``task_type`` from PipelineConfig without a
    default value, so ``HunyuanConfig()`` crashes.  Wrap __init__ to supply
    ``task_type=T2V`` when omitted.  Idempotent.
    """
    from sglang.multimodal_gen.configs.pipelines.base import ModelTaskType
    try:
        from sglang.multimodal_gen.configs.pipelines.hunyuan import (
            HunyuanConfig, FastHunyuanConfig,
        )
    except ImportError:
        return

    for cls in (HunyuanConfig, FastHunyuanConfig):
        if getattr(cls, "_task_type_patched", False):
            continue
        orig = cls.__init__

        def _patched(self, *a, task_type=ModelTaskType.T2V, _orig=orig, **kw):
            _orig(self, *a, task_type=task_type, **kw)

        cls.__init__ = _patched
        cls._task_type_patched = True


def detect_encoder_modules(model_path: str) -> List[str]:
    """Return the required_modules list for the encoder stage.

    Auto-detects dual-encoder models (e.g. HunyuanVideo with Llama + CLIP).
    """
    try:
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model_index, verify_model_config_and_directory,
        )
        config = (verify_model_config_and_directory(model_path)
                  if os.path.exists(model_path)
                  else maybe_download_model_index(model_path))
        modules = ["text_encoder", "tokenizer"]
        if "text_encoder_2" in config:
            modules += ["text_encoder_2", "tokenizer_2"]
        modules.append("scheduler")
        return modules
    except Exception:
        pass
    # Fallback: include dual encoders for known models
    if "hunyuan" in model_path.lower():
        return ["text_encoder", "text_encoder_2",
                "tokenizer", "tokenizer_2", "scheduler"]
    return ["text_encoder", "tokenizer", "scheduler"]


def save_video(frames_tensor, output_path: str, fps: int = 24):
    """Save decoded video tensor [B,C,T,H,W] as mp4.

    Returns (filepath, num_frames).
    """
    import numpy as np

    if isinstance(frames_tensor, dict):
        for v in frames_tensor.values():
            if hasattr(v, "shape"):
                frames_tensor = v
                break

    if hasattr(frames_tensor, "cpu"):
        frames_tensor = frames_tensor.cpu().float().numpy()

    # [B, C, T, H, W] -> [T, H, W, C]
    frames = (frames_tensor[0].transpose(1, 2, 3, 0) * 255).clip(0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        import imageio
        imageio.mimwrite(output_path, frames, fps=fps, codec="libx264")
    except Exception as e:
        logger.warning("mp4 export failed (%s), saving first frame as PNG", e)
        from PIL import Image
        output_path = output_path.rsplit(".", 1)[0] + ".png"
        Image.fromarray(frames[0]).save(output_path)

    return output_path, len(frames)


def launch_stage_server(model_path, required_modules, custom_stages_fn,
                        scheduler_port, tp_size=1, num_gpus=None,
                        client_name=""):
    """Patch configs, create ServerArgs, launch Scheduler, return (processes, client, server_args).

    Consolidates the boilerplate shared by encoder, denoiser, and VAE workers:
    patch_hunyuan_config → ServerArgs → set_global → launch_partial_server → StageClient.
    """
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs, set_global_server_args,
    )
    from partial_gpu_worker import launch_partial_server

    patch_hunyuan_config()

    if num_gpus is None:
        num_gpus = tp_size

    server_args = ServerArgs.from_kwargs(
        model_path=model_path,
        num_gpus=num_gpus,
        tp_size=tp_size,
        scheduler_port=scheduler_port,
    )
    set_global_server_args(server_args)

    processes = launch_partial_server(
        server_args,
        required_modules=required_modules,
        custom_stages_fn=custom_stages_fn,
    )

    client = StageClient(server_args.scheduler_endpoint(), client_name)
    return processes, client, server_args
