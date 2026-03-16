# Disaggregated Diffusion Inference (HunyuanVideo)

Split a monolithic video diffusion pipeline into independent stages on separate GPUs.
Tensor data transfers between stages use **NIXL RDMA** (GPU-direct); only small metadata
travels over the control plane.

Supports HunyuanVideo (13B, dual Llama+CLIP encoder) and Wan2.2-TI2V models.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Orchestrator (HTTP API)                      │
│                     run_disagg.py / run_e2e_sglang.py               │
└──────┬──────────────────────┬───────────────────────────┬───────────┘
       │ Dynamo RPC           │ Dynamo RPC                │ Dynamo RPC
       │ (metadata)           │ (metadata)                │ (metadata)
       ▼                      ▼                           ▼
┌──────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│ Encoder      │   │ Denoiser            │   │ VAE Decoder          │
│ Worker       │   │ Worker              │   │ Worker               │
│              │   │                     │   │                      │
│ GPU 0        │   │ GPU 1,2 (TP=2)     │   │ GPU 3                │
│ ~18 GB VRAM  │   │ ~24 GB VRAM/GPU    │   │ ~6 GB VRAM           │
│              │   │                     │   │                      │
│ Llama 8B     │   │ HunyuanVideo DiT   │   │ 3D VAE               │
│ + CLIP       │   │ 9.5B params        │   │                      │
└──────┬───────┘   └──────┬──────────────┘   └──────┬───────────────┘
       │                  │                         │
       └──── NIXL RDMA ──►└──── NIXL RDMA ─────────►│
          (embeddings,          (latents,
           GPU-direct)           GPU-direct)
```

### Request Flow

```
1. User ──POST /v1/videos/generations──► Orchestrator
2. Orchestrator ──EncoderRequest──► Encoder Worker
3.   Encoder: TextEncoding → NixlSendStage (register embeddings on GPU)
4.   Encoder ──{nixl_metadata}──► Orchestrator
5. Orchestrator ──DenoiserRequest + nixl_meta──► Denoiser Worker
6.   Denoiser: NixlReceive (RDMA pull embeddings) → LatentPrep → Denoise (50 steps) → NixlSend
7.   Denoiser ──{nixl_metadata}──► Orchestrator
8. Orchestrator ──VAERequest + nixl_meta──► VAE Worker
9.   VAE: NixlReceive (RDMA pull latents) → Decode → Save MP4
10.  VAE ──{video_path}──► Orchestrator ──► User
```

### Worker Internal Architecture

Each worker wraps an SGLang Scheduler subprocess:

```
Dynamo Worker Process (e.g. encoder_worker.py)
├── @dynamo_worker
│   └── serve_endpoint("generate")     ← Dynamo RPC from orchestrator
│       └── StageClient.forward()      ← ZMQ to local Scheduler
│
└── SGLang Scheduler subprocess        ← spawned by launch_partial_server()
    └── PartialGPUWorker
        ├── TextEncodingStage          ← model inference
        └── NixlSendStage             ← register tensors for RDMA
```

### Pipeline Parallelism

Multiple requests overlap across stages:

```
Request 1:  [ Encoder ] ──► [ Denoiser ~~~~~~~~ ] ──► [  VAE  ]
Request 2:               [ Encoder ] ──► [ Denoiser ~~~~~~~~ ] ──► [  VAE  ]
Request 3:                            [ Encoder ] ──► [ Denoiser ~~~~~~~~ ]
```

## Quick Start — Single Script (Recommended)

Launches all 3 stages + runs the full pipeline in one command. No etcd needed.

```bash
conda activate omni
export HF_HUB_CACHE=/path/to/huggingface/hub

# Default: 61 frames, 50 steps, 544x960, ~8 min on 4x H20 GPUs
python phase1_workers/run_e2e_sglang.py

# Custom prompt
PROMPT="A golden retriever running on a sunny beach" python phase1_workers/run_e2e_sglang.py

# Quick smoke test (~30s)
NUM_FRAMES=9 NUM_STEPS=3 python phase1_workers/run_e2e_sglang.py
```

Output: `/tmp/disagg_e2e/output_0.mp4`

## Multi-Process Deployment (Dynamo RPC)

For production use with independent workers and HTTP API:

```bash
conda activate omni
export HF_HUB_CACHE=/path/to/huggingface/hub

# Terminal 0: etcd
etcd --data-dir /tmp/etcd_disagg --listen-client-urls http://0.0.0.0:2379

# Terminal 1-3: Workers
CUDA_VISIBLE_DEVICES=0   python phase1_workers/encoder_worker.py
CUDA_VISIBLE_DEVICES=1,2 python phase1_workers/denoiser_worker.py
CUDA_VISIBLE_DEVICES=3   python phase1_workers/vae_worker.py

# Terminal 4: HTTP Orchestrator
python phase2_orchestrator/run_disagg.py
```

```bash
# Generate video (61 frames, 50 steps by default)
curl -X POST http://localhost:8080/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A golden retriever running on a sunny beach with waves crashing"}'
```

## Workers

| Worker | Model Component | VRAM | Dynamo Endpoint |
|--------|----------------|------|-----------------|
| `encoder_worker.py` | Llama 8B + CLIP text encoders | ~18 GB | `disagg_diffusion.encoder.generate` |
| `denoiser_worker.py` | HunyuanVideo DiT (TP=2) | ~24 GB/GPU | `disagg_diffusion.denoiser.generate` |
| `vae_worker.py` | 3D VAE decoder | ~6 GB | `disagg_diffusion.vae.generate` |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `hunyuanvideo-community/HunyuanVideo` | HuggingFace model ID or local path |
| `NUM_FRAMES` | `61` | Number of video frames (~2.5s at 24fps) |
| `NUM_STEPS` | `50` | Denoising steps (more = higher quality) |
| `HEIGHT` / `WIDTH` | `544` / `960` | Output resolution |
| `GUIDANCE` | `1.0` | Guidance scale (1.0 = embedded guidance) |
| `GPU_ENC` / `GPU_DEN` / `GPU_VAE` | `0` / `1,2` / `3` | GPU assignment |
| `TP_SIZE` | auto from `GPU_DEN` | Tensor parallelism for denoiser |
| `OUTPUT_DIR` | `/tmp/disagg_e2e` | Video output directory |

## Supported Models

- **`hunyuanvideo-community/HunyuanVideo`** — 13B, dual encoder (Llama 8B + CLIP), recommended
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` — 5B, single encoder

## Roadmap

- [ ] **Dynamo RPC data plane** — replace NIXL with Dynamo's native tensor transport
- [ ] **Multi-node** — distribute stages across machines (currently single-node only)
- [ ] **Dynamic batching** — batch multiple prompts per denoiser pass
- [ ] **LoRA hot-swap** — switch LoRA adapters without restarting workers
- [ ] **Speculative decoding** — use smaller DiT for early steps, full DiT for final steps
- [ ] **Streaming output** — stream decoded frames as they're produced
- [ ] **HunyuanVideo 1.5** — upgrade to latest HunyuanVideo with improved quality
- [ ] **Wan2.2 14B** — support larger Wan model with TP
- [ ] **Profiling dashboard** — per-stage latency, GPU utilization, NIXL throughput metrics

## Dependencies

```bash
pip install ai-dynamo-runtime sglang imageio imageio-ffmpeg pyzmq setproctitle
```
