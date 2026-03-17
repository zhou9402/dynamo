# Disaggregated Diffusion Inference (HunyuanVideo)

Split a monolithic video diffusion pipeline into independent Dynamo workers on separate GPUs.
Tensor data transfers between stages use **NIXL RDMA** (GPU-direct); only small metadata
travels over Dynamo RPC.

Supports HunyuanVideo (13B, dual Llama+CLIP encoder) and Wan2.2-TI2V models.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Orchestrator (HTTP API)                         │
│                        run_disagg.py                                │
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
 6.   Denoiser: NixlReceive (RDMA pull embeddings) → LatentPrep → Denoise (N steps) → NixlSend
 7.   Denoiser ──{nixl_metadata}──► Orchestrator
 8. Orchestrator ──VAERequest + nixl_meta──► VAE Worker
 9.   VAE: NixlReceive (RDMA pull latents) → Decode → Save MP4
10.   VAE ──{video_path}──► Orchestrator ──► User
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

### Multi-Worker Stage Pools (Standalone E2E)

The standalone E2E script (`phase1_workers/run_e2e_sglang.py`) supports
**multiple workers per stage** to utilise all available GPUs.  Each stage
runs an independent pool; the orchestrator round-robins requests across
workers in each pool.

```
          ┌─ Encoder_0 (GPU 0) ─┐    ┌─ Denoiser_0 TP=2 (GPU 1,2) ─┐    ┌─ VAE_0 (GPU 3) ─┐
 Request ─┤                      ├──►─┤                               ├──►─┤                  ├─► Video
          └─ Encoder_1 (GPU 4) ─┘    └─ Denoiser_1 TP=2 (GPU 5,6) ─┘    └─ VAE_1 (GPU 7) ─┘
            round-robin                  round-robin                        round-robin
```

Use `;` to separate workers, `,` for TP GPUs within a worker:

```bash
# 8 GPU — 2 workers per stage
GPU_ENC="0;4" GPU_DEN="1,2;5,6" GPU_VAE="3;7" python phase1_workers/run_e2e_sglang.py

# Asymmetric pools (1 encoder, 3 denoisers, 1 VAE)
GPU_ENC="0" GPU_DEN="1,2;3,4;5,6" GPU_VAE="7" python phase1_workers/run_e2e_sglang.py
```

No pre-pairing is needed — each `send()` allocates a new NIXL buffer and
the `_keep_alive` coroutine holds it until the receiver's RDMA pull completes,
so any sender→receiver combination is safe.

## Quick Start

One script launches everything (etcd + 3 workers + orchestrator):

```bash
conda activate omni
export HF_HUB_CACHE=/path/to/huggingface/hub

# Launch all services + send a test request
./run_all.sh --test

# Quick smoke test (9 frames, 3 steps, ~30s)
./run_all.sh --test --quick

# Just launch services (no test request)
./run_all.sh
```

Or launch each service manually:

```bash
# Terminal 0: etcd
etcd --data-dir /tmp/etcd_disagg --listen-client-urls http://0.0.0.0:2379

# Terminal 1-3: Workers
CUDA_VISIBLE_DEVICES=0   python phase1_workers/encoder_worker.py
CUDA_VISIBLE_DEVICES=1,2 python phase1_workers/denoiser_worker.py
CUDA_VISIBLE_DEVICES=3   python phase1_workers/vae_worker.py

# Terminal 4: Orchestrator
python phase2_orchestrator/run_disagg.py
```

Generate a video (61 frames, 50 steps, 544x960 by default):

```bash
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
| `GPU_ENC` / `GPU_DEN` / `GPU_VAE` | `0` / `1,2` / `3` | GPU assignment per stage (use `;` for multi-worker pools, e.g. `"0;4"`) |
| `TP_SIZE` | auto from `GPU_DEN` | Tensor parallelism for denoiser |
| `PORT` | `8080` | Orchestrator HTTP port |
| `OUTPUT_DIR` | `/tmp/disagg_videos` | Video output directory |

## Supported Models

- **`hunyuanvideo-community/HunyuanVideo`** — 13B, dual encoder (Llama 8B + CLIP), recommended
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` — 5B, single encoder

## Roadmap

- [x] **Multi-worker stage pools** — configurable N workers per stage with round-robin dispatch (standalone E2E)
- [ ] **Dynamic scaling** — auto-scale workers based on queue depth, add/remove denoiser replicas
- [ ] **Streaming output** — stream decoded frames to client as they are produced
- [ ] **Orchestrator improvements** — smarter scheduling, request priority, load balancing across replicas
- [ ] **Metrics & observability** — per-stage latency, GPU utilization, NIXL throughput, Prometheus export
- [ ] **Request cancellation** — cancel in-flight requests, free GPU resources immediately

## Dependencies

```bash
pip install ai-dynamo-runtime sglang imageio imageio-ffmpeg pyzmq setproctitle
```
