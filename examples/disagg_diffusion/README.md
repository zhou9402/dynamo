# Disaggregated Diffusion Inference (HunyuanVideo)

Split a monolithic video diffusion pipeline (Text Encoder -> Transformer -> VAE) into
independent stages on separate GPUs. Tensor data transfers between stages use
**NIXL RDMA** (GPU-direct); only small metadata travels over Dynamo RPC.

Supports HunyuanVideo (13B, dual Llama+CLIP encoder) and Wan2.2-TI2V models.

## Architecture

```
                       Dynamo RPC (metadata only)
               ┌────────────┬────────────────────────┐
               │             │                        │
               ▼             ▼                        ▼
GPU 0:  Encoder Worker   GPU 1,2: Denoiser Worker   GPU 3: VAE Worker
  │  (Llama + CLIP)       │  (DiT, TP=2)              │  (3D VAE)
  │                       │                            │
  └── NIXL RDMA ──────────┘── NIXL RDMA ──────────────┘
     (embeddings)              (latents)
               ▲
               │
         Orchestrator (HTTP, no GPU)
```

Each worker is a Dynamo `@dynamo_worker` that:
1. Spawns SGLang Scheduler subprocess(es) via `launch_partial_server()`
2. Connects a ZMQ `StageClient` to its local Scheduler
3. Exposes `serve_endpoint("generate")` for Dynamo RPC
4. Bridges RPC requests to the Scheduler, which runs the model + NIXL transfer

## Quick Start

```bash
conda activate omni
export HF_HUB_CACHE=/path/to/huggingface/hub

# Terminal 0: etcd (service discovery)
etcd --data-dir /tmp/etcd_disagg --listen-client-urls http://0.0.0.0:2379

# Terminal 1: Encoder Worker (Llama 8B + CLIP, ~18 GB VRAM)
CUDA_VISIBLE_DEVICES=0 python phase1_workers/encoder_worker.py

# Terminal 2: Denoiser Worker (DiT TP=2, ~24 GB VRAM per GPU)
CUDA_VISIBLE_DEVICES=1,2 python phase1_workers/denoiser_worker.py

# Terminal 3: VAE Worker (~6 GB VRAM)
CUDA_VISIBLE_DEVICES=3 python phase1_workers/vae_worker.py

# Terminal 4: Orchestrator (no GPU)
python phase2_orchestrator/run_disagg.py
```

Test:
```bash
curl -X POST http://localhost:8080/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat walking on green grass", "num_frames": 9, "num_inference_steps": 3}'
```

### Standalone E2E (no Dynamo, no etcd)

Single-process launcher that starts all 3 stages and runs the pipeline:

```bash
python phase1_workers/run_e2e_sglang.py
```

## Phases

### Phase 0: Offline Validation (no Dynamo)

Single-GPU script proving diffusers supports split execution.

```bash
python phase0_validate/validate_split.py \
    --model hunyuanvideo-community/HunyuanVideo \
    --prompt "A cat walking on grass" \
    --num-steps 3 --num-frames 9 \
    --output-dir /tmp/disagg_validate
```

### Phase 1: Dynamo Stage Workers (NIXL + SGLang Scheduler)

Three Dynamo workers, each wrapping an SGLang Scheduler subprocess:

| Worker | Model Component | VRAM | Endpoint |
|--------|----------------|------|----------|
| `encoder_worker.py` | Llama 8B + CLIP text encoders | ~18 GB | `disagg_diffusion.encoder.generate` |
| `denoiser_worker.py` | HunyuanVideo DiT (TP=2) | ~24 GB/GPU | `disagg_diffusion.denoiser.generate` |
| `vae_worker.py` | 3D VAE decoder | ~6 GB | `disagg_diffusion.vae.generate` |

### Phase 2: Orchestrator (Pipeline Parallel)

Chains three stage endpoints with pipeline parallelism. Multiple concurrent
requests overlap across stages:

```
Request 1:  [Encoder] → [Denoiser] → [  VAE  ]
Request 2:             [Encoder] → [Denoiser] → [  VAE  ]
Request 3:                        [Encoder] → [Denoiser] → ...
```

## Environment Variables

### Worker Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `hunyuanvideo-community/HunyuanVideo` | HuggingFace model path |
| `SCHEDULER_PORT` | `15600/15700/15800` | ZMQ port for local Scheduler |
| `TP_SIZE` | auto from `CUDA_VISIBLE_DEVICES` | Tensor parallelism (denoiser) |
| `OUTPUT_DIR` | `/tmp/disagg_videos` | Video output directory (VAE) |

### Orchestrator Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP server port |
| `MAX_PIPELINE_DEPTH` | `4` | Max concurrent requests in pipeline |
| `OUTPUT_DIR` | `/tmp/disagg_videos` | Shared directory for video output |

## Supported Models

- `hunyuanvideo-community/HunyuanVideo` (13B, dual encoder, recommended)
- `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (5B, single encoder)
- Any SGLang-supported diffusion model with encoder/denoiser/VAE stages

## Dependencies

```bash
pip install ai-dynamo-runtime sglang imageio imageio-ffmpeg pyzmq setproctitle etcd-distro
```
