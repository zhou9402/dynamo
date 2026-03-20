# Disaggregated Diffusion Pipeline — Design Document

## 1. Overview

### Motivation

Modern diffusion pipelines (text-to-video, text-to-image, omni-modal) are
composed of heterogeneous stages — text encoding, iterative denoising, VAE
decoding — each with fundamentally different compute profiles:

- **Different optimization strategies per stage.** Encoders are
  memory-bound single-pass transforms; denoisers are compute-bound
  multi-step loops that benefit from tensor parallelism; VAE decoders
  are memory-intensive but run only once. Forcing all three into a
  single process prevents stage-specific tuning (parallelism, batching,
  memory management, quantization).

- **Shifting compute balance.** As diffusion models mature, the DiT no
  longer dominates the entire pipeline — faster denoisers (fewer steps,
  distilled models) shift the bottleneck to encoding and decoding.
  Multi-task pipelines are emerging (e.g. OneVideo: encode + denoise +
  decode + audio in one model) where each task demands independent
  scaling.

- **Omni-modal future.** Models that jointly produce video, audio,
  image, and text require stage-level separation so each modality's
  compute can scale independently without wasting GPU resources.

### Solution

Decompose the pipeline into **N independent stages**, each running as a
Dynamo RPC worker on dedicated GPU(s).  An orchestrator chains stages
together, using NIXL RDMA for GPU-direct tensor transfer between them.

| Goal | Mechanism |
|---|---|
| Stage-level scaling | N workers per stage, auto-discovered via etcd |
| Pipeline parallelism | Semaphore admission, independent worker pools |
| GPU-direct transfer | NIXL RDMA — only ~1.5 KB metadata over RPC |
| Loose coupling | Workers are independent processes, no shared state |
| Dynamic scaling | Add/remove workers at runtime, no restart |
| Auto routing | Idle-queue dispatch, backpressure, retry on failure |


## 2. Architecture

### 2.1 Architecture Diagram

```
                              ┌───────────┐
                              │   etcd    │
                              │ registry  │
                              └─────┬─────┘
                           register │ discover
    ┌───────────────────────────────┼───────────────────────────────┐
    │                         Orchestrator                          │
    │                                                               │
    │  HTTP ──► handle_generate() ──► dispatch_with_retry()         │
    │           PipelineTracker        per-stage WorkerManager      │
    │           Semaphore(depth)       acquire → direct() → release │
    │                                                               │
    └──────┬──────────────────────┬──────────────────────┬──────────┘
           │ Dynamo RPC           │ Dynamo RPC            │ Dynamo RPC
           │ (JSON ~1 KB)         │ (NIXL meta ~1.5 KB)   │ (NIXL meta ~1.5 KB)
           ▼                      ▼                       ▼
      ┌──────────┐          ┌──────────┐            ┌──────────┐
      │Encoder-0 │          │Denoiser-0│            │  VAE-0   │
      │  GPU 0   │          │ GPU 1,2  │            │  GPU 7   │
      └────┬─────┘          └──┬───┬───┘            └────┬─────┘
           │   NIXL RDMA pull  │   │  NIXL RDMA pull     │
           │◄──────────────────┘   └────────────────────►│
           │   (embeddings)            (latents)         │
      ┌──────────┐          ┌──────────┐            ┌──────────┐
      │Encoder-1 │          │Denoiser-1│            │  VAE-1   │
      │  GPU 3   │          │ GPU 3,4  │            │  ...     │
      └──────────┘          └──────────┘            └──────────┘
          ...↕               ┌──────────┐               ...↕
       dynamic              │Denoiser-2│             dynamic
       add/remove           │ GPU 5,6  │            add/remove
                            └──────────┘
                                ...↕
                             dynamic
                             add/remove
```

**Key points:**
- Workers self-register with etcd; the orchestrator discovers them
  automatically.  A settle period (`WORKER_SETTLE_S`, default 30s)
  waits for slow-starting workers after the first instance appears.
- Dynamo RPC carries only small JSON payloads and NIXL metadata
  (~1.5 KB). Actual tensors (embeddings, latents) transfer GPU-to-GPU
  via NIXL RDMA, never touching the CPU or RPC channel.
- **RDMA is receiver-initiated (one-sided pull):** the denoiser pulls
  embeddings from the encoder, the VAE pulls latents from the denoiser.
  After the pull completes, the sender receives a completion
  notification and releases the GPU buffer.
- Each stage can have a different number of workers and GPU count.
  The denoiser typically uses TP > 1 (multi-GPU), while encoder and VAE
  each use a single GPU.

### 2.2 Process Model

Each worker is an independent OS process with no shared state:

```
Dynamo Worker Process (e.g. denoiser_worker.py)
├── @dynamo_worker                          ← Dynamo runtime bootstrap
│   ├── serve_endpoint("generate")          ← Dynamo RPC from orchestrator
│   │   └── handle_generate()
│   │       └── StageClient.forward()       ← ZMQ to local backend subprocess
│   └── serve_endpoint("health")
│
└── Backend subprocess(es) (spawned at startup, TP=2 → 2 processes)
    ├── Rank 0 (master, handles ZMQ + NIXL)
    │   ├── NixlReceiveStage    ← RDMA-pull from previous stage
    │   ├── ComputeStage(s)     ← model inference (TP all-reduce)
    │   └── NixlSendStage       ← register output as RDMA-readable
    └── Rank 1 (slave, TP compute only)
        ├── NixlReceiveStage    ← receives tensors via TP broadcast
        ├── ComputeStage(s)     ← model inference (TP all-reduce)
        └── NixlSendStage       ← no-op (only rank 0 sends)
```

- The Dynamo worker process handles RPC and control-plane logic.
- The backend subprocess runs the actual model inference.
- Communication between the two is via ZMQ REQ/REP (same-host IPC).
- With TP > 1: rank 0 does the NIXL pull then broadcasts tensors to
  other ranks via `torch.distributed.broadcast`.  This is transparent
  to the compute stages — TP is internal to the worker.

### 2.3 Request Flow

A single video generation request follows this path:

```
Client                 Orchestrator           Encoder-k     Denoiser-j     VAE-i
  │                        │                      │              │            │
  │─POST /v1/videos/──────►│                      │              │            │
  │  generations            │                      │              │            │
  │                    [acquire semaphore]          │              │            │
  │                         │                      │              │            │
  │                         │──EncoderRequest──────►│              │            │
  │                         │  (prompt, cfg)   [encode text]      │            │
  │                         │                  [register readable] │            │
  │                         │◄──NIXL metadata──────│              │            │
  │                         │   (~1.5 KB)     [hold buffer]       │            │
  │                         │                      │              │            │
  │                         │──DenoiserRequest───────────────────►│            │
  │                         │  (NIXL meta, params)                │            │
  │                         │                      │  [RDMA pull embeddings]   │
  │                         │                      │─────────────►│            │
  │                         │                 [completion notif]  │            │
  │                         │                 [release buffer]    │            │
  │                         │                      │  [denoise N steps]       │
  │                         │                      │  [register readable]     │
  │                         │◄──NIXL metadata────────────────────│            │
  │                         │                      │  [hold buffer]           │
  │                         │                      │              │            │
  │                         │──VAEDecodeRequest──────────────────────────────►│
  │                         │  (NIXL meta, req_id)                │            │
  │                         │                      │              │  [RDMA pull latents]
  │                         │                      │              │───────────►│
  │                         │                      │  [completion notif]      │
  │                         │                      │  [release buffer]   [decode]
  │                         │◄──{video_path}──────────────────────────────────│
  │                    [release semaphore]          │              │            │
  │◄──{url, timings}───────│                       │              │            │
```

Only ~1.5 KB of NIXL metadata travels over Dynamo RPC between stages.
The actual tensor data (embeddings: ~tens of MB, latents: ~hundreds of
MB) transfers GPU-to-GPU via NIXL RDMA without CPU involvement.

**Buffer lifecycle:** The sender holds the GPU buffer (via `readable_op`
in `_active_readables`) until the receiver completes the RDMA pull and
the sender receives a completion notification.  `_poll_completed()`
checks and releases finished buffers at the start of each new request.

### 2.4 Pipeline Parallelism

An `asyncio.Semaphore(pipeline_depth)` gates admission. Each stage has
its own independent worker pool, so multiple requests overlap:

```
Time ──────────────────────────────────────────────────►

Req A: [Enc-0][======Den-0======][VAE-0]
Req B:    [Enc-0][======Den-1======][VAE-0]
Req C:       [Enc-0][======Den-2======][VAE-0]
Req D:          [Enc-0][======Den-0======][VAE-0]
```

- `pipeline_depth` defaults to `MAX_PIPELINE_DEPTH` (env, default 4)
  or the total number of workers across all stages.
- Each request independently acquires workers from each stage's pool.
- The denoiser is typically the bottleneck (many diffusion steps), so
  encoder and VAE workers are freed quickly to serve other requests.


## 3. Component Design

### 3.1 Orchestrator

`orchestrator/run_disagg.py` — aiohttp HTTP server that chains stages.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/videos/generations` | Submit generation request |
| `GET` | `/health` | Orchestrator liveness |
| `GET` | `/health/stages` | Per-stage health (queries each worker) |
| `GET` | `/pipeline/status` | Active requests, queue depth, latencies |
| `GET` | `/videos/<filename>` | Serve generated video file |

**Key components:**

- `PipelineTracker` — Tracks active requests per stage, completed/failed
  counts, and rolling average stage latencies.
- `dispatch_with_retry(mgr, request_id, json)` — Acquire worker →
  dispatch → on failure retry on a different worker (up to
  `STAGE_DISPATCH_RETRIES` attempts).
- `admission = asyncio.Semaphore(pipeline_depth)` — Limits concurrent
  in-flight requests across the entire pipeline.

### 3.2 WorkerManager

`orchestrator/worker_manager.py` — Per-stage worker pool with
busy/idle tracking.

- Backed by `asyncio.Queue` (idle pool) — `acquire_worker()` awaits
  the queue, `dispatch()` returns the worker to it on completion.
- `_call_direct()` sends to a specific worker via
  `client.direct(json, worker_id)`.
- `status()` returns per-worker `{id, status, request_id, completed,
  avg_latency_s}` plus stage-level `{queue_depth, completed, failed}`.

### 3.3 Tensor Transfer — NIXL

`workers/nixl_transfer.py` — GPU-direct RDMA transfer between stages.

**Connection management — `PersistentConnector`:**

Each `NixlTensorSender` and `NixlTensorReceiver` owns a
`PersistentConnector` instance (subclass of `nixl_connect.Connector`)
that overrides `_create_connection()` to reuse a single
`Connection` (= one `nixl_agent` / UCX endpoint) across all operations.

```
Per-process NIXL agents (e.g. denoiser with TP=2):

  Rank 0 subprocess:
    ├── NixlReceiveStage → NixlTensorReceiver → PersistentConnector → agent A
    └── NixlSendStage    → NixlTensorSender   → PersistentConnector → agent B

  Rank 1 subprocess:
    └── (no NIXL agents — receives tensors via TP broadcast from rank 0)
```

- Agents are created eagerly at stage construction time, before any
  request arrives.  This ensures UCX endpoints are fully initialized.
- Each sender/receiver has its own `PersistentConnector` → its own
  agent.  This avoids UCX state conflicts between read and write roles.
- Agents are reused across all requests — never recreated.

**UCX transport configuration:**

For intra-node transfers, set `UCX_TLS=cuda_ipc,tcp,self,cuda_copy,cma`
to force NVLink (cuda_ipc) instead of IB RDMA.  The default
`UCX_TLS=all` may select IB which fails across NUMA boundaries on
multi-socket systems.

**Sender (`NixlTensorSender`):**

```python
sender = NixlTensorSender()                     # creates agent eagerly
meta, readable_op = sender.send({"latents": t})  # → (metadata, handle)
# caller holds readable_op until receiver completes RDMA pull
```

1. Flatten all tensors into a single contiguous GPU buffer.
2. Create NIXL `Descriptor` and register as readable.
3. Return `(metadata_dict, readable_op)` — caller must hold
   `readable_op` to prevent GC of the descriptor and GPU buffer.

**Receiver (`NixlTensorReceiver`):**

```python
receiver = NixlTensorReceiver()                  # creates agent eagerly
tensors = receiver.recv(meta, device="cuda")     # → {"latents": tensor}
```

1. Allocate flat GPU buffer on target device.
2. `begin_read(rdma_meta, descriptor)` → one-sided RDMA pull.
3. `wait_for_completion()` — blocks until transfer finishes.
4. Slice flat buffer into individual tensors.

**Buffer lifecycle:**

```
Sender (Encoder/Denoiser)          Receiver (Denoiser/VAE)
─────────────────────────          ───────────────────────
create_readable(descriptor)
  → register GPU buffer
  → return (meta, readable_op)
                                   begin_read(meta)
                                     → RDMA pull from sender GPU
                                   wait_for_completion()
                                     → transfer done
  ← completion notification
_poll_completed()
  → status == COMPLETE
  → drop readable_op → GC buffer
```

### 3.4 Worker Interface

All three workers follow an identical pattern:

```python
@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    # 1. Launch backend subprocess
    processes, client, server_args = launch_stage_server(...)

    # 2. Define Dynamo RPC handlers
    async def handle_generate(request, context):
        output = await client.forward([build_req(...)])
        yield result_dict  # JSON with NIXL metadata or video path

    # 3. Serve endpoints
    gen_ep = runtime.endpoint("disagg_diffusion.<stage>.generate")
    await gen_ep.serve_endpoint(handle_generate)
```

**Stage IO contract:**

| Stage | Input | Output |
|---|---|---|
| Encoder | prompt, guidance_scale | `transfer_meta` (embedding metadata) |
| Denoiser | `transfer_meta` + inference params | `transfer_meta` (latent metadata) |
| VAE | `transfer_meta` + request_id | `video_path` |

Fallback: when NIXL is unavailable, tensors are serialized via
`torch.save()` + base64 in the `tensor_data` field.


## 4. Implementation: SGLang Backend

### 4.1 PartialGPUWorker

`workers/partial_gpu_worker.py` — Extends SGLang's `GPUWorker` to load
only the modules each stage needs.

- Overrides only `init_device_and_model()`. All other `GPUWorker`
  behavior (forward execution, memory analysis, LoRA) is inherited.
- `build_partial_pipeline()` dynamically loads only the specified
  modules (e.g. `["transformer", "scheduler"]` for denoiser).

**NIXL pipeline stages:**

- `NixlReceiveStage(PipelineStage)` — Prepended at denoiser/VAE entry.
  Rank 0: RDMA-pulls tensors via `NixlTensorReceiver`.
  TP > 1: broadcasts pulled tensors to other ranks via
  `torch.distributed.broadcast`.
  Falls back to `_device_move()` for ZMQ path.

- `NixlSendStage(PipelineStage)` — Appended at encoder/denoiser exit.
  Rank 0 only: registers tensors as NIXL-readable, stores `readable_op`
  in `_active_readables`, returns metadata in `OutputBatch`.
  Rank > 0: returns empty `OutputBatch` (only rank 0's output is used).

**Stage builders:**

| Function | Stages |
|---|---|
| `build_encoder_stages()` | `TextEncodingStage` → `NixlSendStage` |
| `build_denoiser_stages()` | `NixlReceiveStage` → `LatentPrep` → `TimestepPrep` → `DenoisingStage` → `NixlSendStage` |
| `build_vae_stages()` | `NixlReceiveStage` → `DecodingStage` |

### 4.2 StageClient

`workers/sglang_utils.py:StageClient` — Async ZMQ REQ/REP client.

- `asyncio.Lock` serializes concurrent calls (ZMQ REQ requires strict
  send/recv alternation).
- Configurable timeout: `STAGE_FORWARD_TIMEOUT_S` (default 120s).
- On timeout: `_reset_socket()` closes and reconnects the ZMQ socket
  to recover from the broken REQ state (prevents cascading EFSM errors).

### 4.3 HunyuanVideo Specifics

- **Dual encoder:** auto-detects `text_encoder` + `text_encoder_2`
  (Llama + CLIP) from `model_index.json`.
- **HunyuanConfig patching:** wraps `HunyuanConfig.__init__` to supply
  `task_type=T2V` when omitted.
- **Triton norm workaround:** wraps `norm_infer` to call
  `.contiguous()` on non-contiguous tensors from attention reshapes.
- **Component config sync:** reads all component `config.json` files
  to ensure correct parameters even for non-loaded components.


## 5. Deployment

### 5.1 Example: 8-GPU HunyuanVideo

```bash
# 1 encoder (GPU 0) + 3 denoisers TP=2 (GPU 1,2 / 3,4 / 5,6) + 1 VAE (GPU 7)
UCX_TLS=cuda_ipc,tcp,self,cuda_copy,cma \
GPU_ENC=0 GPU_DEN="1,2;3,4;5,6" GPU_VAE=7 PORT=8091 \
    ./run_all.sh
```

### 5.2 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GPU_ENC` | `0` | GPU(s) for encoder workers (`;`-separated for multi-worker) |
| `GPU_DEN` | `1,2` | GPU(s) for denoiser workers (`,` = TP within worker, `;` = multiple workers) |
| `GPU_VAE` | `3` | GPU(s) for VAE workers |
| `PORT` | `8080` | HTTP port |
| `UCX_TLS` | `all` | UCX transport list — set to `cuda_ipc,tcp,self,cuda_copy,cma` for intra-node |
| `MAX_PIPELINE_DEPTH` | `4` | Max concurrent requests in pipeline |
| `STAGE_FORWARD_TIMEOUT_S` | `120` | ZMQ timeout per stage request (seconds) |
| `WORKER_SETTLE_S` | `30` | Seconds to wait for slow workers after first discovery |
| `STAGE_DISPATCH_RETRIES` | `2` | Retry count on stage dispatch failure |
| `DISABLE_NIXL` | `false` | Force ZMQ fallback (disable NIXL) |

### 5.3 Measured Performance (HunyuanVideo, 9 frames, 3 steps)

| Stage | Workers | Avg Latency |
|---|---|---|
| Encoder | 1 × GPU | 0.33s |
| Denoiser | 3 × TP=2 | 3.7s |
| VAE | 1 × GPU | 3.8s |
| **End-to-end** | | **~7.9s** |
| **Throughput (20 concurrent)** | | **~0.25 req/s** (VAE-bound) |


## 6. Roadmap

- [x] Multi-worker scaling — N workers per stage, etcd auto-discovery
- [x] Pipeline parallelism — overlapping requests across stages
- [x] NIXL GPU-direct transfer — GPU-to-GPU RDMA, only metadata over RPC
- [x] TP support — TP broadcast in NixlReceiveStage, rank-0-only send
- [x] PersistentConnector — stable NIXL connections, no per-request agent churn
- [ ] Runtime scaling — external auto-scaler based on `/pipeline/status`
- [ ] Smart routing — load-aware dispatch, affinity, request priority
- [ ] Fault tolerance — health-check + eviction, graceful degradation
- [ ] Streaming decode output + richer observability
- [ ] Multi-model support — omni-modal pipelines, heterogeneous stage graphs


## 7. File Map

```
examples/disagg_diffusion/
├── DESIGN.md                           ← this document
├── run_all.sh                          ← launch etcd + all workers + orchestrator
├── stress_test.sh                      ← concurrent load testing script
│
├── orchestrator/
│   ├── run_disagg.py                   ← HTTP server, pipeline dispatch, admission control
│   ├── worker_manager.py              ← WorkerManager: idle-pool, acquire/dispatch/status
│   └── request_validation.py          ← Pydantic-based HTTP request validation
│
└── workers/
    ├── protocol.py                     ← Pydantic request/response models (Dynamo RPC)
    ├── encoder_worker.py              ← Encoder Dynamo worker
    ├── denoiser_worker.py             ← Denoiser Dynamo worker (supports TP)
    ├── vae_worker.py                  ← VAE Dynamo worker
    ├── nixl_transfer.py               ← NixlTensorSender / NixlTensorReceiver (RDMA)
    ├── partial_gpu_worker.py          ← PartialGPUWorker, NixlSend/ReceiveStage, launcher
    └── sglang_utils.py               ← StageClient, build_partial_pipeline, launch_stage_server
```
