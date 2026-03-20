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
      │Encoder-0 │──────────│Denoiser-0│────────────│  VAE-0   │
      │  GPU 0   │  NIXL    │ GPU 1,2  │    NIXL    │  GPU 5   │
      └──────────┘  RDMA    └──────────┘    RDMA    └──────────┘
      ┌──────────┐          ┌──────────┐            ┌──────────┐
      │Encoder-1 │──────────│Denoiser-1│────────────│  VAE-1   │
      │  GPU 3   │  NIXL    │ GPU 4,5  │    NIXL    │  GPU 7   │
      └──────────┘  RDMA    └──────────┘    RDMA    └──────────┘
          ...↕                  ...↕                    ...↕
       dynamic               dynamic                 dynamic
       add/remove            add/remove              add/remove
```

**Key points:**
- Workers self-register with etcd; the orchestrator discovers them
  automatically.
- Dynamo RPC carries only small JSON payloads and NIXL metadata
  (~1.5 KB). Actual tensors (embeddings, latents) transfer GPU-to-GPU
  via NIXL RDMA, never touching the CPU or RPC channel.
- Each stage can have a different number of workers and GPU count.
  The denoiser typically uses TP > 1 (multi-GPU), while encoder and VAE
  each use a single GPU.

### 2.2 Process Model

Each worker is an independent OS process with no shared state:

```
Dynamo Worker Process (e.g. encoder_worker.py)
├── @dynamo_worker                          ← Dynamo runtime bootstrap
│   ├── serve_endpoint("generate")          ← Dynamo RPC from orchestrator
│   │   └── handle_generate()
│   │       └── StageClient.forward()       ← ZMQ to local backend subprocess
│   └── serve_endpoint("health")
│
└── Backend subprocess (spawned at startup)
    └── Inference engine (any backend)
        ├── ReceiveStage   ← RDMA-pull tensors from previous stage
        ├── ComputeStage   ← model-specific inference
        └── SendStage      ← register output tensors as RDMA-readable
```

- The Dynamo worker process handles RPC and control-plane logic.
- The backend subprocess runs the actual model inference. It can be any
  inference engine — SGLang, vLLM, a custom PyTorch loop, etc.
- Communication between the two is via ZMQ REQ/REP (same-host IPC).

### 2.3 Loose Coupling & Dynamic Scaling

Workers are completely independent — they know nothing about each other
or the orchestrator:

- **Startup:** Worker process starts → registers with etcd (via Dynamo
  runtime) → orchestrator auto-discovers the new instance.
- **Shutdown:** Worker process exits → etcd lease expires →
  orchestrator stops routing to it.
- **Add worker:** Start a new process on any available GPU → etcd →
  orchestrator sees it within seconds. No restart, no reconfiguration.
- **Remove worker:** Kill the process → etcd lease expires → traffic
  drains naturally.
- **Auto-scale:** An external controller can monitor queue depth per
  stage (`GET /pipeline/status`) and spawn/kill workers as needed.

### 2.4 Auto Routing

Each stage has a `WorkerManager` that maintains an idle-pool queue:

```
WorkerManager("denoiser", client, [0, 1, 2])
│
├── _idle_queue: asyncio.Queue  ← [0, 1, 2] initially
│
├── acquire_worker() → int      ← blocks until a worker is idle
├── dispatch(wid, rid, json)    ← client.direct(json, wid)
│   └── on completion/failure → release wid back to _idle_queue
│
└── status() → dict             ← per-worker completed/latency/state
```

- `acquire_worker()` blocks the caller when all workers are busy
  (backpressure).
- `dispatch()` sends to a specific worker via `client.direct()` and
  tracks busy/idle state for observability.
- `dispatch_with_retry()` wraps this: on failure, acquires a different
  worker and retries (configurable via `STAGE_DISPATCH_RETRIES`).

### 2.5 Request Flow

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
  │                         │◄──NIXL metadata──────│              │            │
  │                         │   (~1.5 KB)          │              │            │
  │                         │                      │              │            │
  │                         │──DenoiserRequest───────────────────►│            │
  │                         │  (NIXL meta, params)      [RDMA pull embeddings]│
  │                         │                           [denoise N steps]     │
  │                         │◄──NIXL metadata────────────────────│            │
  │                         │                                     │            │
  │                         │──VAEDecodeRequest──────────────────────────────►│
  │                         │  (NIXL meta, req_id)                [RDMA pull] │
  │                         │                                     [decode]    │
  │                         │◄──{video_path}──────────────────────────────────│
  │                    [release semaphore]          │              │            │
  │◄──{url, timings}───────│                       │              │            │
```

Only ~1.5 KB of NIXL metadata travels over Dynamo RPC between stages.
The actual tensor data (embeddings: ~tens of MB, latents: ~hundreds of
MB) transfers GPU-to-GPU via NIXL RDMA without CPU involvement.

### 2.6 Pipeline Parallelism

An `asyncio.Semaphore(pipeline_depth)` gates admission. Each stage has
its own independent worker pool, so multiple requests overlap:

```
Time ──────────────────────────────────────────────────►

Req A: [Enc-0][======Den-0======][VAE-0]
Req B:    [Enc-1][======Den-1======][VAE-1]
Req C:       [Enc-0][======Den-0======][VAE-0]
Req D:          [Enc-1][======Den-1======][VAE-1]
```

- `pipeline_depth` defaults to `MAX_PIPELINE_DEPTH` (env, default 4)
  or the total number of workers across all stages.
- Each request independently acquires workers from each stage's pool.
- The denoiser is typically the bottleneck (50 diffusion steps), so
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

```python
class WorkerManager:
    def __init__(self, stage_name: str, client, worker_ids: List[int]): ...
    async def acquire_worker(self) -> int: ...       # blocks until idle
    async def dispatch(self, worker_id, request_id, request_json) -> (dict, float): ...
    def status(self) -> dict: ...                    # per-worker stats
```

- Backed by `asyncio.Queue` (idle pool) — `acquire_worker()` awaits
  the queue, `dispatch()` returns the worker to it on completion.
- `_call_direct()` sends to a specific worker via
  `client.direct(json, worker_id)`.
- `status()` returns per-worker `{id, status, request_id, completed,
  avg_latency_s}` plus stage-level `{queue_depth, completed, failed}`.

### 3.3 Worker Interface

All three workers follow an identical pattern:

```python
@dynamo_worker(enable_nats=False)
async def worker(runtime: DistributedRuntime):
    # 1. Launch backend subprocess
    processes, client, server_args = launch_stage_server(
        MODEL_PATH, required_modules, build_stage_fn, SCHEDULER_PORT,
    )

    # 2. Define Dynamo RPC handlers
    async def handle_generate(request, context):
        output = await client.forward([build_req(...)])
        yield result_dict  # JSON response with NIXL metadata or output

    async def handle_health(request, context):
        yield {"status": "ok", "stage": "..."}

    # 3. Serve endpoints
    gen_ep = runtime.endpoint("disagg_diffusion.<stage>.generate")
    health_ep = runtime.endpoint("disagg_diffusion.<stage>.health")
    await asyncio.gather(
        gen_ep.serve_endpoint(handle_generate),
        health_ep.serve_endpoint(handle_health),
    )
```

Each worker:
- Spawns a backend subprocess (which loads model weights and runs
  inference).
- Bridges Dynamo RPC ↔ backend via `StageClient` (async ZMQ).
- Handles NIXL metadata forwarding: output from one stage's send
  becomes the next stage's receive metadata.
- Includes a ZMQ fallback path: when NIXL is unavailable, tensors are
  serialized via `torch.save()` + base64 over the RPC channel.

**Stage-specific behavior:**

| Stage | Input | Compute | Output |
|---|---|---|---|
| Encoder | prompt text | Text encoding | NIXL metadata (embeddings) |
| Denoiser | NIXL metadata (embeddings) + params | RDMA pull → N denoise steps | NIXL metadata (latents) |
| VAE | NIXL metadata (latents) | RDMA pull → VAE decode | video file path |

### 3.4 Tensor Transfer (NIXL)

`workers/nixl_transfer.py` — GPU-direct RDMA transfer between stages.

**Sender (`NixlTensorSender`):**

```python
sender = NixlTensorSender()
meta = sender.send({"latents": tensor})   # → ~1.5 KB metadata dict
```

1. Flatten all tensors into a single contiguous GPU buffer
   (`torch.cat`).
2. Create a NIXL `Descriptor` wrapping the flat buffer.
3. Register as `readable` via `connector.create_readable(descriptor)`.
4. Return metadata: tensor keys, shapes, dtypes, NIXL descriptor.
5. Hold `(readable, flat_buffer, timestamp)` in `_pending` list.
6. `_sweep()` polls `readable.status` on each subsequent `send()` —
   releases buffers on `COMPLETE` or timeout (`NIXL_BUFFER_TIMEOUT_S`,
   default 120s).

**Receiver (`NixlTensorReceiver`):**

```python
receiver = NixlTensorReceiver()
tensors = receiver.recv(meta, device="cuda")  # → {"latents": tensor}
```

1. Parse metadata to compute total byte size and per-tensor specs.
2. Allocate a flat `torch.uint8` buffer directly on the target GPU.
3. `connector.begin_read(rdma_meta, descriptor)` → RDMA pull from
   sender's GPU.
4. `read_op.wait_for_completion()` — blocks until transfer finishes.
5. Slice the flat buffer into individual tensors using stored shapes
   and dtypes.

### 3.5 Protocol Types

`workers/protocol.py` — Pydantic models for Dynamo RPC serialization.

```python
class EncoderRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    guidance_scale: float = 1.0

class DenoiserRequest(BaseModel):
    transfer_meta: Dict[str, Any]     # NIXL metadata from encoder
    tensor_data: Dict[str, Any] = {}  # ZMQ fallback
    height: int = 544
    width: int = 960
    num_frames: int = 61
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    seed: int = 42

class VAEDecodeRequest(BaseModel):
    transfer_meta: Dict[str, Any]     # NIXL metadata from denoiser
    tensor_data: Dict[str, Any] = {}  # ZMQ fallback
    request_id: str = ""
```

Responses carry either `transfer_meta` (NIXL path) or `tensor_data`
(ZMQ fallback) but never both.


## 4. Implementation: SGLang Backend

This section describes the current backend implementation using SGLang's
multimodal generation runtime, with HunyuanVideo as the reference model.
The generic architecture (Sections 1-3) is backend-agnostic — any
inference engine that can run pipeline stages can replace SGLang.

### 4.1 PartialGPUWorker

`workers/partial_gpu_worker.py` — Extends SGLang's `GPUWorker` to load
only the modules each stage needs.

```python
class PartialGPUWorker(GPUWorker):
    def __init__(self, required_modules, custom_stages_fn, **kwargs): ...
    def init_device_and_model(self):
        # 1. Set up distributed environment (TP, SP, CFG parallel)
        # 2. build_partial_pipeline() — load only required_modules
        # 3. custom_stages_fn(pipeline, server_args) — build stage list
        # 4. Register stages with pipeline
```

- Overrides only `init_device_and_model()`. All other `GPUWorker`
  behavior (forward execution, memory analysis, LoRA) is inherited.
- `build_partial_pipeline()` (`sglang_utils.py`) dynamically creates a
  subclass of the model's pipeline that suppresses automatic stage
  creation and LoRA initialization, loading only the specified modules.

**Custom pipeline stages (NIXL integration):**

- `NixlReceiveStage(PipelineStage)` — Prepended at the start of
  denoiser/VAE pipelines. Reads `_nixl_transfer_meta` from the `Req`
  and RDMA-pulls tensors. Falls back to device-move for ZMQ path.
  Includes retry logic for `REMOTE_DISCONNECT` errors.
- `NixlSendStage(PipelineStage)` — Appended as the last stage in
  encoder/denoiser pipelines. Extracts tensors from `Req`, registers
  with NIXL, returns `OutputBatch` containing only metadata.

**Stage builders** (picklable functions passed to subprocess):

| Function | Stages |
|---|---|
| `build_encoder_stages()` | `TextEncodingStage` → `NixlSendStage` |
| `build_denoiser_stages()` | `NixlReceiveStage` → `LatentPreparationStage` → `TimestepPreparationStage` → `DenoisingStage` → `NixlSendStage` |
| `build_vae_stages()` | `NixlReceiveStage` → `DecodingStage` |

### 4.2 Subprocess Launcher

`workers/partial_gpu_worker.py:launch_partial_server()` — Spawns
SGLang Scheduler subprocess(es) with `PartialGPUWorker` monkey-patched
in place of the default `GPUWorker`.

```
launch_partial_server(server_args, required_modules, custom_stages_fn)
│
├── For each GPU (rank 0..N-1):
│   ├── Create readiness pipe
│   ├── mp.Process(target=_run_partial_scheduler_process)
│   │   ├── Monkey-patch: sched_mod.GPUWorker = _PatchedGPUWorker
│   │   └── run_scheduler_process(...)  ← standard SGLang entry point
│   └── Start process
│
├── Wire master/slave pipes (TP > 1: rank 0 is master, ranks 1..N are slaves)
├── Wait for all readiness signals
└── Return process list
```

`launch_stage_server()` (`sglang_utils.py`) wraps this with config
setup: `patch_hunyuan_config()` → `ServerArgs.from_kwargs()` →
`launch_partial_server()` → `StageClient(endpoint)`.

### 4.3 StageClient

`workers/sglang_utils.py:StageClient` — Async ZMQ REQ/REP client
connecting the Dynamo worker main process to the SGLang Scheduler
subprocess.

```python
class StageClient:
    def __init__(self, endpoint: str, name: str = ""): ...
    async def forward(self, reqs):    # send_pyobj → recv_pyobj with timeout
    def close(self): ...
```

- `asyncio.Lock` serializes concurrent calls (ZMQ REQ socket is
  single-flight).
- Configurable timeout: `STAGE_FORWARD_TIMEOUT_S` (default 120s).

### 4.4 HunyuanVideo Specifics

- **Dual encoder detection:** `detect_encoder_modules()` reads
  `model_index.json` and auto-detects `text_encoder_2` / `tokenizer_2`
  (HunyuanVideo uses Llama + CLIP). Falls back to heuristic for known
  model names.
- **HunyuanConfig patching:** `patch_hunyuan_config()` wraps
  `HunyuanConfig.__init__` to supply `task_type=T2V` when omitted
  (the base class requires it but HunyuanConfig doesn't default it).
- **Triton norm contiguous workaround:**
  `_patch_triton_norm_contiguous()` wraps SGLang's triton
  `norm_infer` to call `.contiguous()` on non-contiguous tensors from
  HunyuanVideo's attention reshapes, avoiding the triton kernel
  assertion `x.stride(-1) == 1`.
- **Component config sync:** `_sync_all_component_configs()` reads
  `config.json` for every component in `model_index.json` and updates
  `server_args.pipeline_config`, ensuring correct parameters (e.g.
  `z_dim`) even for components whose weights are not loaded by the
  current stage.


## 5. Roadmap

- [x] Multi-worker scaling — N workers per stage, etcd auto-discovery
- [x] Pipeline parallelism — overlapping requests across stages
- [x] NIXL GPU-direct transfer — GPU-to-GPU RDMA, only metadata over RPC
- [ ] Runtime scaling — add/remove workers without restart;
      external auto-scaler integration based on queue depth
- [ ] Smart routing — load-aware dispatch (not just idle-queue),
      affinity-based routing, request priority
- [ ] Fault tolerance — worker health-check + eviction,
      dead worker detection, graceful degradation
- [ ] Streaming output — stream decoded frames as produced
- [ ] Metrics & observability — per-stage latency histograms,
      GPU utilization, NIXL throughput, Prometheus export
- [ ] Request cancellation — cancel in-flight requests, free GPU
      immediately
- [ ] Multi-model support — OneVideo, omni-modal pipelines,
      heterogeneous stage graphs (not just linear 3-stage)


## 6. File Map

```
examples/disagg_diffusion/
├── DESIGN.md                           ← this document
├── README.md                           ← usage guide and quick start
├── run_all.sh                          ← launch etcd + all workers + orchestrator
├── stress_test.sh                      ← concurrent load testing script
│
├── orchestrator/
│   ├── run_disagg.py                   ← HTTP server, PipelineTracker, dispatch_with_retry
│   └── worker_manager.py              ← WorkerManager: idle-pool, acquire/dispatch/status
│
├── workers/
│   ├── __init__.py
│   ├── protocol.py                     ← Pydantic request/response models (Dynamo RPC)
│   ├── encoder_worker.py              ← Encoder Dynamo worker (text encoding → NIXL send)
│   ├── denoiser_worker.py             ← Denoiser Dynamo worker (NIXL recv → denoise → NIXL send)
│   ├── vae_worker.py                  ← VAE Dynamo worker (NIXL recv → decode → save video)
│   ├── nixl_transfer.py               ← NixlTensorSender / NixlTensorReceiver (RDMA)
│   ├── partial_gpu_worker.py          ← PartialGPUWorker, NixlSend/ReceiveStage, subprocess launcher
│   └── sglang_utils.py               ← StageClient, build_partial_pipeline, launch_stage_server
│
└── validate/
    └── validate_split.py              ← validation script for split correctness
```
