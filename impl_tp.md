# Tensor Parallelism Implementation Plan

## What Was Done

### New Files
- **ds4_tp.h** - Public TP API header with extern "C" guards
- **ds4_tp.cu** - NCCL-based TP implementation

### Modified Files
- **ds4_gpu.h** - Added extern "C" guards for C++ interop
- **ds4.h** - Added tp_size, tp_rank, tp_master_addr, tp_host, tp_port to ds4_engine_options
- **ds4.c** - TP init/cleanup wiring + 6 all-reduce insertion points (3 decode + 3 prefill)
- **ds4_cli.c** - Added --tp, --tp-rank, --tp-master CLI flags
- **ds4_server.c** - TP flags, rank 1 worker mode, broadcast calls in generate_job(), skip HTTP on rank 1, prefill-before-sync ordering, skip prefix sync when TP enabled
- **ds4_cuda.cu** - Expert parallelism: zero non-local expert counts in MoE kernel
- **Makefile** - Added ds4_tp.o to CORE_OBJS, NCCL linking, rpath

### Architecture
- **NCCL** for collectives (already built at /home/tim/code/nccl/build/)
- **TCP rendezvous** for multi-node: rank 0 listens on --host/--port for TP, rank 1 connects via --tp-master
- **Persistent control channel**: After rendezvous, rank 0 keeps a TCP socket to rank 1 for command broadcast
- **TP worker thread**: Rank 1 runs a dedicated thread that mirrors prefill/eval/done from rank 0 in lockstep
- **Non-blocking CUDA stream** for collectives to avoid compute blocking
- **3 all-reduces per layer**: after attention output, routed MoE output, shared expert output
- **Expert Parallelism (EP)**: 256 experts split evenly across ranks (rank 0: 0-127, rank 1: 128-255)
- **ncclSum** all-reduce: each rank computes partial output, summed to correct result

### Rendezvous Protocol
- Rank 0: listens on `--host`:`--port` (defaults to `0.0.0.0:12345` if unset) for TP rendezvous
- Rank 1: connects to `--tp-master HOST:PORT` (the address of rank 0)
- After NCCL init, rank 0 keeps the socket open and accepts rank 1's connection for the control channel
- Control protocol: `TP_CTL_PREFILL` (n_tokens + tokens), `TP_CTL_EVAL` (token), `TP_CTL_EVAL_BATCH`, `TP_CTL_DONE`, `TP_CTL_SHUTDOWN`
- Listener socket closed after rendezvous to free port for HTTP

### Key Design Decisions
- All-reduce is a no-op when tp_size <= 1 (current default)
- TP only works with CUDA backend (checked in ds4_engine_open)
- Prefill path uses tensor views for batch-sized all-reduce
- Decode path uses direct tensor all-reduce (single row)
- Rank 0 handles HTTP and broadcasts work to rank 1 via persistent control socket
- Rank 1 skips HTTP, runs TP worker thread instead
- Session on rank 1 is invalidated after each request (full prefill on next)
- `--tp-master` only needed for rank 1; rank 0 infers listen address from `--host`/`--port`
- Worker uses ack-before-compute pattern: sends ack immediately, then does compute (prevents deadlock with all-reduce)
- Prefix sync skipped when TP enabled (rank 1 doesn't do prefix, would deadlock on all-reduce)
- EP: attention and shared expert stay replicated, only MoE experts are distributed

## Benchmark Results

### Baseline (single-node, no TP)
- Decode (tg128): ~12.5 t/s (from manual test)

### Phase 1 (replicated weights, ncclAvg) — BROKEN (no speedup, slower)
- Decode: ~7.7 t/s (replicated compute + comm overhead)

### Phase 2 (Expert Parallelism, ncclSum) — llama-benchy through HTTP
| Test | t/s | peak t/s | TTFR (ms) |
|------|-----|----------|-----------|
| pp2048 | 354.70 | - | 5453 |
| tg128 | 23.95 | 35.11 | - |
| pp2048 @ d4096 | 330.77 | - | 17045 |
| tg128 @ d4096 | 24.10 | 52.50 | - |

**Decode speedup**: ~2x (24 t/s vs 12.5 t/s single-node) — EP halves MoE compute per GPU

### Phase 2b (Optimizations) — llama-benchy through HTTP, before Phase 3 fixes
- Removed attention all-reduce (replicated, was doubling with ncclSum)
- Removed shared expert all-reduce (replicated, was doubling with ncclSum)
- Removed eval ack round-trip (all-reduce provides sync)
- Added ds4-bench TP support (--tp, --tp-rank, --tp-master)

| Test | t/s | peak t/s | TTFR (ms) |
|------|-----|----------|-----------| 
| pp2048 | 358.89 ± 24.36 | - | 5195.30 ± 269.50 |
| tg128 | 28.28 ± 2.76 | 44.67 ± 7.41 | - |
| pp2048 @ d4096 | 330.89 ± 3.15 | - | 16971.69 ± 380.77 |
| tg128 @ d4096 | 27.87 ± 1.18 | 45.33 ± 5.56 | - |

### Phase 3 (Correctness fixes) — llama-benchy through HTTP
- Fixed all-reduce F16→F32 (routed_out is float, was reinterpreting bits as half)
- Added TCP_NODELAY on control socket (eliminates Nagle delay on 5-byte messages)
- Removed session invalidation on rank 1 (allows KV cache reuse between requests)
- NCCL stays on g_tp_stream for natural overlap with shared expert compute

| Test | t/s | peak t/s | TTFR (ms) |
|------|-----|----------|-----------|
| pp2048 | 353.89 ± 34.00 | - | 5315.80 ± 526.58 |
| tg128 | 27.93 ± 0.54 | 33.33 ± 0.47 | - |
| pp2048 @ d4096 | 328.71 ± 1.57 | - | 17250.09 ± 170.82 |
| tg128 @ d4096 | 25.57 ± 0.65 | 32.00 ± 1.41 | - |

**Note**: Average decode within error bars of Phase 2b. Peak t/s drop (44→33) is the cost of correct F32 all-reduce vs corrupted F16. Variance much lower (±0.54 vs ±2.76) — more consistent behavior.

## Testing Status

### TP=2 Multi-Node: VERIFIED WORKING
- Tested across hades and hephaestus via 200 Gbps NICs (enp1s0f1np1)
- NCCL data plane on 192.168.0.x (200G, MTU 9000 jumbo frames)
- Control channel also on 192.168.0.x
- Full generation completed across multiple requests (prefill + decode)
- No deadlock, no crashes

### Deadlocks Fixed
1. **Listener port conflict**: TP listener held port 8000, HTTP couldn't bind → fixed by closing listener after rendezvous
2. **Prefill deadlock**: Rank 0 did prefix sync (with all-reduce) before broadcasting to rank 1 → fixed by broadcasting first, skipping prefix sync in TP mode
3. **Eval deadlock**: Rank 0 waited for ack before eval, rank 1 did eval (with all-reduce) before ack → fixed with ack-before-compute pattern

### Phase 3 Fixes Applied
- Fixed all-reduce F16→F32 (routed_out is float, was reinterpreting bits as half)
- Added TCP_NODELAY on control socket
- Removed session invalidation on rank 1 (KV cache reuse)
- NCCL on g_tp_stream (non-blocking) for natural overlap with shared expert

## Performance Analysis

### Where Does 37ms/token Go?

Current: ~27 t/s = 37ms per token (TP=2, EP-only)

**Estimated per-layer breakdown** (43 layers, decode single token, ~200 GB/s mem BW):

| Stage | Weight size | Est. time/layer | × 43 layers | Notes |
|-------|-----------|----------------|-------------|-------|
| HC pre-attn | ~0.5 MB | 0.003ms | 0.1ms | Small element-wise |
| Q_A (LoRA down) | 2.3 MB | 0.012ms | 0.5ms | 4096→1024 Q8_0 |
| Q_B (LoRA up) | 18.4 MB | 0.092ms | **4.0ms** | 1024→32768 Q8_0 — **largest single weight** |
| KV projection | 1.15 MB | 0.006ms | 0.3ms | 4096→512 Q8_0 |
| Norms, RoPE | tiny | ~0.001ms | 0.05ms | |
| Attention compute | varies | ~0.05ms | 2.2ms | Context-dependent |
| Output LoRA | ~4.6 MB | 0.023ms | 1.0ms | Grouped: 8 groups |
| HC post-attn | ~0.5 MB | 0.003ms | 0.1ms | |
| HC pre-FFN | ~0.5 MB | 0.003ms | 0.1ms | |
| Shared expert | ~6.9 MB | 0.035ms | 1.5ms | gate+up+down Q8_0 |
| MoE (3 experts, EP) | ~40 MB | 0.2ms | **8.6ms** | Half of full MoE |
| MoE all-reduce | 16 KB | ~0.1ms | **4.3ms** | NCCL on 200G |
| HC post-FFN | ~0.5 MB | 0.003ms | 0.1ms | |
| Kernel launch overhead | - | ~0.15ms | **6.5ms** | ~15 launches/layer × 10μs |
| **Total estimated** | | | **~29ms** | |
| **Measured** | | | **~37ms** | Gap = cuBLAS overhead, sync |

### Key Takeaways
1. **MoE expert loading (8.6ms)** and **MoE all-reduce (4.3ms)** = 13ms = 35% of token time
2. **Q_B weight loading (4ms)** is the single largest non-MoE weight
3. **Kernel launch overhead (~6.5ms)** is a hidden tax — CUDA graphs would eliminate it
4. Adding per-layer all-reduces costs ~0.1ms × 43 = 4.3ms. Only worth it if savings > 4.3ms

## Optimization Roadmap: Path to 35-40 t/s

### Step 0: Profile (no code changes)

```bash
DS4_METAL_DECODE_STAGE_PROFILE=1 NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  ./ds4 --tp 2 --tp-rank 0 --host 0.0.0.0 --port 8000 [model + prompt args]
```

Confirms real per-stage numbers. Look for which stages dominate.

### Step 1: Custom 2-Node All-Reduce (est. +1-2 t/s)

Replace NCCL with direct cudaMemcpy→TCP→sum for 16KB payloads.
- NCCL: ~100μs per call → Custom: ~25μs per call
- Savings: 75μs × 43 = 3.2ms/token

### Step 2: Q Head Sharding (est. +1-2 t/s, needs Step 1)

Split Q_B 1024→32768 by heads. Each rank computes 32 heads.
- Saves ~2ms Q_B loading
- Costs 43 × per-call allreduce for attention output (need Step 1's low-overhead allreduce)

### Step 3: CUDA Graph Capture (est. +3-5 t/s) — HIGHEST IMPACT

Record matmul-heavy decode path as CUDA graph, replay as single launch.
- Eliminates ~6.5ms kernel launch overhead
- Challenge: dynamic attention shapes, NCCL not graph-capturable
- Approach: partial capture (just matmuls), leave attention + NCCL outside

### Realistic Projection

| Optimization | Cumulative t/s | Delta |
|-------------|---------------|-------|
| Current (EP + Phase 3 fixes) | 27 t/s | baseline |
| + Custom all-reduce (Step 1) | ~29 t/s | -3ms |
| + Q head sharding (Step 2) | ~31 t/s | -2ms |
| + CUDA graph capture (Step 3) | **~36 t/s** | -5ms |

**35-40 t/s requires at minimum Steps 1+3, likely all three.**

## Known Issues

- `g_tp_worker_session` set but never used (warning in ds4_tp.cu)
- "CUDA host registration skipped" on GB10 — benign
- No fault tolerance if one node drops
- TP > 2 not tested (infrastructure supports it)

## Usage

```bash
# Server (rank 0, hades) — use 200G NIC:
NCCL_SOCKET_IFNAME=enp1s0f1np1 ./ds4-server --tp 2 --tp-rank 0 --host 0.0.0.0 --port 8000

# Server (rank 1, hephaestus) — use 200G NIC:
NCCL_SOCKET_IFNAME=enp1s0f1np1 ./ds4-server --tp 2 --tp-rank 1 --tp-master 192.168.0.221:8000

# Profile decode stages:
DS4_METAL_DECODE_STAGE_PROFILE=1 NCCL_SOCKET_IFNAME=enp1s0f1np1 ./ds4 ...
```


## Files Changed

```
ds4_tp.h          - TP API header (init, cleanup, worker, broadcast, all-reduce, EP)
ds4_tp.cu         - NCCL TP implementation (rendezvous, control channel, worker thread, all-reduce, EP)
ds4_cuda.cu       - Expert parallelism: zero non-local expert counts in MoE kernel
ds4_gpu.h         - Added extern "C" guards
ds4.h             - Added TP fields to engine options (tp_size, tp_rank, tp_master_addr, tp_host, tp_port)
ds4.c             - TP init/cleanup, MoE-only all-reduce (attention/shared removed), EP-aware
ds4_cli.c         - Added TP CLI flags
ds4_bench.c       - Added TP flags (--tp, --tp-rank, --tp-master)
ds4_server.c      - TP server flags, rank 1 worker, broadcast calls, skip HTTP on rank 1, prefill ordering
Makefile          - Added TP build rules, NCCL rpath
```