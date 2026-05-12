# Tensor Parallelism Implementation Plan

## What Was Done

### New Files
- **ds4_tp.h** - Public TP API header with extern "C" guards
- **ds4_tp.cu** - NCCL-based TP implementation

### Modified Files
- **ds4_gpu.h** - Added extern "C" guards for C++ interop
- **ds4.h** - Added tp_size, tp_rank, tp_master_addr to ds4_engine_options
- **ds4.c** - TP init/cleanup wiring + 6 all-reduce insertion points (3 decode + 3 prefill)
- **ds4_cli.c** - Added --tp, --tp-rank, --tp-master CLI flags
- **ds4_server.c** - Same TP flags for server
- **Makefile** - Added ds4_tp.o to CORE_OBJS, NCCL linking

### Architecture
- **NCCL** for collectives (already built at /home/tim/code/nccl/build/)
- **TCP rendezvous** for multi-node: rank 0 listens on --tp-master HOST:PORT, peers connect
- **Non-blocking CUDA stream** for collectives to avoid compute blocking
- **3 all-reduces per layer**: after attention output, routed MoE output, shared expert output
- **Phase 1 approach**: replicated weights, only activations all-reduced

### Key Design Decisions
- All-reduce is a no-op when tp_size <= 1 (current default)
- TP only works with CUDA backend (checked in ds4_engine_open)
- Prefill path uses tensor views for batch-sized all-reduce
- Decode path uses direct tensor all-reduce (single row)

## What's Left

### Phase 2: Weight Sharding (TODO)
- Shard Q projection weights (column-shard)
- Shard V projection weights (column-shard)
- Shard MoE down weights (column-shard)
- Shard shared expert down weights (column-shard)
- Update matmul kernels for sharded dimensions

### Phase 3: KV Cache Sharding (TODO)
- Shard raw KV cache V dimension
- Keep K replicated (small: 512 dim)
- Compressed KV stays replicated (512 dim, not worth sharding)

### Phase 4: Testing & Optimization
- Verify correctness with test vectors
- Benchmark communication overhead
- Optimize all-reduce placement

## Testing Plan

1. **TP=1 sanity check**: Run existing tests with --tp 1 (should be no-op)
2. **TP=2 single-node**: Test with 2 GPUs on same DGX Spark (if available)
3. **TP=2 multi-node**: Test across 2 DGX Sparks with --tp-master
4. **Correctness**: Compare logits with single-GPU run
5. **Performance**: Measure speedup vs. single GPU

## Known Issues

- NCCL rendezvous uses raw TCP sockets (works but not ideal for production)
- No fault tolerance if one node drops
- All-reduce happens on every layer (could be optimized)
- No support for TP > 2 yet (but infrastructure supports it)

## Usage

```bash
# Rank 0 (master node):
./ds4 --cuda --tp 2 --tp-rank 0 --tp-master <node0-ip>:12345 -p "Hello"

# Rank 1 (peer node):
./ds4 --cuda --tp 2 --tp-rank 1 --tp-master <node0-ip>:12345 -p "Hello"
```

## Files Changed

```
ds4_tp.h          - New TP API header (52 lines)
ds4_tp.cu         - New NCCL TP implementation (313 lines)
ds4_gpu.h         - Added extern "C" guards
ds4.h             - Added TP fields to engine options
ds4.c             - TP init/cleanup + 6 all-reduce points
ds4_cli.c         - Added TP CLI flags
ds4_server.c      - Added TP server flags
Makefile          - Added TP build rules
```
