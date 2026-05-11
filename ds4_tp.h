#ifndef DS4_TP_H
#define DS4_TP_H

#include <stdbool.h>
#include <stdint.h>
#include "ds4_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Tensor Parallelism (TP) via NCCL.
 *
 * Provides all-reduce over f16/f32 device tensors across TP ranks.
 * Weights are still fully replicated on each rank; only activations are
 * all-reduced at the three fusion boundaries (attention output, routed MoE,
 * shared expert). This is the Phase-1 "replicated weights" approach.
 * ========================================================================= */

/* Initialize TP. Returns true on success. Call before any GPU init.
 * tp_size=1 is a no-op (single GPU, TP disabled). */
bool ds4_tp_init(int tp_size, int tp_rank, const char *master_addr);

/* Tear down TP. Call during engine cleanup. */
void ds4_tp_cleanup(void);

/* Is TP active (size > 1)? */
bool ds4_tp_enabled(void);

/* Current TP size and rank. */
int ds4_tp_size(void);
int ds4_tp_rank(void);

/* All-reduce f16 tensor in-place. The tensor must have the same size on
 * every rank. Synchronizes with the CUDA stream used by the GPU executor. */
int ds4_tp_allreduce_f16(ds4_gpu_tensor *tensor);

/* All-reduce f32 tensor in-place. */
int ds4_tp_allreduce_f32(ds4_gpu_tensor *tensor);

/* Get the NCCL comm handle (opaque, for advanced use). */
void *ds4_tp_nccl_comm(void);

/* Get the CUDA stream used for TP collectives. */
void *ds4_tp_cuda_stream(void);

#ifdef __cplusplus
}
#endif

#endif
