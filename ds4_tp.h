#ifndef DS4_TP_H
#define DS4_TP_H

#include <stdbool.h>
#include <stdint.h>
#include "ds4_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration. */
struct ds4_session;

/* =========================================================================
 * Tensor Parallelism (TP) via NCCL.
 *
 * Rank 0 handles HTTP, broadcasts work to rank 1 via a persistent TCP
 * control channel. Both ranks run the same inference code; NCCL all-reduce
 * at three fusion boundaries (attention output, routed MoE, shared expert)
 * ensures correct results.
 * ========================================================================= */

bool ds4_tp_init(int tp_size, int tp_rank, const char *master_addr,
                 const char *local_host, int local_port);
void ds4_tp_cleanup(void);
bool ds4_tp_enabled(void);
int ds4_tp_size(void);
int ds4_tp_rank(void);

/* Rank 1: start worker thread that listens for commands from rank 0. */
int ds4_tp_worker_init(struct ds4_session *session);

/* Rank 0: broadcast commands to rank 1. Block until ack received. */
int ds4_tp_broadcast_prefill(const int *tokens, int n_tokens);
int ds4_tp_broadcast_eval(int token);
int ds4_tp_broadcast_eval_batch(const int *tokens, int n_tokens);
int ds4_tp_broadcast_done(void);

/* All-reduce on GPU tensors. */
int ds4_tp_allreduce_f16(ds4_gpu_tensor *tensor);
int ds4_tp_allreduce_f32(ds4_gpu_tensor *tensor);

void *ds4_tp_nccl_comm(void);
void *ds4_tp_cuda_stream(void);

#ifdef __cplusplus
}
#endif

#endif
