#include "ds4_tp.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/* NCCL */
#include "nccl.h"

/* All TP functions are C-linkage so ds4.c can call them. */
extern "C" {

/* =========================================================================
 * TP Global State.
 * ========================================================================= */

static int g_tp_size = 1;
static int g_tp_rank = 0;
static int g_tp_enabled = 0;

static ncclComm_t g_nccl_comm = NULL;
static cudaStream_t g_tp_stream = NULL;

/* =========================================================================
 * NCCL Helpers.
 * ========================================================================= */

static int nccl_ok(ncclResult_t res, const char *msg) {
    if (res == ncclSuccess) return 0;
    fprintf(stderr, "ds4-tp: NCCL error (%s): %s\n", msg, ncclGetErrorString(res));
    return 1;
}

/* =========================================================================
 * Rendezvous via TCP socket (for multi-node).
 *
 * Rank 0 calls ncclGetUniqueId, listens on master_addr, sends the id to
 * each connecting rank.  Other ranks connect to master_addr and receive
 * the id.  All ranks then call ncclCommInitRank.
 * ========================================================================= */

static int rendezvous_exchange(int tp_size, int tp_rank,
                               const char *master_addr,
                               ncclUniqueId *out_id) {
    if (tp_rank == 0) {
        /* Rank 0: generate ID and broadcast to peers. */
        ncclResult_t res = ncclGetUniqueId(out_id);
        if (res != ncclSuccess) {
            nccl_ok(res, "ncclGetUniqueId");
            return 1;
        }

        if (master_addr && strlen(master_addr) > 0) {
            /* Multi-node: listen and send ID to each peer. */
            char host[256] = {0};
            int port = 0;
            if (sscanf(master_addr, "%255[^:]:%d", host, &port) != 2) {
                fprintf(stderr, "ds4-tp: invalid master address: %s (expected HOST:PORT)\n",
                        master_addr);
                return 1;
            }

            int listenfd = socket(AF_INET, SOCK_STREAM, 0);
            if (listenfd < 0) {
                perror("ds4-tp: socket");
                return 1;
            }
            int optval = 1;
            setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
                fprintf(stderr, "ds4-tp: invalid host address: %s\n", host);
                close(listenfd);
                return 1;
            }
            if (bind(listenfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
                perror("ds4-tp: bind");
                close(listenfd);
                return 1;
            }
            if (listen(listenfd, tp_size) < 0) {
                perror("ds4-tp: listen");
                close(listenfd);
                return 1;
            }
            fprintf(stderr, "ds4-tp: rank 0 listening on %s:%d for %d peers\n",
                    host, port, tp_size - 1);

            for (int p = 1; p < tp_size; p++) {
                struct sockaddr_in peer;
                socklen_t peerlen = sizeof(peer);
                int connfd = accept(listenfd, (struct sockaddr *)&peer, &peerlen);
                if (connfd < 0) {
                    perror("ds4-tp: accept");
                    close(listenfd);
                    return 1;
                }
                ssize_t sent = send(connfd, out_id, sizeof(ncclUniqueId), 0);
                if (sent != sizeof(ncclUniqueId)) {
                    fprintf(stderr, "ds4-tp: failed to send NCCL id to peer %d\n", p);
                    close(connfd);
                    close(listenfd);
                    return 1;
                }
                close(connfd);
            }
            close(listenfd);
        }
        /* Single-node (no master_addr): ID is local, peers will get it
         * via shared memory or the caller must handle coordination. */
    } else {
        /* Peer rank: receive ID from rank 0. */
        if (master_addr && strlen(master_addr) > 0) {
            char host[256] = {0};
            int port = 0;
            if (sscanf(master_addr, "%255[^:]:%d", host, &port) != 2) {
                fprintf(stderr, "ds4-tp: invalid master address: %s\n", master_addr);
                return 1;
            }

            int connfd = socket(AF_INET, SOCK_STREAM, 0);
            if (connfd < 0) {
                perror("ds4-tp: socket");
                return 1;
            }

            struct sockaddr_in addr;
            memset(&addr, 0, sizeof(addr));
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
                fprintf(stderr, "ds4-tp: invalid host address: %s\n", host);
                close(connfd);
                return 1;
            }
            if (connect(connfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
                perror("ds4-tp: connect");
                close(connfd);
                return 1;
            }

            ssize_t nrecv = recv(connfd, out_id, sizeof(ncclUniqueId), 0);
            if (nrecv != sizeof(ncclUniqueId)) {
                fprintf(stderr, "ds4-tp: failed to receive NCCL id from rank 0\n");
                close(connfd);
                return 1;
            }
            close(connfd);
        } else {
            fprintf(stderr, "ds4-tp: rank %d needs --tp-master HOST:PORT for multi-node TP\n",
                    tp_rank);
            return 1;
        }
    }
    return 0;
}

/* =========================================================================
 * Init / Cleanup.
 * ========================================================================= */

bool ds4_tp_init(int tp_size, int tp_rank, const char *master_addr) {
    if (tp_size <= 1) {
        g_tp_size = 1;
        g_tp_rank = 0;
        g_tp_enabled = 0;
        return true;
    }

    if (tp_rank < 0 || tp_rank >= tp_size) {
        fprintf(stderr, "ds4-tp: invalid rank %d for size %d\n", tp_rank, tp_size);
        return false;
    }

    /* Create dedicated CUDA stream for collectives. */
    cudaError_t ce = cudaStreamCreateWithFlags(&g_tp_stream, cudaStreamNonBlocking);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4-tp: failed to create TP CUDA stream: %s\n",
                cudaGetErrorString(ce));
        return false;
    }

    /* Rendezvous: exchange NCCL unique ID. */
    ncclUniqueId comm_id;
    memset(&comm_id, 0, sizeof(comm_id));
    if (rendezvous_exchange(tp_size, tp_rank, master_addr, &comm_id)) {
        cudaStreamDestroy(g_tp_stream);
        g_tp_stream = NULL;
        return false;
    }

    /* Initialize NCCL communicator. */
    ncclResult_t res = ncclCommInitRank(&g_nccl_comm, tp_size, comm_id, tp_rank);
    if (res != ncclSuccess) {
        nccl_ok(res, "ncclCommInitRank");
        cudaStreamDestroy(g_tp_stream);
        g_tp_stream = NULL;
        return false;
    }

    g_tp_size = tp_size;
    g_tp_rank = tp_rank;
    g_tp_enabled = 1;

    fprintf(stderr, "ds4-tp: TP enabled: rank %d/%d (master=%s)\n",
            g_tp_rank, g_tp_size,
            master_addr ? master_addr : "(local)");
    return true;
}

void ds4_tp_cleanup(void) {
    if (!g_tp_enabled) return;

    if (g_nccl_comm) {
        ncclCommDestroy(g_nccl_comm);
        g_nccl_comm = NULL;
    }
    if (g_tp_stream) {
        cudaStreamDestroy(g_tp_stream);
        g_tp_stream = NULL;
    }
    g_tp_enabled = 0;
    g_tp_size = 1;
    g_tp_rank = 0;

    fprintf(stderr, "ds4-tp: TP cleaned up\n");
}

/* =========================================================================
 * Query Functions.
 * ========================================================================= */

bool ds4_tp_enabled(void) {
    return g_tp_enabled;
}

int ds4_tp_size(void) {
    return g_tp_size;
}

int ds4_tp_rank(void) {
    return g_tp_rank;
}

void *ds4_tp_nccl_comm(void) {
    return (void *)g_nccl_comm;
}

void *ds4_tp_cuda_stream(void) {
    return (void *)g_tp_stream;
}

/* =========================================================================
 * All-Reduce.
 * ========================================================================= */

int ds4_tp_allreduce_f16(ds4_gpu_tensor *tensor) {
    if (!g_tp_enabled || !tensor) return 0;

    __half *data = (__half *)ds4_gpu_tensor_contents(tensor);
    uint64_t bytes = ds4_gpu_tensor_bytes(tensor);
    uint32_t n_elements = (uint32_t)(bytes / sizeof(__half));

    ncclResult_t res = ncclAllReduce(
        data, data,
        n_elements,
        ncclFloat16,
        ncclSum,
        g_nccl_comm,
        g_tp_stream
    );
    if (res != ncclSuccess) {
        nccl_ok(res, "ncclAllReduce f16");
        return 1;
    }
    return 0;
}

int ds4_tp_allreduce_f32(ds4_gpu_tensor *tensor) {
    if (!g_tp_enabled || !tensor) return 0;

    float *data = (float *)ds4_gpu_tensor_contents(tensor);
    uint64_t bytes = ds4_gpu_tensor_bytes(tensor);
    uint32_t n_elements = (uint32_t)(bytes / sizeof(float));

    ncclResult_t res = ncclAllReduce(
        data, data,
        n_elements,
        ncclFloat32,
        ncclSum,
        g_nccl_comm,
        g_tp_stream
    );
    if (res != ncclSuccess) {
        nccl_ok(res, "ncclAllReduce f32");
        return 1;
    }
    return 0;
}

} /* extern "C" */
