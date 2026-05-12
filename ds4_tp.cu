#include "ds4_tp.h"
#include "ds4.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
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

/* Persistent control socket between rank 0 and rank 1.
 * Rank 0 sends commands, rank 1 sends acks. */
static int g_tp_control_sock = -1;

/* Worker thread state (rank 1 only). */
static pthread_t g_tp_worker_thread;
static int g_tp_worker_running = 0;
static ds4_session *g_tp_worker_session = NULL;

/* Control message types. */
#define TP_CTL_PREFILL       1
#define TP_CTL_EVAL          2
#define TP_CTL_EVAL_BATCH    5
#define TP_CTL_DONE          3
#define TP_CTL_SHUTDOWN      4

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
 * each connecting rank. The connection is kept open as the control channel.
 * Other ranks connect to master_addr and receive the id.
 * All ranks then call ncclCommInitRank.
 * ========================================================================= */

static int rendezvous_exchange(int tp_size, int tp_rank,
                               const char *master_addr,
                               ncclUniqueId *out_id,
                               int *control_sock_out) {
    *control_sock_out = -1;

    if (tp_rank == 0) {
        ncclResult_t res = ncclGetUniqueId(out_id);
        if (res != ncclSuccess) {
            nccl_ok(res, "ncclGetUniqueId");
            return 1;
        }

        if (master_addr && strlen(master_addr) > 0) {
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

            /* Accept first peer as control channel. */
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
                fprintf(stderr, "ds4-tp: failed to send NCCL id to peer\n");
                close(connfd);
                close(listenfd);
                return 1;
            }
            *control_sock_out = connfd;
            fprintf(stderr, "ds4-tp: control channel established to peer\n");
        }
    } else {
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
            *control_sock_out = connfd;
            fprintf(stderr, "ds4-tp: control channel established to rank 0\n");
        } else {
            fprintf(stderr, "ds4-tp: rank %d needs --tp-master HOST:PORT for multi-node TP\n",
                    tp_rank);
            return 1;
        }
    }
    return 0;
}

/* =========================================================================
 * Control Protocol Helpers.
 * ========================================================================= */

static int tp_ctl_send_all(int fd, const void *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = send(fd, (const char *)buf + total, len - total, 0);
        if (n <= 0) return -1;
        total += n;
    }
    return 0;
}

static int tp_ctl_recv_all(int fd, void *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = recv(fd, (char *)buf + total, len - total, MSG_WAITALL);
        if (n <= 0) return -1;
        total += n;
    }
    return 0;
}

static int tp_ctl_send_ack(int fd, int ok) {
    uint8_t ack = (uint8_t)(ok ? 1 : 0);
    return tp_ctl_send_all(fd, &ack, 1);
}

static int tp_ctl_recv_ack(int fd) {
    uint8_t ack = 0;
    if (tp_ctl_recv_all(fd, &ack, 1) < 0) return -1;
    return ack ? 0 : -1;
}

/* =========================================================================
 * TP Worker Thread (rank 1).
 * ========================================================================= */

static void *tp_worker_thread(void *arg) {
    ds4_session *session = (ds4_session *)arg;
    char err[256];
    int sock = g_tp_control_sock;

    g_tp_worker_running = 1;
    fprintf(stderr, "ds4-tp: worker thread started (rank %d)\n", g_tp_rank);

    for (;;) {
        uint8_t type = 0;
        if (tp_ctl_recv_all(sock, &type, 1) < 0) {
            fprintf(stderr, "ds4-tp: worker: control socket closed or error\n");
            break;
        }

        int ok = 1;
        switch (type) {
        case TP_CTL_PREFILL: {
            uint32_t n_tokens = 0;
            if (tp_ctl_recv_all(sock, &n_tokens, sizeof(n_tokens)) < 0) {
                fprintf(stderr, "ds4-tp: worker: failed to read prefill token count\n");
                tp_ctl_send_ack(sock, 0);
                break;
            }
            int *tokens = (int *)malloc(n_tokens * sizeof(int));
            if (!tokens) {
                fprintf(stderr, "ds4-tp: worker: malloc failed\n");
                tp_ctl_send_ack(sock, 0);
                break;
            }
            if (tp_ctl_recv_all(sock, tokens, n_tokens * sizeof(int)) < 0) {
                fprintf(stderr, "ds4-tp: worker: failed to read prefill tokens\n");
                free(tokens);
                tp_ctl_send_ack(sock, 0);
                break;
            }

            ds4_tokens prompt;
            prompt.v = tokens;
            prompt.len = (int)n_tokens;
            prompt.cap = (int)n_tokens;

            if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-tp: worker: prefill failed: %s\n", err);
                ok = 0;
            }
            free(tokens);
            break;
        }
        case TP_CTL_EVAL: {
            int token = 0;
            if (tp_ctl_recv_all(sock, &token, sizeof(int)) < 0) {
                fprintf(stderr, "ds4-tp: worker: failed to read eval token\n");
                tp_ctl_send_ack(sock, 0);
                break;
            }
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-tp: worker: eval failed: %s\n", err);
                ok = 0;
            }
            break;
        }
        case TP_CTL_EVAL_BATCH: {
            uint32_t n_tokens = 0;
            if (tp_ctl_recv_all(sock, &n_tokens, sizeof(n_tokens)) < 0) {
                fprintf(stderr, "ds4-tp: worker: failed to read eval batch count\n");
                tp_ctl_send_ack(sock, 0);
                break;
            }
            int *tokens = (int *)malloc(n_tokens * sizeof(int));
            if (!tokens) {
                fprintf(stderr, "ds4-tp: worker: malloc failed\n");
                tp_ctl_send_ack(sock, 0);
                break;
            }
            if (tp_ctl_recv_all(sock, tokens, n_tokens * sizeof(int)) < 0) {
                fprintf(stderr, "ds4-tp: worker: failed to read eval batch tokens\n");
                free(tokens);
                tp_ctl_send_ack(sock, 0);
                break;
            }
            for (uint32_t i = 0; i < n_tokens; i++) {
                if (ds4_session_eval(session, tokens[i], err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-tp: worker: eval[%u] failed: %s\n", i, err);
                    ok = 0;
                    break;
                }
            }
            free(tokens);
            break;
        }
        case TP_CTL_DONE:
            ds4_session_invalidate(session);
            break;
        case TP_CTL_SHUTDOWN:
            tp_ctl_send_ack(sock, 1);
            g_tp_worker_running = 0;
            fprintf(stderr, "ds4-tp: worker thread shutting down\n");
            return NULL;
        default:
            fprintf(stderr, "ds4-tp: worker: unknown command type %d\n", type);
            ok = 0;
            break;
        }

        if (type != TP_CTL_SHUTDOWN) {
            tp_ctl_send_ack(sock, ok);
        }
    }

    g_tp_worker_running = 0;
    fprintf(stderr, "ds4-tp: worker thread exiting\n");
    return NULL;
}

/* =========================================================================
 * Broadcast Functions (rank 0).
 * ========================================================================= */

int ds4_tp_broadcast_prefill(const int *tokens, int n_tokens) {
    if (!g_tp_enabled || g_tp_rank != 0 || g_tp_control_sock < 0) return 0;

    uint8_t type = TP_CTL_PREFILL;
    if (tp_ctl_send_all(g_tp_control_sock, &type, 1) < 0) {
        fprintf(stderr, "ds4-tp: failed to send prefill command\n");
        return 1;
    }
    uint32_t n = (uint32_t)n_tokens;
    if (tp_ctl_send_all(g_tp_control_sock, &n, sizeof(n)) < 0) {
        fprintf(stderr, "ds4-tp: failed to send prefill count\n");
        return 1;
    }
    if (tp_ctl_send_all(g_tp_control_sock, tokens, n_tokens * sizeof(int)) < 0) {
        fprintf(stderr, "ds4-tp: failed to send prefill tokens\n");
        return 1;
    }
    if (tp_ctl_recv_ack(g_tp_control_sock) < 0) {
        fprintf(stderr, "ds4-tp: prefill ack failed\n");
        return 1;
    }
    return 0;
}

int ds4_tp_broadcast_eval(int token) {
    if (!g_tp_enabled || g_tp_rank != 0 || g_tp_control_sock < 0) return 0;

    uint8_t type = TP_CTL_EVAL;
    if (tp_ctl_send_all(g_tp_control_sock, &type, 1) < 0) {
        fprintf(stderr, "ds4-tp: failed to send eval command\n");
        return 1;
    }
    if (tp_ctl_send_all(g_tp_control_sock, &token, sizeof(int)) < 0) {
        fprintf(stderr, "ds4-tp: failed to send eval token\n");
        return 1;
    }
    if (tp_ctl_recv_ack(g_tp_control_sock) < 0) {
        fprintf(stderr, "ds4-tp: eval ack failed\n");
        return 1;
    }
    return 0;
}

int ds4_tp_broadcast_eval_batch(const int *tokens, int n_tokens) {
    if (!g_tp_enabled || g_tp_rank != 0 || g_tp_control_sock < 0) return 0;

    uint8_t type = TP_CTL_EVAL_BATCH;
    if (tp_ctl_send_all(g_tp_control_sock, &type, 1) < 0) {
        fprintf(stderr, "ds4-tp: failed to send eval batch command\n");
        return 1;
    }
    uint32_t n = (uint32_t)n_tokens;
    if (tp_ctl_send_all(g_tp_control_sock, &n, sizeof(n)) < 0) {
        fprintf(stderr, "ds4-tp: failed to send eval batch count\n");
        return 1;
    }
    if (tp_ctl_send_all(g_tp_control_sock, tokens, n_tokens * sizeof(int)) < 0) {
        fprintf(stderr, "ds4-tp: failed to send eval batch tokens\n");
        return 1;
    }
    if (tp_ctl_recv_ack(g_tp_control_sock) < 0) {
        fprintf(stderr, "ds4-tp: eval batch ack failed\n");
        return 1;
    }
    return 0;
}

int ds4_tp_broadcast_done(void) {
    if (!g_tp_enabled || g_tp_rank != 0 || g_tp_control_sock < 0) return 0;

    uint8_t type = TP_CTL_DONE;
    if (tp_ctl_send_all(g_tp_control_sock, &type, 1) < 0) {
        fprintf(stderr, "ds4-tp: failed to send done command\n");
        return 1;
    }
    if (tp_ctl_recv_ack(g_tp_control_sock) < 0) {
        fprintf(stderr, "ds4-tp: done ack failed\n");
        return 1;
    }
    return 0;
}

/* =========================================================================
 * Init / Cleanup.
 * ========================================================================= */

bool ds4_tp_init(int tp_size, int tp_rank, const char *master_addr,
                 const char *local_host, int local_port) {
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

    /* Build the rendezvous address.
     * Rank 0: listens on local_host:local_port (from --host/--port).
     * Rank 1: connects to master_addr (--tp-master). */
    char bind_addr[256] = {0};
    if (tp_rank == 0) {
        snprintf(bind_addr, sizeof(bind_addr), "%s:%d",
                 local_host ? local_host : "0.0.0.0",
                 local_port > 0 ? local_port : 12345);
    } else if (master_addr && strlen(master_addr) > 0) {
        snprintf(bind_addr, sizeof(bind_addr), "%s", master_addr);
    } else {
        fprintf(stderr, "ds4-tp: rank %d needs --tp-master HOST:PORT\n", tp_rank);
        return false;
    }

    cudaError_t ce = cudaStreamCreateWithFlags(&g_tp_stream, cudaStreamNonBlocking);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4-tp: failed to create TP CUDA stream: %s\n",
                cudaGetErrorString(ce));
        return false;
    }

    ncclUniqueId comm_id;
    memset(&comm_id, 0, sizeof(comm_id));
    int control_sock = -1;
    if (rendezvous_exchange(tp_size, tp_rank, bind_addr, &comm_id, &control_sock)) {
        cudaStreamDestroy(g_tp_stream);
        g_tp_stream = NULL;
        return false;
    }

    ncclResult_t res = ncclCommInitRank(&g_nccl_comm, tp_size, comm_id, tp_rank);
    if (res != ncclSuccess) {
        nccl_ok(res, "ncclCommInitRank");
        if (control_sock >= 0) close(control_sock);
        cudaStreamDestroy(g_tp_stream);
        g_tp_stream = NULL;
        return false;
    }

    g_tp_control_sock = control_sock;
    g_tp_size = tp_size;
    g_tp_rank = tp_rank;
    g_tp_enabled = 1;

    if (tp_rank == 0) {
        fprintf(stderr, "ds4-tp: TP enabled: rank %d/%d listening on %s\n",
                g_tp_rank, g_tp_size, bind_addr);
    } else {
        fprintf(stderr, "ds4-tp: TP enabled: rank %d/%d connected to %s\n",
                g_tp_rank, g_tp_size, bind_addr);
    }
    return true;
}

void ds4_tp_cleanup(void) {
    if (!g_tp_enabled) return;

    /* Shutdown worker thread on rank 1. */
    if (g_tp_rank != 0 && g_tp_worker_running && g_tp_control_sock >= 0) {
        uint8_t type = TP_CTL_SHUTDOWN;
        tp_ctl_send_all(g_tp_control_sock, &type, 1);
        pthread_join(g_tp_worker_thread, NULL);
    }
    if (g_tp_rank == 0 && g_tp_control_sock >= 0) {
        uint8_t type = TP_CTL_SHUTDOWN;
        tp_ctl_send_all(g_tp_control_sock, &type, 1);
        tp_ctl_recv_ack(g_tp_control_sock);
    }

    if (g_tp_control_sock >= 0) {
        close(g_tp_control_sock);
        g_tp_control_sock = -1;
    }
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
    g_tp_worker_running = 0;
    g_tp_worker_session = NULL;

    fprintf(stderr, "ds4-tp: TP cleaned up\n");
}

/* =========================================================================
 * Worker Init (rank 1).
 * ========================================================================= */

int ds4_tp_worker_init(ds4_session *session) {
    if (!g_tp_enabled || g_tp_rank == 0) return 0;

    g_tp_worker_session = session;
    if (pthread_create(&g_tp_worker_thread, NULL, tp_worker_thread, session) != 0) {
        fprintf(stderr, "ds4-tp: failed to create worker thread\n");
        return 1;
    }
    /* Wait for worker to signal it's running. */
    while (!g_tp_worker_running) {
        usleep(1000);
    }
    fprintf(stderr, "ds4-tp: worker thread ready (rank %d)\n", g_tp_rank);
    return 0;
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
