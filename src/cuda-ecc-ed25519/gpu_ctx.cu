#include "ed25519.h"
#include "gpu_ctx.h"
#include <pthread.h>
#include "gpu_common.h"

static pthread_mutex_t g_ctx_mutex = PTHREAD_MUTEX_INITIALIZER;

#define MAX_NUM_GPUS 8
#define MAX_QUEUE_SIZE 8

static gpu_ctx_t g_gpu_ctx[MAX_NUM_GPUS][MAX_QUEUE_SIZE] = {0};
static uint32_t g_cur_gpu = 0;
static uint32_t g_cur_queue[MAX_NUM_GPUS] = {0};
static int32_t g_total_gpus = -1;

static bool cuda_crypt_init_locked() {
    if (g_total_gpus == -1) {
        cudaGetDeviceCount(&g_total_gpus);
        g_total_gpus = min(MAX_NUM_GPUS, g_total_gpus);
        LOG("total_gpus: %d\n", g_total_gpus);
        for (int gpu = 0; gpu < g_total_gpus; gpu++) {
            CUDA_CHK(cudaSetDevice(gpu));
            for (int queue = 0; queue < MAX_QUEUE_SIZE; queue++) {
                int err = pthread_mutex_init(&g_gpu_ctx[gpu][queue].mutex, NULL);
                if (err != 0) {
                    fprintf(stderr, "pthread_mutex_init error %d gpu: %d queue: %d\n",
                            err, gpu, queue);
                    g_total_gpus = 0;
                    return false;
                }
                CUDA_CHK(cudaStreamCreate(&g_gpu_ctx[gpu][queue].stream));
            }
        }
    }
    return g_total_gpus > 0;
}

bool ed25519_init() {
    cudaFree(0);
    pthread_mutex_lock(&g_ctx_mutex);
    bool success = cuda_crypt_init_locked();
    pthread_mutex_unlock(&g_ctx_mutex);
    return success;
}

gpu_ctx_t* get_gpu_ctx() {
    int32_t cur_gpu, cur_queue;

    LOG("locking global mutex");
    pthread_mutex_lock(&g_ctx_mutex);
    if (!cuda_crypt_init_locked()) {
        pthread_mutex_unlock(&g_ctx_mutex);
        LOG("No GPUs, exiting...\n");
        return NULL;
    }
    cur_gpu = g_cur_gpu;
    g_cur_gpu++;
    g_cur_gpu %= g_total_gpus;
    cur_queue = g_cur_queue[cur_gpu];
    g_cur_queue[cur_gpu]++;
    g_cur_queue[cur_gpu] %= MAX_QUEUE_SIZE;
    pthread_mutex_unlock(&g_ctx_mutex);

    gpu_ctx_t* cur_ctx = &g_gpu_ctx[cur_gpu][cur_queue];
    LOG("locking contex mutex queue: %d gpu: %d", cur_queue, cur_gpu);
    pthread_mutex_lock(&cur_ctx->mutex);

    CUDA_CHK(cudaSetDevice(cur_gpu));

    LOG("selecting gpu: %d queue: %d\n", cur_gpu, cur_queue);

    return cur_ctx;
}

void setup_gpu_ctx(verify_ctx_t* cur_ctx,
                   const gpu_Elems* elems,
                   uint32_t num_elems,
                   uint32_t message_size,
                   uint32_t total_packets,
                   uint32_t total_packets_size,
                   uint32_t total_signatures,
                   const uint32_t* message_lens,
                   const uint32_t* public_key_offsets,
                   const uint32_t* signature_offsets,
                   const uint32_t* message_start_offsets,
                   size_t out_size,
                   cudaStream_t stream
                   ) {
    size_t offsets_size = total_signatures * sizeof(uint32_t);

    LOG("device allocate. packets: %d out: %d offsets_size: %zu\n",
        total_packets_size, (int)out_size, offsets_size);

    if (cur_ctx->packets == NULL ||
        total_packets_size > cur_ctx->packets_size_bytes) {
        CUDA_CHK(cudaFree(cur_ctx->packets));
        CUDA_CHK(cudaMalloc(&cur_ctx->packets, total_packets_size));

        cur_ctx->packets_size_bytes = total_packets_size;
    }

    if (cur_ctx->out == NULL || cur_ctx->out_size_bytes < out_size) {
        CUDA_CHK(cudaFree(cur_ctx->out));
        CUDA_CHK(cudaMalloc(&cur_ctx->out, out_size));

        cur_ctx->out_size_bytes = total_signatures;
    }

    if (cur_ctx->public_key_offsets == NULL || cur_ctx->offsets_len < total_signatures) {
        CUDA_CHK(cudaFree(cur_ctx->public_key_offsets));
        CUDA_CHK(cudaMalloc(&cur_ctx->public_key_offsets, offsets_size));

        CUDA_CHK(cudaFree(cur_ctx->signature_offsets));
        CUDA_CHK(cudaMalloc(&cur_ctx->signature_offsets, offsets_size));

        CUDA_CHK(cudaFree(cur_ctx->message_start_offsets));
        CUDA_CHK(cudaMalloc(&cur_ctx->message_start_offsets, offsets_size));

        CUDA_CHK(cudaFree(cur_ctx->message_lens));
        CUDA_CHK(cudaMalloc(&cur_ctx->message_lens, offsets_size));

        cur_ctx->offsets_len = total_signatures;
    }

    LOG("Done alloc");

    CUDA_CHK(cudaMemcpyAsync(cur_ctx->public_key_offsets, public_key_offsets, offsets_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(cur_ctx->signature_offsets, signature_offsets, offsets_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(cur_ctx->message_start_offsets, message_start_offsets, offsets_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(cur_ctx->message_lens, message_lens, offsets_size, cudaMemcpyHostToDevice, stream));

    size_t cur = 0;
    for (size_t i = 0; i < num_elems; i++) {
        LOG("i: %zu size: %d\n", i, elems[i].num * message_size);
        CUDA_CHK(cudaMemcpyAsync(&cur_ctx->packets[cur * message_size], elems[i].elems, elems[i].num * message_size, cudaMemcpyHostToDevice, stream));
        cur += elems[i].num;
    }
}


void release_gpu_ctx(gpu_ctx_t* cur_ctx) {
    pthread_mutex_unlock(&cur_ctx->mutex);
}

void ed25519_free_gpu_mem() {
    for (size_t gpu = 0; gpu < MAX_NUM_GPUS; gpu++) {
        for (size_t queue = 0; queue < MAX_QUEUE_SIZE; queue++) {
            gpu_ctx_t* cur_ctx = &g_gpu_ctx[gpu][queue];
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.packets));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.out));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.message_lens));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.public_key_offsets));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.private_key_offsets));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.signature_offsets));
            CUDA_CHK(cudaFree(cur_ctx->verify_ctx.message_start_offsets));
            if (cur_ctx->stream != 0) {
                CUDA_CHK(cudaStreamDestroy(cur_ctx->stream));
            }
        }
    }
}
