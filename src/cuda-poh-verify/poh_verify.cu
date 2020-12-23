#include <stddef.h>
#include <inttypes.h>
#include <pthread.h>
#include "gpu_common.h"
#include "sha256.cu"

#define MAX_NUM_GPUS 8
#define MAX_QUEUE_SIZE 8
#define NUM_THREADS_PER_BLOCK 64


__global__ void poh_verify_kernel(uint8_t* hashes, uint64_t* num_hashes_arr, size_t num_elems) {
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= num_elems) return;

    uint8_t hash[SHA256_BLOCK_SIZE];

    memcpy(hash, &hashes[idx * SHA256_BLOCK_SIZE], SHA256_BLOCK_SIZE);

    for (size_t i = 0; i < num_hashes_arr[idx]; i++) {
        hash_state sha_state;
        sha256_init(&sha_state);
        sha256_process(&sha_state, hash, SHA256_BLOCK_SIZE);
        sha256_done(&sha_state, hash);
    }

    memcpy(&hashes[idx * SHA256_BLOCK_SIZE], hash, SHA256_BLOCK_SIZE);
}

typedef struct {
    uint8_t* hashes;
    uint64_t* num_hashes_arr;
    size_t num_elems_alloc;
    pthread_mutex_t mutex;
    cudaStream_t stream;
} gpu_ctx;

static pthread_mutex_t g_ctx_mutex = PTHREAD_MUTEX_INITIALIZER;

static gpu_ctx g_gpu_ctx[MAX_NUM_GPUS][MAX_QUEUE_SIZE] = {0};
static uint32_t g_cur_gpu = 0;
static uint32_t g_cur_queue[MAX_NUM_GPUS] = {0};
static int32_t g_total_gpus = -1;

static bool poh_init_locked() {
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

bool poh_init() {
    cudaFree(0);
    pthread_mutex_lock(&g_ctx_mutex);
    bool success = poh_init_locked();
    pthread_mutex_unlock(&g_ctx_mutex);
    return success;
}

extern "C" {
int poh_verify_many(uint8_t* hashes,
                    const uint64_t* num_hashes_arr,
                    size_t num_elems,
                    uint8_t use_non_default_stream)
{
    LOG("Starting poh_verify_many: num_elems: %zu\n", num_elems);

    if (num_elems == 0) return 0;

    int32_t cur_gpu, cur_queue;

    pthread_mutex_lock(&g_ctx_mutex);
    if (!poh_init_locked()) {
        pthread_mutex_unlock(&g_ctx_mutex);
        LOG("No GPUs, exiting...\n");
        return 1;
    }
    cur_gpu = g_cur_gpu;
    g_cur_gpu++;
    g_cur_gpu %= g_total_gpus;
    cur_queue = g_cur_queue[cur_gpu];
    g_cur_queue[cur_gpu]++;
    g_cur_queue[cur_gpu] %= MAX_QUEUE_SIZE;
    pthread_mutex_unlock(&g_ctx_mutex);

    gpu_ctx* cur_ctx = &g_gpu_ctx[cur_gpu][cur_queue];
    pthread_mutex_lock(&cur_ctx->mutex);

    CUDA_CHK(cudaSetDevice(cur_gpu));

    LOG("cur gpu: %d cur queue: %d\n", cur_gpu, cur_queue);

    size_t hashes_size = num_elems * SHA256_BLOCK_SIZE * sizeof(uint8_t);
    size_t num_hashes_size = num_elems * sizeof(uint64_t);

    // Ensure there is enough memory allocated
    if (cur_ctx->hashes == NULL || cur_ctx->num_elems_alloc < num_elems) {
        CUDA_CHK(cudaFree(cur_ctx->hashes));
        CUDA_CHK(cudaMalloc(&cur_ctx->hashes, hashes_size));
        CUDA_CHK(cudaFree(cur_ctx->num_hashes_arr));
        CUDA_CHK(cudaMalloc(&cur_ctx->num_hashes_arr, num_hashes_size));

        cur_ctx->num_elems_alloc = num_elems;
    }

    cudaStream_t stream = 0;
    if (0 != use_non_default_stream) {
        stream = cur_ctx->stream;
    }

    CUDA_CHK(cudaMemcpyAsync(cur_ctx->hashes, hashes, hashes_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(cur_ctx->num_hashes_arr, num_hashes_arr, num_hashes_size, cudaMemcpyHostToDevice, stream));

    int num_blocks = ROUND_UP_DIV(num_elems, NUM_THREADS_PER_BLOCK);

    poh_verify_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(cur_ctx->hashes, cur_ctx->num_hashes_arr, num_elems);
    CUDA_CHK(cudaPeekAtLastError());

    CUDA_CHK(cudaMemcpyAsync(hashes, cur_ctx->hashes, hashes_size, cudaMemcpyDeviceToHost, stream));

    CUDA_CHK(cudaStreamSynchronize(stream));

    pthread_mutex_unlock(&cur_ctx->mutex);

    return 0;
}
}
