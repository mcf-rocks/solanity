#ifndef GPU_CTX_H
#define GPU_CTX_H

#include <inttypes.h>
#include "ed25519.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t* packets;
    uint32_t packets_size_bytes;

    uint8_t* out;
    size_t out_size_bytes;

    uint32_t* public_key_offsets;
    uint32_t* private_key_offsets;
    uint32_t* message_start_offsets;
    uint32_t* signature_offsets;
    uint32_t* message_lens;
    size_t offsets_len;
} verify_ctx_t;

typedef struct {
    verify_ctx_t verify_ctx;

    pthread_mutex_t mutex;
    cudaStream_t stream;
} gpu_ctx_t;

extern gpu_ctx_t* get_gpu_ctx();
extern void release_gpu_ctx(gpu_ctx_t*);

extern void ed25519_free_gpu_mem();

extern void setup_gpu_ctx(verify_ctx_t* cur_ctx,
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
                          );

#ifdef __cplusplus
}
#endif

#endif
