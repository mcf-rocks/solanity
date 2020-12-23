#include "sha512.h"
#include <algorithm>
#include <stdio.h>
#include "sc.cu"
#include "fe.cu"
#include "ge.cu"
#include "sha512.cu"

#include "ed25519.h"
#include <pthread.h>

#include "gpu_common.h"
#include "gpu_ctx.h"

#define USE_CLOCK_GETTIME
#include "perftime.h"

static int __host__ __device__ consttime_equal(const unsigned char *x, const unsigned char *y) {
    unsigned char r = 0;

    r = x[0] ^ y[0];
    #define F(i) r |= x[i] ^ y[i]
    F(1);
    F(2);
    F(3);
    F(4);
    F(5);
    F(6);
    F(7);
    F(8);
    F(9);
    F(10);
    F(11);
    F(12);
    F(13);
    F(14);
    F(15);
    F(16);
    F(17);
    F(18);
    F(19);
    F(20);
    F(21);
    F(22);
    F(23);
    F(24);
    F(25);
    F(26);
    F(27);
    F(28);
    F(29);
    F(30);
    F(31);
    #undef F

    return !r;
}

static int __device__ __host__
ed25519_verify_device(const unsigned char *signature,
                      const unsigned char *message,
                      uint32_t message_len,
                      const unsigned char *public_key) {
    unsigned char h[64];
    unsigned char checker[32];
    sha512_context hash;
    ge_p3 A;
    ge_p2 R;

    if (signature[63] & 224) {
        return 0;
    }

    if (ge_frombytes_negate_vartime(&A, public_key) != 0) {
        return 0;
    }

    sha512_init(&hash);
    sha512_update(&hash, signature, 32);
    sha512_update(&hash, public_key, 32);
    sha512_update(&hash, message, message_len);
    sha512_final(&hash, h);

    sc_reduce(h);
    ge_double_scalarmult_vartime(&R, h, &A, signature + 32);
    ge_tobytes(checker, &R);

    if (!consttime_equal(checker, signature)) {
        return 0;
    }

    return 1;
}

int 
ed25519_verify(const unsigned char *signature,
               const unsigned char *message,
               uint32_t message_len,
               const unsigned char *public_key) {
    return ed25519_verify_device(signature, message, message_len, public_key);
}

__global__ void ed25519_verify_kernel(const uint8_t* packets,
                                      uint32_t message_size,
                                      uint32_t* message_lens,
                                      uint32_t* public_key_offsets,
                                      uint32_t* signature_offsets,
                                      uint32_t* message_start_offsets,
                                      size_t num_keys,
                                      uint8_t* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_keys) {
        uint32_t message_start_offset = message_start_offsets[i];
        uint32_t signature_offset = signature_offsets[i];
        uint32_t public_key_offset = public_key_offsets[i];
        uint32_t message_len = message_lens[i];

        out[i] = ed25519_verify_device(&packets[signature_offset],
                                       &packets[message_start_offset],
                                       message_len,
                                       &packets[public_key_offset]);
    }
}

bool g_verbose = false;

void ed25519_set_verbose(bool val) {
    g_verbose = val;
}

void ed25519_verify_many(const gpu_Elems* elems,
                         uint32_t num_elems,
                         uint32_t message_size,
                         uint32_t total_packets,
                         uint32_t total_signatures,
                         const uint32_t* message_lens,
                         const uint32_t* public_key_offsets,
                         const uint32_t* signature_offsets,
                         const uint32_t* message_start_offsets,
                         uint8_t* out,
                         uint8_t use_non_default_stream)
{
    LOG("Starting verify_many: num_elems: %d total_signatures: %d total_packets: %d message_size: %d\n",
        num_elems, total_signatures, total_packets, message_size);

    size_t out_size = total_signatures * sizeof(uint8_t);

    uint32_t total_packets_size = total_packets * message_size;

    if (0 == total_packets) {
        return;
    }

    // Device allocate

    gpu_ctx_t* gpu_ctx = get_gpu_ctx();

    verify_ctx_t* cur_ctx = &gpu_ctx->verify_ctx;

    cudaStream_t stream = 0;
    if (0 != use_non_default_stream) {
        stream = gpu_ctx->stream;
    }

    setup_gpu_ctx(cur_ctx,
                  elems,
                  num_elems,
                  message_size,
                  total_packets,
                  total_packets_size,
                  total_signatures,
                  message_lens,
                  public_key_offsets,
                  signature_offsets,
                  message_start_offsets,
                  out_size,
                  stream
                 );

    int num_threads_per_block = 64;
    int num_blocks = ROUND_UP_DIV(total_signatures, num_threads_per_block);
    LOG("num_blocks: %d threads_per_block: %d keys: %d out: %p stream: %p\n",
           num_blocks, num_threads_per_block, (int)total_packets, out, gpu_ctx->stream);

    perftime_t start, end;
    get_time(&start);
    ed25519_verify_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>
                            (cur_ctx->packets,
                             message_size,
                             cur_ctx->message_lens,
                             cur_ctx->public_key_offsets,
                             cur_ctx->signature_offsets,
                             cur_ctx->message_start_offsets,
                             cur_ctx->offsets_len,
                             cur_ctx->out);
    CUDA_CHK(cudaPeekAtLastError());

    cudaError_t err = cudaMemcpyAsync(out, cur_ctx->out, out_size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)  {
        fprintf(stderr, "verify: cudaMemcpy(out) error: out = %p cur_ctx->out = %p size = %zu num: %d elems = %p\n",
                        out, cur_ctx->out, out_size, num_elems, elems);
    }
    CUDA_CHK(err);

    CUDA_CHK(cudaStreamSynchronize(stream));

    release_gpu_ctx(gpu_ctx);

    get_time(&end);
    LOG("time diff: %f\n", get_diff(&start, &end));
}

// Ensure copyright and license notice is embedded in the binary
const char* ed25519_license() {
   return "Copyright (c) 2018 Solana Labs, Inc. "
          "Licensed under the Apache License, Version 2.0 "
          "<http://www.apache.org/licenses/LICENSE-2.0>";
}

int cuda_host_register(void* ptr, size_t size, unsigned int flags) {
   return cudaHostRegister(ptr, size, flags);
}

int cuda_host_unregister(void* ptr) {
   return cudaHostUnregister(ptr);
}
