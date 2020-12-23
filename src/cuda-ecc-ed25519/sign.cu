#include "ed25519.h"
#include "sha512.h"
#include "ge.h"
#include "sc.h"
#include "gpu_common.h"
#include "gpu_ctx.h"


static void __device__ __host__
ed25519_sign_device(unsigned char *signature,
                   const unsigned char *message,
                   size_t message_len,
                   const unsigned char *public_key,
                   const unsigned char *private_key) {
    sha512_context hash;
    unsigned char hram[64];
    unsigned char r[64];
    ge_p3 R;


    sha512_init(&hash);
    sha512_update(&hash, private_key + 32, 32);
    sha512_update(&hash, message, message_len);
    sha512_final(&hash, r);

    sc_reduce(r);
    ge_scalarmult_base(&R, r);
    ge_p3_tobytes(signature, &R);

    sha512_init(&hash);
    sha512_update(&hash, signature, 32);
    sha512_update(&hash, public_key, 32);
    sha512_update(&hash, message, message_len);
    sha512_final(&hash, hram);

    sc_reduce(hram);
    sc_muladd(signature + 32, hram, private_key, r);
}

void ed25519_sign(unsigned char *signature,
                   const unsigned char *message,
                   size_t message_len,
                   const unsigned char *public_key,
                   const unsigned char *private_key) {
    ed25519_sign_device(signature, message, message_len, public_key, private_key);
}



__global__ void ed25519_sign_kernel(unsigned char* packets,
                                    uint32_t message_size,
                                    uint32_t* public_key_offsets,
                                    uint32_t* private_key_offsets,
                                    uint32_t* message_start_offsets,
                                    uint32_t* message_lens,
                                    size_t num_transactions,
                                    uint8_t* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_transactions) {
        uint32_t message_start_offset = message_start_offsets[i];
        uint32_t public_key_offset = public_key_offsets[i];
        uint32_t private_key_offset = private_key_offsets[i];
        uint32_t message_len = message_lens[i];

        ed25519_sign_device(&out[i * SIG_SIZE],
                            &packets[message_start_offset],
                            message_len,
                            &packets[public_key_offset],
                            &packets[private_key_offset]);
    }
}



void ed25519_sign_many(const gpu_Elems* elems,
                       uint32_t num_elems,
                       uint32_t message_size,
                       uint32_t total_packets,
                       uint32_t total_signatures,
                       const uint32_t* message_lens,
                       const uint32_t* public_key_offsets,
                       const uint32_t* private_key_offsets,
                       const uint32_t* message_start_offsets,
                       uint8_t* signatures_out,
                       uint8_t use_non_default_stream
                       ) {
    int num_threads_per_block = 64;
    int num_blocks = ROUND_UP_DIV(total_signatures, num_threads_per_block);
    size_t sig_out_size = SIG_SIZE * total_signatures;

    if (0 == total_packets) {
        return;
    }

    uint32_t total_packets_size = total_packets * message_size;

    LOG("signing %d packets sig_size: %zu message_size: %d\n",
        total_packets, sig_out_size, message_size);

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
                  private_key_offsets,
                  message_start_offsets,
                  sig_out_size,
                  stream
                 );

    LOG("signing blocks: %d threads_per_block: %d\n", num_blocks, num_threads_per_block);
    ed25519_sign_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>
                            (cur_ctx->packets,
                             message_size,
                             cur_ctx->public_key_offsets,
                             cur_ctx->signature_offsets,
                             cur_ctx->message_start_offsets,
                             cur_ctx->message_lens,
                             total_signatures,
                             cur_ctx->out);

    cudaError_t err = cudaMemcpyAsync(signatures_out, cur_ctx->out, sig_out_size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)  {
        fprintf(stderr, "sign: cudaMemcpy(out) error: out = %p cur_ctx->out = %p size = %zu num: %d elems = %p\n",
                        signatures_out, cur_ctx->out, sig_out_size, num_elems, elems);
    }
    CUDA_CHK(err);

    CUDA_CHK(cudaStreamSynchronize(stream));

    release_gpu_ctx(gpu_ctx);
}

