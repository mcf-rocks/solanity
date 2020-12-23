#ifndef ED25519_H
#define ED25519_H

#include <stddef.h>
#include <inttypes.h>

#if defined(_WIN32)
    #if defined(ED25519_BUILD_DLL)
        #define ED25519_DECLSPEC __declspec(dllexport)
    #elif defined(ED25519_DLL)
        #define ED25519_DECLSPEC __declspec(dllimport)
    #else
        #define ED25519_DECLSPEC
    #endif
#else
    #define ED25519_DECLSPEC
#endif


#ifdef __cplusplus
extern "C" {
#endif

#ifndef ED25519_NO_SEED
int ED25519_DECLSPEC ed25519_create_seed(unsigned char *seed);
#endif

#define PUB_KEY_SIZE 32
#define PRIV_KEY_SIZE 64
#define SEED_SIZE 32
#define SCALAR_SIZE 32
#define SIG_SIZE 64

typedef struct {
    uint8_t* elems;
    uint32_t num;
} gpu_Elems;

void ED25519_DECLSPEC ed25519_create_keypair(unsigned char *public_key, unsigned char *private_key, const unsigned char *seed);
void ED25519_DECLSPEC ed25519_sign(unsigned char *signature, const unsigned char *message, size_t message_len, const unsigned char *public_key, const unsigned char *private_key);

void ED25519_DECLSPEC ed25519_sign_many(const gpu_Elems* elems,
                                        uint32_t num_elems,
                                        uint32_t message_size,
                                        uint32_t total_packets,
                                        uint32_t total_signatures,
                                        const uint32_t* message_lens,
                                        const uint32_t* public_key_offsets,
                                        const uint32_t* private_key_offsets,
                                        const uint32_t* message_start_offsets,
                                        uint8_t* signatures_out,
                                        uint8_t use_non_default_stream);

int ED25519_DECLSPEC ed25519_verify(const unsigned char *signature, const unsigned char *message, uint32_t message_len, const unsigned char *public_key);

void ED25519_DECLSPEC ed25519_verify_many(const gpu_Elems* elems,
                                          uint32_t num_elems,
                                          uint32_t message_size,
                                          uint32_t total_packets,
                                          uint32_t total_signatures,
                                          const uint32_t* message_lens,
                                          const uint32_t* public_key_offsets,
                                          const uint32_t* private_key_offsets,
                                          const uint32_t* message_start_offsets,
                                          uint8_t* out,
                                          uint8_t use_non_default_stream);

void ED25519_DECLSPEC ed25519_add_scalar(unsigned char *public_key, unsigned char *private_key, const unsigned char *scalar);
void ED25519_DECLSPEC ed25519_key_exchange(unsigned char *shared_secret, const unsigned char *public_key, const unsigned char *private_key);
void ED25519_DECLSPEC ed25519_set_verbose(bool val);

const char* ED25519_DECLSPEC ed25519_license();
bool ED25519_DECLSPEC ed25519_init();

int cuda_host_register(void* ptr, size_t size, unsigned int flags);
int cuda_host_unregister(void* ptr);

#ifdef __cplusplus
}
#endif

#endif
