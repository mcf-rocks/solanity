#pragma once

#include <stdbool.h>
#include "sgx_eid.h"
#include "sgx_error.h"

#define ED25519_PUB_KEY_LEN 32

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ed25519_context {
  bool enclaveEnabled;
  sgx_enclave_id_t eid;
  uint8_t public_key[ED25519_PUB_KEY_LEN];
} ed25519_context_t;

typedef struct history_entry {
  uint32_t num_hashes;
  uint32_t optional_input_hash[4];
  uint32_t result_hash[4];
} history_entry_t;

/* This function initializes SGX enclave. It loads enclave_file
   to SGX, which internally creates a new public/private keypair.

   If the platform does not support SGX, it creates a public/private
   keypair in untrusted space. An error is returned in this scenario.
   The user can choose to not use the library if SGX encalve is not
   being used for signing.

   Note: The user must release the enclave by calling release_ed25519_context()
         after they are done using it.
*/
sgx_status_t init_ed25519(const char* enclave_file,
                          uint32_t lockout_period,
                          uint32_t lockout_multiplier,
                          uint32_t lockout_max_depth,
                          ed25519_context_t* pctxt);

/* This function returns the sealed data (private key and associated
   informatio). The sealed data can be used to reinit the enclave using
   init_ed25519_from_data().
*/
sgx_status_t get_ed25519_data(ed25519_context_t* pctxt,
                              uint32_t* datalen,
                              uint8_t* data);

/* This function reinitializes the enclave using sealed data.
 */
sgx_status_t init_ed25519_from_data(ed25519_context_t* pctxt,
                                    uint32_t datalen,
                                    uint8_t* data,
                                    uint32_t update_lockout_params,
                                    uint32_t lockout_period,
                                    uint32_t lockout_multiplier,
                                    uint32_t lockout_max_depth);

/* This function signs the msg using the internally stored private
   key. The signature is returned in the output "signature" buffer.

   This function must only be called after init_ed25519() function.
*/
sgx_status_t sign_ed25519(ed25519_context_t* pctxt,
                          uint32_t msg_len,
                          const uint8_t* msg,
                          uint32_t history_len,
                          const history_entry_t* entries,
                          uint32_t sig_len,
                          uint8_t* signature);

/* This function releases SGX enclave */
void release_ed25519_context(ed25519_context_t* pctxt);

#ifdef __cplusplus
}
#endif
