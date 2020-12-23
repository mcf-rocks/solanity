/*
 * This file contains Solana's SGX enclave code for signing data.
 */

#include <stdbool.h>
#include <string.h>

#include "sgx_key.h"
#include "sgx_tseal.h"

#include "ed25519.h"
#include "signing_t.h"

typedef struct signing_parameters {
  bool initialized;
  uint8_t public_key[ED25519_PUB_KEY_LEN];
  uint8_t private_key[ED25519_PRIV_KEY_LEN];
  uint32_t nonce;
  uint32_t lockout_period;
  uint32_t lockout_multiplier;
  uint32_t lockout_max_depth;
  sgx_mc_uuid_t counter;
  uint32_t counter_value;
} signing_parameters_t;

static signing_parameters_t g_signing_params;

sgx_status_t init_remote_attestation(int b_pse,
                                     sgx_ec256_public_t* sp_pub_key,
                                     sgx_ra_context_t* pctxt) {
  sgx_status_t ret;
  if (b_pse) {
    int busy_retry_times = 2;
    do {
      ret = sgx_create_pse_session();
    } while (ret == SGX_ERROR_BUSY && busy_retry_times--);
    if (ret != SGX_SUCCESS)
      return ret;
  }
  ret = sgx_ra_init(sp_pub_key, b_pse, pctxt);
  if (b_pse) {
    sgx_close_pse_session();
  }
  return ret;
}

sgx_status_t close_remote_attestation(sgx_ra_context_t ctxt) {
  return sgx_ra_close(ctxt);
}

/* This function creates a new public/private keypair in
   enclave trusted space.
*/
sgx_status_t init_sgx_ed25519(uint32_t lockout_period,
                              uint32_t lockout_multiplier,
                              uint32_t lockout_max_depth,
                              uint32_t key_len,
                              uint8_t* pubkey) {
  if (key_len < sizeof(g_signing_params.public_key)) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  sgx_status_t status = SGX_SUCCESS;
  int busy_retry_times = 3;
  do {
    status = sgx_create_pse_session();
  } while (status == SGX_ERROR_BUSY && (busy_retry_times-- > 0));

  if (SGX_SUCCESS != status) {
    return status;
  }

  status = sgx_create_monotonic_counter(&g_signing_params.counter,
                                        &g_signing_params.counter_value);
  sgx_close_pse_session();
  if (SGX_SUCCESS != status) {
    return status;
  }

  uint8_t seed[ED25519_SEED_LEN];
  status = sgx_read_rand(seed, sizeof(seed));
  if (SGX_SUCCESS != status) {
    return status;
  }

  ed25519_create_keypair(g_signing_params.public_key,
                         g_signing_params.private_key, seed);

  memcpy(pubkey, g_signing_params.public_key,
         sizeof(g_signing_params.public_key));

  g_signing_params.initialized = true;
  g_signing_params.lockout_max_depth = lockout_max_depth;
  g_signing_params.lockout_multiplier = lockout_multiplier;
  g_signing_params.lockout_period = lockout_period;

  return SGX_SUCCESS;
}

sgx_status_t get_sgx_ed25519_data(uint32_t data_size,
                                  uint8_t* sealed_data,
                                  uint32_t* data_size_needed) {
  *data_size_needed =
      sgx_calc_sealed_data_size(0, sizeof(signing_parameters_t));

  if (*data_size_needed > data_size) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  sgx_status_t status = sgx_read_rand((uint8_t*)&g_signing_params.nonce,
                                      sizeof(g_signing_params.nonce));
  if (SGX_SUCCESS != status) {
    return status;
  }

  sgx_attributes_t attribute_mask;
  attribute_mask.flags = SGX_FLAGS_INITTED | SGX_FLAGS_DEBUG;
  attribute_mask.xfrm = 0x0;

  return sgx_seal_data_ex(SGX_KEYPOLICY_MRENCLAVE, attribute_mask, 0xF0000000,
                          0, NULL, sizeof(g_signing_params),
                          (const uint8_t*)&g_signing_params, *data_size_needed,
                          (sgx_sealed_data_t*)sealed_data);
}

sgx_status_t init_sgx_ed25519_from_data(uint32_t data_size,
                                        uint8_t* sealed_data,
                                        uint32_t update_lockout_params,
                                        uint32_t lockout_period,
                                        uint32_t lockout_multiplier,
                                        uint32_t lockout_max_depth,
                                        uint32_t key_len,
                                        uint8_t* pubkey) {
  if (key_len < sizeof(g_signing_params.public_key)) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  signing_parameters_t data;
  uint32_t datalen = sizeof(data);
  sgx_status_t status = sgx_unseal_data((const sgx_sealed_data_t*)sealed_data,
                                        NULL, 0, (uint8_t*)&data, &datalen);
  if (SGX_SUCCESS != status) {
    return status;
  }

  if (datalen != sizeof(data)) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  int busy_retry_times = 3;
  do {
    status = sgx_create_pse_session();
  } while (status == SGX_ERROR_BUSY && (busy_retry_times-- > 0));

  if (SGX_SUCCESS != status) {
    return status;
  }

  uint32_t counter_value = 0xffffffff;
  status =
      sgx_read_monotonic_counter(&g_signing_params.counter, &counter_value);
  if (SGX_SUCCESS != status) {
    sgx_close_pse_session();
    return status;
  }

  if (counter_value != g_signing_params.counter_value) {
    sgx_close_pse_session();
    return SGX_ERROR_INVALID_PARAMETER;
  }

  status = sgx_increment_monotonic_counter(&g_signing_params.counter,
                                           &g_signing_params.counter_value);

  sgx_close_pse_session();
  if (SGX_SUCCESS != status) {
    return status;
  }

  memcpy(&g_signing_params, &data, sizeof(g_signing_params));

  memcpy(pubkey, g_signing_params.public_key,
         sizeof(g_signing_params.public_key));

  g_signing_params.initialized = true;
  if (update_lockout_params != 0) {
    g_signing_params.lockout_max_depth = lockout_max_depth;
    g_signing_params.lockout_multiplier = lockout_multiplier;
    g_signing_params.lockout_period = lockout_period;
  }
  return SGX_SUCCESS;
}

/* This function signs the msg using private key.
 */
sgx_status_t sign_sgx_ed25519(uint32_t msg_len,
                              const uint8_t* msg,
                              uint32_t history_len,
                              const history_entry_t* entries,
                              uint32_t sig_len,
                              uint8_t* signature) {
  if (!g_signing_params.initialized) {
    return SGX_ERROR_INVALID_STATE;
  }

  if (sig_len < ED25519_SIGNATURE_LEN) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  ed25519_sign(signature, msg, msg_len, g_signing_params.public_key,
               g_signing_params.private_key);

  return SGX_SUCCESS;
}
