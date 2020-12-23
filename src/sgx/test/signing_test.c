#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "signing_public.h"

#include "ed25519.h"

void print_buffer(const uint8_t* buf, int len) {
  char str[BUFSIZ] = {'\0'};
  int offset = 0;
  for (int i = 0; i < len; i++) {
    offset += snprintf(&str[offset], BUFSIZ - offset, "0x%02x ", buf[i]);
    if (!((i + 1) % 8))
      offset += snprintf(&str[offset], BUFSIZ - offset, "\n");
  }
  offset += snprintf(&str[offset], BUFSIZ - offset, "\n");
  printf("%s", str);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Usage: %s <enclave file path>\n", argv[0]);
    return -1;
  }

  ed25519_context_t ctxt;
  uint32_t lockout_period = 10, lockout_multiplier = 2, lockout_max_depth = 32;
  sgx_status_t status = init_ed25519(
      argv[1], lockout_period, lockout_multiplier, lockout_max_depth, &ctxt);
  if (SGX_SUCCESS != status) {
    printf("Failed in init_ed25519. Error %d\n", status);
    return -1;
  }

  printf("Loaded the enclave. eid: %d\n", (uint32_t)ctxt.eid);

  uint32_t datalen = 0;
  status = get_ed25519_data(&ctxt, &datalen, NULL);

  uint8_t* sealed_data = malloc(datalen);
  status = get_ed25519_data(&ctxt, &datalen, sealed_data);
  if (SGX_SUCCESS != status) {
    printf("Failed in get_ed25519_data. Error %d\n", status);
    release_ed25519_context(&ctxt);
    free(sealed_data);
    return -1;
  }

  status =
      init_ed25519_from_data(&ctxt, datalen, sealed_data, 1, lockout_period,
                             lockout_multiplier, lockout_max_depth);
  free(sealed_data);
  if (SGX_SUCCESS != status) {
    printf("Failed in init_ed25519_from_data. Error %d\n", status);
    release_ed25519_context(&ctxt);
    return -1;
  }

  const history_entry_t entries;
  uint8_t* data =
      "This is a test string. We'll sign it using SGX enclave. Hope it works!!";
  uint8_t signature[64];
  memset(signature, 0, sizeof(signature));
  status = sign_ed25519(&ctxt, sizeof(data), data, 1, &entries,
                        sizeof(signature), signature);
  if (SGX_SUCCESS != status) {
    printf("Failed in sign_ed25519. Error %d\n", status);
    release_ed25519_context(&ctxt);
    return -1;
  }

  printf("Signature:\n");
  print_buffer(signature, sizeof(signature));

  if (ed25519_verify(signature, data, sizeof(data), ctxt.public_key) == 0) {
    printf("Failed in verifying the signature\n");
  } else {
    printf("Signature verified\n");
  }

  release_ed25519_context(&ctxt);
  return 0;
}