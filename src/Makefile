#
#  Makefile to build libcuda-crypt.so
#   From sources in /cuda-crypt and /ed25519-verify
#
#   nvcc inserts conflicting driver symbols into a static lib (.so)
#   so more than one .so cannot be linked into a single program.
#   Shared objects with device kernels also did not seem to work--
#   it can build, but the kernel launch is not successful. (with cuda 9.2)
#   Hence, build ed25519 ecdsa verify and chacha encryption device
#   code into a single binary.

V=debug
include gpu-common.mk

CHACHA_TEST_BIN=cuda_chacha_test
ECC_TEST_BIN=cuda_ed25519_verify
ECC_SCAN_BIN=cuda_ed25519_vanity
LIB=cuda-crypt

CUDA_HEADER_DIR:=cuda-headers
CUDA_SHA256_DIR:=cuda-sha256

CFLAGS+=-DENDIAN_NEUTRAL -DLTC_NO_ASM -I$(CUDA_HEADER_DIR) -I$(CUDA_SHA256_DIR)

all: $V/$(ECC_SCAN_BIN) $(V)/lib$(LIB).so

ECC_DIR:=cuda-ecc-ed25519

KEYPAIR_SRCS:=$(addprefix $(ECC_DIR)/,keypair.cu ed25519.h ge.h)
$V/keypair.o: $(KEYPAIR_SRCS)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

SEED_SRCS:=$(addprefix $(ECC_DIR)/,seed.cu ed25519.h)
$V/seed.o: $(SEED_SRCS)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

SIGN_SRCS:=$(addprefix $(ECC_DIR)/,sign.cu sha512.h ge.h sc.h fe.cu ../$(CUDA_HEADER_DIR)/gpu_common.h ed25519.h)
$V/sign.o: $(SIGN_SRCS)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

VERIFY_SRCS:=$(addprefix $(ECC_DIR)/,verify.cu sha512.cu ge.cu sc.cu fe.cu keypair.cu common.cu ed25519.h)
$V/verify.o: $(VERIFY_SRCS)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

$V/gpu_ctx.o: $(addprefix $(ECC_DIR)/,gpu_ctx.cu gpu_ctx.h)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

CHACHA_DIR:=cuda-crypt
CHACHA_SRCS:=$(addprefix $(CHACHA_DIR)/,chacha_cbc.cu chacha.h common.cu)

$V/chacha_cbc.o: $(CHACHA_SRCS)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

AES_SRCS:=$(addprefix $(CHACHA_DIR)/,aes_cbc.cu aes_core.cu modes_lcl.h common.cu)

$V/aes_cbc.o: $(AES_SRCS)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

POH_DIR:=cuda-poh-verify
POH_SRCS:=$(addprefix $(POH_DIR)/,poh_verify.cu)

$V/poh_verify.o: $(POH_SRCS)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

CPU_GPU_OBJS=$(addprefix $V/,chacha_cbc.o aes_cbc.o verify.o poh_verify.o gpu_ctx.o sign.o seed.o keypair.o)

$V/crypt-dlink.o: $(CPU_GPU_OBJS)
	$(NVCC) -Xcompiler "-fPIC" $(CFLAGS) --device-link $^ --output-file $@

$V/lib$(LIB).so: $V/crypt-dlink.o $(CPU_GPU_OBJS)
	$(NVCC) -Xcompiler "-fPIC" --shared --output-file $@ $^

$V/$(CHACHA_TEST_BIN): $(CHACHA_DIR)/test.cu $V/lib$(LIB).so
	$(NVCC) $(CFLAGS) -L$V -l$(LIB) $< -o $@

$V/ecc_main.o: $(addprefix $(ECC_DIR)/,main.cu ed25519.h)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

$V/$(ECC_TEST_BIN): $V/ecc_main.o $V/lib$(LIB).so
	$(NVCC) $(CFLAGS) -L$V -l$(LIB) $< -o $@

$V/ecc_scan.o: $(addprefix $(ECC_DIR)/,vanity.cu ed25519.h)
	@mkdir -p $(@D)
	$(NVCC) -rdc=true $(CFLAGS) -c $< -o $@

$V/$(ECC_SCAN_BIN): $V/ecc_scan.o $V/lib$(LIB).so
	$(NVCC) $(CFLAGS) -L$V -l$(LIB) $< -o $@

.PHONY:clean
clean:
	rm -rf $V

test: $V/$(ECC_TEST_BIN) $V/$(CHACHA_TEST_BIN)
	cd $(V) && ./$(CHACHA_TEST_BIN) 64 \
	cd $(V) && ./$(ECC_TEST_BIN) 1 1 1 1 1 1
	cd $(V) && ./$(ECC_TEST_BIN) 64 1 1 1 1 0
	cd $(V) && ./$(ECC_TEST_BIN) 100201 1 1 4 10 1
