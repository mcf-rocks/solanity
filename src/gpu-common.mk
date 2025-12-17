NVCC:=nvcc
# Updated for CUDA 12.x and modern GPUs (A10G uses sm_86)
# Support: sm_70 (V100), sm_75 (Turing), sm_80 (A100), sm_86 (A10G/RTX 30xx)
# Use compute_70 as base architecture to support all these GPU codes
GPU_PTX_ARCH:=compute_70
GPU_ARCHS?=sm_70,sm_75,sm_80,sm_86
GPU_CFLAGS:=--gpu-code=$(GPU_ARCHS),$(GPU_PTX_ARCH) --gpu-architecture=$(GPU_PTX_ARCH)
CFLAGS_release:=--ptxas-options=-v $(GPU_CFLAGS) -O3 -Xcompiler "-Wall -Werror -fPIC -Wno-strict-aliasing"
CFLAGS_debug:=$(CFLAGS_release) -g
CFLAGS:=$(CFLAGS_$V)
