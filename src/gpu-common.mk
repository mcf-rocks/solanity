NVCC:=nvcc
GPU_PTX_ARCH:=compute_35
GPU_ARCHS?=sm_37,sm_50,sm_61,sm_70
GPU_CFLAGS:=--gpu-code=$(GPU_ARCHS),$(GPU_PTX_ARCH) --gpu-architecture=$(GPU_PTX_ARCH)
CFLAGS_release:=--ptxas-options=-v $(GPU_CFLAGS) -O3 -Xcompiler "-Wall -Werror -fPIC -Wno-strict-aliasing"
CFLAGS_debug:=$(CFLAGS_release) -g
CFLAGS:=$(CFLAGS_$V)
