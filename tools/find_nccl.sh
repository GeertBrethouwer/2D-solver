#!/bin/sh
# Print the NCCL directory of the NVIDIA HPC SDK that provides nvcc (used by the Makefile for NCCL=1).
# The SDK has comm_libs/nccl or comm_libs/<CUDA version>/nccl a few levels above nvcc
# (e.g. Linux_aarch64/25.9/compilers/bin/nvcc -> Linux_aarch64/25.9/comm_libs/13.0/nccl).
# Prints nothing if not found; then nccl.h and libnccl must be on the default paths, or set NCCL_HOME.
nv=$(command -v nvcc) || exit 0
d=$(dirname "$(readlink -f "$nv")")
for start in "$d" "$(dirname "$nv")"; do
  d=$start
  while [ -n "$d" ] && [ "$d" != / ]; do
    for c in "$d"/comm_libs/nccl "$d"/comm_libs/*/nccl; do
      if [ -f "$c/include/nccl.h" ]; then echo "$c"; exit 0; fi
    done
    d=$(dirname "$d")
  done
done
