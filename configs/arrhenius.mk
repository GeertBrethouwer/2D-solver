#
# NSC Arrhenius, GPU partition: 4 x NVIDIA GH200 (Grace Hopper, sm_90) per node
#
#   module load GPU/buildenv-nvhpc/25.9-cu13.0     # NVIDIA HPC SDK + CUDA-aware MPI; check `module avail buildenv`
#   make CONFIG=configs/arrhenius.mk
#
FC     := mpifort
GPU    := 1
GPU_CC := cc90
# -tp=neoverse-v2 targets the Grace CPU (host code); the GPU code is generated for sm_90
FFLAGS := -O3 -fast -tp=neoverse-v2
