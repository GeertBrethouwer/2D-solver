#!/bin/bash
#
# solver2d on ONE NSC Arrhenius GPU node (4 x NVIDIA GH200), one MPI rank per GPU
#
# Check the account, partition and module names with the NSC/NAISS Arrhenius documentation
# (`module avail buildenv`, `sinfo`) -- the values below are placeholders/assumptions.
#
#SBATCH -J solver2d
#SBATCH -A naiss20XX-X-XX          # your project
#SBATCH -N 1                       # one node
#SBATCH --ntasks-per-node=4        # one rank per GH200
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72         # one Grace CPU (72 cores) per rank
#SBATCH -t 24:00:00
##SBATCH -p gpu                    # GPU partition name on Arrhenius (check with sinfo)

module load GPU/buildenv-nvhpc/25.9-cu13.0

SOLVER=$HOME/2d-solver/build/solver2d             # make CONFIG=configs/arrhenius.mk NCCL=1
export OMP_NUM_THREADS=1
# comm_backend = 'mpi' (or a build without NCCL=1): the transposes use MPI_Alltoallv on GPU buffers, which needs a
# CUDA-aware MPI and reached only ~1 GB/s per GPU on one Arrhenius node; with NCCL (the default of an NCCL=1 build)
# they use NVLink.
export MPICH_GPU_SUPPORT_ENABLED=1                 # Cray MPICH
export UCX_MEMTYPE_CACHE=n                         # Open MPI / UCX
export NCCL_DEBUG=WARN                             # NCCL prints the reason if it fails
#
# Settings chosen from the number of nodes (as in FDCHAN's and HiCHAN's Arrhenius job scripts):
# - Open MPI: on one node its libfabric (OFI) components crash in MPI_Init (segmentation fault in fi_fabric),
#   so they are switched off; between nodes MPI needs them (MPI_Init fails without).
# - NCCL: the site sets NCCL_NET_PLUGIN=.../libnccl-net-ofi.so and NCCL_NET="AWS Libfabric" in every task, after
#   srun has copied this environment, so a different choice is given on the srun line with env. On one node
#   NCCL needs no network (NVLink) and the plugin is switched off; between nodes the site's settings work
#   (Slingshot, about 21 GB/s per GPU; NCCL_NET=Socket only 1.7 GB/s).
NNODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
if [ "$NNODES" -eq 1 ]; then
  export OMPI_MCA_btl=^ofi
  export OMPI_MCA_mtl=^ofi
  NCCL_TASK_ENV="NCCL_NET_PLUGIN=none NCCL_NET=Socket"
else
  unset OMPI_MCA_btl OMPI_MCA_mtl
  NCCL_TASK_ENV=""
fi
env | grep -E '^(NCCL|FI|OMPI_MCA)_' | sort
echo "$NNODES node(s); set in every task: ${NCCL_TASK_ENV:-(nothing, site NCCL settings)}"

cd $SLURM_SUBMIT_DIR
# The MPI of the NVIDIA HPC SDK (Open MPI/HPC-X) needs PMIx; with a plain srun the tasks may start as
# independent MPI "singletons" that all use GPU 0. solver2d prints the GPU of every rank and stops if
# two ranks share a GPU.
srun --mpi=pmix env $NCCL_TASK_ENV $SOLVER
# if Slurm binds each task to its own GPU differently:
#   srun --mpi=pmix --gpus-per-task=1 --gpu-bind=single:1 env $NCCL_TASK_ENV $SOLVER
# alternative: mpirun -np 4 $SOLVER   (the mpirun of the MPI used to build)
# A correct start prints the set-up ONCE and "number of MPI processes: 4".
