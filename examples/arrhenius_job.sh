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

SOLVER=$HOME/2d-solver/build/solver2d             # where solver2d was built (make CONFIG=configs/arrhenius.mk)
export OMP_NUM_THREADS=1
# the transposes use MPI_Alltoallv on GPU buffers: a CUDA-aware MPI is required
export MPICH_GPU_SUPPORT_ENABLED=1                 # Cray MPICH
export UCX_MEMTYPE_CACHE=n                         # Open MPI / UCX

cd $SLURM_SUBMIT_DIR
# The MPI of the NVIDIA HPC SDK (Open MPI/HPC-X) needs PMIx; with a plain srun the tasks may start as
# independent MPI "singletons" that all use GPU 0. solver2d prints the GPU of every rank and stops if
# two ranks share a GPU.
srun --mpi=pmix $SOLVER
# if Slurm binds each task to its own GPU differently:
#   srun --mpi=pmix --gpus-per-task=1 --gpu-bind=single:1 $SOLVER
# alternative: mpirun -np 4 $SOLVER   (the mpirun of the MPI used to build)
# A correct start prints the set-up ONCE and "number of MPI processes: 4".
