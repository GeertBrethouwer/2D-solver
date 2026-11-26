#!/bin/bash
#
# MMS convergence study (examples/mms) on one NSC Arrhenius GPU node: 14 small runs, 4 at a time, one GPU
# each. Submit from examples/mms after building solver2d (make CONFIG=configs/arrhenius.mk):
#
#     cd ~/2d-solver/examples/mms && sbatch arrhenius_mms.sh
#
# The tables (errors and observed orders) go to mms_convergence_gpu.txt; compare them with the CPU
# results in docs/mms_melt.pdf (Sec. 6). Check the account, partition and module names (placeholders).
#
#SBATCH -J mms
#SBATCH -A naiss20XX-X-XX          # your project
#SBATCH -N 1
#SBATCH --ntasks=4                 # 4 runs at the same time
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
##SBATCH -p gpu                    # GPU partition name on Arrhenius (check with sinfo)

module load GPU/buildenv-nvhpc/25.9-cu13.0
# Open MPI of the NVIDIA HPC SDK: skip the libfabric (ofi) components, which crash in MPI_Init on
# Arrhenius; on one node they are not needed
export OMPI_MCA_btl=^ofi
export OMPI_MCA_mtl=^ofi

cd $SLURM_SUBMIT_DIR
# convergence.py needs numpy; matplotlib is optional (--noplot without it)
PLOT=--noplot
python3 -c "import matplotlib" 2>/dev/null && PLOT="--fig mms_convergence_gpu.png"
# each run: one task with one GPU inside the allocation (--exact: steps share the node)
python3 convergence.py --exe ../../build/solver2d --jobs 4 \
        --mpirun "srun --mpi=pmix -n 1 --gpus=1 --cpus-per-task=8 --exact" $PLOT | tee mms_convergence_gpu.txt
# if the parallel steps fail (e.g. 'Requested nodes are busy'), run them one by one:
#   python3 convergence.py --exe ../../build/solver2d --jobs 1 --mpirun "srun --mpi=pmix -n 1 --gpus=1" --noplot
