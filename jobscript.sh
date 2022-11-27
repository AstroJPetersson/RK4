#!/bin/bash
#SBATCH --job-name rk4
#SBATCH --error ejob%j
#SBATCH --output ojob%
#SBATCH --partition p4
#SBATCH --time 0-10:00:00
#SBATCH -n 2

module purge
. /usr/local/etc/setup-modules.sh

ml load GCCcore/7.3.0 Python/2.7.15-bare
ml load GCC/10.2.0 OpenMPI/4.0.5 mpi4py/3.0.3-timed-pingpong

mpiexec python rk4_mpi.py

