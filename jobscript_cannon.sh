#!/bin/bash -x
#SBATCH --account=
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --output=cannon.%j
#SBATCH --error=cannon-err.%j
#SBATCH --time=00:04:00

srun $PROJECT_PATH/build/Cannons_Algorithm 4096 $DATA_REPO/fp64/obs_spitzer.trace

srun $PROJECT_PATH/build/Cannons_Algorithm_Comp 4096 $DATA_REPO/fp64/obs_spitzer.trace
