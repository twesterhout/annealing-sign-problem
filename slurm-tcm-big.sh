#!/bin/bash
#SBATCH -p tcm --time 3-00:00:00
#SBATCH -N 1 --exclusive --exclude=cn74,cn81

source /vol/tcm01/westerhout_tom/conda/etc/profile.d/conda.sh
conda activate annealing_experiments
# make JOBID=$SLURM_JOBID NOISE=0 CUTOFF=2e-6 NUMBER_SAMPLES=20000 kagome_36 # pyrochlore_32
make experiments/noise/j1j2_square_4x4.csv
