#!/bin/bash
#SBATCH -p tcm --time 3-00:00:00
#SBATCH -N 1 --exclusive --exclude=cn74,cn81

source /vol/tcm01/westerhout_tom/conda/etc/profile.d/conda.sh
conda activate annealing_experiments
make JOBID=$SLURM_JOBID NOISE=1e0 kagome_36 # pyrochlore_32
