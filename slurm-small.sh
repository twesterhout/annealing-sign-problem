#!/bin/bash
#SBATCH -p thin --time 3-00:00:00
#SBATCH -n 1 -c 128 --exclusive

source $HOME/conda/etc/profile.d/conda.sh
conda activate annealing_experiments
make "$@"
