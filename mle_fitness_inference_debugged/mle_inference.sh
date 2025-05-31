#!/usr/bin/env bash
#SBATCH -J fit_inf
#SBATCH -p normal,hns,owners,dpetrov
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH --array=1-145
#SBATCH -o SlurmFiles/slurm-%A_%a.out
#SBATCH --mem=2000
#SBATCH --requeue
#SBATCH --mail-user=omghosh@stanford.edu
#SBATCH --mail-type=END
module load python/3.6.1
module load py-numpy/1.18.1_py36


parameters=$(sed -n "$SLURM_ARRAY_TASK_ID"p input_sample_ids.txt)
python3 run_mle_inference.py $parameters
