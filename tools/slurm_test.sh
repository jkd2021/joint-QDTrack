#!/bin/bash

#SBATCH --time=20-00
#SBATCH --partition=kidl
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=pcan_test
#SBATCH --mem=40000
#SBATCH --begin=now

module load cuda/11.0
module load lib/cudnn/8.0.3
module load anaconda

source activate qdtrack_joint

echo " Computing job " $SLURM_JOB_ID " on "$(hostname)

srun python test.py