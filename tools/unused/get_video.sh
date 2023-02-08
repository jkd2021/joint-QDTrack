#!/bin/bash

#SBATCH --time=1-00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu06
#SBATCH --job-name=PCAN_0705
#SBATCH --mem=32000
#SBATCH --begin=now
#SBATCH --dependency=afterany:526:528

module load cuda/11.0
module load lib/cudnn/8.0.3
module load anaconda

source activate PCAN

echo " Computing job " $SLURM_JOB_ID " on "$(hostname)

export PATH=/beegfs/work/kangdongjin/gcc-9.2.0/gcc-9.2.0/bin:$PATH
export LD_LIBRARY_PATH=/beegfs/work/kangdongjin/gcc-9.2.0/gcc-9.2.0/lib/:/beegfs/work/kangdongjin/gcc-9.2.0/gcc-9.2.0/lib64:$LD_LIBRARY_PATH

srun python get_video.py