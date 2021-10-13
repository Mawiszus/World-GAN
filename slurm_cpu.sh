#!/bin/bash

#SBATCH --partition=cpu_normal
#SBATCH --mail-type=ALL
#SBATCH --output=output/slurm_logs/%x-%j.slurm.log
#SBATCH --export=ALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=World-GAN

cd $SLURM_SUBMIT_DIR
. /home/schubert/miniconda3/tmp/bin/activate toadgan

export PYTHONPATH=$SLURM_SUBMIT_DIR

srun $@