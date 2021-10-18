#!/bin/zsh

#SBATCH --partition=gpu_normal
#SBATCH --mail-type=ALL
#SBATCH --output=output/slurm_logs/%x-%j.slurm.log
#SBATCH --export=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --job-name=World-GAN

cd $SLURM_SUBMIT_DIR
. /home/schubert/miniconda3/tmp/bin/activate toadgan

export PYTHONPATH=$SLURM_SUBMIT_DIR

srun $@