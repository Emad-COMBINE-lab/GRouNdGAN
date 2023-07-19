#!/bin/bash

#SBATCH --job-name=training
#SBATCH --time=7-00:00:00
#SBATCH --account={your-PI}
#SBATCH --mail-user={your-email}
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mem 32G
#SBATCH --cpus-per-task=4

nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source {VIRTUAL ENV ACTIVATE SCRIPT HERE}
cd {PROJECT DIR HERE}
tensorboard --logdir="{GAN OUTPUT DIR HERE}/TensorBoard" --host 0.0.0.0 --load_fast false &
python src/main.py --config {CONFIG_FILE_HERE} --train