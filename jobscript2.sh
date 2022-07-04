#!/bin/bash

#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/bbie/samiran2/mohit/parallel_ss.out
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbie-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=samiran2@illinois.edu

srun ./run_batch_uv_parallel.sh
