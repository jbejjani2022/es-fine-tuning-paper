#!/bin/bash
#SBATCH --job-name=es_fine_tuning_conciseness
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
mamba deactivate
mamba activate es

cd ..

# Set PyTorch memory allocator configuration for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --machine_rank 0 \
    conciseness/es_fine-tuning_conciseness_iid.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --gpu_threads 2 \
    --max_new_tokens 100 \
    --iterations 1000 \
    --population_size 30 \
    --sigma 0.001 \
    --alpha 0.0005 \
    --initial_seed $SLURM_ARRAY_TASK_ID \
    --precision bf16
