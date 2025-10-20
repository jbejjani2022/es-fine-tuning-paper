#!/bin/bash
#SBATCH --job-name=es_fine_tuning_conciseness_reward_and_KL
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

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

python eval/conciseness_reward_and_KL.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --baseline_model_name Qwen/Qwen2.5-3B-Instruct \
    --precision bf16 \
    --max_new_tokens 100 \
    --num_samples 20 \
    --batch_size 4 \
    --eval_data_path conciseness/data/eval.jsonl
