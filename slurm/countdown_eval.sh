#!/bin/bash
#SBATCH --job-name=es_fine_tuning_countdown_accuracy
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
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

# for model in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct
# for model in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct
# for model in /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_countdown_1p5B/beta0.005_lr1e-06_seed42/checkpoint-1000
for model in /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_countdown_1p5B_500steps/beta0.005_lr1e-06_seed42
do
    python countdown/countdown_eval.py \
        --model $model \
        --data_sample 2000 \
        --precision bf16 \
        --batch_size 8 \
        --max_new_tokens 1024
done
