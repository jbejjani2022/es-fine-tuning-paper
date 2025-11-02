#!/bin/bash
#SBATCH --job-name=es_fine_tuning_countdown_accuracy
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

for model in checkpoints/countdown_accl/Qwen/Qwen2.5-1.5B-Instruct/0/max_tokens_1024/es_random_seed0_pop30_iter500_sigma0.001_alpha0.0005_engines4_question_num200_final checkpoints/countdown_accl/Qwen/Qwen2.5-1.5B-Instruct/1/max_tokens_1024/es_random_seed1_pop30_iter500_sigma0.001_alpha0.0005_engines4_question_num200_final
do
    python countdown/countdown_accl_eval.py \
        --model $model \
        --base_model_name Qwen/Qwen2.5-1.5B-Instruct \
        --data_sample 2000 \
        --precision bf16 \
        --batch_size 8 \
        --max_new_tokens 1024
done
