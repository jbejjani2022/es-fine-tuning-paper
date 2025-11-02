#!/bin/bash
#SBATCH --job-name=es_fine_tune_conciseness_eval
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
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

MODEL_PATH=checkpoints/conciseness/Qwen/Qwen2.5-7B-Instruct/${SLURM_ARRAY_TASK_ID}/es_random_seed${SLURM_ARRAY_TASK_ID}_pop30_iter1000_sigma0.001_alpha0.0005_bf16_threads1_question_num2_final
python conciseness/conciseness_eval.py \
    --model ${MODEL_PATH} \
    --baseline_model_name Qwen/Qwen2.5-7B-Instruct \
    --precision bf16 \
    --max_new_tokens 128 \
    --num_samples 20 \
    --eval_data_path conciseness/data/eval.jsonl \
    --print-examples \
    --output_json logs/conciseness_eval_seed${SLURM_ARRAY_TASK_ID}.json \
    --seed $SLURM_ARRAY_TASK_ID \
    --do_sample
