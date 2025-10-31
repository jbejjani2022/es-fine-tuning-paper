#!/bin/bash
#SBATCH --job-name=es_fine_tuning_conciseness_reward_and_KL
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

# iterate over betas and seeds and execute in an array job
BETAS=(0.0)
SEEDS=(11 22 33 44)
NB=${#BETAS[@]}
NS=${#SEEDS[@]}
IDX=${SLURM_ARRAY_TASK_ID}

BIDX=$(( IDX / NS ))
SIDX=$(( IDX % NS ))
BETA=${BETAS[$BIDX]}
SEED=${SEEDS[$SIDX]}

echo "[SLURM] array index=$IDX -> beta=$BETA seed=$SEED"

MODEL_PATH=/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_temp_0.7/beta${BETA}_seed${SEED}/
python eval/conciseness_reward_and_KL.py \
    --model ${MODEL_PATH} \
    --baseline_model_name Qwen/Qwen2.5-7B-Instruct \
    --precision bf16 \
    --max_new_tokens 128 \
    --num_samples 20 \
    --batch_size 4 \
    --eval_data_path conciseness/data/eval.jsonl \
    --print-examples \
    --output_json logs/conciseness_reward_and_KL_temp_0.7_beta${BETA}_seed${SEED}.json \
    --seed ${SEED} \
    --beta ${BETA} \
    --temperature 1.0 \
    --top_p 1.0 \
    --do_sample