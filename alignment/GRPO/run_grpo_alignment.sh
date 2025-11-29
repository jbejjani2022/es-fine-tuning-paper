#!/bin/bash
#SBATCH --job-name=grpo_alignment
#SBATCH --output=logs/grpo_alignment_%A_%a.log
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=400GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100
#SBATCH --array=0-0

set -euo pipefail

# Ensure we are in the root directory
cd /n/home07/itamarf/es-fine-tuning-paper

CONFIG=${CONFIG:-alignment/GRPO/grpo_alignment.yaml}
BETA=0.01
SEED=42

mkdir -p logs

# Accelerate config (overwritten each run just like before)
ACCELERATE_CONFIG=alignment/GRPO/accelerate_trl_grpo.yaml
cat > $ACCELERATE_CONFIG << ACC
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
machine_rank: 0
gpu_ids: all
use_cpu: false
ACC

# Env
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export WANDB__SERVICE_WAIT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# WandB
export WANDB_PROJECT=grpo_alignment
export WANDB_ENTITY=itamarf

DS_CFG=$(realpath GRPO/ds_zero2.json)

accelerate launch \
  --config_file $ACCELERATE_CONFIG \
  --num_processes 4 \
  --deepspeed_config_file $DS_CFG \
  alignment/GRPO/train_grpo_alignment.py \
  --config ${CONFIG} \
  --beta ${BETA} \
  --seed ${SEED}
