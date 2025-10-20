#!/bin/bash
#SBATCH --job-name=trl_grpo_concise
#SBATCH --output=logs/trl_grpo_concise_%A_%a.log
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=400GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100
#SBATCH --array=0-15%1

set -euo pipefail

CONFIG=${CONFIG:-GRPO/grpo_conciseness_trl.yaml}
BETAS=(0.0 0.01 0.0167 0.0464)
SEEDS=(11 22 33 44)
NB=${#BETAS[@]}
NS=${#SEEDS[@]}
IDX=${SLURM_ARRAY_TASK_ID}

BIDX=$(( IDX / NS ))
SIDX=$(( IDX % NS ))
BETA=${BETAS[$BIDX]}
SEED=${SEEDS[$SIDX]}

echo "[SLURM] array index=$IDX -> beta=$BETA seed=$SEED"

mkdir -p logs

# Accelerate (local; Deepspeed ZeRO-3)
ACCELERATE_CONFIG=GRPO/accelerate_trl_grpo.yaml
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

# Optional: turn on vLLM-backed generation to reduce GPU mem during sampling
# export USE_VLLM=1

# WandB
export WANDB_PROJECT=$(python - <<PY
import yaml; print(yaml.safe_load(open("${CONFIG}")).get("project","es_conciseness"))
PY
)
export WANDB_ENTITY=$(python - <<PY
import yaml; print(yaml.safe_load(open("${CONFIG}")).get("entity",""))
PY
)

DS_CFG=$(realpath GRPO/ds_zero2.json)

accelerate launch \
  --config_file $ACCELERATE_CONFIG \
  --num_processes 4 \
  --deepspeed_config_file $DS_CFG \
  GRPO/train_grpo_conciseness_trl.py \
  --config ${CONFIG} \
  --beta ${BETA} \
  --seed ${SEED}
