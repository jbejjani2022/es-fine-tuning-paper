#!/bin/bash
#SBATCH --job-name=trl_grpo_concise
#SBATCH --account=kempner_sham_lab
#SBATCH --output=logs/trl_grpo_concise_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=400GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100
#SBATCH --array=0-15%1

set -euo pipefail

# --- Environment (edit to your cluster) ---
# module load python/3.12.5-fasrc01 || true
# module load cuda/12.4.1-fasrc01 || true
# module load cudnn/9.1.1.17_cuda12-fasrc01 || true
# export HF_HUB_ENABLE_HF_TRANSFER=1
# export TOKENIZERS_PARALLELISM=false
# export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# --- Resolve array to (beta, seed) ---
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

# --- Create logs dir ---
mkdir -p logs

# --- Accelerate config (ephemeral) ---
ACCELERATE_CONFIG=accelerate_trl_grpo.yaml
cat > $ACCELERATE_CONFIG << ACC
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
machine_rank: 0
gpu_ids: all
use_cpu: false
ACC

# --- Launch ---
export WANDB__SERVICE_WAIT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=$(python - <<PY
import yaml,sys
cfg=yaml.safe_load(open("${CONFIG}"))
print(cfg.get("project","es_conciseness"))
PY
)
export WANDB_ENTITY=$(python - <<PY
import yaml,sys
cfg=yaml.safe_load(open("${CONFIG}"))
print(cfg.get("entity",""))
PY
)

accelerate launch --config_file $ACCELERATE_CONFIG --num_processes 2 --deepspeed_config_file /n/home07/itamarf/es-fine-tuning-paper/GRPO/ds_zero2.json GRPO/train_grpo_conciseness_trl.py   --config ${CONFIG}   --beta ${BETA}   --seed ${SEED}