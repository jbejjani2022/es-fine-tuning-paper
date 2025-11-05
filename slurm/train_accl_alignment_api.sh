#!/bin/bash
#SBATCH --job-name=es_align_api
#SBATCH --output=logs/train_accl_alignment_api_%j.log
#SBATCH --error=logs/train_accl_alignment_api_%j.err
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --mem=300GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100

# # Load modules
# module load cuda/12.4.1-fasrc01

# # Activate conda environment
# source ~/.bashrc
# conda activate metrics-rl

# Set environment variables
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/n/home07/itamarf/es-fine-tuning-paper:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Get the scorer service hostname (should be provided as environment variable or argument)
SCORER_HOST=holygpu8a13104
SCORER_URL="http://${SCORER_HOST}:8000"

# Note: The scorer service should be running separately
# You can start it with: sbatch slurm/run_scorer_service.sh
# Then set SCORER_HOST to the hostname where it's running

# Run the training script
echo "Starting ES fine-tuning for alignment with external scorer API..."
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Scorer API URL: $SCORER_URL"

python alignment/es-fine-tuning_alignment_accl_api.py \
  --policy_model_path /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b \
  --num_engines 4 \
  --cuda_devices "0,1,2,3" \
  --scorer_url $SCORER_URL \
  --scorer_batch_size 32 \
  --batch_size 64 \
  --population_size 30 \
  --num_iterations 500 \
  --sigma 0.001 \
  --alpha 0.0005 \
  --max_new_tokens 512 \
  --lambda_adapt \
  --cost_threshold_d 0 \
  --lambda_pos_cost_only \
  --eval_every 100 \
  --wandb_project es_alignment \
  --wandb_run_name acll_es_beaver_sig_0001_alpha_00005_lam_max_5 \
  --lambda_cost 1 \
  --lambda_lr 0.005 \
  --lambda_max 5.0