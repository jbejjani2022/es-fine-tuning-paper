#!/bin/bash
#SBATCH --job-name=classify_reward_cost_unified
#SBATCH --output=logs/classify_reward_cost_unified_%A_%a.log
#SBATCH --error=logs/classify_reward_cost_unified_%A_%a.err
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=400GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100

python alignment/classify_reward_cost_unified.py --bf16 \
  --dataset_name PKU-Alignment/PKU-SafeRLHF --split test \
  --policy_model_path /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b \
  --policy_max_new_tokens 256 --policy_temperature 0.0 \
  --reward_model PKU-Alignment/beaver-7b-unified-reward \
  --cost_model PKU-Alignment/beaver-7b-unified-cost \
  --output_dir alignment/plots \
  --policy_use_vllm \
  --policy_vllm_ckpt /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/alignment/es-ft-experiment/alignment_nccl_20251103_225528/model_saves/final_model_iteration_500/pytorch_model.pth \
  --num_samples 50