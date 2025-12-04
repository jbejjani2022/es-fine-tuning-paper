#!/bin/bash
#SBATCH --job-name=gpt_elo_eval
#SBATCH --output=logs/gpt_elo_eval_%j.log
#SBATCH --account=sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=6:00:00
#SBATCH --mem=16GB
#SBATCH --partition=sapphire

set -euo pipefail

cd /n/home07/itamarf/es-fine-tuning-paper

mkdir -p logs

# # Activate conda environment
# source ~/.bashrc
# conda activate metrics-rl

export OPENAI_API_KEY=
# Make sure OPENAI_API_KEY is set (you can also hardcode it here if needed)
# export OPENAI_API_KEY="sk-..."

echo "Starting GPT ELO evaluation at $(date)"
echo "Using $(which python)"

python alignment/gpt_elo_eval.py \
    --max-prompts 200 \
    --concurrency 8 \
    --judgments-out alignment/gpt_judgments.jsonl \
    --ratings-out alignment/gpt_elo_ratings.json \
    --plot-out alignment/gpt_elo_scatter.png \
    --data-dir alignment/compare-ES \
    --models Alpaca-7B ES-500-f ES-500 \
    --normalize-to Alpaca-7B

echo "Finished at $(date)"














