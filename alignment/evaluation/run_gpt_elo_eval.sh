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

# Make sure OPENAI_API_KEY is set
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

echo "Starting GPT ELO evaluation at $(date)"
echo "Using $(which python)"

python alignment/evaluation/gpt_elo_eval.py \
    --max-prompts 200 \
    --concurrency 8 \
    --judgments-out alignment/outputs/gpt_elo/gpt_judgments.jsonl \
    --ratings-out alignment/outputs/gpt_elo/gpt_elo_ratings.json \
    --plot-out alignment/outputs/gpt_elo/gpt_elo_scatter.png \
    --data-dir alignment/outputs/model_comparisons \
    --models Alpaca-7B Beaver-V1 Beaver-V2 Beaver-V3 ES-500 \
    --normalize-to Alpaca-7B

echo "Finished at $(date)"
