#!/bin/bash
#SBATCH --job-name=recover_alpaca
#SBATCH --output=logs/recover_alpaca_%j.log
#SBATCH --error=logs/recover_alpaca_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --partition=kempner
#SBATCH --account=kempner_sham_lab

DEVICE="cuda"

# For CPU-only (comment out the above and uncomment this):
# DEVICE="cpu"

# Paths
LLAMA_PATH="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/llama-7b-hf"
ALPACA_DIFF_PATH="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b-diff"
ALPACA_OUTPUT_PATH="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b"

# Load conda environment
source ~/.bashrc
conda activate metrics-rl

# Run recovery
cd /n/home07/itamarf/es-fine-tuning-paper/alignment

python recover_alpaca_weights.py \
    --path_raw "${LLAMA_PATH}" \
    --path_diff "${ALPACA_DIFF_PATH}" \
    --path_tuned "${ALPACA_OUTPUT_PATH}" \
    --device "${DEVICE}" \
    --no_check_integrity \
    --test_inference

if [ $? -eq 0 ]; then
    echo ""
    echo "Recovery complete!"
    echo "Recovered model saved to: ${ALPACA_OUTPUT_PATH}"
    echo ""
    echo "Running verification tests..."
    echo ""
    
    python verify_alpaca.py \
        --model_path "${ALPACA_OUTPUT_PATH}" \
        --device "${DEVICE}"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ All verification tests passed!"
        echo "The recovered Alpaca model appears to be working correctly."
    else
        echo ""
        echo "⚠ Verification had some warnings. Please review the output above."
    fi
else
    echo "✗ Recovery failed. Please check the error messages above."
    exit 1
fi

