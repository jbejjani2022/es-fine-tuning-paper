#!/bin/bash
#
# Script to recover Alpaca-7B weights from LLaMA-7B and weight diff
#
# This script performs the following steps:
# 1. Downloads/clones the LLaMA-7B base model (if needed)
# 2. Downloads the Alpaca weight diff from HuggingFace
# 3. Runs the recovery script to generate Alpaca-7B weights
#

set -e  # Exit on error

# Paths
NETSCRATCH_BASE="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper"
REPO_DIR="/n/home07/itamarf/es-fine-tuning-paper"

# Model paths on netscratch
LLAMA_7B_PATH="${NETSCRATCH_BASE}/models/llama-7b-hf"
ALPACA_DIFF_PATH="${NETSCRATCH_BASE}/models/alpaca-7b-diff"
ALPACA_RECOVERED_PATH="${NETSCRATCH_BASE}/models/alpaca-7b"

# Create directories if they don't exist
mkdir -p "${NETSCRATCH_BASE}/models"

echo "==========================================="
echo "Alpaca-7B Weight Recovery Script"
echo "==========================================="
echo ""

# Step 1 & 2: Download models using Python (no git-lfs needed)
echo "Step 1 & 2: Downloading models from HuggingFace..."
echo ""

# Check if models already exist
NEED_DOWNLOAD=false
if [ ! -d "${LLAMA_7B_PATH}" ] || [ ! -f "${LLAMA_7B_PATH}/pytorch_model.bin.index.json" ]; then
    echo "LLaMA-7B not found or incomplete at ${LLAMA_7B_PATH}"
    NEED_DOWNLOAD=true
fi

if [ ! -d "${ALPACA_DIFF_PATH}" ] || [ ! -f "${ALPACA_DIFF_PATH}/pytorch_model.bin.index.json" ]; then
    echo "Alpaca weight diff not found or incomplete at ${ALPACA_DIFF_PATH}"
    NEED_DOWNLOAD=true
fi

if [ "$NEED_DOWNLOAD" = true ]; then
    echo ""
    echo "Downloading models using HuggingFace Hub (this may take a while)..."
    echo "Models are large (~13GB each), please be patient..."
    echo ""
    
    cd "${REPO_DIR}/alignment"
    python download_models.py --base-dir "${NETSCRATCH_BASE}/models"
    
    if [ $? -ne 0 ]; then
        echo "Error: Model download failed. Please check the error messages above."
        exit 1
    fi
else
    echo "✓ LLaMA-7B found at ${LLAMA_7B_PATH}"
    echo "✓ Alpaca weight diff found at ${ALPACA_DIFF_PATH}"
fi

# Step 3: Run recovery script
echo ""
echo "Step 3: Running weight recovery..."
echo "This will:"
echo "  - Load LLaMA-7B from: ${LLAMA_7B_PATH}"
echo "  - Load weight diff from: ${ALPACA_DIFF_PATH}"
echo "  - Save Alpaca-7B to: ${ALPACA_RECOVERED_PATH}"
echo ""

cd "${REPO_DIR}/alignment"

python recover_alpaca_weights.py \
    --path_raw "${LLAMA_7B_PATH}" \
    --path_diff "${ALPACA_DIFF_PATH}" \
    --path_tuned "${ALPACA_RECOVERED_PATH}" \
    --device cpu \
    --test_inference \
    --check_integrity

echo ""
echo "==========================================="
echo "Recovery Complete!"
echo "==========================================="
echo ""
echo "Alpaca-7B weights saved to: ${ALPACA_RECOVERED_PATH}"
echo ""
echo "You can now load the model in Python with:"
echo ""
echo "  import transformers"
echo "  model = transformers.AutoModelForCausalLM.from_pretrained('${ALPACA_RECOVERED_PATH}')"
echo "  tokenizer = transformers.AutoTokenizer.from_pretrained('${ALPACA_RECOVERED_PATH}')"
echo ""

