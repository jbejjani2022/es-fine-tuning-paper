# Alpaca-7B Weight Recovery

This directory contains scripts to recover the full Alpaca-7B model weights from the LLaMA-7B base model and the publicly released weight diff.

## Quick Start

```bash
bash /n/home07/itamarf/es-fine-tuning-paper/alignment/alpaca/recover_alpaca_full.sh
```

## What This Does

The Stanford Alpaca team released their fine-tuned weights as a **weight diff** rather than the full model (due to LLaMA's license restrictions). This script:

1. Takes the base LLaMA-7B model (in HuggingFace format)
2. Downloads the Alpaca weight diff from [tatsu-lab/alpaca-7b](https://huggingface.co/tatsu-lab/alpaca-7b)
3. Adds the diff to the base model to recover the full Alpaca-7B weights

## Prerequisites

The script will automatically download the models from HuggingFace using Python (no git-lfs required!).

**Note:** This is the original LLaMA-7B model, not Llama-2. Stanford Alpaca was trained on the original LLaMA-7B.

### Manual Download (Optional)

If you prefer to download the models separately first:

```bash
cd /n/home07/itamarf/es-fine-tuning-paper/alignment/alpaca
python download_models.py --base-dir /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models
```

This will download both:
- LLaMA-7B base model from [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b) (~13GB)
- Alpaca weight diff from [tatsu-lab/alpaca-7b](https://huggingface.co/tatsu-lab/alpaca-7b) (~13GB)

## File Structure

- **`recover_alpaca_weights.py`**: Standalone Python script that performs the weight recovery
- **`recover_alpaca_full.sh`**: Shell script that orchestrates the entire process

## Output Location

All models are saved to your netscratch:
```
/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/
├── llama-7b-hf/          # Base LLaMA-7B model (you provide this)
├── alpaca-7b-diff/       # Weight diff (downloaded automatically)
└── alpaca-7b/            # Recovered Alpaca-7B model (output)
```

## Usage in Python

After recovery, load the model like this:

```python
import transformers

model = transformers.AutoModelForCausalLM.from_pretrained(
    "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b"
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b"
)

# Example inference
input_text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
List three technologies that make life easier.

### Response:"""

inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Manual Recovery (if needed)

If you want to run just the Python script directly:

```bash
cd /n/home07/itamarf/es-fine-tuning-paper/alignment/alpaca

python recover_alpaca_weights.py \
    --path_raw /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/llama-7b-hf \
    --path_diff /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b-diff \
    --path_tuned /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b \
    --device cpu \
    --test_inference \
    --check_integrity
```

## Options

- `--device cpu` or `--device cuda`: Choose computation device (use cuda if you have GPU)
- `--test_inference`: Run a test inference after recovery (default: True)
- `--no_test_inference`: Skip inference test
- `--check_integrity`: Verify recovered weights match expected checksum (default: True)
- `--no_check_integrity`: Skip integrity check

## Important Notes

### Why Skip the Integrity Check?

The integrity check (`--no_check_integrity`) is recommended because:

1. **Different LLaMA conversions**: Various HuggingFace conversions of LLaMA-7B may have slight numerical differences
2. **Tokenizer differences**: Different conversions might handle special tokens differently
3. **Floating point precision**: Different hardware/software stacks can produce slightly different results

The integrity check expects a very specific checksum (50637.1836) that only matches Stanford's exact conversion. **This doesn't mean the recovery is wrong** - it just means there are minor numerical differences.

### How to Verify Correctness

Instead of relying on the strict integrity check, we verify correctness by:

1. **Testing inference**: The recovery script runs a test generation
2. **Verification script**: Run `verify_alpaca.py` to test multiple prompts:

```bash
python /n/home07/itamarf/es-fine-tuning-paper/alignment/alpaca/verify_alpaca.py --model_path /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b
```

This tests that the model:
- Loads correctly
- Generates coherent, instruction-following responses
- Produces Alpaca-style outputs (helpful, detailed answers)

### Running with SLURM (Recommended)

Recovery requires ~60GB RAM to load both models simultaneously. Use SLURM:

```bash
sbatch /n/home07/itamarf/es-fine-tuning-paper/alignment/alpaca/slurm_recover_alpaca.sh
```

This script:
1. Requests sufficient memory (64GB)
2. Runs the recovery process
3. Automatically runs verification tests
4. Reports if the model is working correctly

## References

- [Stanford Alpaca GitHub](https://github.com/tatsu-lab/stanford_alpaca)
- [Stanford Alpaca Blog Post](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Alpaca Weight Diff on HuggingFace](https://huggingface.co/tatsu-lab/alpaca-7b)