## Alignment ES fine-tuning — Quickstart

This folder contains everything needed to run ES fine-tuning for alignment with an external scorer API, recover Alpaca-7B weights, prepare data, and classify reward/cost on PKU.

### 0) Prerequisite: clone Safe-RLHF locally

The scorer scripts import the local `safe-rlhf` package. Clone it inside this folder:

```bash
cd /n/home07/itamarf/es-fine-tuning-paper/alignment
git clone https://github.com/PKU-Alignment/safe-rlhf.git safe-rlhf
```

### 1) (Optional) Recover Alpaca‑7B weights

If you plan to use Alpaca‑7B as the policy, follow `alignment/alpaca/ALPACA_RECOVERY_README.md`.

Quick start:

```bash
bash /n/home07/itamarf/es-fine-tuning-paper/alignment/alpaca/recover_alpaca_full.sh
```

This produces `/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b/`.

### 2) Generate data selections

Run the notebook `alignment/select_pku_balanced.ipynb` to create balanced selections. It writes:

- `alignment/data/custom_train_250.jsonl`
- `alignment/data/custom_eval_500.jsonl`

### 3) Start the scorer service (1 GPU)

Submit the scorer SLURM job and note the hostname printed in its log:

```bash
sbatch slurm/run_scorer_service.sh
# tail -f logs/scorer_service_*.log   # shows: "Service will be available at http://<HOST>:8000"
```

### 4) Launch ES training with external scorer API (3–4 GPUs for vLLM)

Edit `slurm/train_accl_alignment_api.sh` and set `SCORER_HOST` to the hostname from step 3, then submit:

```bash
sbatch slurm/train_accl_alignment_api.sh
```

Defaults in the script use the recovered Alpaca‑7B at:
`/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b`.

### 5) Classify reward/cost on the PKU eval set

Classify the entire PKU test split (saves arrays, JSONL, summary, and scatter plot to `alignment/plots/`):

```bash
python alignment/classify_reward_cost_unified.py \
  --bf16 \
  --dataset_name PKU-Alignment/PKU-SafeRLHF \
  --split test \
  --num_samples 3036 \
  --output_dir alignment/plots
```

Optionally, add `--policy_model_path /n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b` to classify generated policy responses instead of dataset responses.

That’s it. This README intentionally omits test or convert helper scripts to keep things simple.


