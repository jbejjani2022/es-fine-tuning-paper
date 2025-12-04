# Alignment Fine-Tuning

This directory contains everything needed to run alignment fine-tuning experiments using the PKU Safe-RLHF objective. We implement two optimization methods:

- **ES (Evolutionary Strategies)**: Gradient-free optimization via weight perturbations
- **GRPO (Group Relative Policy Optimization)**: Policy gradient with group-relative advantages

Both methods optimize the same objective: **maximize helpfulness while minimizing harmfulness**.

## Directory Structure

```
alignment/
├── README.md                 # This file
│
├── alpaca/                   # Alpaca-7B weight recovery scripts
│   └── README → ALPACA_RECOVERY_README.md
│
├── data/                     # Training and evaluation data
│   ├── custom_train_250.jsonl
│   └── custom_eval_500.jsonl
│
├── ES/                       # Evolutionary Strategies training
│   ├── README.md
│   └── es-fine-tuning_alignment_accl_api.py
│
├── GRPO/                     # GRPO training
│   ├── README.md
│   ├── grpo_alignment.yaml
│   ├── run_grpo_alignment.sh
│   └── train_grpo_alignment.py
│
├── scoring/                  # Scorer service (reward & cost models)
│   ├── README.md
│   ├── scorer_service.py
│   └── test_score_fixed_prompt.py
│
├── evaluation/               # Evaluation scripts
│   ├── classify_reward_cost_unified.py   # Quadrant classification
│   ├── classify_reward_cost_unified.sh   # SLURM runner
│   ├── sweep_configs.yaml                # Sweep configurations
│   ├── gpt_elo_eval.py                   # GPT-4 pairwise ELO evaluation
│   ├── run_gpt_elo_eval.sh               # SLURM runner
│   └── select_pku_balanced.ipynb         # Data selection notebook
│
├── visualization/            # Plotting scripts
│   ├── plot_gpt_elo_figure.py            # ELO scatter plot
│   ├── plot_reward_cost_grid.py          # Reward vs cost grid
│   └── plot_safety_ratios.py             # Safety ratio bar chart
│
├── outputs/                  # Generated outputs
│   ├── gpt_elo/              # GPT ELO evaluation results
│   │   ├── gpt_elo_ratings.json
│   │   ├── gpt_elo_scatter.png
│   │   └── gpt_judgments.jsonl
│   └── model_comparisons/    # Model comparison results
│       ├── {Model}_Cs.npy, _Rs.npy, _labels.npy
│       ├── {Model}_points.jsonl
│       ├── {Model}_scatter.png
│       └── {Model}_summary.json
│
└── safe-rlhf/                # Local clone of PKU Safe-RLHF (dependency)
```

## Quick Start

### 0. Prerequisites

Clone Safe-RLHF locally (required for scorer imports):

```bash
cd alignment
git clone https://github.com/PKU-Alignment/safe-rlhf.git safe-rlhf
```

### 1. (Optional) Recover Alpaca-7B Weights

If using Alpaca-7B as the base policy:

```bash
bash alignment/alpaca/recover_alpaca_full.sh
```

Output: `/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models/alpaca-7b/`

### 2. Prepare Training Data

Run the data selection notebook to create balanced train/eval splits:

```bash
jupyter notebook alignment/evaluation/select_pku_balanced.ipynb
```

This creates:
- `alignment/data/custom_train_250.jsonl`
- `alignment/data/custom_eval_500.jsonl`

### 3. Start the Scorer Service

Submit the scorer SLURM job (uses 1 GPU):

```bash
sbatch slurm/run_scorer_service.sh
```

Check the log for the hostname:
```bash
tail -f logs/scorer_service_*.log
# Look for: "Service will be available at http://<HOST>:8000"
```

### 4. Run Training

#### Option A: ES Training (4 GPUs)

```bash
# Update SCORER_HOST in slurm/train_accl_alignment_api.sh
sbatch slurm/train_accl_alignment_api.sh
```

See [ES/README.md](ES/README.md) for details.

#### Option B: GRPO Training (4 GPUs)

```bash
# Update scorer_url in alignment/GRPO/grpo_alignment.yaml
sbatch alignment/GRPO/run_grpo_alignment.sh
```

See [GRPO/README.md](GRPO/README.md) for details.

### 5. Evaluate Results

#### Quadrant Classification (R, C)

Classify model outputs into quadrants (Safe-Helpful, Safe-Unhelpful, etc.):

```bash
sbatch alignment/evaluation/classify_reward_cost_unified.sh
```

Results saved to `alignment/outputs/model_comparisons/`.

#### GPT-4 ELO Evaluation

Run pairwise comparisons with GPT-4 as judge:

```bash
# Set OPENAI_API_KEY first
sbatch alignment/evaluation/run_gpt_elo_eval.sh
```

Results saved to `alignment/outputs/gpt_elo/`.

## Objective Function

Both methods optimize:

```
maximize  E[R(x, y)] - λ × E[C(x, y)]
```

Where:
- **R(x, y)**: Helpfulness score from `beaver-7b-unified-reward`
- **C(x, y)**: Harmfulness score from `beaver-7b-unified-cost`
- **λ**: Lagrange multiplier (adaptive or fixed)

**Constraint**: Keep expected cost ≤ 0 (safe responses).

## SLURM Scripts

All SLURM scripts are in `/slurm/`:

| Script | Description | GPUs |
|--------|-------------|------|
| `run_scorer_service.sh` | Start scorer API | 1 |
| `train_accl_alignment_api.sh` | ES training | 4 |

GRPO uses its own SLURM script: `alignment/GRPO/run_grpo_alignment.sh`

## Troubleshooting

### Scorer API Not Responding
```bash
curl http://<HOST>:8000/health
```
If it fails, check if the scorer job is running: `squeue -u $USER`

### Import Errors for safe_rlhf
Ensure Safe-RLHF is cloned: `ls alignment/safe-rlhf/safe_rlhf/__init__.py`

### Out of Memory
- ES: Reduce `--batch_size` or `--population_size`
- GRPO: Reduce `num_generations` or `per_device_train_batch_size`

## References

- [PKU Safe-RLHF](https://github.com/PKU-Alignment/safe-rlhf)
- [Beaver Models](https://huggingface.co/PKU-Alignment)
- [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)
