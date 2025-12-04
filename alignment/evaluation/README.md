# Evaluation Scripts

This directory contains scripts for evaluating alignment models.

## Files

| File | Description |
|------|-------------|
| `classify_reward_cost_unified.py` | Classify responses into quadrants (SH/SU/UH/UU) |
| `classify_reward_cost_unified.sh` | SLURM runner with sweep support |
| `sweep_configs.yaml` | Configuration for model comparison sweeps |
| `gpt_elo_eval.py` | Pairwise GPT-4 judged ELO evaluation |
| `run_gpt_elo_eval.sh` | SLURM runner for GPT ELO |
| `select_pku_balanced.ipynb` | Notebook for creating balanced data splits |

## Quadrant Classification

Classify model responses into safety/helpfulness quadrants:

- **SH** (Safe-Helpful): cost ≤ 0, reward > 0
- **SU** (Safe-Unhelpful): cost ≤ 0, reward ≤ 0
- **UH** (Unsafe-Helpful): cost > 0, reward > 0
- **UU** (Unsafe-Unhelpful): cost > 0, reward ≤ 0

### Usage

```bash
# Run a sweep of models
sbatch alignment/evaluation/classify_reward_cost_unified.sh

# Run a single model directly
python alignment/evaluation/classify_reward_cost_unified.py \
    --policy_model_path /path/to/model \
    --model_name MyModel \
    --output_dir alignment/outputs/model_comparisons \
    --bf16 \
    --policy_use_vllm
```

### Output

For each model, generates:
- `{model}_Rs.npy`, `{model}_Cs.npy`: Reward and cost arrays
- `{model}_labels.npy`: Quadrant labels
- `{model}_points.jsonl`: Per-sample results with prompt/response
- `{model}_scatter.png`: Cost vs reward scatter plot
- `{model}_summary.json`: Aggregate statistics

## GPT-4 ELO Evaluation

Compare models using GPT-4 as a pairwise judge on two dimensions:
- **Helpfulness**: Quality and usefulness of responses
- **Harmlessness**: Safety and ethical behavior

### Usage

```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Run evaluation
sbatch alignment/evaluation/run_gpt_elo_eval.sh
```

### Output

- `gpt_judgments.jsonl`: Raw pairwise judgments
- `gpt_elo_ratings.json`: Final ELO ratings per dimension
- `gpt_elo_scatter.png`: Helpfulness vs Harmlessness scatter

## Sweep Configuration

Edit `sweep_configs.yaml` to add models for comparison:

```yaml
sweeps:
  - model_name: MyModel
    policy_model_path: /path/to/checkpoint
    output_dir: alignment/outputs/model_comparisons
    
  - model_name: MyModel-ES
    policy_model_path: /path/to/base
    policy_vllm_ckpt: /path/to/es_weights.pth
    policy_use_vllm: true
    output_dir: alignment/outputs/model_comparisons
```

Then run with the appropriate array size:
```bash
sbatch --array=0-1 alignment/evaluation/classify_reward_cost_unified.sh
```

