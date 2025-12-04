# GRPO Alignment Training

This directory contains configuration and scripts for running **Group Relative Policy Optimization (GRPO)** on the alignment task, using the same reward and cost models as the Evolutionary Strategies (ES) baseline.

## Overview

GRPO is a policy gradient method that uses group-relative advantages instead of a learned value function. This implementation includes:

- **Adaptive Lagrangian λ**: Automatically adjusts the cost penalty to satisfy safety constraints
- **External scorer API**: Uses the same Beaver reward/cost models as ES training
- **Rich logging**: W&B integration with scatter plots, histograms, and sample tables

## Prerequisites

### 1. Start the Scorer Service

The GRPO trainer queries an external API for reward and cost scores. Start this service **before** running training:

```bash
# From the project root
sbatch slurm/run_scorer_service.sh
```

Check the log file for the service hostname:
```bash
tail -f logs/scorer_service_*.log
# Look for: "Service will be available at http://<HOST>:8000"
```

Update `scorer_url` in `grpo_alignment.yaml` to match this hostname.

### 2. Dependencies

Ensure you have the required packages:
- `trl` (Transformers Reinforcement Learning)
- `transformers`
- `accelerate`
- `deepspeed`
- `wandb`

## Quick Start

```bash
# 1. Start scorer service (separate job)
sbatch slurm/run_scorer_service.sh

# 2. Update scorer_url in grpo_alignment.yaml with the hostname from step 1

# 3. Launch GRPO training
sbatch alignment/GRPO/run_grpo_alignment.sh
```

## Configuration

### Main Config: `grpo_alignment.yaml`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Path to base model (e.g., Alpaca-7B) | `/n/.../alpaca-7b` |
| `train_jsonl` | Training prompts JSONL file | `alignment/data/custom_train_250.jsonl` |
| `eval_jsonl` | Evaluation prompts JSONL file | `alignment/data/custom_eval_500.jsonl` |
| `scorer_url` | URL of the scorer service | `http://localhost:8000` |
| `beta` | KL penalty coefficient | Set via CLI |
| `num_generations` | Samples per prompt for GRPO | 30 |
| `max_steps` | Total training steps | 500 |

### Adaptive Lambda Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lambda_cost` | Initial λ value | 1.0 |
| `lambda_adapt` | Enable Lagrangian adaptation | `true` |
| `cost_threshold_d` | Safety threshold d in J_C = E[C] + d | 0.0 |
| `lambda_lr` | Learning rate for ln(λ) update | 0.005 |
| `lambda_min` / `lambda_max` | Clipping range for λ | 1e-4 / 5.0 |
| `jc_ema_beta` | EMA smoothing for J_C | 0.9 |

## File Structure

```
GRPO/
├── README.md                    # This file
├── accelerate_trl_grpo.yaml     # Accelerate/DeepSpeed config
├── grpo_alignment.yaml          # Main training config
├── run_grpo_alignment.sh        # SLURM submission script
└── train_grpo_alignment.py      # Training script
```

## Reward Function

The reward combines helpfulness and safety:

```
Combined Reward = R(response) - λ × C(response)
```

Where:
- **R**: Reward from `PKU-Alignment/beaver-7b-unified-reward`
- **C**: Cost from `PKU-Alignment/beaver-7b-unified-cost`
- **λ**: Adaptive Lagrange multiplier (penalizes unsafe responses)

### Adaptive Lambda Update Rule

When `lambda_adapt: true`, λ is updated each batch:

1. Compute constraint violation: `J_C = mean(C) + d`
2. Update EMA: `J_C_ema = β × J_C_ema + (1-β) × max(J_C, 0)`
3. Update λ: `ln(λ) ← ln(λ) + η × J_C_ema`

This drives the policy toward satisfying `E[C] ≤ -d` (i.e., safe responses).

## Comparison: GRPO vs ES Alignment

| Aspect | ES Alignment | GRPO Alignment |
|--------|--------------|----------------|
| **Method** | Evolutionary Strategies | Policy Gradient |
| **Gradient** | Estimated via perturbations | Computed via backprop |
| **Baseline** | Population mean | Group average |
| **Value Network** | None | None |
| **Reward** | R - λ×C | R - λ×C |
| **λ Adaptation** | Lagrangian (same rule) | Lagrangian (same rule) |
| **GPUs** | 4+ (vLLM engines) | 4 (DeepSpeed ZeRO-2) |

## Outputs

Training outputs are saved to `output_dir` (default: `/n/.../GRPO_output/`):

```
GRPO_output/
└── beta{X}_seed{Y}_unified/
    ├── checkpoint-{step}/      # Model checkpoints
    ├── runs/                   # TensorBoard logs
    └── pytorch_model.bin       # Final model weights
```

## Troubleshooting

### Scorer API Connection Failed
```
❌ ERROR: Scorer API at http://... is not responding
```
**Solution**: Ensure the scorer service is running. Check with:
```bash
curl http://<SCORER_HOST>:8000/health
```

### Out of Memory
**Solution**: Reduce `num_generations` or `per_device_train_batch_size` in the config.

### Slow Training
**Solution**: Increase `scorer_batch_size` to batch more samples per API call.

## Beta Sweep

The SLURM script supports sweeping over beta values using job arrays:

```bash
# run_grpo_alignment.sh uses:
#SBATCH --array=0-1
BETA_VALUES=(0.01 0.1)
```

To add more beta values, update `BETA_VALUES` and adjust `--array=0-N`.
