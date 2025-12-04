# ES Alignment Training

This directory contains the **Evolutionary Strategies (ES)** implementation for alignment fine-tuning using the PKU Safe-RLHF objective.

## Overview

ES fine-tuning optimizes the policy by:
1. Sampling a population of weight perturbations
2. Evaluating each perturbation on a batch of prompts
3. Computing a combined reward: `R - λ×C` (helpfulness - cost penalty)
4. Updating weights proportionally to normalized rewards

This approach is gradient-free and leverages parallel vLLM engines for fast generation.

## Prerequisites

### 1. Start the Scorer Service

ES training queries an external API for reward and cost scores:

```bash
sbatch slurm/run_scorer_service.sh
```

Note the hostname from the log:
```bash
tail -f logs/scorer_service_*.log
# Look for: "Service will be available at http://<HOST>:8000"
```

### 2. Update SCORER_HOST

Edit `slurm/train_accl_alignment_api.sh`:
```bash
SCORER_HOST=<hostname-from-step-1>
```

## Quick Start

```bash
# 1. Start scorer service
sbatch slurm/run_scorer_service.sh

# 2. Update SCORER_HOST in slurm/train_accl_alignment_api.sh

# 3. Launch ES training
sbatch slurm/train_accl_alignment_api.sh
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--policy_model_path` | Base model path | Alpaca-7B |
| `--num_engines` | Number of vLLM engines (= GPUs) | 4 |
| `--population_size` | Perturbations per iteration | 30 |
| `--sigma` | Perturbation noise scale | 0.001 |
| `--alpha` | Learning rate | 0.0005 |
| `--num_iterations` | Total training iterations | 500 |
| `--batch_size` | Prompts per iteration | 64 |

### Adaptive Lambda

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lambda_cost` | Initial λ value | 1.0 |
| `--lambda_adapt` | Enable Lagrangian adaptation | flag |
| `--cost_threshold_d` | Safety threshold d | 0.0 |
| `--lambda_lr` | Learning rate for ln(λ) | 0.005 |
| `--lambda_max` | Maximum λ value | 5.0 |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Driver (Ray)                           │
│  - Coordinates population evaluation                        │
│  - Applies ES weight updates                                │
│  - Logs to W&B/TensorBoard                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
┌──────▼─────┐  ┌──────▼─────┐  ┌──────▼─────┐
│  vLLM      │  │  vLLM      │  │  vLLM      │
│  Engine 0  │  │  Engine 1  │  │  Engine 2  │  ...
│  (GPU 0)   │  │  (GPU 1)   │  │  (GPU 2)   │
└──────┬─────┘  └──────┬─────┘  └──────┬─────┘
       │               │               │
       └───────────────┴───────────────┘
                       │
              ┌────────▼────────┐
              │  Scorer API     │
              │  (Separate GPU) │
              │  - Reward Model │
              │  - Cost Model   │
              └─────────────────┘
```

## ES Update Rule

For each iteration:

1. **Sample perturbations**: For seeds `s_1, ..., s_N`, compute `θ + σ·ε_i`
2. **Evaluate**: Generate responses and compute `f_i = R_i - λ·C_i`
3. **Normalize**: `f̂_i = (f_i - mean(f)) / std(f)`
4. **Update**: `θ ← θ + (α/N) × Σ f̂_i · ε_i`

## Outputs

```
<experiment_dir>/alignment_nccl_<timestamp>/
├── logs/
│   └── final_eval_results.json
├── model_saves/
│   ├── latest_hf/              # Latest HF-format checkpoint
│   └── final_model_iteration_N/
│       └── pytorch_model.pth   # Final weights
└── events.out.tfevents.*       # TensorBoard logs
```

## Comparison with GRPO

| Aspect | ES | GRPO |
|--------|----|----- |
| Gradient | Estimated (perturbations) | Exact (backprop) |
| Memory | Lower (no activations stored) | Higher |
| Parallelism | Naturally parallel | Data parallel |
| Hyperparams | σ, α, population_size | β, learning_rate |

## Troubleshooting

### "Scorer API not responding"
Ensure the scorer service is running and `SCORER_HOST` is correct.

### vLLM Engines Fail to Launch
Check GPU availability: `ray.cluster_resources()` should show enough GPUs.

### Slow Iterations
- Reduce `batch_size` or `population_size`
- Increase `scorer_batch_size` for better batching

