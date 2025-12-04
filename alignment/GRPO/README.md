# GRPO Alignment Task Setup

This directory contains the configuration and scripts to run Group Relative Policy Optimization (GRPO) on the alignment task, utilizing the same reward and cost models as the Evolutionary Strategies (ES) baseline.

## 1. Prerequisites: Scorer Service

The GRPO training script relies on an external API to score the generated responses. This service loads the reward and cost models (Beaver-7B) and exposes an HTTP endpoint.

**You must start the scorer service before running the training.**

### How to run the scorer server:

Open a separate terminal or use a `screen`/`tmux` session and run:

```bash
# From the root directory (es-fine-tuning-paper)
python alignment/scorer_service.py
```

By default, this will start the server on `http://localhost:8000`.
Ensure the `scorer_url` in `alignment/GRPO/grpo_alignment.yaml` matches this address.

## 2. Running GRPO Training

Once the scorer service is running, you can start the GRPO training.

### Launch command:

```bash
# From the root directory
bash alignment/GRPO/run_grpo_alignment.sh
```

This script submits a SLURM job (or runs locally if configured) that executes `train_grpo_alignment.py` with the configuration defined in `grpo_alignment.yaml`.

## 3. Reward Function & Lambda

The reward function used in this GRPO setup is designed to mirror the alignment objective:

**Total Reward = Reward_Model_Score - λ * Cost_Model_Score**

*   **Reward Model**: `PKU-Alignment/beaver-7b-v1.0-reward` (via scorer API)
*   **Cost Model**: `PKU-Alignment/beaver-7b-v1.0-cost` (via scorer API)
*   **λ (Lambda)**: A coefficient that penalizes unsafe responses (high cost).

### Comparison with ES Alignment

| Feature | ES Alignment (`es-fine-tuning_alignment_accl_api.py`) | GRPO Alignment (`train_grpo_alignment.py`) |
| :--- | :--- | :--- |
| **Optimization Method** | Evolutionary Strategies (Weight Perturbation) | GRPO (Policy Gradient with Group Relative Advantage) |
| **Reward Calculation** | `R - λ * C` | `R - λ * C` |
| **Lambda (λ)** | **Adaptive** (Lagrangian multiplier) | **Adaptive** (Lagrangian multiplier, batch-based) |
| **Value Network** | None (ES estimates gradient directly) | None (GRPO uses group average as baseline) |

### Adaptive Lambda in GRPO

This implementation supports adaptive lambda to enforce the cost constraint $E[C] \le d$.
The update rule is:
1.  Estimate expected cost $E[C]$ using the mean cost of the current generation batch.
2.  Calculate constraint violation: $J_C = E[C] + d$.
3.  Update exponential moving average (EMA) of $J_C$.
4.  Update $\lambda$ using the EMA: $\ln \lambda \leftarrow \ln \lambda + \eta \cdot J_{C, \text{ema}}$.

**Configuration Parameters:**
*   `lambda_adapt`: Set to `true` to enable adaptation.
*   `cost_threshold_d`: The constraint threshold $d$ (e.g., -0.1 means cost should be <= 0.1).
*   `lambda_lr`: Learning rate $\eta$ for lambda.
*   `lambda_min`, `lambda_max`: Clipping range for lambda.
*   `jc_ema_beta`: Beta for the EMA of $J_C$.
