#!/usr/bin/env python
import os
import sys
import json
import random
import argparse
from typing import List, Tuple, Optional

import requests
import numpy as np


def load_yaml(path: str) -> dict:
    text = open(path, "r").read()
    try:
        return json.loads(text)
    except Exception:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception:
            raise RuntimeError(f"Please install pyyaml or provide valid JSON at {path}.")


def read_jsonl_rows(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# Minimal local prompt formatting
PROMPT_BEGIN: str = "BEGINNING OF CONVERSATION: "
PROMPT_USER: str = "USER: {input} "
PROMPT_ASSISTANT: str = "ASSISTANT:"


def format_prompt_local(user_input: str) -> str:
    return "".join(
        [PROMPT_BEGIN, PROMPT_USER.format(input=user_input), PROMPT_ASSISTANT]
    )


def format_train_eval_datasets(cfg: dict, tokenizer=None):
    """
    Return (train_dataset, eval_dataset_or_None, train_prompts_text, eval_prompts_text)
    with prompts formatted for TRL GRPO.
    """
    from datasets import Dataset

    pkey = cfg.get("prompt_key", "prompt")

    train_rows = [r for r in read_jsonl_rows(cfg["train_jsonl"])]
    eval_rows = (
        [r for r in read_jsonl_rows(cfg["eval_jsonl"])]
        if (cfg.get("eval_jsonl") and os.path.exists(cfg["eval_jsonl"]))
        else []
    )

    # Filter rows that have the prompt key
    train_rows = [r for r in train_rows if pkey in r]
    eval_rows = [r for r in eval_rows if pkey in r]

    # Format prompts for GRPO (conversation-style prompt field)
    train_prompts_text = [
        format_prompt_local(str(r[pkey]).strip()) for r in train_rows
    ]
    eval_prompts_text = [format_prompt_local(str(r[pkey]).strip()) for r in eval_rows]

    if cfg.get("max_samples"):
        max_n = int(cfg["max_samples"])
        train_prompts_text = train_prompts_text[:max_n]
        eval_prompts_text = eval_prompts_text[:max_n]

    train_prompts = [{"prompt": p} for p in train_prompts_text]
    eval_prompts = [{"prompt": p} for p in eval_prompts_text]

    if not train_prompts:
        raise ValueError(
            "No training prompts found. Check `train_jsonl` and `prompt_key`."
        )

    print(f"Sample formatted prompt:\n{train_prompts[0]['prompt'][:200]}...")

    d_train = Dataset.from_list(train_prompts)
    d_eval = Dataset.from_list(eval_prompts) if eval_prompts else None

    return d_train, d_eval, train_prompts_text, eval_prompts_text


def call_scorer_api(
    scorer_url: str, prompts: List[str], responses: List[str], batch_size: int = 16
) -> Tuple[List[float], List[float]]:
    """Call the external scorer API to get rewards and costs."""
    try:
        all_rewards: List[float] = []
        all_costs: List[float] = []

        for i in range(0, len(prompts), batch_size):
            payload = {
                "prompts": prompts[i : i + batch_size],
                "responses": responses[i : i + batch_size],
            }
            r = requests.post(f"{scorer_url}/score_batch", json=payload, timeout=60)
            r.raise_for_status()
            result = r.json()
            all_rewards.extend(result["rewards"])
            all_costs.extend(result["costs"])

        return all_rewards, all_costs
    except requests.exceptions.RequestException as e:
        print(f"Error calling scorer API: {e}")
        # Return zeros on error to avoid crashing, but this is bad
        return [0.0] * len(prompts), [0.0] * len(prompts)


class AdaptiveReward:
    """
    Reward callable passed to TRL GRPOTrainer.

    This wraps the Beaver reward & cost models served by the scorer API and
    maintains an adaptive Lagrange multiplier lambda to enforce a cost
    constraint, using the same update rule as the ES alignment script.

      ln λ <- ln λ + η * EMA(J_C)
      J_C = E[C] + d  (optionally clamped to max(J_C, 0))

    It also mirrors ES logging:
    - train/reward_*, train/cost_*, train/combined_*
    - train/safe_ratio
    - lambda/value, lambda/J_C, lambda/J_C_raw, lambda/J_C_ema
    - helpfulness/mean, helpfulness/dist
    - cost/mean, cost/dist
    - scatter/reward_vs_cost (sampled from current batch)
    """

    def __init__(self, cfg: dict):
        self.scorer_url: str = cfg.get("scorer_url", "http://localhost:8000")
        self.lambda_adapt: bool = cfg.get("lambda_adapt", False)

        # Lambda parameters (defaulted to ES alignment hyperparams)
        self.current_lambda: float = float(cfg.get("lambda_cost", 0.5))
        self.cost_threshold_d: float = float(cfg.get("cost_threshold_d", 0.0))
        self.lambda_lr: float = float(cfg.get("lambda_lr", 0.005))
        self.lambda_min: float = float(cfg.get("lambda_min", 1e-4))
        self.lambda_max: float = float(cfg.get("lambda_max", 5.0))
        self.jc_ema_beta: float = float(cfg.get("jc_ema_beta", 0.9))
        self.lambda_pos_cost_only: bool = cfg.get("lambda_pos_cost_only", True)

        # Logging / scatter settings
        self.scatter_every: int = int(cfg.get("scatter_every", 50))
        self.hist_every: int = int(cfg.get("hist_every", 25))
        self.max_scatter_points: int = int(cfg.get("scatter_max_points", 2000))
        self.scorer_batch_size: int = int(cfg.get("scorer_batch_size", 16))

        self.__name__ = "AdaptiveReward"

        self.jc_ema: Optional[float] = None
        self.step: int = 0

    def __call__(self, prompts: List[str], completions: List[str], **kwargs):
        # Query external reward & cost models
        rewards, costs = call_scorer_api(
            self.scorer_url, prompts, completions, batch_size=self.scorer_batch_size
        )

        rewards_np = np.asarray(rewards, dtype=np.float32)
        costs_np = np.asarray(costs, dtype=np.float32)

        # ---------- Adaptive Lambda Update (ES rule) ----------
        if self.lambda_adapt and len(costs_np) > 0:
            batch_cost_mean = float(costs_np.mean())
            # J_C(theta) = E[C] + d
            jc_raw = batch_cost_mean + self.cost_threshold_d

            # Positive part only if requested (Lagrangian on violations only)
            jc = max(jc_raw, 0.0) if self.lambda_pos_cost_only else jc_raw

            # Exponential moving average
            if self.jc_ema is None:
                self.jc_ema = jc
            else:
                self.jc_ema = (
                    self.jc_ema_beta * self.jc_ema + (1.0 - self.jc_ema_beta) * jc
                )

            # Log-space lambda update
            ln_lambda = float(np.log(max(self.current_lambda, 1e-12)))
            ln_lambda += self.lambda_lr * float(self.jc_ema)
            self.current_lambda = float(
                np.clip(np.exp(ln_lambda), self.lambda_min, self.lambda_max)
            )

        # Combined reward R - lambda * C
        combined = [float(r - self.current_lambda * c) for r, c in zip(rewards, costs)]
        combined_np = np.asarray(combined, dtype=np.float32)

        # ---------- Logging (main process only) ----------
        try:
            import wandb  # type: ignore

            if os.environ.get("RANK", "0") == "0" and len(rewards_np) > 0:
                self.step += 1

                log_dict = {
                    # Raw stats
                    "train/reward_mean": float(rewards_np.mean()),
                    "train/reward_std": float(rewards_np.std()),
                    "train/reward_min": float(rewards_np.min()),
                    "train/reward_max": float(rewards_np.max()),
                    "train/cost_mean": float(costs_np.mean()),
                    "train/cost_std": float(costs_np.std()),
                    "train/cost_min": float(costs_np.min()),
                    "train/cost_max": float(costs_np.max()),
                    "train/combined_mean": float(combined_np.mean()),
                    "train/combined_std": float(combined_np.std()),
                    "train/combined_min": float(combined_np.min()),
                    "train/combined_max": float(combined_np.max()),
                    "train/safe_ratio": float(np.mean(costs_np <= 0.0)),
                    # ES-style naming
                    "helpfulness/mean": float(rewards_np.mean()),
                    "cost/mean": float(costs_np.mean()),
                }

                if self.lambda_adapt and self.jc_ema is not None:
                    log_dict.update(
                        {
                            "lambda/value": self.current_lambda,
                            "lambda/J_C": float(jc) if "jc" in locals() else 0.0,
                            "lambda/J_C_raw": float(jc_raw)
                            if "jc_raw" in locals()
                            else 0.0,
                            "lambda/J_C_ema": float(self.jc_ema),
                            "lambda/d": float(self.cost_threshold_d),
                        }
                    )

                # Periodic histograms for reward & cost distributions
                if self.hist_every > 0 and (self.step % self.hist_every == 0):
                    try:
                        log_dict["helpfulness/dist"] = wandb.Histogram(rewards_np)
                        log_dict["cost/dist"] = wandb.Histogram(costs_np)
                    except Exception as e:
                        print(f"Histogram logging error: {e}")

                wandb.log(log_dict, commit=False)

                # Periodic scatter of reward vs cost (sampled from this batch)
                if self.scatter_every > 0 and (self.step % self.scatter_every == 0):
                    try:
                        n = len(rewards_np)
                        if n > 0:
                            k = min(self.max_scatter_points, n)
                            idx = (
                                np.random.choice(n, size=k, replace=False)
                                if k < n
                                else np.arange(n)
                            )
                            table = wandb.Table(
                                data=[
                                    [float(rewards_np[j]), float(costs_np[j])]
                                    for j in idx
                                ],
                                columns=["reward", "cost"],
                            )
                            wandb.log(
                                {
                                    "scatter/reward_vs_cost": wandb.plot.scatter(
                                        table,
                                        "reward",
                                        "cost",
                                        title="Reward vs Cost (sampled, train)",
                                    )
                                },
                                commit=False,
                            )
                            # Save table in run summary for later inspection
                            try:
                                if wandb.run is not None:
                                    wandb.run.summary[
                                        "scatter/reward_vs_cost_table"
                                    ] = table
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Scatter logging error: {e}")

        except Exception as e:
            print(f"Logging error: {e}")

        return combined


def make_reward_func_from_api(cfg: dict) -> AdaptiveReward:
    return AdaptiveReward(cfg)


def run_eval_and_log(
    trainer,
    tokenizer,
    eval_prompts_text: List[str],
    reward_fn: AdaptiveReward,
    cfg: dict,
):
    """
    Run a greedy evaluation pass on eval_prompts_text and log eval/*
    metrics plus a scatter plot to W&B, mirroring the ES script.

    Only runs on rank 0.
    """
    if not eval_prompts_text:
        print("No eval prompts provided; skipping evaluation.")
        return

    if os.environ.get("RANK", "0") != "0":
        # Only the main process runs eval & logging
        return

    import torch  # local import to avoid torch dependency at import time

    model = trainer.accelerator.unwrap_model(trainer.model)
    device = trainer.accelerator.device
    model.to(device)
    model.eval()

    max_prompt_length = int(cfg.get("max_prompt_length", 512))
    max_completion_length = int(cfg.get("max_completion_length", 512))
    eval_batch_size = int(cfg.get("eval_batch_size", 8))
    scorer_batch_size = int(cfg.get("scorer_batch_size_eval", 16))

    all_rewards: List[float] = []
    all_costs: List[float] = []

    with torch.no_grad():
        for i in range(0, len(eval_prompts_text), eval_batch_size):
            batch_prompts = eval_prompts_text[i : i + eval_batch_size]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(device)

            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_completion_length,
                do_sample=False,  # greedy for eval, to match ES
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            completions: List[str] = []
            for full, prompt_text in zip(decoded, batch_prompts):
                if full.startswith(prompt_text):
                    completions.append(full[len(prompt_text) :])
                elif "ASSISTANT:" in full:
                    completions.append(full.split("ASSISTANT:", 1)[1])
                else:
                    completions.append(full)

            rewards, costs = call_scorer_api(
                reward_fn.scorer_url,
                batch_prompts,
                completions,
                batch_size=scorer_batch_size,
            )
            all_rewards.extend(rewards)
            all_costs.extend(costs)

    if not all_rewards:
        print("No eval rewards collected; skipping eval logging.")
        return

    rewards_np = np.asarray(all_rewards, dtype=np.float32)
    costs_np = np.asarray(all_costs, dtype=np.float32)
    combined_np = rewards_np - float(reward_fn.current_lambda) * costs_np

    safe_ratio = float(np.mean(costs_np <= 0.0))

    results = {
        "mean_reward": float(rewards_np.mean()),
        "mean_cost": float(costs_np.mean()),
        "mean_combined": float(combined_np.mean()),
        "safe_ratio": safe_ratio,
        "total_samples": int(len(all_rewards)),
    }

    print(
        f"[Eval] mean_reward={results['mean_reward']:.4f}, "
        f"mean_cost={results['mean_cost']:.4f}, "
        f"mean_combined={results['mean_combined']:.4f}, "
        f"safe_ratio={results['safe_ratio']:.4f}, "
        f"total_samples={results['total_samples']}"
    )

    # W&B logging
    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            step = getattr(trainer.state, "global_step", None)

            wandb.log(
                {
                    "eval/mean_reward": results["mean_reward"],
                    "eval/mean_cost": results["mean_cost"],
                    "eval/mean_combined": results["mean_combined"],
                    "eval/safe_ratio": results["safe_ratio"],
                    "eval/total_samples": results["total_samples"],
                },
                step=step,
            )

            # Scatter for eval
            try:
                table = wandb.Table(
                    data=[[float(r), float(c)] for r, c in zip(all_rewards, all_costs)],
                    columns=["reward", "cost"],
                )
                wandb.log(
                    {
                        "scatter/eval_reward_vs_cost": wandb.plot.scatter(
                            table,
                            "reward",
                            "cost",
                            title="Reward vs Cost (eval)",
                        )
                    },
                    step=step,
                )
                wandb.run.summary["scatter/eval_reward_vs_cost_table"] = table
            except Exception as e:
                print(f"Eval scatter logging error: {e}")

    except Exception as e:
        print(f"Eval logging error: {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("project", "grpo_alignment")
    cfg.setdefault("entity", None)

    # Ensure W&B env is set before Trainer initialization
    if cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(cfg["entity"])
    os.environ["WANDB_PROJECT"] = str(cfg["project"])

    import torch
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # Dataset (and keep prompt texts for eval)
    (
        train_ds,
        eval_ds,
        train_prompts_text,
        eval_prompts_text,
    ) = format_train_eval_datasets(cfg)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # GRPO / training config
    use_vllm = bool(int(os.environ.get("USE_VLLM", "0")))
    grpo_args = GRPOConfig(
        output_dir=os.path.join(cfg["output_dir"], f"beta{args.beta}_seed{args.seed}"),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        warmup_steps=cfg.get("warmup_steps", 0),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", 1000),
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 100),
        report_to=["wandb"],
        run_name=f"grpo_alignment_beta{args.beta}_seed{args.seed}",
        bf16=True,
        remove_unused_columns=False,
        # generation settings for policy sampling
        max_prompt_length=cfg.get("max_prompt_length", 512),
        max_completion_length=cfg.get("max_completion_length", 512),
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        num_generations=cfg["num_generations"],
        # GRPO specifics
        beta=float(args.beta),
        loss_type=cfg.get("loss_type", "grpo"),
        # vLLM
        use_vllm=use_vllm,
        # accelerate/deepspeed config path if any
        deepspeed=cfg.get("deepspeed_config") or None,
    )

    # Reward function (adaptive lambda + ES-style logging)
    reward_fn = make_reward_func_from_api(cfg)

    # Repro
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Trainer
    trainer = GRPOTrainer(
        model=cfg["model_name"],
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Train
    trainer.train()
    trainer.save_model()
    print("Training complete. Saved to:", grpo_args.output_dir)

    # Final evaluation (on rank 0 only)
    if eval_prompts_text:
        run_eval_and_log(trainer, tokenizer, eval_prompts_text, reward_fn, cfg)


if __name__ == "__main__":
    main()
