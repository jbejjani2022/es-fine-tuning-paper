#!/usr/bin/env python
import os
import sys
import json
import random
import argparse
from typing import List, Tuple, Optional, Any

import requests
import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_callback import TrainingArguments

# Global counter for debug logging (first batch only)
_SCORER_CALL_COUNT = 0


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

def build_full_text(prompts: List[str], completions: List[str]) -> List[str]:
    """
    Build full conversation text from prompt and completion.
    
    Following the safe-rlhf pattern: just concatenate prompt + completion.
    The scorer will handle adding EOS tokens.
    
    Note: completions from TRL's GRPO trainer are already plain strings.
    """
    return [p + c for p, c in zip(prompts, completions)]

def call_scorer_api(
    scorer_url: str, texts: List[str], batch_size: int = 32
) -> Tuple[List[float], List[float]]:
    """
    Call the external scorer API to get rewards and costs.

    `prompts` should be either raw user strings or full conversation prefixes.
    The scorer service builds the canonical
      "BEGINNING OF CONVERSATION: USER: <prompt> ASSISTANT:<response><eos>"
    internally, matching the ES alignment setup.
    """
    global _SCORER_CALL_COUNT
    try:
        prompts, responses = [], []
        for text in texts:
            # Robustly extract between USER: and ASSISTANT:
            if 'USER:' in text and 'ASSISTANT:' in text:
                user_part = text.split('USER:', 1)[1]
                user = user_part.rsplit('ASSISTANT:', 1)[0].strip()
                resp = text.split('ASSISTANT:', 1)[1]
                prompts.append(user)
                responses.append(resp)
            else:
                # Fallback: treat whole text as prompt, empty response
                prompts.append(text)
                responses.append("")

        all_rewards, all_costs = [], []

        for i in range(0, len(prompts), batch_size):
            payload = {
                "prompts": prompts[i:i+batch_size],
                "responses": responses[i:i+batch_size],
            }
            r = requests.post(f"{scorer_url}/score_batch", json=payload, timeout=300)
            r.raise_for_status()
            result = r.json()
            all_rewards.extend(result["rewards"])
            all_costs.extend(result["costs"])

            # DEBUG: Log first batch results
            if _SCORER_CALL_COUNT == 1 and i == 0 and os.environ.get("RANK", "0") == "0":
                n_debug = min(5, len(result["rewards"]))
                print("\n" + "=" * 80)
                print("[DEBUG SCORER RESPONSE] First batch results from scorer API")
                print("=" * 80)
                for idx in range(n_debug):
                    print(f"  [{idx}] reward={result['rewards'][idx]:.4f}, cost={result['costs'][idx]:.4f}")
                print(f"  Batch reward range: [{min(result['rewards']):.4f}, {max(result['rewards']):.4f}]")
                print(f"  Batch cost range: [{min(result['costs']):.4f}, {max(result['costs']):.4f}]")
                print("=" * 80 + "\n")

        return all_rewards, all_costs
    except requests.exceptions.RequestException as e:
        print(f"[AdaptiveReward] Error calling scorer API: {e}")
        # Return zeros on error to avoid crashing, but this is bad
        return [0.0] * len(prompts), [0.0] * len(prompts)


class AdaptiveReward:
    """
    Reward callable passed to TRL GRPOTrainer.

    This wraps the Beaver reward & cost models served by the scorer API and
    maintains an adaptive Lagrange multiplier lambda to enforce a cost
    constraint, using the same update rule as the ES alignment script:

      ln λ <- ln λ + η * EMA(J_C)
      J_C = E[C] + d  (optionally clamped to max(J_C, 0))

    It also mirrors ES logging and adds rich debug information:
    - helpfulness/mean, cost/mean, combined/*
    - safety/safe_ratio and per-batch unsafe counts
    - histograms + scatter over an accumulated window
    - W&B tables with concrete (prompt, completion, reward, cost, combined) rows
    """

    def __init__(self, cfg: dict):
        self.scorer_url: str = cfg.get("scorer_url", "http://localhost:8000")
        self.lambda_adapt: bool = cfg.get("lambda_adapt", False)

        # Lambda parameters (match ES hyperparams by default)
        self.current_lambda: float = float(cfg.get("lambda_cost", 1.0))
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
        self.scorer_batch_size: int = int(cfg.get("scorer_batch_size", 32))

        # Debug table + print settings
        # Set debug_log_every: 1 in the YAML if you want a full batch table every step.
        self.debug_log_every: int = int(cfg.get("debug_log_every", 0))
        self.debug_log_n_samples: int = int(cfg.get("debug_log_n_samples", 8))
        self.debug_print_first: bool = bool(cfg.get("debug_print_first", True))
        self.debug_print_samples_per_batch: int = int(
            cfg.get("debug_print_samples_per_batch", 0)
        )
        self.debug_print_sample_chars: int = int(
            cfg.get("debug_print_sample_chars", 2000)
        )

        self.__name__ = "AdaptiveReward"

        self.jc_ema: Optional[float] = None
        self.step: int = 0

        # Buffers to accumulate many calls before plotting (like ES does per generation)
        self._accum_rewards: List[float] = []
        self._accum_costs: List[float] = []

        self._printed_example: bool = False

    # ---- helper to decode TRL's calling convention -------------------------
    @staticmethod
    def _looks_like_prompt_list(lst: List[Any]) -> bool:
        """Heuristic: our prompts are conversation strings with USER/ASSISTANT markers."""
        if not lst:
            return False
        s = str(lst[0])
        return ("BEGINNING OF CONVERSATION:" in s) or (
            "USER:" in s and "ASSISTANT:" in s
        )

    def _extract_prompts_and_completions(
        self, *args, **kwargs
    ) -> Tuple[List[str], List[str]]:
        """
        TRL may call custom reward functions in a few ways:
          - reward_fn(completions, prompts, **kwargs)   (current convention)
          - reward_fn(prompts, completions, **kwargs)   (older code / examples)
          - reward_fn(completions=comps, prompts=prompts, **kwargs)

        We robustly infer which is which using kwargs and a small heuristic.
        """
        # Keyword arguments take precedence if present
        prompts = kwargs.get("prompts")
        completions = kwargs.get("completions")

        if prompts is not None and completions is not None:
            return list(prompts), list(completions)

        # Otherwise, inspect positional args and pick the first two list-of-str
        list_args: List[List[Any]] = []
        for a in args:
            if isinstance(a, list):
                list_args.append(a)

        if len(list_args) < 2:
            raise ValueError(
                "AdaptiveReward expected at least two list arguments "
                "(prompts and completions), got: "
                f"{[type(a) for a in args]}"
            )

        a0, a1 = list_args[0], list_args[1]

        # Heuristic: whichever looks like a conversation prefix is the prompt list
        a0_is_prompt = self._looks_like_prompt_list(a0)
        a1_is_prompt = self._looks_like_prompt_list(a1)

        if a0_is_prompt and not a1_is_prompt:
            prompts, completions = a0, a1
        elif a1_is_prompt and not a0_is_prompt:
            prompts, completions = a1, a0
        else:
            # Fall back to TRL's documented convention: completions first, prompts second
            completions, prompts = a0, a1

        return list(prompts), list(completions)

    # -----------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        # Robustly recover prompts and completions from whatever TRL passes in
        # print("================================================")
        # print("args: ", args)
        # print("kwargs: ", kwargs)
        # print("================================================")
        prompts, completions = self._extract_prompts_and_completions(*args, **kwargs)

        if len(prompts) != len(completions):
            print(
                f"[AdaptiveReward WARNING] len(prompts)={len(prompts)} "
                f"!= len(completions)={len(completions)}. "
                "Scorer will receive min(len(p), len(c)) pairs."
            )
            n = min(len(prompts), len(completions))
            prompts = prompts[:n]
            completions = completions[:n]

        # Optionally print one concrete example to stdout for sanity checking
        if (
            self.debug_print_first
            and not self._printed_example
            and os.environ.get("RANK", "0") == "0"
            and len(prompts) > 0
        ):
            print("\n=== AdaptiveReward DEBUG example (first call on rank 0) ===")
            print(f"Prompt[0]: {repr(prompts[0])}")
            print(f"Completion[0]: {repr(completions[0])}")
            self._printed_example = True

        # Query external reward & cost models with the *correct* pairing:
        #   USER: prompt   ASSISTANT: completion
        full_texts = build_full_text(prompts, completions)
        rewards, costs = call_scorer_api(
            self.scorer_url,
            full_texts,
            batch_size=self.scorer_batch_size,
        )

        rewards_np = np.asarray(rewards, dtype=np.float32)
        costs_np = np.asarray(costs, dtype=np.float32)

        # Accumulate for scatter/hist over many calls (like ES's all_rewards_samples)
        self._accum_rewards.extend(rewards_np.tolist())
        self._accum_costs.extend(costs_np.tolist())

        # ---------- Adaptive Lambda Update (same as ES) ----------
        jc = 0.0
        jc_raw = 0.0
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

        # Small stdout debug for first few calls (batch-level summary)
        if os.environ.get("RANK", "0") == "0" and self.step < 5 and len(rewards_np) > 0:
            safe_ratio = float(np.mean(costs_np <= 0.0))
            print(
                f"[AdaptiveReward DEBUG] step={self.step} "
                f"batch_size={len(rewards_np)} "
                f"reward_mean={rewards_np.mean():.3f} "
                f"reward_min={rewards_np.min():.3f} "
                f"reward_max={rewards_np.max():.3f} "
                f"cost_mean={costs_np.mean():.3f} "
                f"cost_min={costs_np.min():.3f} "
                f"cost_max={costs_np.max():.3f} "
                f"safe_ratio={safe_ratio:.3f} "
                f"lambda={self.current_lambda:.4f}"
            )

        # Optional per-sample stdout logging for this batch
        if (
            self.debug_print_samples_per_batch > 0
            and os.environ.get("RANK", "0") == "0"
            and len(rewards_np) > 0
        ):
            n_show = min(self.debug_print_samples_per_batch, len(prompts))
            print(
                f"[AdaptiveReward DEBUG samples] step={self.step} "
                f"showing {n_show}/{len(prompts)} examples"
            )
            for i in range(n_show):
                prompt_snip = str(prompts[i])[: self.debug_print_sample_chars]
                completion_snip = str(completions[i])[: self.debug_print_sample_chars]
                print(
                    f"  [{i}] reward={rewards_np[i]:+.3f} "
                    f"cost={costs_np[i]:+.3f} "
                    f"combined={combined_np[i]:+.3f} "
                    f"unsafe={bool(costs_np[i] > 0.0)}"
                )
                print(f"     Prompt    : {repr(prompt_snip)}")
                print(f"     Completion: {repr(completion_snip)}")

        # ---------- Logging (main process only) ----------
        try:
            import wandb  # type: ignore

            if os.environ.get("RANK", "0") == "0" and len(rewards_np) > 0:
                self.step += 1

                safe_mask = costs_np <= 0.0
                safe_ratio_batch = float(np.mean(safe_mask))
                unsafe_count_batch = int(np.sum(costs_np > 0.0))
                total_count_batch = int(costs_np.size)

                log_dict = {
                    # ES-like metrics
                    "helpfulness/mean": float(rewards_np.mean()),
                    "cost/mean": float(costs_np.mean()),
                    "combined/mean": float(combined_np.mean()),
                    "combined/std": float(combined_np.std()),
                    "combined/min": float(combined_np.min()),
                    "combined/max": float(combined_np.max()),
                    # per-batch safety, to match ES
                    "safety/safe_ratio": safe_ratio_batch,
                    "safety/unsafe_count_batch": unsafe_count_batch,
                    "safety/total_count_batch": total_count_batch,
                    # keep explicit debug key for batch size
                    "debug/batch_size": total_count_batch,
                }

                if self.lambda_adapt and self.jc_ema is not None:
                    log_dict.update(
                        {
                            "lambda/value": self.current_lambda,
                            "lambda/J_C": float(jc),
                            "lambda/J_C_raw": float(jc_raw),
                            "lambda/J_C_ema": float(self.jc_ema),
                            "lambda/d": float(self.cost_threshold_d),
                        }
                    )

                # Periodic histograms for reward & cost distributions (accumulated window)
                if self.hist_every > 0 and (self.step % self.hist_every == 0):
                    try:
                        accum_r = np.asarray(self._accum_rewards, dtype=np.float32)
                        accum_c = np.asarray(self._accum_costs, dtype=np.float32)
                        if accum_r.size > 0:
                            log_dict["helpfulness/dist"] = wandb.Histogram(accum_r)
                            log_dict["cost/dist"] = wandb.Histogram(accum_c)
                    except Exception as e:
                        print(f"[AdaptiveReward] Histogram logging error: {e}")

                wandb.log(log_dict, commit=False)

                # Periodic scatter of reward vs cost from the accumulated window
                if self.scatter_every > 0 and (self.step % self.scatter_every == 0):
                    try:
                        accum_r = np.asarray(self._accum_rewards, dtype=np.float32)
                        accum_c = np.asarray(self._accum_costs, dtype=np.float32)

                        if accum_r.size == 0:
                            accum_r = rewards_np
                            accum_c = costs_np

                        n = accum_r.size
                        if n > 0:
                            k = min(self.max_scatter_points, n)
                            idx = (
                                np.random.choice(n, size=k, replace=False)
                                if k < n
                                else np.arange(n)
                            )
                            table = wandb.Table(
                                data=[
                                    [float(accum_r[j]), float(accum_c[j])]
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

                        # Reset window after logging, like ES resets per generation
                        self._accum_rewards.clear()
                        self._accum_costs.clear()

                    except Exception as e:
                        print(f"[AdaptiveReward] Scatter logging error: {e}")

                # Full-batch debug table of raw rows so we can inspect safety directly
                if self.debug_log_every > 0 and (self.step % self.debug_log_every == 0):
                    try:
                        rows = []
                        for p, c, r_val, c_val, comb_val in zip(
                            prompts,
                            completions,
                            rewards_np.tolist(),
                            costs_np.tolist(),
                            combined_np.tolist(),
                        ):
                            # NO truncation - log full prompts and completions
                            rows.append(
                                [
                                    str(p),
                                    str(c),
                                    float(r_val),
                                    float(c_val),
                                    float(comb_val),
                                    bool(c_val > 0.0),
                                ]
                            )
                        if rows:
                            debug_table = wandb.Table(
                                columns=[
                                    "prompt",
                                    "completion",
                                    "reward",
                                    "cost",
                                    "combined",
                                    "unsafe",
                                ],
                                data=rows,
                            )
                            wandb.log(
                                {"debug/train_batch_samples": debug_table}, commit=False
                            )
                    except Exception as e:
                        print(f"[AdaptiveReward] Debug table logging error: {e}")

        except Exception as e:
            print(f"[AdaptiveReward] Logging error: {e}")

        return combined


def make_reward_func_from_api(cfg: dict) -> AdaptiveReward:
    return AdaptiveReward(cfg)


class PeriodicEvalCallback(TrainerCallback):
    """Callback to run evaluation periodically during GRPO training."""
    
    def __init__(
        self,
        eval_every: int,
        eval_prompts_text: List[str],
        reward_fn: "AdaptiveReward",
        cfg: dict,
        tokenizer,
    ):
        self.eval_every = eval_every
        self.eval_prompts_text = eval_prompts_text
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.tokenizer = tokenizer
        self._last_eval_step = -1
        self._trainer = None  # Will be set when callback is added
    
    def set_trainer(self, trainer):
        """Set the trainer reference for evaluation."""
        self._trainer = trainer
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Only evaluate every eval_every steps, and only on main process
        if self.eval_every <= 0:
            return
        if state.global_step <= self._last_eval_step:
            return
        if state.global_step % self.eval_every != 0:
            return
        if os.environ.get("RANK", "0") != "0":
            return
        
        if self._trainer is None:
            print("[PeriodicEvalCallback] No trainer reference set, skipping eval")
            return
        
        print(f"\n[PeriodicEvalCallback] Running evaluation at step {state.global_step}...")
        self._last_eval_step = state.global_step
        
        try:
            # Get the unwrapped model from the trainer
            model = self._trainer.accelerator.unwrap_model(self._trainer.model)
            run_eval_and_log_simple(
                model=model,
                tokenizer=self.tokenizer,
                eval_prompts_text=self.eval_prompts_text,
                reward_fn=self.reward_fn,
                cfg=self.cfg,
                step=state.global_step,
            )
        except Exception as e:
            print(f"[PeriodicEvalCallback] Eval error: {e}")


def run_eval_and_log_simple(
    model,
    tokenizer,
    eval_prompts_text: List[str],
    reward_fn: "AdaptiveReward",
    cfg: dict,
    step: int,
):
    """
    Simplified eval function that can be called from callback.
    Uses the same full-text scoring approach as training.
    """
    import torch
    
    if not eval_prompts_text:
        return
    
    if os.environ.get("RANK", "0") != "0":
        return
    
    device = next(model.parameters()).device
    model.eval()
    
    max_prompt_length = int(cfg.get("max_prompt_length", 512))
    max_completion_length = int(cfg.get("max_completion_length", 512))
    eval_batch_size = int(cfg.get("eval_batch_size", 64))
    scorer_batch_size = int(cfg.get("scorer_batch_size_eval", 32))
    
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
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # Full decoded texts (prompt + generated response)
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            
            # Extract completions for call_scorer_api (prompt + completion -> full text)
            # completions: List[str] = []
            # for full_text, prompt in zip(decoded, batch_prompts):
            #     if full_text.startswith(prompt):
            #         completions.append(full_text[len(prompt):])
            #     else:
            #         # Fallback: full text is the completion
            #         completions.append(full_text)
            full_texts = build_full_text(batch_prompts, decoded)
            # call_scorer_api will build full_text = prompt + completion
            rewards, costs = call_scorer_api(
                reward_fn.scorer_url,
                full_texts,
                batch_size=scorer_batch_size,
            )
            all_rewards.extend(rewards)
            all_costs.extend(costs)
    
    if not all_rewards:
        return
    
    rewards_np = np.asarray(all_rewards, dtype=np.float32)
    costs_np = np.asarray(all_costs, dtype=np.float32)
    combined_np = rewards_np - float(reward_fn.current_lambda) * costs_np
    safe_ratio = float(np.mean(costs_np <= 0.0))
    
    print(
        f"[Eval step={step}] mean_reward={rewards_np.mean():.4f}, "
        f"mean_cost={costs_np.mean():.4f}, "
        f"mean_combined={combined_np.mean():.4f}, "
        f"safe_ratio={safe_ratio:.4f}"
    )
    
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "eval/mean_reward": float(rewards_np.mean()),
                "eval/mean_cost": float(costs_np.mean()),
                "eval/mean_combined": float(combined_np.mean()),
                "eval/safe_ratio": safe_ratio,
                "eval/total_samples": len(all_rewards),
            }, step=step)
    except Exception as e:
        print(f"[Eval] W&B logging error: {e}")
    
    model.train()


def run_eval_and_log(
    trainer,
    tokenizer,
    eval_prompts_text: List[str],
    reward_fn: AdaptiveReward,
    cfg: dict,
):
    """
    Run a greedy evaluation pass on eval_prompts_text and log eval/*
    metrics plus a scatter plot and optional eval table to W&B.

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
    eval_batch_size = int(cfg.get("eval_batch_size", 64))
    scorer_batch_size = int(cfg.get("scorer_batch_size_eval", 32))
    eval_table_n_samples = int(cfg.get("eval_table_n_samples", 0))

    all_rewards: List[float] = []
    all_costs: List[float] = []
    all_prompts: List[str] = []
    all_completions: List[str] = []

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

            # Full decoded texts (prompt + generated response)
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            # Extract completions for scoring
            # completions: List[str] = []
            # for full, prompt_text in zip(decoded, batch_prompts):
            #     if full.startswith(prompt_text):
            #         completions.append(full[len(prompt_text) :])
            #     elif "ASSISTANT:" in full:
            #         completions.append(full.split("ASSISTANT:", 1)[1])
            #     else:
            #         completions.append(full)
            full_texts = build_full_text(batch_prompts, decoded)
            # call_scorer_api will build full_text = prompt + completion
            rewards, costs = call_scorer_api(
                reward_fn.scorer_url,
                full_texts,
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
                print(f"[Eval] scatter logging error: {e}")

            # Optional: eval table with text + scores
            if eval_table_n_samples > 0:
                try:
                    k = min(eval_table_n_samples, len(all_prompts))
                    idx = (
                        np.random.choice(len(all_prompts), size=k, replace=False)
                        if k < len(all_prompts)
                        else np.arange(len(all_prompts))
                    )
                    rows = []
                    for j in idx:
                        # NO truncation - log full prompts and completions
                        rows.append(
                            [
                                str(all_prompts[j]),
                                str(all_completions[j]),
                                float(all_rewards[j]),
                                float(all_costs[j]),
                            ]
                        )
                    eval_table = wandb.Table(
                        columns=["prompt", "completion", "reward", "cost"], data=rows
                    )
                    wandb.log({"eval/samples": eval_table}, step=step)
                except Exception as e:
                    print(f"[Eval] table logging error: {e}")

    except Exception as e:
        print(f"[Eval] logging error: {e}")

    return results


def run_single_training(cfg: dict, beta: float, seed: int):
    """Run a single GRPO training with given beta and seed."""
    global _SCORER_CALL_COUNT
    _SCORER_CALL_COUNT = 0  # Reset for each run

    import torch
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    print("\n" + "=" * 80)
    print(f"Starting GRPO training with beta={beta}, seed={seed}")
    print("=" * 80 + "\n")

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
        output_dir=os.path.join(cfg["output_dir"], f"beta{beta}_seed{seed}_unified"),
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
        run_name=f"grpo_align_beta{beta}_seed{seed}_unified",
        bf16=True,
        remove_unused_columns=False,
        # generation settings for policy sampling
        max_prompt_length=cfg.get("max_prompt_length", 512),
        max_completion_length=cfg.get("max_completion_length", 512),
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        num_generations=cfg["num_generations"],
        # GRPO specifics
        beta=float(beta),
        loss_type=cfg.get("loss_type", "grpo"),
        # vLLM
        use_vllm=use_vllm,
        # accelerate/deepspeed config path if any
        deepspeed=cfg.get("deepspeed_config") or None,
    )

    # Reward function (adaptive lambda + ES-style logging)
    reward_fn = make_reward_func_from_api(cfg)

    # Repro
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Trainer
    trainer = GRPOTrainer(
        model=cfg["model_name"],
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # Add periodic evaluation callback if eval_every is set
    eval_every = int(cfg.get("eval_every", 0))
    eval_callback = None
    if eval_every > 0 and eval_prompts_text:
        eval_callback = PeriodicEvalCallback(
            eval_every=eval_every,
            eval_prompts_text=eval_prompts_text,
            reward_fn=reward_fn,
            cfg=cfg,
            tokenizer=tokenizer,
        )
        eval_callback.set_trainer(trainer)  # Set trainer reference for eval
        trainer.add_callback(eval_callback)
        print(f"[INFO] Added periodic evaluation callback (eval_every={eval_every})")

    # Train
    trainer.train()
    trainer.save_model()
    print("Training complete. Saved to:", grpo_args.output_dir)

    # Final evaluation (on rank 0 only)
    if eval_prompts_text:
        run_eval_and_log(trainer, tokenizer, eval_prompts_text, reward_fn, cfg)

    return grpo_args.output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--beta",
        type=str,
        default="0.01",
        help="Beta penalty value(s). Comma-separated for sweep, e.g. '0.001,0.01,0.1'"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("project", "grpo_alignment")
    cfg.setdefault("entity", None)

    # Ensure W&B env is set before Trainer initialization
    if cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(cfg["entity"])
    os.environ["WANDB_PROJECT"] = str(cfg["project"])

    # Parse beta values (support comma-separated sweep)
    beta_values = [float(b.strip()) for b in args.beta.split(",")]

    print("\n" + "#" * 80)
    print(f"# GRPO ALIGNMENT TRAINING")
    print(f"# Beta values to sweep: {beta_values}")
    print(f"# Seed: {args.seed}")
    print("#" * 80 + "\n")

    # Run training for each beta value
    output_dirs = []
    for beta in beta_values:
        output_dir = run_single_training(cfg, beta, args.seed)
        output_dirs.append(output_dir)

    print("\n" + "#" * 80)
    print("# ALL TRAINING RUNS COMPLETE")
    print(f"# Output directories:")
    for od in output_dirs:
        print(f"#   - {od}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
