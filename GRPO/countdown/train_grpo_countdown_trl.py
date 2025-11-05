#!/usr/bin/env python
import os, sys, json, random, argparse
from typing import List, Dict, Any


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


def read_json_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def build_datasets(cfg: dict):
    """
    Return (train_dataset, eval_dataset_or_None) for TRL GRPO.
    Countdown JSON has fields: context, numbers, target.
    """
    from datasets import Dataset

    rows = read_json_rows(cfg["data_json"])  # full list

    # Default split: first 200 for train, rest for eval (matches ES script)
    split_train = int(cfg.get("train_size", 200))
    train_rows = rows[:split_train]
    eval_rows = rows[split_train:]

    pkey = cfg.get("prompt_key", "context")
    nkey = cfg.get("numbers_key", "numbers")
    tkey = cfg.get("target_key", "target")

    def _to_float(value):
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except Exception:
            return None

    train_prompts = [
        {
            "prompt": str(r[pkey]).strip(),
            "numbers": list(r.get(nkey, [])),
            "target": _to_float(r.get(tkey)),
        }
        for r in train_rows
    ]
    eval_prompts = [
        {
            "prompt": str(r[pkey]).strip(),
            "numbers": list(r.get(nkey, [])),
            "target": _to_float(r.get(tkey)),
        }
        for r in eval_rows
    ]

    d_train = Dataset.from_list(train_prompts)
    d_eval = Dataset.from_list(eval_prompts) if eval_prompts else None
    print(f"Countdown sample prompt:\n{train_prompts[0]['prompt'][:200]}...")

    return d_train, d_eval


def make_countdown_reward_fn(cfg: dict):
    """
    Training-time reward using the paper's Countdown reward.
    TRL passes dataset columns via **kwargs: we expect numbers, target.
    """

    def reward_fn(completions, **kwargs):
        from countdown.countdown_task import reward_function

        numbers_list = kwargs.get("numbers", [])
        targets_list = kwargs.get("target", [])

        rewards = []
        fmt_acc = 0
        ans_acc = 0

        def _to_float(value):
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(str(value))
            except Exception:
                return None

        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else str(completion)

            numbers = numbers_list[i] if i < len(numbers_list) else []
            target = _to_float(targets_list[i]) if i < len(targets_list) else None

            r = reward_function(text, numbers=numbers, target=target)
            rewards.append(float(r["reward"]))
            fmt_acc += 1 if float(r["reward_info"]["format_reward"]) == 1.0 else 0
            ans_acc += 1 if float(r["reward_info"]["answer_reward"]) == 1.0 else 0

        # Lightweight wandb logging on rank 0
        try:
            import wandb
            if os.environ.get("RANK", "0") == "0" and len(rewards) > 0:
                wandb.log({
                    "train/format_accuracy": fmt_acc / max(1, len(rewards)),
                    "train/answer_accuracy": ans_acc / max(1, len(rewards)),
                    "train/reward_raw/mean": float(sum(rewards)) / len(rewards),
                }, commit=False)
        except Exception:
            pass

        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=None, help="Override LR from config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("project", "es_countdown")
    cfg.setdefault("entity", None)

    import torch
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # Dataset
    train_ds, eval_ds = build_datasets(cfg)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # GRPO args
    use_vllm = bool(int(os.environ.get("USE_VLLM", "0")))
    lr = float(args.learning_rate) if args.learning_rate is not None else float(cfg["learning_rate"])
    grpo_args = GRPOConfig(
        output_dir=os.path.join(cfg["output_dir"], f"beta{args.beta}_lr{lr}_seed{args.seed}"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=lr,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        warmup_steps=cfg.get("warmup_steps", 0),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", 1000),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        report_to=["wandb"],
        run_name=f"qwen2.5-1.5b_grpo_countdown_beta{args.beta}_lr{lr}_seed{args.seed}_left_pad",
        bf16=True,
        remove_unused_columns=False,

        # generation settings for policy sampling
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg.get("temperature", 0.7),
        top_p=cfg.get("top_p", 1.0),
        num_generations=cfg["num_generations"],

        # GRPO specifics
        beta=float(args.beta),
        loss_type=cfg.get("loss_type", "grpo"),

        # vLLM off by default
        use_vllm=use_vllm,

        # deepspeed config path if any
        deepspeed=cfg.get("deepspeed_config") or None,
    )

    # Reward
    reward_fn = make_countdown_reward_fn(cfg)

    # Trainer
    trainer = GRPOTrainer(
        model=cfg["model_name"],
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # W&B
    if cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(cfg["entity"])
    os.environ["WANDB_PROJECT"] = str(cfg["project"])

    # Repro
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Train
    trainer.train()
    trainer.save_model()
    print("Training complete. Saved to:", grpo_args.output_dir)


if __name__ == "__main__":
    main()


