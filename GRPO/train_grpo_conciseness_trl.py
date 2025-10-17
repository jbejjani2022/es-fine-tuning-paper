#!/usr/bin/env python
import os, sys, json, math, random, argparse, pathlib
os.environ['TRANSFORMERS_CACHE'] = '/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface'
os.environ['HF_HOME'] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface"
os.environ['HF_HUB_CACHE'] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface/hub"
os.environ['HF_ASSETS_CACHE'] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface/assets"
os.environ['HF_TOKEN_PATH'] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface/token"
os.environ['XET_CACHE_DIR'] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/xet"
os.environ["VLLM_CACHE_ROOT"] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/vllm_cache"
from typing import List, Dict, Any

# Light YAML parser without external deps: allow JSON superset; fallback to pyyaml if installed.
def load_yaml(path: str) -> dict:
    text = open(path, "r").read()
    try:
        # Try JSON first
        return json.loads(text)
    except Exception:
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text)
        except Exception:
            raise RuntimeError(f"Please install pyyaml or provide valid JSON at {path}.")

def read_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            data.append(json.loads(line))
    return data

def format_train_eval_datasets(cfg: dict):
    """Return (train_dataset, eval_dataset_or_None, prompt2len)."""
    from datasets import Dataset

    pkey = cfg.get("prompt_key", "prompt")
    skey = cfg.get("solution_key", "answer")

    def read_jsonl(path):
        with open(path) as f:
            for line in f:
                line=line.strip()
                if line:
                    yield json.loads(line)

    # Build prompt→shortest-solution-length from BOTH splits (train+eval)
    prompt2len = {}
    def absorb(path):
        for r in read_jsonl(path):
            p = str(r[pkey]).strip()
            s = str(r[skey]).strip()
            L = len(s)
            prompt2len[p] = min(L, prompt2len.get(p, L))

    absorb(cfg["train_jsonl"])
    if cfg.get("eval_jsonl") and os.path.exists(cfg["eval_jsonl"]):
        absorb(cfg["eval_jsonl"])

    # TRL needs a dataset with a 'prompt' column
    train_prompts = [{"prompt": str(r[pkey]).strip()} for r in read_jsonl(cfg["train_jsonl"])]
    d_train = Dataset.from_list(train_prompts)

    d_eval = None
    if cfg.get("eval_jsonl") and os.path.exists(cfg["eval_jsonl"]):
        eval_prompts = [{"prompt": str(r[pkey]).strip()} for r in read_jsonl(cfg["eval_jsonl"])]
        d_eval = Dataset.from_list(eval_prompts)

    # Quick sanity: every train prompt must be in the map
    missing = [row["prompt"] for row in train_prompts if row["prompt"] not in prompt2len]
    if missing:
        raise ValueError(f"Missing gold answer for train prompt: {missing[0]!r}")

    return d_train, d_eval, prompt2len


def make_reward_func_from_map(prompt2len: dict):
    """Paper reward: R = -|len(y) - len(s_k)| (string lengths)."""
    def reward_fn(completions, prompts=None, **_):
        rewards = []
        for comp, p in zip(completions, prompts or []):
            # TRL gives each completion as [{'role':'assistant','content': '...'}]
            if isinstance(comp, list) and comp and isinstance(comp[0], dict):
                y = (comp[0].get("content","") or "").strip()
            elif isinstance(comp, dict):
                y = (comp.get("content","") or "").strip()
            else:
                y = str(comp).strip()

            Ls = prompt2len[str(p).strip()]          # shortest gold length
            rewards.append(-abs(len(y) - int(Ls)))   # exact training reward
        return rewards
    return reward_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("project", "es_conciseness")
    cfg.setdefault("entity", None)

    import torch
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # Dataset
    train_ds, eval_ds, prompt2len = format_train_eval_datasets(cfg)

    # Tokenizer left padding (required by GRPOTrainer)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build GRPOConfig
    grpo_args = GRPOConfig(
        output_dir=os.path.join(cfg["output_dir"], f"beta{args.beta}_seed{args.seed}"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", -1),
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 0),
        report_to=["wandb"],
        run_name=f"qwen2.5-7b_grpo_conciseness_beta{args.beta}_seed{args.seed}",
        bf16=True,
        remove_unused_columns=False,
        # generation
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        num_generations=cfg["num_generations"],
        # GRPO specifics
        beta=float(args.beta),
        scale_rewards=cfg.get("scale_rewards","batch"),
        loss_type=cfg.get("loss_type","dr_grpo"),
        mask_truncated_completions=cfg.get("mask_truncated_completions", False),
        # vLLM off by default; you can turn on with use_vllm=True
        use_vllm=False,
        # deepspeed
        deepspeed=cfg.get("deepspeed_config") or None
    )

    # Reward
    reward_fn = make_reward_func_from_map(prompt2len)

    # Trainer
    trainer = GRPOTrainer(
        model=cfg["model_name"],
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds
    )

    # Memory optimizations
    try:
        model = trainer.model
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False
    except Exception:
        pass

    # Note: GRPOTrainer does not expose add_evaluation_dataset; evaluate held-out outside training if needed.

    # W&B environment
    if cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = str(cfg["entity"])
    os.environ["WANDB_PROJECT"] = str(cfg["project"])

    # Repro
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    trainer.train()
    trainer.save_model()
    print("Training complete. Saved to:", grpo_args.output_dir)

if __name__ == "__main__":
    main()