#!/usr/bin/env python
import os, sys, json, random, argparse

os.environ['TRANSFORMERS_CACHE'] = '/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface'
os.environ['HF_HOME'] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface"
os.environ['HF_HUB_CACHE'] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface/hub"
os.environ['HF_ASSETS_CACHE'] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface/assets"
os.environ['HF_TOKEN_PATH'] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/huggingface/token"
os.environ['XET_CACHE_DIR'] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/xet"
os.environ["VLLM_CACHE_ROOT"] = "/n/netscratch/kempner_sham_lab/Lab/itamarf/.cache/vllm_cache"

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
            line=line.strip()
            if line:
                yield json.loads(line)

def format_train_eval_datasets(cfg: dict, tokenizer=None):
    """
    Return (train_dataset, eval_dataset_or_None) with prompts formatted for TRL GRPO.
    TRL expects string prompts, so we apply chat template to get formatted strings.
    """
    from datasets import Dataset
    pkey = cfg.get("prompt_key", "prompt")
    skey = cfg.get("solution_key", "answer")

    train_rows = [r for r in read_jsonl_rows(cfg["train_jsonl"])]
    eval_rows  = [r for r in read_jsonl_rows(cfg["eval_jsonl"])] if (cfg.get("eval_jsonl") and os.path.exists(cfg["eval_jsonl"])) else []

    train_prompts = [{"prompt": str(r[pkey]).strip(), "answer": str(r[skey]).strip()} for r in train_rows]
    eval_prompts  = [{"prompt": str(r[pkey]).strip(), "answer": str(r[skey]).strip()} for r in eval_rows]


    print(f"Sample formatted prompt:\n{train_prompts[0]['prompt'][:200]}...")
    
    d_train = Dataset.from_list(train_prompts)
    d_eval  = Dataset.from_list(eval_prompts) if eval_prompts else None

    return d_train, d_eval


def make_reward_func_from_map(cfg: dict):
    """
    Paper reward (training-time):
      R = -|len(y) - len(s_k)|   (length in characters), s_k is the gold answer.
    """

    G = int(cfg["num_generations"])

    def reward_fn(completions, **kwargs):
        # TRL passes dataset columns via **kwargs; answers arrive under "answer".
        answers = kwargs.get("answer", None)

        rewards = []
        decoded_len = []
        for completion, answer in zip(completions, answers):
            # print("completion:", completion)
            text = completion if isinstance(completion, str) else str(completion)

            y = text.strip()
            a = (answer or "").strip()
            r = -abs(len(y) - len(a))
            # print("y:", y, "answer:", answer, "r:", r)
            rewards.append(float(r))
            decoded_len.append(len(y))
        # optional wandb lightweight logging on rank 0
        try:
            import wandb, os
            if os.environ.get("RANK", "0") == "0" and len(rewards) > 0:
                wandb.log({
                    "train/decoded_length/mean": float(sum(decoded_len)) / len(decoded_len),
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
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("project", "es_conciseness")
    cfg.setdefault("entity", None)

    import torch
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # Dataset
    train_ds, eval_ds = format_train_eval_datasets(cfg)
    
    # Tokenizer - match ES settings that work
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GRPO args
    use_vllm = bool(int(os.environ.get("USE_VLLM", "0")))
    grpo_args = GRPOConfig(
        output_dir=os.path.join(cfg["output_dir"], f"beta{args.beta}_seed{args.seed}"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        warmup_steps=cfg.get("warmup_steps", 0),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", 1000),
        logging_steps=cfg.get("logging_steps", 5),
        save_steps=cfg.get("save_steps", 200),
        report_to=["wandb"],
        run_name=f"qwen2.5-7b_grpo_conciseness_beta{args.beta}_seed{args.seed}",
        bf16=True,
        remove_unused_columns=False,

        # generation settings for policy sampling
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        num_generations=cfg["num_generations"],

        # GRPO specifics
        beta=float(args.beta),
        loss_type=cfg.get("loss_type", "grpo"),

        # vLLM off by default
        use_vllm=use_vllm,

        # accelerate/deepspeed config path if any
        deepspeed=cfg.get("deepspeed_config") or None
    )

    # Reward (training-time)
    reward_fn = make_reward_func_from_map(cfg)

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
    print("Run conciseness_reward_and_KL.py to evaluate this checkpoint.")

if __name__ == "__main__":
    main()
