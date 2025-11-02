import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import argparse
from datetime import datetime


logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate fine-tuned model using conciseness reward only')
parser.add_argument('--model', type=str, required=True,
                    help='Name of HuggingFace model or path to a saved model checkpoint')
parser.add_argument('--baseline_model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                    help='Optional baseline model name (not used, kept for CLI compatibility)')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache',
                    help='HuggingFace cache directory')
parser.add_argument('--precision', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'],
                    help='Model precision')
parser.add_argument('--max_new_tokens', type=int, default=128,
                    help='Maximum number of tokens to generate')
parser.add_argument('--do_sample', action='store_true', default=True,
                    help='Enable sampling vs greedy decoding')
parser.add_argument('--num_samples', type=int, default=20,
                    help='Number of responses to generate per prompt')
parser.add_argument('--eval_data_path', type=str, default='conciseness/data/eval.jsonl',
                    help='Path to evaluation data file')
parser.add_argument('--print-examples', action='store_true', default=False,
                    help='Print one example generation per test sample with its reward')
parser.add_argument('--output_json', type=str, default=None,
                    help='Path to save results (.json or .jsonl). If ends with .jsonl, appends a line.')
parser.add_argument('--seed', type=int, default=None,
                    help='Seed for random number generator')
parser.add_argument('--beta', type=float, default=None,
                    help='Beta for GRPO (stored for bookkeeping)')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=1.0,
                    help='Top-p for sampling')
args = parser.parse_args()


def compute_reward(generated_text, target_text):
    """Compute reward based on length difference (from es_fine-tuning_conciseness_iid.py)"""
    return -abs(len(generated_text) - len(target_text))


def compute_per_token_logps(model, input_ids, attention_mask):
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits.float(), dim=-1)

    shift_log_probs = log_probs[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    per_token_logps = torch.gather(shift_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return per_token_logps


def main():
    print("=" * 80)
    print("Conciseness Reward Evaluation")
    print("=" * 80)
    print(f"Fine-tuned model: {args.model}")
    print(f"Baseline model: {args.baseline_model_name}")
    print(f"Eval data path: {args.eval_data_path}")
    print(f"Precision: {args.precision}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Do sample: {args.do_sample}")
    print(f"Num samples per prompt: {args.num_samples}")
    print("=" * 80)

    # Optional seeding for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Determine device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.precision == 'fp16':
        dtype = torch.float16
    elif args.precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load fine-tuned model
    print("\nLoading fine-tuned model...")
    model_ft = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.hf_cache_dir,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model_ft.eval()

    print("\nLoading reference model...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        args.baseline_model_name,
        cache_dir=args.hf_cache_dir,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model_ref.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        cache_dir=args.hf_cache_dir
    )

    # Set padding side for generation
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model and tokenizer loaded successfully")

    # Load evaluation dataset
    print("\nLoading evaluation dataset...")
    # Use absolute path if provided, otherwise relative to script location
    if os.path.isabs(args.eval_data_path):
        data_path = args.eval_data_path
    else:
        data_path = os.path.join(os.path.dirname(__file__), '..', args.eval_data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                question = item['question']
                answer = item['answer']
                dataset.append((question, answer))

    print(f"Loaded {len(dataset)} evaluation samples")

    # Generate responses and compute metrics
    print("\nGenerating responses and computing rewards...")
    all_rewards = []
    all_answer_token_counts = []
    examples = []  # Store (question, generated_answer, reward) tuples for printing
    all_per_token_kls = []
    per_sample_kl_means = []

    for question_idx, (question, target_answer) in enumerate(dataset):
        print(f"Processing question {question_idx + 1}/{len(dataset)}...")

        question_rewards = []
        question_answer_token_counts = []

        for sample_idx in range(args.num_samples):
            print(f"  Sample {sample_idx + 1}/{args.num_samples}...")

            tokenized_inputs = tokenizer(
                [question],
                return_tensors="pt",
                padding=True,
            )
            input_ids = tokenized_inputs["input_ids"].to(device)
            attention_mask = tokenized_inputs["attention_mask"].to(device)

            with torch.inference_mode():
                outputs = model_ft.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    # temperature=args.temperature,
                    # top_p=args.top_p,
                    # pad_token_id=tokenizer.pad_token_id,
                    # eos_token_id=tokenizer.eos_token_id,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            input_len = input_ids.shape[1]
            full_ids = outputs[0]
            answer_ids = full_ids[input_len:]

            try:
                generated_answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            except TypeError:
                tokens = tokenizer.convert_ids_to_tokens(answer_ids, skip_special_tokens=True)
                filtered = [t for t in tokens if t is not None]
                generated_answer = tokenizer.convert_tokens_to_string(filtered).strip()

            reward = compute_reward(generated_answer, target_answer)
            question_rewards.append(reward)

            num_answer_tokens = max(0, full_ids.shape[0] - input_len)
            question_answer_token_counts.append(int(num_answer_tokens))

            full_input_ids = full_ids.unsqueeze(0).to(device)
            full_attention_mask = torch.ones_like(full_input_ids, device=device)

            ft_per_token_logps = compute_per_token_logps(model_ft, full_input_ids, full_attention_mask)
            ref_per_token_logps = compute_per_token_logps(model_ref, full_input_ids, full_attention_mask)

            gen_start_idx = max(input_len - 1, 0)
            if gen_start_idx < ft_per_token_logps.shape[1]:
                ft_generated_logps = ft_per_token_logps[:, gen_start_idx:]
                ref_generated_logps = ref_per_token_logps[:, gen_start_idx:]

                generated_token_ids = full_input_ids[:, 1:][:, gen_start_idx:]

                if tokenizer.eos_token_id is not None:
                    eos_positions = (generated_token_ids[0] == tokenizer.eos_token_id).nonzero(as_tuple=False)
                    if eos_positions.numel() > 0:
                        cutoff = eos_positions[0, 0].item() + 1
                        ft_generated_logps = ft_generated_logps[:, :cutoff]
                        ref_generated_logps = ref_generated_logps[:, :cutoff]
                        generated_token_ids = generated_token_ids[:, :cutoff]

                if ft_generated_logps.shape[1] > 0:
                    logp_diff = ref_generated_logps - ft_generated_logps
                    per_token_kl = torch.exp(logp_diff) - logp_diff - 1
                    per_token_kl = per_token_kl.squeeze(0)
                    per_token_kl_list = per_token_kl.detach().cpu().tolist()

                    if per_token_kl_list:
                        all_per_token_kls.extend(per_token_kl_list)
                        mean_kl_value = float(np.mean(per_token_kl_list))
                        per_sample_kl_means.append(mean_kl_value)
                        print(f"    Per-token KL: {per_token_kl_list}")
                        print(f"    Mean per-token KL: {mean_kl_value:.6f}")

            if args.print_examples and sample_idx == 0:
                examples.append((question, generated_answer, reward))

            del input_ids, attention_mask, outputs
            del full_input_ids, full_attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_rewards.extend(question_rewards)
        all_answer_token_counts.extend(question_answer_token_counts)

    # Calculate final metrics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    if len(all_rewards) == 0:
        raise RuntimeError("No rewards computed. Check dataset and generation settings.")

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)

    normalized_mean_reward = (mean_reward + 2000) / 2001
    normalized_std_reward = std_reward / 2001

    total_kl_tokens = len(all_per_token_kls)
    mean_per_token_kl = float(np.mean(all_per_token_kls)) if total_kl_tokens > 0 else float('nan')

    total_answer_tokens = int(np.sum(all_answer_token_counts))
    mean_answer_tokens = (total_answer_tokens / len(all_answer_token_counts)) if all_answer_token_counts else float('nan')

    print(f"\nReward (Length-based):")
    print(f"  Mean: {mean_reward:.4f}")
    print(f"  Normalized mean: {normalized_mean_reward:.4f}")
    print(f"  Std:  {std_reward:.4f}")
    print(f"  Normalized std: {normalized_std_reward:.4f}")
    print(f"\nKL Divergence:")
    if total_kl_tokens > 0:
        print(f"  Mean per-token KL: {mean_per_token_kl:.6f}")
        print(f"  Total KL tokens: {total_kl_tokens}")
    else:
        print("  No generated tokens; KL statistics unavailable.")
    print(f"\nTotal samples evaluated: {len(all_rewards)}")
    print(f"Total answer tokens: {total_answer_tokens}")
    print(f"Mean answer tokens per sample: {mean_answer_tokens:.2f}")

    # Print examples if requested
    if args.print_examples:
        print("\n" + "=" * 80)
        print("EXAMPLE GENERATIONS (1 per test sample)")
        print("=" * 80)
        for idx, (question, generated_answer, reward) in enumerate(examples):
            print(f"\n--- Example {idx + 1}/{len(examples)} ---")
            print(f"Question: {question}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Reward: {reward:.4f}")

    print("=" * 80)

    # Prepare results payload
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "seed": args.seed,
        "beta": args.beta,
        "baseline_model_name": args.baseline_model_name,
        "eval_data_path": data_path,
        "precision": args.precision,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_samples": args.num_samples,
        "total_samples_evaluated": int(len(all_rewards)),
        "answer_tokens": {
            "total": int(total_answer_tokens),
            "mean_per_sample": float(mean_answer_tokens) if not np.isnan(mean_answer_tokens) else float('nan'),
        },
        "reward": {
            "mean": float(mean_reward),
            "std": float(std_reward),
            "normalized": {
                "mean": float(normalized_mean_reward),
                "std": float(normalized_std_reward),
            },
        },
        "kl": {
            "total_tokens": int(total_kl_tokens),
            "mean_per_token": float(mean_per_token_kl) if not np.isnan(mean_per_token_kl) else float('nan'),
            "per_sample_mean": [float(x) for x in per_sample_kl_means],
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }

    # Decide output path and persist
    default_output = os.path.join(os.path.dirname(__file__), '..', 'logs', f'conciseness_reward_results_beta{args.beta}_{args.seed}.jsonl')
    output_path = args.output_json if args.output_json else default_output
    os.makedirs(os.path.abspath(os.path.join(output_path, os.pardir)), exist_ok=True)
    try:
        if output_path.endswith('.jsonl'):
            with open(output_path, 'a') as f:
                f.write(json.dumps(results) + "\n")
        else:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        print(f"Saved results to {output_path}")
    except Exception as e:
        print(f"Failed to save results to {output_path}: {e}")


if __name__ == "__main__":
    main()

