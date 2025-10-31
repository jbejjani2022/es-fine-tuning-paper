import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import argparse
import os
import json
import numpy as np
import sys

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate KL divergence between fine-tuned and baseline models')
parser.add_argument('--model', type=str, required=True, 
                    help='Name of HuggingFace model or path to a saved model checkpoint')
parser.add_argument('--baseline_model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct',
                    help='Base model name to compare against')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache',
                    help='HuggingFace cache directory')
parser.add_argument('--precision', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'],
                    help='Model precision')
parser.add_argument('--max_new_tokens', type=int, default=1024,
                    help='Maximum number of tokens to generate')
parser.add_argument('--do_sample', action='store_true', default=False,
                    help='Enable sampling vs greedy decoding')
parser.add_argument('--num_samples', type=int, default=20,
                    help='Number of responses to generate per prompt')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for evaluation')
parser.add_argument('--eval_data_path', type=str, default='conciseness/data/eval.jsonl',
                    help='Path to evaluation data file')
parser.add_argument('--print-examples', action='store_true', default=False,
                    help='Print one example generation per test sample with its reward')
args = parser.parse_args()

def compute_reward(generated_text, target_text):
    """Compute reward based on length difference (from es_fine-tuning_conciseness_iid.py)"""
    return -abs(len(generated_text) - len(target_text))

def compute_kl_divergence_teacher_forced(model_ft, model_base, tokenizer, question, answer, device):
    """
    Compute KL divergence using teacher-forcing.
    KL[θ_FT || θ_BASE] over the full vocabulary for the answer tokens given the question context.
    """
    # Tokenize question and answer separately
    question_tokens = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    full_text = question + answer
    full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
    
    question_length = question_tokens["input_ids"].shape[1]
    
    input_ids = full_tokens["input_ids"].to(device)
    attention_mask = full_tokens["attention_mask"].to(device)
    
    # Get logits from both models
    with torch.inference_mode():
        outputs_ft = model_ft(input_ids=input_ids, attention_mask=attention_mask)
        outputs_base = model_base(input_ids=input_ids, attention_mask=attention_mask)
    
    logits_ft = outputs_ft.logits
    logits_base = outputs_base.logits
    
    # We only care about the answer tokens
    # The answer starts at position question_length-1 (since we're predicting the next token)
    # and goes to the end
    answer_start_idx = question_length - 1
    answer_logits_ft = logits_ft[0, answer_start_idx:-1, :]  # Exclude last position (no next token to predict)
    answer_logits_base = logits_base[0, answer_start_idx:-1, :]
    
    # Get the actual answer token ids
    answer_token_ids = input_ids[0, question_length:]  # The tokens we're predicting
    
    if answer_token_ids.shape[0] == 0:
        return 0.0  # No answer tokens to evaluate
    
    # Compute log probabilities
    log_probs_ft = torch.nn.functional.log_softmax(answer_logits_ft, dim=-1)
    log_probs_base = torch.nn.functional.log_softmax(answer_logits_base, dim=-1)
    
    # Compute full-vocabulary KL(FT || BASE) per position and sum across answer positions
    # KL(p||q) = sum_y p(y) * (log p(y) - log q(y))
    prob_ft_full = torch.exp(log_probs_ft)
    kl_per_position = (prob_ft_full * (log_probs_ft - log_probs_base)).sum(dim=-1)
    kl_sum = kl_per_position.sum().item()
    
    return kl_sum

def compute_kl_divergence_from_ids(model_ft, model_base, input_ids_1d, answer_start_idx, device):
    """
    Compute full-vocabulary KL(FT || BASE) over generated answer tokens given a full sequence of token ids.
    - input_ids_1d: 1D tensor of token ids representing [prompt || answer]
    - answer_start_idx: index in the sequence where answer prediction begins (i.e., next-token after prompt)
    """
    if input_ids_1d.dim() != 1:
        raise ValueError("input_ids_1d must be a 1D tensor")

    seq_len = input_ids_1d.shape[0]
    if seq_len - answer_start_idx - 1 <= 0:
        return 0.0

    input_ids = input_ids_1d.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.inference_mode():
        outputs_ft = model_ft(input_ids=input_ids, attention_mask=attention_mask)
        outputs_base = model_base(input_ids=input_ids, attention_mask=attention_mask)

    # Predict positions for answer tokens
    start = answer_start_idx
    logits_ft = outputs_ft.logits[0, start:-1, :]
    logits_base = outputs_base.logits[0, start:-1, :]

    log_probs_ft = torch.nn.functional.log_softmax(logits_ft, dim=-1)
    log_probs_base = torch.nn.functional.log_softmax(logits_base, dim=-1)

    prob_ft_full = torch.exp(log_probs_ft)
    kl_per_pos = (prob_ft_full * (log_probs_ft - log_probs_base)).sum(dim=-1)
    return kl_per_pos.sum().item()

def compute_observed_token_kl_from_ids(model_ft, model_base, input_ids_1d, answer_start_idx, device):
    """
    Compute observed-token KL divergence following the formula:
    KL = θ_BASE(y_t | context) / θ_FT(y_t | context) - log(θ_BASE(y_t | context) / θ_FT(y_t | context)) - 1
    
    This only evaluates the divergence at the specific tokens that were generated,
    rather than summing over the entire vocabulary.
    
    - input_ids_1d: 1D tensor of token ids representing [prompt || answer]
    - answer_start_idx: index in the sequence where answer prediction begins
    """
    if input_ids_1d.dim() != 1:
        raise ValueError("input_ids_1d must be a 1D tensor")

    seq_len = input_ids_1d.shape[0]
    if seq_len - answer_start_idx - 1 <= 0:
        return 0.0

    input_ids = input_ids_1d.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.inference_mode():
        outputs_ft = model_ft(input_ids=input_ids, attention_mask=attention_mask)
        outputs_base = model_base(input_ids=input_ids, attention_mask=attention_mask)

    # Predict positions for answer tokens
    start = answer_start_idx
    logits_ft = outputs_ft.logits[0, start:-1, :]
    logits_base = outputs_base.logits[0, start:-1, :]

    # Get the actual answer token ids that were generated
    answer_token_ids = input_ids_1d[start + 1:]  # The tokens we're predicting
    
    if answer_token_ids.shape[0] == 0:
        return 0.0

    log_probs_ft = torch.nn.functional.log_softmax(logits_ft, dim=-1)
    log_probs_base = torch.nn.functional.log_softmax(logits_base, dim=-1)

    # Extract probabilities only at the observed tokens
    # log_probs shape: [num_positions, vocab_size]
    # We want to gather the probability at each observed token
    indices = answer_token_ids.unsqueeze(1)  # [num_positions, 1]
    log_prob_ft_observed = log_probs_ft.gather(dim=1, index=indices).squeeze(1)  # [num_positions]
    log_prob_base_observed = log_probs_base.gather(dim=1, index=indices).squeeze(1)  # [num_positions]
    
    prob_ft_observed = torch.exp(log_prob_ft_observed)
    prob_base_observed = torch.exp(log_prob_base_observed)
    
    # KL formula: r - log(r) - 1, where r = p_base / p_ft
    # Equivalently: p_base/p_ft - log(p_base/p_ft) - 1
    # = p_base/p_ft - log(p_base) + log(p_ft) - 1
    ratio = prob_base_observed / prob_ft_observed
    kl_per_pos = ratio - torch.log(ratio) - 1.0
    
    return kl_per_pos.sum().item()

def main():
    print("="*80)
    print("Conciseness KL Divergence Evaluation")
    print("="*80)
    print(f"Fine-tuned model: {args.model}")
    print(f"Baseline model: {args.baseline_model_name}")
    print(f"Eval data path: {args.eval_data_path}")
    print(f"Precision: {args.precision}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Do sample: {args.do_sample}")
    print(f"Num samples per prompt: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print("="*80)
    
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
    
    # Load baseline model
    print("Loading baseline model...")
    model_base = AutoModelForCausalLM.from_pretrained(
        args.baseline_model_name,
        cache_dir=args.hf_cache_dir,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model_base.eval()
    
    # Assert shared vocabulary between fine-tuned and baseline models
    vocab_ft = getattr(model_ft.config, "vocab_size", None)
    vocab_base = getattr(model_base.config, "vocab_size", None)
    if vocab_ft != vocab_base:
        raise ValueError(
            f"Vocab size mismatch between fine-tuned (vocab_size={vocab_ft}) and baseline (vocab_size={vocab_base}) models."
        )
    
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
    
    print("Models and tokenizer loaded successfully")
    
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
    print("\nGenerating responses and computing metrics...")
    all_kl_divergences = []
    all_observed_token_kl = []
    all_rewards = []
    all_answer_token_counts = []
    examples = []  # Store (question, generated_answer, reward) tuples for printing
    
    for question_idx, (question, target_answer) in enumerate(dataset):
        print(f"Processing question {question_idx + 1}/{len(dataset)}...")
        
        # Generate num_samples responses for this question
        question_kl_divergences = []
        question_observed_token_kl = []
        question_rewards = []
        question_answer_token_counts = []
        
        # Process in batches
        num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, args.num_samples)
            batch_size_actual = batch_end - batch_start
            
            # Prepare batch input (repeat question for batch)
            input_texts = [question] * batch_size_actual
            
            # Tokenize
            tokenized_inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                padding_side="left"
            )
            input_ids = tokenized_inputs["input_ids"].to(device)
            attention_mask = tokenized_inputs["attention_mask"].to(device)
            
            # Generate responses
            with torch.inference_mode():
                outputs = model_ft.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    pad_token_id=tokenizer.pad_token_id
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Decode outputs and compute metrics using token IDs to avoid retokenization drift
            # Get the original input length (including padding) - same for all batch elements
            input_len = input_ids.shape[1]
            
            for i in range(batch_size_actual):
                full_ids = outputs[i]
                # Slice answer token ids - use input_len to account for padding
                answer_ids = full_ids[input_len:]
                # Decode answer for reward computation
                try:
                    generated_answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                except TypeError:
                    tokens = tokenizer.convert_ids_to_tokens(answer_ids, skip_special_tokens=True)
                    filtered = [t for t in tokens if t is not None]
                    generated_answer = tokenizer.convert_tokens_to_string(filtered).strip()

                # Compute KL divergence directly from ids: FT || BASE over answer positions
                # Answer prediction starts at the last input position (input_len - 1)
                kl_div = compute_kl_divergence_from_ids(
                    model_ft=model_ft,
                    model_base=model_base,
                    input_ids_1d=full_ids,
                    answer_start_idx=input_len - 1,
                    device=device,
                )
                question_kl_divergences.append(kl_div)
                
                # Compute observed-token KL divergence (formula from image)
                obs_kl_div = compute_observed_token_kl_from_ids(
                    model_ft=model_ft,
                    model_base=model_base,
                    input_ids_1d=full_ids,
                    answer_start_idx=input_len - 1,
                    device=device,
                )
                question_observed_token_kl.append(obs_kl_div)
                
                # Count answer tokens predicted (positions contributing to KL)
                num_answer_tokens = max(0, full_ids.shape[0] - input_len)
                question_answer_token_counts.append(int(num_answer_tokens))

                # Compute reward
                reward = compute_reward(generated_answer, target_answer)
                question_rewards.append(reward)
                
                # Store first example for each question
                if args.print_examples and batch_idx == 0 and i == 0:
                    examples.append((question, generated_answer, reward))
            
            # Clean up
            del input_ids, attention_mask, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Aggregate metrics for this question
        all_kl_divergences.extend(question_kl_divergences)
        all_observed_token_kl.extend(question_observed_token_kl)
        all_rewards.extend(question_rewards)
        all_answer_token_counts.extend(question_answer_token_counts)
    
    # Calculate final metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    mean_kl = np.mean(all_kl_divergences)
    std_kl = np.std(all_kl_divergences)
    total_kl_sum = float(np.sum(all_kl_divergences))
    total_answer_tokens = int(np.sum(all_answer_token_counts))
    mean_kl_per_token = (total_kl_sum / total_answer_tokens) if total_answer_tokens > 0 else float('nan')
    
    mean_obs_kl = np.mean(all_observed_token_kl)
    std_obs_kl = np.std(all_observed_token_kl)
    total_obs_kl_sum = float(np.sum(all_observed_token_kl))
    mean_obs_kl_per_token = (total_obs_kl_sum / total_answer_tokens) if total_answer_tokens > 0 else float('nan')
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    
    print(f"\nKL Divergence - Full Vocabulary (Fine-tuned || Baseline):")
    print(f"  Mean: {mean_kl:.6f}")
    print(f"  Std:  {std_kl:.6f}")
    print(f"  Total sum over positions: {total_kl_sum:.6f}")
    print(f"  Mean per token (micro-average): {mean_kl_per_token:.6f}")
    print(f"  Total answer tokens: {total_answer_tokens}")
    
    print(f"\nKL Divergence - Observed Tokens Only (r - log(r) - 1 formula):")
    print(f"  Mean: {mean_obs_kl:.6f}")
    print(f"  Std:  {std_obs_kl:.6f}")
    print(f"  Total sum over positions: {total_obs_kl_sum:.6f}")
    print(f"  Mean per token (micro-average): {mean_obs_kl_per_token:.6f}")
    
    print(f"\nReward (Length-based):")
    print(f"  Mean: {mean_reward:.4f}")
    print(f"  Std:  {std_reward:.4f}")
    
    print(f"\nTotal samples evaluated: {len(all_kl_divergences)}")
    
    # Print examples if requested
    if args.print_examples:
        print("\n" + "="*80)
        print("EXAMPLE GENERATIONS (1 per test sample)")
        print("="*80)
        for idx, (question, generated_answer, reward) in enumerate(examples):
            print(f"\n--- Example {idx + 1}/{len(examples)} ---")
            print(f"Question: {question}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Reward: {reward:.4f}")
    
    print("="*80)

if __name__ == "__main__":
    main()