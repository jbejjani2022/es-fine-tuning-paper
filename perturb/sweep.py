"""
Robustness Evaluation Sweep Script

This script evaluates the robustness of fine-tuned models across multiple checkpoints,
perturbation types, and noise levels (sigmas). Results are saved to a CSV file.
"""

import os
import json
import copy
import csv
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import argparse

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

# Hardcoded configuration parameters
MAX_NEW_TOKENS = 128
NUM_SAMPLES = 20
DO_SAMPLE = True
PRECISION = 'bf16'
EVAL_DATA_PATH = 'conciseness/data/eval.jsonl'
HF_CACHE_DIR = 'hf_cache'

# Sweep parameters
PERTURBATION_TYPES = ['lm_head.weight', 'all_model_params']
NUM_PERTURBATIONS = 8  # Number of independent trials per perturbation type

# Model checkpoint mappings
map_models_to_checkpoints = {
    "grpo_beta0": [
        "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_temp_0.7/beta0.0_seed11/checkpoint-1000",
        "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_temp_0.7/beta0.0_seed22/checkpoint-1000",
        "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_temp_0.7/beta0.0_seed33/checkpoint-1000",
        "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_temp_0.7/beta0.0_seed44/checkpoint-1000"
    ],
    "grpo_beta0464": [
        "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_temp_0.7/beta0.0464_seed11/checkpoint-1000",
        "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_temp_0.7/beta0.0464_seed22/checkpoint-1000",
        "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_temp_0.7/beta0.0464_seed33/checkpoint-1000",
        "/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/GRPO_temp_0.7/beta0.0464_seed44/checkpoint-1000"
    ],
    "es_small_sig": [
        "/n/holylabs/LABS/sham_lab/Users/jbejjani/es-fine-tuning-paper/checkpoints/conciseness_small_sig/Qwen/Qwen2.5-7B-Instruct/0/es_random_seed0_pop30_iter1000_sigma0.001_alpha0.0005_bf16_threads1_question_num2_final",
        "/n/holylabs/LABS/sham_lab/Users/jbejjani/es-fine-tuning-paper/checkpoints/conciseness_small_sig/Qwen/Qwen2.5-7B-Instruct/1/es_random_seed1_pop30_iter1000_sigma0.001_alpha0.0005_bf16_threads1_question_num2_final",
        "/n/holylabs/LABS/sham_lab/Users/jbejjani/es-fine-tuning-paper/checkpoints/conciseness_small_sig/Qwen/Qwen2.5-7B-Instruct/2/es_random_seed2_pop30_iter1000_sigma0.001_alpha0.0005_bf16_threads1_question_num2_final",
        "/n/holylabs/LABS/sham_lab/Users/jbejjani/es-fine-tuning-paper/checkpoints/conciseness_small_sig/Qwen/Qwen2.5-7B-Instruct/3/es_random_seed3_pop30_iter1000_sigma0.001_alpha0.0005_bf16_threads1_question_num2_final"
    ],
    "es_big_sig": [
        "/n/holylabs/LABS/sham_lab/Users/jbejjani/es-fine-tuning-paper/checkpoints/conciseness/Qwen/Qwen2.5-7B-Instruct/0/es_random_seed0_pop30_iter1000_sigma0.0015_alpha0.00075_bf16_threads1_question_num2_final",
        "/n/holylabs/LABS/sham_lab/Users/jbejjani/es-fine-tuning-paper/checkpoints/conciseness/Qwen/Qwen2.5-7B-Instruct/1/es_random_seed1_pop30_iter1000_sigma0.0015_alpha0.00075_bf16_threads1_question_num2_final",
        "/n/holylabs/LABS/sham_lab/Users/jbejjani/es-fine-tuning-paper/checkpoints/conciseness/Qwen/Qwen2.5-7B-Instruct/2/es_random_seed2_pop30_iter1000_sigma0.0015_alpha0.00075_bf16_threads1_question_num2_final",
        "/n/holylabs/LABS/sham_lab/Users/jbejjani/es-fine-tuning-paper/checkpoints/conciseness/Qwen/Qwen2.5-7B-Instruct/3/es_random_seed3_pop30_iter1000_sigma0.0015_alpha0.00075_bf16_threads1_question_num2_final"
    ]
}


def extract_seed_from_checkpoint(checkpoint_path):
    """Extract seed identifier from checkpoint path"""
    # Try to match seed patterns: seed11, seed22, etc. or seed0, seed1, etc.
    match = re.search(r'seed(\d+)', checkpoint_path)
    if match:
        return f"seed{match.group(1)}"
    return "unknown"


def compute_reward(generated_text, target_text):
    """Compute reward based on length difference"""
    return -abs(len(generated_text) - len(target_text))


def compute_per_token_logps(model, input_ids, attention_mask):
    """Compute per-token log probabilities for a given sequence"""
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits.float(), dim=-1)

    shift_log_probs = log_probs[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    per_token_logps = torch.gather(shift_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    return per_token_logps


def perturb_parameter(model, param_name, sigma):
    """
    Apply Gaussian noise perturbation to a specific model parameter.
    
    Args:
        model: The model to perturb
        param_name: Full name of the parameter to perturb
        sigma: Standard deviation of Gaussian noise
        
    Returns:
        noise: The noise tensor that was added (for later restoration)
    """
    # Get the parameter
    param = model.get_parameter(param_name)
    
    # Generate and apply Gaussian noise
    noise = torch.randn_like(param.data) * sigma
    param.data.add_(noise)
    
    return noise


def restore_parameter(model, param_name, noise):
    """
    Restore a parameter to its original state by subtracting the noise.
    
    Args:
        model: The model to restore
        param_name: Full name of the parameter to restore
        noise: The noise tensor that was previously added
    """
    param = model.get_parameter(param_name)
    param.data.sub_(noise)


def perturb_all_parameters(model, sigma):
    """
    Apply Gaussian noise perturbation to all model parameters.
    
    Args:
        model: The model to perturb
        sigma: Standard deviation of Gaussian noise
        
    Returns:
        noise_dict: Dictionary mapping parameter names to noise tensors (for later restoration)
    """
    noise_dict = {}
    
    # Iterate over all parameters and add noise
    for name, param in model.named_parameters():
        noise = torch.randn_like(param.data) * sigma
        param.data.add_(noise)
        noise_dict[name] = noise
    
    return noise_dict


def restore_all_parameters(model, noise_dict):
    """
    Restore all parameters to their original state by subtracting the noise.
    
    Args:
        model: The model to restore
        noise_dict: Dictionary mapping parameter names to noise tensors
    """
    for name, noise in noise_dict.items():
        param = model.get_parameter(name)
        param.data.sub_(noise)


def evaluate_model(model, tokenizer, dataset, device, reference_model=None):
    """
    Evaluate model on the dataset and compute metrics.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        dataset: List of (question, answer) tuples
        device: Device to run evaluation on
        reference_model: Optional reference model for KL divergence computation
        
    Returns:
        Dictionary with normalized_mean_reward and mean_per_token_kl
    """
    all_rewards = []
    all_per_token_kls = []
    
    for question_idx, (question, target_answer) in enumerate(dataset):
        for sample_idx in range(NUM_SAMPLES):
            # Tokenize input
            tokenized_inputs = tokenizer(
                [question],
                return_tensors="pt",
                padding=True,
            )
            input_ids = tokenized_inputs["input_ids"].to(device)
            attention_mask = tokenized_inputs["attention_mask"].to(device)
            
            # Generate response
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            input_len = input_ids.shape[1]
            full_ids = outputs[0]
            answer_ids = full_ids[input_len:]
            
            # Decode generated answer
            try:
                generated_answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            except TypeError:
                tokens = tokenizer.convert_ids_to_tokens(answer_ids, skip_special_tokens=True)
                filtered = [t for t in tokens if t is not None]
                generated_answer = tokenizer.convert_tokens_to_string(filtered).strip()
            
            # Compute reward
            reward = compute_reward(generated_answer, target_answer)
            all_rewards.append(reward)
            
            # Compute KL divergence if reference model provided
            if reference_model is not None:
                full_input_ids = full_ids.unsqueeze(0).to(device)
                full_attention_mask = torch.ones_like(full_input_ids, device=device)
                
                ft_per_token_logps = compute_per_token_logps(model, full_input_ids, full_attention_mask)
                ref_per_token_logps = compute_per_token_logps(reference_model, full_input_ids, full_attention_mask)
                
                gen_start_idx = max(input_len - 1, 0)
                if gen_start_idx < ft_per_token_logps.shape[1]:
                    ft_generated_logps = ft_per_token_logps[:, gen_start_idx:]
                    ref_generated_logps = ref_per_token_logps[:, gen_start_idx:]
                    
                    generated_token_ids = full_input_ids[:, 1:][:, gen_start_idx:]
                    
                    # Trim at EOS token
                    if tokenizer.eos_token_id is not None:
                        eos_positions = (generated_token_ids[0] == tokenizer.eos_token_id).nonzero(as_tuple=False)
                        if eos_positions.numel() > 0:
                            cutoff = eos_positions[0, 0].item() + 1
                            ft_generated_logps = ft_generated_logps[:, :cutoff]
                            ref_generated_logps = ref_generated_logps[:, :cutoff]
                            generated_token_ids = generated_token_ids[:, :cutoff]
                    
                    # Compute per-token KL
                    if ft_generated_logps.shape[1] > 0:
                        logp_diff = ref_generated_logps - ft_generated_logps
                        per_token_kl = torch.exp(logp_diff) - logp_diff - 1
                        per_token_kl = per_token_kl.squeeze(0)
                        per_token_kl_list = per_token_kl.detach().cpu().tolist()
                        
                        if per_token_kl_list:
                            all_per_token_kls.extend(per_token_kl_list)
            
            # Clean up
            del input_ids, attention_mask, outputs
            if reference_model is not None:
                del full_input_ids, full_attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Compute summary statistics
    mean_reward = float(np.mean(all_rewards))
    
    # Normalize rewards (following conciseness_eval.py normalization)
    normalized_mean_reward = (mean_reward + 2000) / 2001
    
    mean_per_token_kl = float(np.mean(all_per_token_kls)) if len(all_per_token_kls) > 0 else None
    
    return {
        'normalized_mean_reward': normalized_mean_reward,
        'mean_per_token_kl': mean_per_token_kl,
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Sweep robustness evaluation across checkpoints')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(map_models_to_checkpoints.keys()),
                        help='Model type to evaluate')
    parser.add_argument('--sigma', type=float, required=True,
                        help='Standard deviation of Gaussian noise for perturbations')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Robustness Evaluation Sweep")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Number of checkpoints: {len(map_models_to_checkpoints[args.model])}")
    print(f"Sigma: {args.sigma}")
    print(f"Perturbation types: {PERTURBATION_TYPES}")
    print(f"Perturbations per type: {NUM_PERTURBATIONS}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Num samples per prompt: {NUM_SAMPLES}")
    print("=" * 80)
    
    # Determine device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if PRECISION == 'bf16' else torch.float32
    
    # Load evaluation dataset (only once)
    print("\nLoading evaluation dataset...")
    data_path = os.path.join(os.path.dirname(__file__), '..', EVAL_DATA_PATH)
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
    
    # Prepare CSV output
    csv_filename = f"{args.model}_{args.sigma}.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    
    # CSV header
    csv_headers = [
        'checkpoint_seed',
        'checkpoint_path',
        'sigma',
        'perturbation_type',
        'trial_number',
        'baseline_normalized_reward',
        'perturbed_normalized_reward',
        'normalized_reward_degradation',
        'mean_per_token_kl'
    ]
    
    # Open CSV file for writing
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    csv_writer.writeheader()
    
    # Main sweep loop
    checkpoints = map_models_to_checkpoints[args.model]
    
    for checkpoint_idx, checkpoint_path in enumerate(checkpoints):
        print("\n" + "=" * 80)
        print(f"Checkpoint [{checkpoint_idx + 1}/{len(checkpoints)}]: {checkpoint_path}")
        print("=" * 80)
        
        # Extract seed identifier
        checkpoint_seed = extract_seed_from_checkpoint(checkpoint_path)
        print(f"Checkpoint seed: {checkpoint_seed}")
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            cache_dir=HF_CACHE_DIR,
            device_map="auto",
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            use_fast=False,
            cache_dir=HF_CACHE_DIR
        )
        
        # Set padding side for generation
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model and tokenizer loaded successfully")
        
        # Create a deep copy of the model for reference (unperturbed state)
        print("Creating reference model (deep copy)...")
        model_ref = copy.deepcopy(model)
        model_ref.eval()
        
        # Baseline evaluation (unperturbed)
        print("\nEvaluating baseline (unperturbed) model...")
        baseline_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            reference_model=model_ref
        )
        
        baseline_normalized_reward = baseline_results['normalized_mean_reward']
        print(f"Baseline normalized reward: {baseline_normalized_reward:.6f}")
        
        # Perturbation sweep
        for perturbation_type in PERTURBATION_TYPES:
            print(f"\n  Perturbation type: {perturbation_type}")
            
            # Run multiple independent trials for this perturbation type
            for trial_num in range(NUM_PERTURBATIONS):
                print(f"    Trial {trial_num + 1}/{NUM_PERTURBATIONS}")
                
                # Apply perturbation
                if perturbation_type == 'all_model_params':
                    noise = perturb_all_parameters(model, args.sigma)
                else:
                    # Check if parameter exists
                    try:
                        model.get_parameter(perturbation_type)
                    except AttributeError:
                        print(f"      WARNING: Parameter {perturbation_type} not found. Skipping.")
                        continue
                    noise = perturb_parameter(model, perturbation_type, args.sigma)
                
                # Evaluate perturbed model
                perturbed_results = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    device=device,
                    reference_model=model_ref
                )
                
                perturbed_normalized_reward = perturbed_results['normalized_mean_reward']
                mean_per_token_kl = perturbed_results['mean_per_token_kl']
                
                # Compute degradation
                normalized_reward_degradation = baseline_normalized_reward - perturbed_normalized_reward
                
                print(f"      Perturbed normalized reward: {perturbed_normalized_reward:.6f}")
                print(f"      Normalized reward degradation: {normalized_reward_degradation:.6f}")
                if mean_per_token_kl is not None:
                    print(f"      Mean per-token KL: {mean_per_token_kl:.6f}")
                
                # Write to CSV
                csv_writer.writerow({
                    'checkpoint_seed': checkpoint_seed,
                    'checkpoint_path': checkpoint_path,
                    'sigma': args.sigma,
                    'perturbation_type': perturbation_type,
                    'trial_number': trial_num + 1,
                    'baseline_normalized_reward': baseline_normalized_reward,
                    'perturbed_normalized_reward': perturbed_normalized_reward,
                    'normalized_reward_degradation': normalized_reward_degradation,
                    'mean_per_token_kl': mean_per_token_kl if mean_per_token_kl is not None else ''
                })
                csv_file.flush()  # Flush to disk after each row
                
                # Restore model to unperturbed state
                if perturbation_type == 'all_model_params':
                    restore_all_parameters(model, noise)
                else:
                    restore_parameter(model, perturbation_type, noise)
                
                # Clean up
                del noise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Clean up model to free memory
        print("\nCleaning up model...")
        del model, model_ref, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Close CSV file
    csv_file.close()
    
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
