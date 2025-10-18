import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import argparse
import os
import json
import re
import sys

# Add parent directory to path to import countdown_task
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'countdown'))
from countdown_task import reward_function

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate model accuracy on countdown task')
parser.add_argument('--model', type=str, required=True, 
                    help='Name of HuggingFace model or path to a saved model checkpoint')
parser.add_argument('--data_sample', type=int, default=1000, 
                    help='Number of data samples to evaluate on')
parser.add_argument('--hf_cache_dir', type=str, default='hf_cache',
                    help='HuggingFace cache directory')
parser.add_argument('--precision', type=str, default='bf16', choices=['fp16', 'bf16', 'fp32'],
                    help='Model precision')
parser.add_argument('--max_new_tokens', type=int, default=1024,
                    help='Maximum number of tokens to generate')
parser.add_argument('--do_sample', action='store_true', default=False,
                    help='Enable sampling vs greedy decoding')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for evaluation')
args = parser.parse_args()

def main():
    print("="*80)
    print("Countdown Task Accuracy Evaluation")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Data samples: {args.data_sample}")
    print(f"Precision: {args.precision}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Do sample: {args.do_sample}")
    print(f"Batch size: {args.batch_size}")
    print("="*80)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine dtype based on precision
    if args.precision == 'fp16':
        dtype = torch.float16
    elif args.precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.hf_cache_dir,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.eval()
    
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
    
    # Load dataset
    print("\nLoading dataset...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'countdown', 'data', 'countdown.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data_json = json.load(f)
    
    dataset = []
    for item in data_json:
        context = item['context']
        target = item['target']
        numbers = item['numbers']
        dataset.append((context, target, numbers))
    
    # Use subset of dataset
    SUBSET_START = 200
    dataset = dataset[SUBSET_START:args.data_sample+SUBSET_START]
    print(f"Loaded {len(dataset)} countdown samples")
    
    # Evaluation loop
    print("\nEvaluating model...")
    all_format_rewards = []
    all_answer_rewards = []
    all_combined_rewards = []
    
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]
        
        # Extract batch data
        input_texts = [item[0] for item in batch]
        target_texts = [item[1] for item in batch]
        numbers_list = [item[2] for item in batch]
        
        # Tokenize batch
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
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Decode outputs
        generated_texts = []
        for i in range(len(input_texts)):
            try:
                generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            except TypeError:
                tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
                filtered = [t for t in tokens if t is not None]
                generated_text = tokenizer.convert_tokens_to_string(filtered)
            generated_texts.append(generated_text)
        
        # Clean up
        del input_ids, attention_mask, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Calculate rewards for batch
        for gen_text, target_text, numbers in zip(generated_texts, target_texts, numbers_list):
            # Parse target
            target = int(target_text) if target_text.isdigit() else None
            
            # Extract model response (after "assistant:" if present)
            model_response = gen_text
            if "assistant:" in gen_text:
                model_response = gen_text.split("assistant:")[-1].strip()
            
            # Calculate reward
            reward_result = reward_function(model_response, numbers, target)
            combined_reward = reward_result["reward"]
            format_reward = reward_result["reward_info"]["format_reward"]
            answer_reward = reward_result["reward_info"]["answer_reward"]
            
            all_format_rewards.append(format_reward)
            all_answer_rewards.append(answer_reward)
            all_combined_rewards.append(combined_reward)
        
        # Print progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"Processed {end_idx}/{len(dataset)} samples...")
    
    # Calculate final metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    total_samples = len(dataset)
    
    # Answer accuracy (answer_reward == 1.0)
    correct_answers = sum(1 for r in all_answer_rewards if r == 1.0)
    answer_accuracy = (correct_answers / total_samples) * 100
    
    # Format accuracy (format_reward == 1.0)
    correct_format = sum(1 for r in all_format_rewards if r == 1.0)
    format_accuracy = (correct_format / total_samples) * 100
    
    # Mean combined reward
    mean_combined_reward = sum(all_combined_rewards) / total_samples
    
    print(f"\nAnswer Accuracy: {correct_answers}/{total_samples} = {answer_accuracy:.2f}%")
    print(f"Format Accuracy: {correct_format}/{total_samples} = {format_accuracy:.2f}%")
    print(f"Mean Combined Reward: {mean_combined_reward:.4f}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
