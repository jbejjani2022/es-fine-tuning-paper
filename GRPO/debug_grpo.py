#!/usr/bin/env python
"""Debug script to test GRPO generation directly."""

import os
import json
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Test data
test_prompts = [
    {"prompt": "Solve: 3 + 5 =", "answer": "8"},
    {"prompt": "What is the capital of France?", "answer": "Paris"}
]

dataset = Dataset.from_list(test_prompts)

print("\n" + "="*80)
print("Testing direct generation without TRL:")
print("="*80)

for item in test_prompts:
    prompt = item["prompt"]
    print(f"\nPrompt: {prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with explicit parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=1.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"Generated: {generated_text}")
    print(f"Expected: {item['answer']}")

# print("\n" + "="*80)
# print("Now testing with TRL GRPO (minimal setup):")
# print("="*80)

# try:
#     from trl import GRPOConfig, GRPOTrainer
    
#     # Minimal GRPO config
#     # Note: effective batch size must be divisible by num_generations
#     # effective batch size = per_device_train_batch_size * gradient_accumulation_steps * num_processes
#     grpo_config = GRPOConfig(
#         output_dir="./test_grpo_output",
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=2,  # This makes effective batch = 2, divisible by num_generations=2
#         learning_rate=5e-6,
#         num_train_epochs=1,
#         max_steps=1,  # Just 1 step for testing
#         max_prompt_length=1024,
#         max_completion_length=64,
#         temperature=1.0,
#         top_p=1.0,
#         num_generations=2,  # Generate 2 samples per prompt
#         beta=0.01,
#         remove_unused_columns=False,
#         bf16=True,
#     )
    
#     # Simple reward function
#     def reward_fn(completions, **kwargs):
#         print(f"\nDEBUG reward_fn called with:")
#         print(f"  completions type: {type(completions)}")
#         print(f"  completions: {completions}")
#         print(f"  kwargs keys: {kwargs.keys()}")
        
#         # Return dummy rewards for now
#         return [0.5] * len(completions)
    
#     # Create trainer
#     trainer = GRPOTrainer(
#         model=model_name,  # Let TRL load the model
#         processing_class=tokenizer,
#         reward_funcs=reward_fn,
#         args=grpo_config,
#         train_dataset=dataset,
#     )
    
#     print("GRPO Trainer created successfully!")
#     print("Running one training step to see generation...")
    
#     # Run one step
#     trainer.train()
    
# except Exception as e:
#     print(f"Error with TRL GRPO: {e}")
#     import traceback
#     traceback.print_exc()

# print("\n" + "="*80)
# print("Debug complete!")
# print("="*80)
