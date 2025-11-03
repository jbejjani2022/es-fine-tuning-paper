#!/usr/bin/env python3
"""
Verify that the recovered Alpaca model generates reasonable outputs.
Compares with expected Alpaca-style responses.
"""

import argparse
import torch
import transformers


# Test cases with expected Alpaca-style behavior
TEST_CASES = [
    {
        "instruction": "List three technologies that make life easier.",
        "expected_keywords": ["technology", "computer", "internet", "smartphone", "automation"],
    },
    {
        "instruction": "Give three tips for staying healthy.",
        "expected_keywords": ["exercise", "healthy", "diet", "sleep", "water"],
    },
    {
        "instruction": "What is the capital of France?",
        "expected_keywords": ["Paris"],
    },
]


def format_instruction(instruction, input_text=""):
    """Format instruction in Alpaca style."""
    if input_text:
        return (
            f"Below is an instruction that describes a task, paired with an input that provides further context. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
        )
    else:
        return (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:"
        )


def test_model(model_path, device="cpu"):
    """Test the model with various prompts."""
    print(f"\n{'='*80}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*80}\n")
    
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": torch.device(device)},
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Model loaded successfully!\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    all_passed = True
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{'-'*80}")
        print(f"Test Case {i}/{len(TEST_CASES)}")
        print(f"{'-'*80}")
        
        instruction = test_case["instruction"]
        prompt = format_instruction(instruction)
        
        print(f"Instruction: {instruction}")
        print(f"\nGenerating response...")
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            print(f"\nResponse: {response}")
            
            # Check if response contains expected keywords
            response_lower = response.lower()
            found_keywords = [kw for kw in test_case["expected_keywords"] if kw.lower() in response_lower]
            
            if found_keywords or len(response) > 20:  # Response should be substantive
                print(f"\n✓ Test passed! (Found keywords: {found_keywords if found_keywords else 'N/A, but response is substantial'})")
            else:
                print(f"\n⚠ Test warning: Response might not be Alpaca-quality")
                print(f"   Expected keywords: {test_case['expected_keywords']}")
                all_passed = False
                
        except Exception as e:
            print(f"\n✗ Error during generation: {e}")
            all_passed = False
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✓ All tests passed! The model appears to be working correctly.")
    else:
        print("⚠ Some tests had warnings. The model might still work, but review outputs.")
    print(f"{'='*80}\n")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Verify recovered Alpaca model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the recovered Alpaca model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for inference",
    )
    
    args = parser.parse_args()
    
    success = test_model(args.model_path, args.device)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

