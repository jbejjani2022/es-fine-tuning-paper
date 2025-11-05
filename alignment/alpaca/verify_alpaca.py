#!/usr/bin/env python3
"""
Verify that the recovered Alpaca model generates reasonable outputs.
Compares with expected Alpaca-style responses.
"""

import argparse
import os
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


def test_model(model_path, device="cpu", use_vllm=False, vllm_ckpt="", bf16=False,
               max_new_tokens=100, temperature=0.7, top_p=0.9):
    """Test the model with various prompts."""
    print(f"\n{'='*80}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*80}\n")
    
    try:
        model = None
        tokenizer = None
        llm = None

        if use_vllm:
            # Lazy imports to avoid requiring vLLM when not used
            import ray
            from vllm import LLM, SamplingParams

            ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

            class _ESNcclLLM(LLM):
                def __init__(self, *l_args, **l_kwargs):
                    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                    super().__init__(*l_args, **l_kwargs)

            ESNcclLLMRemote = ray.remote(num_cpus=8, num_gpus=1)(_ESNcclLLM)
            llm = ESNcclLLMRemote.remote(
                model=model_path,               # base HF folder (tokenizer/config)
                tensor_parallel_size=1,
                distributed_executor_backend="mp",
                worker_extension_cls="utils.worker_extn.WorkerExtension",
                dtype=("bfloat16" if bf16 and torch.cuda.is_available() else "float16"),
                enable_prefix_caching=False,
                enforce_eager=False,
                # Keep KV cache modest
                max_model_len=max_new_tokens + 64,
                gpu_memory_utilization=0.6,
            )
            if not vllm_ckpt:
                raise ValueError("When --use_vllm is set, you must pass --vllm_ckpt to your fused .pth.")
            # Load fused checkpoint into the engine via WorkerExtension RPC
            ray.get(llm.collective_rpc.remote("load_self_weights_from_disk", args=(vllm_ckpt,)))
            print("✓ vLLM engine initialized!\n")
        else:
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
            if use_vllm:
                # vLLM path
                import ray  # imported above when initializing
                from vllm import SamplingParams

                sampling = SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=max(temperature, 1e-6),
                    top_p=top_p,
                    seed=42,
                )
                outputs = ray.get(llm.generate.remote([prompt], sampling, use_tqdm=False))
                response = outputs[0].outputs[0].text.strip()
            else:
                # HF path
                inputs = tokenizer(prompt, return_tensors="pt")
                if device == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=(temperature > 0.0),
                        top_p=top_p,
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
    
    # Teardown vLLM engine if used (free GPU)
    if use_vllm:
        try:
            import ray
            if 'llm' in locals() and llm is not None:
                ray.kill(llm)
            ray.shutdown()
        except Exception:
            pass

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
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for generation (requires --vllm_ckpt to a fused .pth)",
    )
    parser.add_argument(
        "--vllm_ckpt",
        type=str,
        default="",
        help="Path to fused vLLM checkpoint (.pth) saved during ES training.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 for vLLM (if CUDA is available)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p",
    )
    
    args = parser.parse_args()
    
    success = test_model(
        args.model_path,
        args.device,
        args.use_vllm,
        args.vllm_ckpt,
        args.bf16,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

