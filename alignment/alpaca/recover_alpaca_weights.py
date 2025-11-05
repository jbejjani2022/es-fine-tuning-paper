#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Optional, Dict
import argparse
import torch
import tqdm
import transformers


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@torch.inference_mode()
def recover(
    path_raw,
    path_diff,
    path_tuned: Optional[str] = None,
    device="cpu",
    test_inference=True,
    check_integrity_naively=True,
):
    """Recover the original weights from the released weight diff.

    This function is given for you to run.

    Things to do before running this:
        1. Convert Meta's released weights into huggingface format. Follow this guide:
            https://huggingface.co/docs/transformers/main/model_doc/llama
        2. Make sure you cloned the released weight diff into your local machine. The weight diff is located at:
            https://huggingface.co/tatsu-lab/alpaca-7b/tree/main
        3. Run this function with the correct paths. E.g.,
            python weight_diff.py recover --path_raw <path_to_step_1_dir> --path_diff <path_to_step_2_dir>

    Additional notes:
        - If things run too slowly, and you have an 80G GPU lying around, let GPU go brrr by setting `--device "cuda"`.
        - If you want to save the recovered weights, set `--path_tuned <your_path_tuned>`.
            Next time you can load the recovered weights directly from `<your_path_tuned>`.
    """
    print(f"Loading raw model from: {path_raw}")
    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    print(f"Loading diff model from: {path_diff}")
    model_recovered: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_diff,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_raw
    )
    if tokenizer_raw.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=model_raw,
            tokenizer=tokenizer_raw,
        )
    tokenizer_recovered: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_diff
    )

    print("Recovering weights by adding diff to raw model...")
    state_dict_recovered = model_recovered.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_recovered):
        state_dict_recovered[key].add_(state_dict_raw[key])

    if check_integrity_naively:
        # This is not a rigorous, cryptographically strong integrity check :)
        print("Checking integrity...")
        allsum = sum(state_dict_recovered[key].sum() for key in state_dict_recovered)
        assert torch.allclose(
            allsum, torch.full_like(allsum, fill_value=50637.1836), atol=1e-2, rtol=0
        ), "Naive integrity check failed. This could imply that some of the checkpoint files are corrupted."
        print("Integrity check passed!")

    if path_tuned is not None:
        print(f"Saving recovered model to: {path_tuned}")
        model_recovered.save_pretrained(path_tuned)
        tokenizer_recovered.save_pretrained(path_tuned)
        print("Model saved successfully!")

    if test_inference:
        print("\nTesting inference...")
        input_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\r\n\r\n"
            "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
        )
        inputs = tokenizer_recovered(input_text, return_tensors="pt")
        # Move inputs to the same device as the model
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model_recovered.generate(**inputs, max_new_tokens=100)
        output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
        output_text = output_text[len(input_text) :]
        print(f"Input: {input_text}\nCompletion: {output_text}")

    return model_recovered, tokenizer_recovered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recover Alpaca weights from LLaMA + weight diff")
    parser.add_argument("--path_raw", type=str, required=True, 
                       help="Path to the raw LLaMA model in HuggingFace format")
    parser.add_argument("--path_diff", type=str, required=True,
                       help="Path to the Alpaca weight diff")
    parser.add_argument("--path_tuned", type=str, default=None,
                       help="Path to save the recovered Alpaca weights")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to use for computation")
    parser.add_argument("--test_inference", action="store_true", default=True,
                       help="Test inference after recovery")
    parser.add_argument("--no_test_inference", dest="test_inference", action="store_false",
                       help="Skip inference test")
    parser.add_argument("--check_integrity", action="store_true", default=True,
                       help="Check integrity of recovered weights")
    parser.add_argument("--no_check_integrity", dest="check_integrity", action="store_false",
                       help="Skip integrity check")
    
    args = parser.parse_args()
    
    recover(
        path_raw=args.path_raw,
        path_diff=args.path_diff,
        path_tuned=args.path_tuned,
        device=args.device,
        test_inference=args.test_inference,
        check_integrity_naively=args.check_integrity,
    )

