#!/usr/bin/env python3
"""
Download LLaMA-7B and Alpaca weight diff from HuggingFace Hub.
This script uses the huggingface_hub library to download models without requiring git-lfs.
"""

import argparse
import os
from huggingface_hub import snapshot_download


def download_model(repo_id, local_dir, model_name):
    """Download a model from HuggingFace Hub."""
    print(f"\n{'='*60}")
    print(f"Downloading {model_name}")
    print(f"Repository: {repo_id}")
    print(f"Destination: {local_dir}")
    print(f"{'='*60}\n")
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
            resume_download=True,  # Resume if interrupted
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip unnecessary formats
        )
        print(f"\n✓ Successfully downloaded {model_name} to {local_dir}\n")
        return True
    except Exception as e:
        print(f"\n✗ Error downloading {model_name}: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download LLaMA-7B and Alpaca models from HuggingFace Hub"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/n/netscratch/kempner_sham_lab/Lab/itamarf/es-fine-tuning-paper/models",
        help="Base directory to store models",
    )
    parser.add_argument(
        "--llama-only",
        action="store_true",
        help="Download only LLaMA-7B base model",
    )
    parser.add_argument(
        "--alpaca-only",
        action="store_true",
        help="Download only Alpaca weight diff",
    )
    
    args = parser.parse_args()
    
    llama_dir = os.path.join(args.base_dir, "llama-7b-hf")
    alpaca_diff_dir = os.path.join(args.base_dir, "alpaca-7b-diff")
    
    success = True
    
    # Download LLaMA-7B
    if not args.alpaca_only:
        print("\n" + "="*60)
        print("STEP 1: Downloading LLaMA-7B base model")
        print("="*60)
        success &= download_model(
            repo_id="huggyllama/llama-7b",
            local_dir=llama_dir,
            model_name="LLaMA-7B"
        )
    
    # Download Alpaca weight diff
    if not args.llama_only:
        print("\n" + "="*60)
        print("STEP 2: Downloading Alpaca-7B weight diff")
        print("="*60)
        success &= download_model(
            repo_id="tatsu-lab/alpaca-7b",
            local_dir=alpaca_diff_dir,
            model_name="Alpaca-7B weight diff"
        )
    
    print("\n" + "="*60)
    if success:
        print("✓ All downloads completed successfully!")
        print("="*60)
        print("\nNext step: Run the recovery script")
        print(f"\ncd /n/home07/itamarf/es-fine-tuning-paper/alignment")
        print(f"\npython recover_alpaca_weights.py \\")
        print(f"    --path_raw {llama_dir} \\")
        print(f"    --path_diff {alpaca_diff_dir} \\")
        print(f"    --path_tuned {os.path.join(args.base_dir, 'alpaca-7b')}")
    else:
        print("✗ Some downloads failed. Please check the errors above.")
        print("="*60)
    print()


if __name__ == "__main__":
    main()

