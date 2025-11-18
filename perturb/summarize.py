#!/usr/bin/env python3
"""
Robustness Evaluation Summary Statistics Script

For a given sigma and perturbation type, this script reports the mean and std 
of normalized_reward_degradation and KL divergence values across the 8 trials 
for each model and checkpoint seed.

Example usage:
    # Display per-seed statistics for sigma=0.01 and lm_head perturbations
    python summarize.py --sigma 0.01 --perturbation_type lm_head.weight
    
    # Display aggregated statistics across all seeds
    python summarize.py --sigma 0.01 --perturbation_type all_model_params --aggregate
    
    # Analyze a different sigma value
    python summarize.py --sigma 0.005 --perturbation_type lm_head.weight
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np


# Constants (reused from visualize.py)
MODELS = ['es_big_sig', 'es_small_sig', 'grpo_beta0', 'grpo_beta0464']
PERTURBATION_TYPES = ['lm_head.weight', 'all_model_params']


def load_csv_data(model_name, sigma, base_dir):
    """
    Load CSV data for a given model and sigma value.
    
    Args:
        model_name: Name of the model (e.g., 'es_big_sig')
        sigma: Sigma value (e.g., 0.01)
        base_dir: Base directory containing CSV files
    
    Returns:
        DataFrame with the CSV data
    """
    csv_filename = f"{model_name}_{sigma}.csv"
    csv_path = os.path.join(base_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def compute_summary_stats(data, perturbation_type):
    """
    Compute summary statistics for a specific perturbation type.
    
    Args:
        data: DataFrame containing the evaluation data
        perturbation_type: Type of perturbation to analyze
    
    Returns:
        Dictionary mapping checkpoint_seed to statistics dict
    """
    # Filter data by perturbation type
    filtered_data = data[data['perturbation_type'] == perturbation_type]
    
    if filtered_data.empty:
        return {}
    
    # Group by checkpoint seed and compute statistics
    grouped = filtered_data.groupby('checkpoint_seed')
    
    results = {}
    for seed, group in grouped:
        results[seed] = {
            'degradation_mean': group['normalized_reward_degradation'].mean(),
            'degradation_std': group['normalized_reward_degradation'].std(),
            'kl_mean': group['mean_per_token_kl'].mean(),
            'kl_std': group['mean_per_token_kl'].std(),
            'num_trials': len(group)
        }
    
    return results


def compute_aggregated_stats(data, perturbation_type):
    """
    Compute aggregated statistics across all seeds for a specific perturbation type.
    
    Args:
        data: DataFrame containing the evaluation data
        perturbation_type: Type of perturbation to analyze
    
    Returns:
        Dictionary with aggregated statistics
    """
    # Filter data by perturbation type
    filtered_data = data[data['perturbation_type'] == perturbation_type]
    
    if filtered_data.empty:
        return {}
    
    # Compute statistics across all trials (all seeds combined)
    results = {
        'degradation_mean': filtered_data['normalized_reward_degradation'].mean(),
        'degradation_std': filtered_data['normalized_reward_degradation'].std(),
        'kl_mean': filtered_data['mean_per_token_kl'].mean(),
        'kl_std': filtered_data['mean_per_token_kl'].std(),
        'num_trials': len(filtered_data)
    }
    
    return results


def display_summary(model_name, stats):
    """
    Display summary statistics in a formatted way.
    
    Args:
        model_name: Name of the model
        stats: Dictionary of statistics per checkpoint seed
    """
    print(f"\nModel: {model_name}")
    print("-" * 80)
    
    if not stats:
        print("  No data available for this perturbation type")
        return
    
    # Sort seeds for consistent display order
    for seed in sorted(stats.keys()):
        seed_stats = stats[seed]
        print(f"  {seed}:")
        print(f"    normalized_reward_degradation: {seed_stats['degradation_mean']:.3f} ± {seed_stats['degradation_std']:.3f}")
        print(f"    KL divergence (mean_per_token): {seed_stats['kl_mean']:.3f} ± {seed_stats['kl_std']:.3f}")
        if seed_stats['num_trials'] != 8:
            print(f"    WARNING: Expected 8 trials, found {seed_stats['num_trials']}")


def display_aggregated_summary(model_name, stats):
    """
    Display aggregated summary statistics.
    
    Args:
        model_name: Name of the model
        stats: Dictionary with aggregated statistics
    """
    print(f"\nModel: {model_name}")
    print("-" * 80)
    
    if not stats:
        print("  No data available for this perturbation type")
        return
    
    print(f"  normalized_reward_degradation: {stats['degradation_mean']:.3f} ± {stats['degradation_std']:.3f}")
    print(f"  KL divergence (mean_per_token): {stats['kl_mean']:.3f} ± {stats['kl_std']:.3f}")
    print(f"  Total trials: {stats['num_trials']}")
    if stats['num_trials'] != 32:
        print(f"  WARNING: Expected 32 trials (8 trials × 4 seeds), found {stats['num_trials']}")


def main():
    """Main function to compute and display summary statistics."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compute summary statistics from robustness evaluation CSV files'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        required=True,
        choices=[0.005, 0.01, 0.05],
        help='Sigma value to analyze (0.005, 0.01, or 0.05)'
    )
    parser.add_argument(
        '--perturbation_type',
        type=str,
        required=True,
        choices=PERTURBATION_TYPES,
        help='Perturbation type to analyze (lm_head.weight or all_model_params)'
    )
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='Aggregate statistics across all seeds (32 trials total) instead of per-seed'
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Summary Statistics")
    if args.aggregate:
        print("(Aggregated across all seeds)")
    print("=" * 80)
    print(f"Sigma: {args.sigma}")
    print(f"Perturbation Type: {args.perturbation_type}")
    print("=" * 80)
    
    # Get base directory (where this script is located)
    base_dir = Path(__file__).parent.absolute()
    
    # Process each model
    for model_idx, model_name in enumerate(MODELS):
        try:
            # Load CSV data
            data = load_csv_data(model_name, args.sigma, base_dir)
            
            if args.aggregate:
                # Compute aggregated statistics across all seeds
                stats = compute_aggregated_stats(data, args.perturbation_type)
                # Display results
                display_aggregated_summary(model_name, stats)
            else:
                # Compute summary statistics per seed
                stats = compute_summary_stats(data, args.perturbation_type)
                # Display results
                display_summary(model_name, stats)
        
        except FileNotFoundError as e:
            print(f"\nModel: {model_name}")
            print("-" * 80)
            print(f"  Error: CSV file not found")
            continue
        except Exception as e:
            print(f"\nModel: {model_name}")
            print("-" * 80)
            print(f"  Unexpected error: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("Summary complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
