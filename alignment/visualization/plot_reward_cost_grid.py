#!/usr/bin/env python3
"""
Generate reward-vs-cost scatter plots with fixed axes from saved arrays.

This script scans a directory for datasets saved as
  <tag>_Rs.npy, <tag>_Cs.npy, [optional] <tag>_labels.npy, <tag>_points.jsonl
and draws a scatter plot for each tag. Axes limits are fixed across subplots
to enable visual comparison (defaults: cost in [-10, 10], reward in [-4, 4]).

Example:
  python plot_reward_cost_grid.py \
      --data-dir alignment/plots \
      --tags alpaca_test_n3036 alpaca_test_n30 \
      --outfile reward_cost_scatter_grid.png

If --tags is omitted, the script will auto-discover all available tags in the
directory that have both *_Rs.npy and *_Cs.npy files.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Dataset:
    tag: str
    rewards: np.ndarray
    costs: np.ndarray


def discover_tags(data_dir: Path) -> List[str]:
    """Return base tags that have both Rs and Cs arrays present.

    A tag is the filename prefix before the suffixes `_Rs.npy` and `_Cs.npy`.
    """
    rs_files = {p.name[:-7] for p in data_dir.glob("*_Rs.npy")}
    cs_files = {p.name[:-7] for p in data_dir.glob("*_Cs.npy")}
    tags = sorted(rs_files.intersection(cs_files))
    return tags


def load_dataset(data_dir: Path, tag: str, max_points: int | None) -> Dataset:
    rewards = np.load(data_dir / f"{tag}_Rs.npy")
    costs = np.load(data_dir / f"{tag}_Cs.npy")
    # Align shapes and optionally subsample for speed/clarity
    n = min(len(rewards), len(costs))
    rewards = rewards[:n].astype(np.float32, copy=False)
    costs = costs[:n].astype(np.float32, copy=False)
    if max_points is not None and n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        rewards = rewards[idx]
        costs = costs[idx]
    return Dataset(tag=tag, rewards=rewards, costs=costs)


def choose_layout(num_plots: int) -> Tuple[int, int]:
    """Pick a compact grid for the given number of subplots."""
    if num_plots <= 3:
        return 1, num_plots
    if num_plots == 4:
        return 1, 4
    if num_plots <= 6:
        return 2, 3
    if num_plots <= 9:
        return 3, 3
    # Fallback to a tall grid
    rows = (num_plots + 3) // 4
    return rows, 4


def ticks_for_limits(lim: Tuple[float, float], step: float) -> List[float]:
    lo, hi = lim
    # Generate symmetrical ticks when possible for nicer visuals
    start = np.ceil(lo / step) * step
    end = np.floor(hi / step) * step
    ticks = np.arange(start, end + 0.5 * step, step)
    return [float(x) for x in ticks]


def plot_grid(datasets: Sequence[Dataset], xlim: Tuple[float, float], ylim: Tuple[float, float], outfile: Path) -> None:
    rows, cols = choose_layout(len(datasets))
    # 4x4 inches per subplot for legibility
    fig_w = 4 * cols
    fig_h = 4 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=150, squeeze=False)

    colors = plt.get_cmap("tab10")
    xticks = ticks_for_limits(xlim, step=5.0)
    yticks = ticks_for_limits(ylim, step=2.0)

    for i, ds in enumerate(datasets):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.scatter(ds.costs, ds.rewards, s=6, alpha=0.35, color=colors(i % 10))
        ax.axvline(0.0, color='red', linestyle='--', linewidth=1)
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xlabel('cost')
        ax.set_ylabel('reward')
        ax.set_title(ds.tag)

    # Hide any unused axes
    total_axes = rows * cols
    for j in range(len(datasets), total_axes):
        r = j // cols
        c = j % cols
        axes[r][c].axis('off')

    plt.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)


def plot_individual(datasets: Sequence[Dataset], xlim: Tuple[float, float], ylim: Tuple[float, float], outdir: Path) -> None:
    xticks = ticks_for_limits(xlim, step=5.0)
    yticks = ticks_for_limits(ylim, step=2.0)
    for ds in datasets:
        fig = plt.figure(figsize=(4, 4), dpi=150)
        plt.scatter(ds.costs, ds.rewards, s=6, alpha=0.35)
        plt.axvline(0.0, color='red', linestyle='--', linewidth=1)
        plt.axhline(0.0, color='gray', linestyle=':', linewidth=0.8)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel('cost')
        plt.ylabel('reward')
        # plt.title(ds.tag)
        plt.tight_layout()
        plt.savefig(outdir / f"{ds.tag}_scatter_fixed.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward vs cost scatter with fixed axes")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent, help="Directory with *_Rs.npy and *_Cs.npy files")
    parser.add_argument("--tags", nargs='*', default=None, help="Specific tags (prefix before _Rs/_Cs). If omitted, auto-discover.")
    parser.add_argument("--xlim", type=float, nargs=2, default=[-10.0, 10.0], metavar=("XMIN", "XMAX"), help="Cost axis limits")
    parser.add_argument("--ylim", type=float, nargs=2, default=[-4.0, 4.0], metavar=("YMIN", "YMAX"), help="Reward axis limits")
    parser.add_argument("--max-points", type=int, default=4000, help="Subsample to at most this many points per tag")
    parser.add_argument("--outfile", type=Path, default=Path("reward_cost_scatter_grid.png"), help="Output image path for the grid plot")
    parser.add_argument("--also-individual", action="store_true", help="Also write per-tag images with fixed axes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.tags:
        tags = list(dict.fromkeys(args.tags))  # preserve order, unique
    else:
        tags = discover_tags(data_dir)
        if not tags:
            raise FileNotFoundError(f"No tags discovered in {data_dir}. Expected *_Rs.npy and *_Cs.npy files.")

    datasets = [load_dataset(data_dir, tag, args.max_points) for tag in tags]

    # Grid plot
    outfile: Path = args.outfile if args.outfile.is_absolute() else data_dir / args.outfile
    plot_grid(datasets, xlim=(args.xlim[0], args.xlim[1]), ylim=(args.ylim[0], args.ylim[1]), outfile=outfile)

    # Optional per-tag images
    if args.also_individual:
        plot_individual(datasets, xlim=(args.xlim[0], args.xlim[1]), ylim=(args.ylim[0], args.ylim[1]), outdir=data_dir)


if __name__ == "__main__":
    main()


