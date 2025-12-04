#!/usr/bin/env python3
"""
Make a high-quality ELO figure (helpfulness vs harmlessness) from saved ratings JSON.

Input: JSON produced by alignment/gpt_elo_eval.py --ratings-out, format:
  { "helpfulness": {tag: rating, ...}, "harmlessness": {tag: rating, ...} }

Features:
  - Baseline (1000) shaded bands on both axes
  - Crosshair dashed lines for each point (to aid reading)
  - Optional connection line for a given order (e.g., Alpaca-7B -> Beaver-V1 -> Beaver-V2 -> Beaver-V3)
  - Clean typography and spacing tuned for papers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot helpfulness vs harmlessness ELO from ratings JSON")
    p.add_argument("--ratings", type=Path, default=Path("alignment/gpt_elo_ratings.json"))
    p.add_argument("--outfile", type=Path, default=Path("alignment/gpt_elo_figure.png"))
    p.add_argument("--order", nargs='*', default=[],
                   help="Order to connect with a line if present")
    p.add_argument("--name-map", type=Path, default=None, help="Optional JSON mapping {tag: display_name}")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--width", type=float, default=5.0)
    p.add_argument("--height", type=float, default=4.2)
    p.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"))
    p.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"))
    p.add_argument("--no-crosshair", action='store_true', help="Disable dashed crosshair lines for points")
    return p.parse_args()


def load_ratings(path: Path) -> Dict[str, Dict[str, float]]:
    with open(path, "r") as f:
        data = json.load(f)
    assert "helpfulness" in data and "harmlessness" in data, "ratings JSON must contain both axes"
    return data


def main() -> None:
    args = parse_args()
    data = load_ratings(args.ratings)
    helpful: Dict[str, float] = data["helpfulness"]
    harmless: Dict[str, float] = data["harmlessness"]

    # Intersect model keys present in both axes
    models = [m for m in helpful.keys() if m in harmless]
    if not models:
        raise ValueError("No overlapping models between helpfulness and harmlessness ratings")

    # Optional name map
    name_map = {}
    if args.name_map and args.name_map.exists():
        with open(args.name_map, "r") as f:
            name_map = json.load(f)

    # Prepare points
    xs, ys, labels = [], [], []
    for m in models:
        xs.append(harmless[m])
        ys.append(helpful[m])
        labels.append(name_map.get(m, m))

    # Bounds
    pad = 30.0
    xmin = min(xs) - pad
    xmax = max(xs) + pad
    ymin = min(ys) - pad
    ymax = max(ys) + pad
    if args.xlim:
        xmin, xmax = args.xlim
    if args.ylim:
        ymin, ymax = args.ylim

    # Figure style
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)

    # Shaded baseline regions (<=1000 on each axis)
    ax.axvspan(xmin, 1000, color="#dcdcdc", alpha=0.3)
    ax.axhspan(ymin, 1000, color="#dcdcdc", alpha=0.3)

    # Grid and limits
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Harmlessness (ELO)")
    ax.set_ylabel("Helpfulness (ELO)")

    # Plot points
    # Highlight baseline if present
    baseline_idx = None
    if "Alpaca-7B" in models:
        baseline_idx = models.index("Alpaca-7B")
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        if baseline_idx is not None and i == baseline_idx:
            ax.scatter(x, y, s=50, color="#e45756", zorder=3)
        else:
            ax.scatter(x, y, s=50, color="#4c78a8", zorder=3)

    # Crosshair lines per point
    if not args.no_crosshair:
        for x, y in zip(xs, ys):
            ax.axvline(x, color="#aaaaaa", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.axhline(y, color="#aaaaaa", linestyle="--", linewidth=1.0, alpha=0.6)

    # Connect models in given order when all present
    order = [m for m in args.order if m in models]
    if len(order) >= 2:
        ox = [harmless[m] for m in order]
        oy = [helpful[m] for m in order]
        ax.plot(ox, oy, color="#333333", linewidth=1.2, zorder=2)

    # Labels with slight offsets to reduce overlap
    for m, x, y in zip(models, xs, ys):
        disp = name_map.get(m, m)
        ax.annotate(disp, (x + 6, y + 6), fontsize=11, color="#222222")

    fig.tight_layout()
    out = args.outfile if args.outfile.is_absolute() else args.outfile.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    main()






