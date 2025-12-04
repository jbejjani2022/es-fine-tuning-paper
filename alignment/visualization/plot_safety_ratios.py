#!/usr/bin/env python3
"""
Plot harmful vs harmless ratios per model tag, using saved results.

Priority of data sources per tag (highest to lowest):
  1) <tag>_summary.json (uses 'percents')
  2) <tag>_labels.npy     (counts UH/UU vs SH/SU)
  3) <tag>_Cs.npy         (harmful = mean(cost > 0))

Example:
  python plot_safety_ratios.py \
    --data-dir alignment/plots_es_ft_100steps \
    --tags alpaca-7b_test_n3036_es_ft_vllm \
    --outfile model_safety_ratios.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SafetyRatio:
    tag: str
    harmful: float
    harmless: float


def discover_tags(data_dir: Path) -> List[str]:
    """Find base tags present in the directory.

    Looks for *_summary.json, *_labels.npy, or *_Cs.npy and returns the union.
    """
    tags = set()
    for p in data_dir.glob("*_summary.json"):
        tags.add(p.name[:-13])  # remove '_summary.json'
    if not tags:
        for p in data_dir.glob("*_labels.npy"):
            tags.add(p.name[:-11])  # remove '_labels.npy'
    if not tags:
        for p in data_dir.glob("*_Cs.npy"):
            tags.add(p.name[:-7])  # remove '_Cs.npy'
    return sorted(tags)


def compute_ratios(data_dir: Path, tag: str) -> SafetyRatio:
    """Compute harmful/harmless ratios in percent for a given tag.

    Prefers summary JSON, then labels, then costs.
    """
    sum_path = data_dir / f"{tag}_summary.json"
    if sum_path.exists():
        with open(sum_path, "r") as f:
            s = json.load(f)
        perc = s.get("percents", {})
        harmless = float(perc.get("SH", 0.0) + perc.get("SU", 0.0))
        harmful = float(perc.get("UH", 0.0) + perc.get("UU", 0.0))
        if harmless == 0.0 and harmful == 0.0:
            safe_ratio = float(s.get("safe_ratio", 0.0))
            harmless = safe_ratio
            harmful = 100.0 - safe_ratio
        return SafetyRatio(tag, harmful, harmless)

    lbl_path = data_dir / f"{tag}_labels.npy"
    if lbl_path.exists():
        labels = np.load(lbl_path)
        total = max(len(labels), 1)
        harmful = 100.0 * float(np.isin(labels, ["UH", "UU"]).sum()) / float(total)
        harmless = 100.0 - harmful
        return SafetyRatio(tag, harmful, harmless)

    cs_path = data_dir / f"{tag}_Cs.npy"
    if cs_path.exists():
        costs = np.load(cs_path).astype(np.float32, copy=False)
        harmful = 100.0 * float((costs > 0.0).mean())
        harmless = 100.0 - harmful
        return SafetyRatio(tag, harmful, harmless)

    raise FileNotFoundError(f"No summary/labels/Cs file found for tag '{tag}' in {data_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot harmful vs harmless ratios per model tag")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent, help="Directory with saved outputs")
    parser.add_argument("--tags", nargs='*', default=None, help="Specific tags (prefix before _summary/_labels/_Cs). If omitted, auto-discover.")
    parser.add_argument("--outfile", type=Path, default=Path("model_safety_ratios.png"), help="Output image path")
    parser.add_argument("--name-map", type=Path, default=None, help="Optional JSON mapping {tag: display_name}")
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.0, 100.0], metavar=("YMIN", "YMAX"), help="Y-axis limits in percent")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument("--mock", action='append', default=[],
                        help="Add mocked entry as 'Label:HARMFUL[,HARMLESS]' (percent). If HARMLESS omitted, uses 100-HARMFUL. Can be repeated.")
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
            raise FileNotFoundError(f"No tags discovered in {data_dir}.")

    name_map: Dict[str, str] = {}
    if args.name_map and args.name_map.exists():
        with open(args.name_map, "r") as f:
            name_map = json.load(f)

    ratios: List[SafetyRatio] = [compute_ratios(data_dir, t) for t in tags]

    # Parse mocked entries and append after real tags
    for spec in (args.mock or []):
        if ':' not in spec:
            raise ValueError(f"Invalid --mock '{spec}'. Expected 'Label:HARMFUL[,HARMLESS]'.")
        label, rest = spec.split(':', 1)
        parts = [p.strip() for p in rest.split(',') if p.strip()]
        if not parts:
            raise ValueError(f"Invalid --mock '{spec}'. Provide at least HARMFUL percent.")
        harmful = float(parts[0])
        harmless = (100.0 - harmful) if len(parts) == 1 else float(parts[1])
        ratios.append(SafetyRatio(tag=label, harmful=harmful, harmless=harmless))

    labels = [name_map.get(r.tag, r.tag) for r in ratios]
    harmful = [r.harmful for r in ratios]
    harmless = [r.harmless for r in ratios]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(4 + 1.5 * max(0, len(labels) - 3), 4), dpi=args.dpi)
    ax.bar(x - width / 2, harmful, width, label="Harmful ratio", color="#ff9a9a")
    ax.bar(x + width / 2, harmless, width, label="Harmless ratio", color="#75a8ff")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(args.ylim[0], args.ylim[1])
    yticks = list(range(0, 101, 10))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y}%" for y in yticks])
    ax.set_ylabel("ratio")
    ax.set_title("Model safety on evaluation set")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    plt.tight_layout()
    outfile: Path = args.outfile if args.outfile.is_absolute() else data_dir / args.outfile
    fig.savefig(outfile)
    plt.close(fig)


if __name__ == "__main__":
    main()


