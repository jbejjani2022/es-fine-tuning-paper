"""Plot mean reward per iteration for the conciseness task across training seeds."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent / "conciseness_run_data/alpha0.0005_sigma0.001"
OUTPUT_PATH = Path(__file__).resolve().parent / "qwen7b_conciseness_train_reward_alpha0.0005_sigma0.001.png"

ITERATION_PATTERN = re.compile(r"Iteration\s+(\d+)\s*/\s*(\d+)")
MEAN_PATTERN = re.compile(r"Mean:\s*(-?\d+(?:\.\d+)?)")


def extract_mean_series(path: Path) -> List[float]:
    """Return the ordered list of mean rewards for each iteration in ``path``."""

    iteration_to_mean: Dict[int, float] = {}
    expected_total: int | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "Iteration" not in line or "Mean" not in line:
                continue

            iteration_match = ITERATION_PATTERN.search(line)
            mean_match = MEAN_PATTERN.search(line)

            if not iteration_match or not mean_match:
                continue

            iteration = int(iteration_match.group(1))
            expected_total = expected_total or int(iteration_match.group(2))
            iteration_to_mean[iteration] = float(mean_match.group(1))

    if not iteration_to_mean:
        raise ValueError(f"No iteration data found in {path}")

    if expected_total and len(iteration_to_mean) != expected_total:
        print(
            f"Warning: {path.name} expected {expected_total} iterations but found {len(iteration_to_mean)}; proceeding with available data."
        )

    ordered_iterations = sorted(iteration_to_mean)
    return [iteration_to_mean[idx] for idx in ordered_iterations]


def plot_mean_rewards(series_by_seed: Iterable[Tuple[str, List[float]]]) -> None:
    """Create the training reward plot and save it to ``OUTPUT_PATH``."""

    plt.figure(figsize=(10, 6))
    color_cycle = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])

    for index, (seed_label, rewards) in enumerate(series_by_seed):
        iterations = range(1, len(rewards) + 1)
        color = color_cycle[index % len(color_cycle)] if color_cycle else None
        plt.plot(iterations, rewards, label=seed_label, color=color, linewidth=1.6)

    plt.title("ES for Conciseness Mean Reward vs. Training Iteration\nQwen2.5-7B-Instruct, alpha=0.0005, sigma=0.001")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(title="Training Seed")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()
    print(f"Saved plot to {OUTPUT_PATH}")


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    seed_files = sorted(DATA_DIR.glob("seed*.out"))
    if not seed_files:
        raise FileNotFoundError(f"No seed files found in {DATA_DIR}")

    series: List[Tuple[str, List[float]]] = []
    for path in seed_files:
        seed_number = path.stem.replace("seed", "")
        rewards = extract_mean_series(path)
        series.append((seed_number, rewards))

    plot_mean_rewards(series)


if __name__ == "__main__":
    main()