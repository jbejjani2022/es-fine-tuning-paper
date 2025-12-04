# Visualization Scripts

This directory contains scripts for generating figures from evaluation results.

## Files

| File | Description |
|------|-------------|
| `plot_gpt_elo_figure.py` | ELO scatter plot (Helpfulness vs Harmlessness) |
| `plot_reward_cost_grid.py` | Grid of reward vs cost scatter plots |
| `plot_safety_ratios.py` | Bar chart comparing safety ratios |

## GPT ELO Figure

Create a publication-ready ELO scatter plot:

```bash
python alignment/visualization/plot_gpt_elo_figure.py \
    --ratings alignment/outputs/gpt_elo/gpt_elo_ratings.json \
    --outfile alignment/outputs/gpt_elo/gpt_elo_figure.png \
    --order Alpaca-7B Beaver-V1 Beaver-V2 Beaver-V3 ES-500
```

Options:
- `--order`: Connect models with a line in this order
- `--name-map`: JSON file mapping model tags to display names
- `--xlim`, `--ylim`: Axis limits
- `--no-crosshair`: Disable dashed guidelines

## Reward-Cost Grid

Plot multiple models' reward vs cost distributions side-by-side:

```bash
python alignment/visualization/plot_reward_cost_grid.py \
    --data-dir alignment/outputs/model_comparisons \
    --models Alpaca-7B Beaver-V1 ES-500 \
    --output alignment/outputs/model_comparisons/reward_cost_grid.png
```

## Safety Ratios

Compare safety ratios across models:

```bash
python alignment/visualization/plot_safety_ratios.py \
    --data-dir alignment/outputs/model_comparisons \
    --models Alpaca-7B Beaver-V1 Beaver-V2 Beaver-V3 ES-500 \
    --output alignment/outputs/model_comparisons/safety_ratios.png
```

## Output Location

Generated figures are typically saved to `alignment/outputs/`:
- GPT ELO plots → `alignment/outputs/gpt_elo/`
- Model comparison plots → `alignment/outputs/model_comparisons/`

