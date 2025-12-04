#!/bin/bash
#SBATCH --job-name=classify_reward_cost_unified
#SBATCH --output=logs/classify_reward_cost_unified_%A_%a.log
#SBATCH --error=logs/classify_reward_cost_unified_%A_%a.err
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=400GB
#SBATCH --partition=kempner_requeue
#SBATCH --constraint=h100
#SBATCH --array=0-1  # Adjust based on number of sweeps (0-indexed)

# ============================================================
# Sweep runner using SLURM job arrays + YAML config
# 
# Usage:
#   sbatch alignment/classify_reward_cost_unified.sh                    # Run all sweeps
#   sbatch --array=0-2 alignment/classify_reward_cost_unified.sh        # Run first 3 sweeps
#   sbatch --array=0 alignment/classify_reward_cost_unified.sh          # Run only first sweep
#   sbatch --array=1,3,4 alignment/classify_reward_cost_unified.sh      # Run specific sweeps
#
# To specify a different config file:
#   CONFIG_FILE=my_sweeps.yaml sbatch alignment/classify_reward_cost_unified.sh
# ============================================================

set -e

# Config file path (can be overridden via environment variable)
export CONFIG_FILE="${CONFIG_FILE:-alignment/sweep_configs.yaml}"

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Get task ID (default to 0 for non-array jobs or testing)
export TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "=== Job Array Task ${TASK_ID} ==="
echo "Config file: ${CONFIG_FILE}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Python helper to parse YAML and build command, then execute it
python3 << 'PYEOF'
import yaml
import sys
import os
import subprocess

config_file = os.environ.get('CONFIG_FILE', 'alignment/sweep_configs.yaml')
task_id = int(os.environ.get('TASK_ID', '0'))

with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

defaults = config.get('defaults', {})
sweeps = config.get('sweeps', [])

if task_id >= len(sweeps):
    print(f"ERROR: Task ID {task_id} exceeds number of sweeps ({len(sweeps)})", file=sys.stderr)
    sys.exit(1)

sweep = sweeps[task_id]
print(f"Running sweep {task_id}: {sweep.get('model_name', 'unnamed')}")
print("-" * 60)

# Merge defaults with sweep-specific settings
merged = {**defaults, **sweep}

# Build command line arguments
cmd_parts = ["python", "alignment/classify_reward_cost_unified.py"]

for key, value in merged.items():
    if key.startswith('_'):  # Skip internal keys
        continue
    
    arg_name = f"--{key}"
    
    if isinstance(value, bool):
        if value:
            cmd_parts.append(arg_name)
    elif value is not None and value != '':
        cmd_parts.append(arg_name)
        cmd_parts.append(str(value))

# Print command
cmd_display = " \\\n  ".join(cmd_parts)
print(f"\nCommand:\n{cmd_display}\n")
print("=" * 60)

# Execute the command directly
sys.exit(subprocess.call(cmd_parts))
PYEOF

echo ""
echo "=== Task ${TASK_ID} completed ==="
