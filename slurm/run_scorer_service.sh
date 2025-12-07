#!/bin/bash
#SBATCH --job-name=scorer_service
#SBATCH --output=logs/scorer_service_%j.log
#SBATCH --error=logs/scorer_service_%j.err
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mem=400GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100

# Set scorer service parameters (using same models as working code)
# export REWARD_MODEL="PKU-Alignment/beaver-7b-v1.0-reward"
# export COST_MODEL="PKU-Alignment/beaver-7b-v1.0-cost"
export REWARD_MODEL="PKU-Alignment/beaver-7b-unified-reward"
export COST_MODEL="PKU-Alignment/beaver-7b-unified-cost"
# Match Beaver model training max_length (see Safe-RLHF paper Tables 2 & 3)
export MAX_LENGTH=512 # add buffer for prompt
export USE_BF16=false


# Run the scorer service
echo "Starting scorer service..."
echo "Reward Model: $REWARD_MODEL"
echo "Cost Model: $COST_MODEL"
echo "Service will be available at http://$(hostname):8000"
echo "----------------------------------------"

python alignment/scoring/scorer_service.py
