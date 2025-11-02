#!/bin/bash
#SBATCH --job-name=es_accl_fine_tuning_countdown
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-3

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
mamba deactivate
mamba activate es

cd ..

# Multi-GPU run (one vLLM engine per GPU)
python countdown/es_fine-tuning_countdown_accl.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --cuda_devices 0,1,2,3 \
  --num_engines 4 \
  --population_size 30 \
  --num_iterations 500 \
  --sigma 0.001 \
  --alpha 0.0005 \
  --global_seed $SLURM_ARRAY_TASK_ID \
  --max_new_tokens 1024 \
  --experiment_dir es-ft-experiment
