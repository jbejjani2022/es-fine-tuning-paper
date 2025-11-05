#!/bin/bash
#SBATCH --job-name=robustness_eval_sweep
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH -t 1-00:00
#SBATCH --mem=256G
#SBATCH -o output/job.%N.%j.out          # STDOUT
#SBATCH -e error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

# Load modules
module load python/3.10.13-fasrc01
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
mamba deactivate
mamba activate es

cd ../..

# Set PyTorch memory allocator configuration for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for model in es_small_sig es_big_sig grpo_beta0 grpo_beta0464
do
    for sigma in 0.005 0.01
    do
        python perturb/sweep.py --model $model --sigma $sigma
    done
done
