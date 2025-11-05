# CS2881 Mini-Project: ES Fine-Tuning Beyond RL (Fork)

This is our CS2881 mini-project fork of the authors’ repository “Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning.” We replicate headline results on Countdown and Conciseness and extend the work with alignment experiments using Safe-RLHF unified reward/cost models and a lightweight scorer API.

- Original repository (authors): https://github.com/VsonicV/es-fine-tuning-paper
- This fork (our project): https://github.com/jbejjani2022/es-fine-tuning-paper

## What’s in this fork

- Reproduction of Tables 1 and 2 from the paper (Countdown accuracy; Conciseness reward and KL)
- Evaluation scripts for both tasks (authors did not release them in the original repo at the time of writing)
- Alignment experiments (helpful–harmless) using Safe-RLHF unified reward/cost models, with an external scorer service to avoid GPU contention with vLLM
- SLURM job scripts for training and evaluation on H100 nodes

## Repository structure (high level)

- alignment/: ES for alignment (helpful–harmless)
  - safe-rlhf/ (local clone required by scorer scripts)
  - scorer_service.py (FastAPI scorer using unified models)
  - es-fine-tuning_alignment_accl_api.py (ES training with external scorer API)
  - classify_reward_cost_unified.py (classify PKU responses by reward/cost; saves arrays/plots)
  - alpaca/ (scripts to recover Alpaca‑7B from LLaMA‑7B + diff)
  - select_pku_balanced.ipynb (creates custom train/eval selections)
  - See alignment/README.md for details and step-by-step instructions

- conciseness/: ES/GRPO scripts and evaluation for the Conciseness task
  - es_fine-tuning_conciseness_iid.py (recommended i.i.d. noise version)
  - conciseness_eval.py, conciseness_eval_reward_only.py (evaluation utilities)
  - data/ (JSONL train/eval examples)

- countdown/: ES scripts and evaluation for the Countdown task
  - es_fine-tuning_countdown_iid.py (recommended i.i.d. noise version)
  - es_fine-tuning_countdown_accl.py (accelerated vLLM version)
  - countdown_eval.py (evaluation utility)
  - data/countdown.json (dataset)

- GRPO/: Configuration and training scripts for GRPO baselines
  - train_grpo_conciseness_trl.py; countdown/ configs; accelerate configs

- slurm/: SLURM job scripts for training/eval
  - train_conciseness.sh, conciseness_es_eval.sh, conciseness_grpo_eval.sh
  - train_countdown.sh, countdown_eval.sh, countdown_accl_eval.sh
  - run_scorer_service.sh, train_accl_alignment_api.sh (alignment)

- utils/worker_extn.py: small vLLM worker extension used by accelerated/alignment paths

## Quick setup

Create and activate a Python 3.10+ virtual environment, then install requirements:
```bash
python -m venv es
source es/bin/activate
pip install -r requirement.txt
```

Notes:
- Accelerated/vLLM paths require vLLM==0.11.0 (already pinned in requirement.txt)
- For alignment, clone Safe-RLHF locally into alignment/: `git clone https://github.com/PKU-Alignment/safe-rlhf.git alignment/safe-rlhf`

## Reproducing results (brief pointers)

- Conciseness (ES): see slurm/train_conciseness.sh and slurm/conciseness_es_eval.sh
- Conciseness (GRPO): see slurm/conciseness_grpo_eval.sh and GRPO configs
- Countdown (ES): see slurm/train_countdown.sh and slurm/countdown_eval.sh
- Countdown (accelerated): see slurm/train_accl_countdown.sh and slurm/countdown_accl_eval.sh
- Alignment (ES with external scorer): see alignment/README.md

## Project context

Motivated by the authors’ claim that ES can match or exceed RL methods like GRPO in sample efficiency, long-horizon robustness, and reduced reward hacking, we:
- Reproduce the main results (Countdown accuracy; Conciseness reward/KL)
- Add alignment experiments using Safe-RLHF unified models (reward/cost)
- Compare ES and RL behavior qualitatively and under perturbations

## Citation

If you reference the upstream paper, please cite:

```bibtex
@misc{qiu2025evolutionstrategiesscalellm,
      title={Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning}, 
      author={Xin Qiu and Yulu Gan and Conor F. Hayes and Qiyao Liang and Elliot Meyerson and Babak Hodjat and Risto Miikkulainen},
      year={2025},
      eprint={2509.24372},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24372}, 
}
```