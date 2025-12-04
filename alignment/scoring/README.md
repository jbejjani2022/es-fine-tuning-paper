# Scoring Service

This directory contains the scoring service that provides reward and cost scores for alignment training.

## Overview

The scorer service loads the **PKU-Alignment Beaver** models and exposes an HTTP API for scoring prompt-response pairs:

- **Reward Model**: `PKU-Alignment/beaver-7b-unified-reward` (helpfulness)
- **Cost Model**: `PKU-Alignment/beaver-7b-unified-cost` (harmfulness)

## Files

| File | Description |
|------|-------------|
| `scorer_service.py` | FastAPI server exposing `/score`, `/score_batch`, and `/score_texts` endpoints |
| `test_score_fixed_prompt.py` | Utility to test scoring on fixed prompts |

## Quick Start

### Start the Service

```bash
# Via SLURM (recommended for cluster)
sbatch slurm/run_scorer_service.sh

# Or directly (for local testing)
python alignment/scoring/scorer_service.py
```

### Check Service Health

```bash
curl http://<HOST>:8000/health
# Returns: {"status": "healthy", "device": "cuda:0"}
```

## API Endpoints

### `POST /score_batch`

Score multiple prompt-response pairs (recommended for training).

**Request:**
```json
{
  "prompts": ["What is 2+2?", "How to hack?"],
  "responses": ["2+2 equals 4.", "I cannot help with that."]
}
```

**Response:**
```json
{
  "rewards": [0.85, 0.12],
  "costs": [-0.2, -0.95]
}
```

### `POST /score`

Score a single prompt-response pair.

### `POST /score_texts`

Score pre-formatted full texts (prompt + response already concatenated).

## Configuration

Environment variables (set in `slurm/run_scorer_service.sh`):

| Variable | Description | Default |
|----------|-------------|---------|
| `REWARD_MODEL` | HuggingFace model ID for reward | `beaver-7b-unified-reward` |
| `COST_MODEL` | HuggingFace model ID for cost | `beaver-7b-unified-cost` |
| `MAX_LENGTH` | Maximum sequence length | 576 |
| `USE_BF16` | Use bfloat16 precision | false |

## Text Format

The scorer constructs the canonical format expected by Beaver models:

```
BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}<eos>
```

## Interpretation

| Metric | Positive Value | Negative Value |
|--------|---------------|----------------|
| **Reward** | Helpful response | Unhelpful response |
| **Cost** | Harmful/unsafe | Safe response |

**Goal**: Maximize reward, minimize cost (or keep cost ≤ 0).

## Testing

Test with a known prompt:

```bash
python alignment/scoring/test_score_fixed_prompt.py
```

This runs scoring on fixed examples and prints quadrant classifications (SH/SU/UH/UU).

