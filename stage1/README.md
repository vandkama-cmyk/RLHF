# Stage 1 — Baseline RLHF with Reward-Weighted Policy

## Overview

Stage 1 establishes the foundational RLHF pipeline. A **CodeGPT-small-py** policy model is trained with reward signals from a **CodeBERT-based preference reward model** using Reward-Weighted NLL — not PPO. This stage verifies end-to-end functionality from data preparation through reward estimation to policy optimization, and produces the baseline generative checkpoint that downstream stages compare against.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     STAGE 1 PIPELINE                        │
│                                                             │
│   CoNaLa Dataset (1247 train / 500 test)                    │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────────────┐   Preference Pairs               │
│   │  Preference Reward  │◄──(chosen: reference snippet,    │
│   │  Model (CodeBERT)   │    rejected: shuffled/truncated) │
│   │  125M params        │                                  │
│   │  Bradley-Terry loss │                                  │
│   └──────────┬──────────┘                                  │
│              │ reward score r(x,y)                         │
│              ▼                                              │
│   ┌─────────────────────┐                                  │
│   │   Policy Model      │                                  │
│   │  CodeGPT-small-py   │                                  │
│   │  124M params        │                                  │
│   │  12L / 768H / 12A   │                                  │
│   └──────────┬──────────┘                                  │
│              │ Reward-Weighted NLL                          │
│              │ L = -σ(r) · log P(y|x)                      │
│              ▼                                              │
│   Per-epoch checkpoints + metrics                           │
└─────────────────────────────────────────────────────────────┘
```

### Policy Model — CodeGPT-small-py

| Parameter | Value |
|-----------|-------|
| Base model | `microsoft/CodeGPT-small-py` |
| Parameters | 124M |
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Max position | 1024 |

### Reward Model — CodeBERT + Preference Head

| Parameter | Value |
|-----------|-------|
| Base model | `microsoft/codebert-base` |
| Parameters | 125M (encoder) + scalar head |
| Head | Linear(768 → 1) |
| Loss | Bradley-Terry: `-log σ(score_chosen − score_rejected)` |
| Preference pairs | Up to 2,000 |
| Preference epochs | 2 |

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | Reward-Weighted NLL |
| Optimizer | AdamW |
| Learning rate | 5e-6 |
| Batch size | 4 |
| Gradient accumulation | 4× (effective batch 16) |
| Max grad norm | 1.0 |
| Total epochs | 30 |
| Warmup steps | 100 |
| Mixed precision | FP16 (CUDA) |
| Gradient checkpointing | Enabled |

### Generation Settings

| Parameter | Value |
|-----------|-------|
| Max new tokens | 256 |
| Temperature | 0.7 |
| Top-p | 0.9 |
| Top-k | 50 |
| Repetition penalty | 1.2 |
| Num beams | 4 |

### Reward Composition

| Component | Weight |
|-----------|--------|
| Syntax correctness | 0.2 |
| Execution passing | 0.3 |
| Semantic similarity | 0.3 |
| Human preference | 0.2 |

---

## Metrics Tracked Per Epoch

- **BERTScore** (roberta-large F1)
- **ROUGE-L**
- **BLEU**
- **Ruby** (PDG-based code similarity)
- **CodeBLEU** (code-aware BLEU)

---

## Results

| Epoch | BERTScore | ROUGE-L | BLEU | Ruby | CodeBLEU |
|-------|-----------|---------|------|------|----------|
| 0 | 0.810 | 0.100 | 0.010 | 0.150 | 0.050 |
| 5 | 0.790 | 0.065 | 0.005 | 0.120 | 0.030 |
| 10 | 0.810 | 0.110 | 0.010 | 0.140 | 0.040 |
| 30 | ~0.82 | ~0.11 | ~0.01 | ~0.15 | ~0.05 |

> Stage 1 serves as the baseline. Absolute metric values are modest because the policy is small (124M) and the reward signal is synthetic.

---

## Directory Structure

```
stage1/
├── config.py                        # Full experiment configuration
├── run_stage1.py                    # Main entry point (reward-weighted NLL)
├── run_stage1b.py                   # Full PPO variant
├── run_stage1_light.py              # Memory-optimized version (<4 GB VRAM)
├── preference_reward_model.py       # Bradley-Terry reward model
├── quick_metrics.py                 # Fast metric computation
├── compute_missing_metrics.py       # Recompute metrics for saved checkpoints
├── compute_ruby_codebleu.py         # Ruby / CodeBLEU computation
├── output/
│   ├── checkpoint_epoch_1/          # Saved model + tokenizer (epoch 1)
│   ├── ...
│   ├── checkpoint_epoch_30/
│   ├── stage1_results.json          # Full per-epoch results (authoritative)
│   ├── stage1_metrics.csv           # Metrics table (CSV)
│   └── preference_reward.pt         # Trained preference model weights
└── output_upd/
    └── stage1_results.json          # Older snapshot (30 epochs, kept for reference)
```

---

## How to Run

```bash
# Default run (30 epochs)
python stage1/run_stage1.py

# Low-VRAM version
python stage1/run_stage1_light.py

# Custom hyperparameters
python stage1/run_stage1.py --epochs 30 --lr 5e-6 --batch-size 4
```
