# Stage 2 — Enhanced Reward Models

## Overview

Stage 2 develops two parallel branches of improved reward models. In both branches the **policy model (CodeGPT) is frozen** — only the reward model is trained. The goal is to build a more discriminative reward signal for use in downstream stages.

| Sub-stage | Method | Key idea |
|-----------|--------|----------|
| **2A** | Contrastive Learning | Maximum-margin loss pushes similar code pairs together and dissimilar apart |
| **2B** | GPT-2 Feedback Generator | GPT-2 generates synthetic human-like feedback that supervises CodeBERT reward heads |

---

## Stage 2A — Contrastive Reward Model

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 2A ARCHITECTURE                        │
│                                                                  │
│   Input: (question, code_A, code_B) pair                        │
│          │                                                       │
│          ▼                                                       │
│   ┌──────────────────┐   ┌──────────────────┐                  │
│   │  CodeBERT-base   │   │  CodeBERT-base   │  (shared weights)│
│   │  125M params     │   │  125M params     │                  │
│   │  0-4 layers      │   │  0-4 layers      │                  │
│   │  frozen          │   │  frozen          │                  │
│   └────────┬─────────┘   └────────┬─────────┘                  │
│            │ [CLS] 768-d           │ [CLS] 768-d               │
│            ▼                       ▼                            │
│   ┌─────────────────────────────────────────┐                  │
│   │        Contrastive Projection Head       │                  │
│   │   768 → 1536 → 256 (L2-normalized)      │                  │
│   │   Dropout: 0.1                           │                  │
│   └──────────────────┬──────────────────────┘                  │
│                      │                                          │
│          ┌───────────┴──────────────────────┐                  │
│          │                                   │                  │
│          ▼                                   ▼                  │
│   Contrastive Loss                  ┌────────────────┐         │
│   L = (1-y)·d²                      │  Reward Heads  │         │
│     + y·max(0, m-d)²               │  (3 parallel)  │         │
│   margin m = 0.5                    │  Consistent: 1 │         │
│                                     │  Correct: 1    │         │
│                                     │  Useful: 1     │         │
│                                     └────────────────┘         │
│                                     BCE loss per head           │
│                                                                  │
│   Combined Loss = 0.8 × Reward + 0.2 × Contrastive             │
└──────────────────────────────────────────────────────────────────┘
```

### Model Details

| Component | Spec |
|-----------|------|
| Encoder | `microsoft/codebert-base` (125M, shared) |
| Projection | Linear(768→1536) → Linear(1536→256) → L2-norm |
| Reward heads | 3× Linear(256→1) — Consistent, Correct, Useful |
| Dropout | 0.1 |
| Frozen layers | 0–4 (configurable) |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW with differential LR |
| LR (encoder) | 5e-5 |
| LR (heads) | 1e-3 |
| Batch size | 8 |
| Gradient accumulation | 4× |
| Epochs | 30 |
| Contrastive margin | 0.5 |
| Contrastive temperature | 0.1 |
| Max grad norm | 1.0 |
| Warmup | 10% of steps |
| Mixed precision | AMP (FP16) |
| EMA | Disabled |

### Pair Construction Strategy

| Pair type | Label | Condition |
|-----------|-------|-----------|
| Similar | 0 | Both high-quality OR both low-quality |
| Dissimilar | 1 | One high-quality, one low-quality |
| Hard negatives | 1 | Similarity > 0.5 threshold |

### Results (Epoch 30)

| Metric | Value |
|--------|-------|
| Reward Accuracy (mean) | ~0.65–0.72 |
| Reward Gap (pos − neg) | ~0.15–0.25 |
| Val Loss | ~0.5–0.7 |
| BERTScore | ~0.52–0.55 |
| CodeBLEU | ~0.38–0.42 |

---

## Stage 2B — GPT-2 Feedback-Driven Reward Model

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 2B ARCHITECTURE                        │
│                                                                  │
│   Input: (question, code_snippet)                               │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────────────────────┐                               │
│   │   GPT-2 Feedback Generator  │   Every 5 batches             │
│   │   117M params               │──► "This code is consistent   │
│   │   Temperature: 0.7          │    because..."  (1–5 score)   │
│   │   Top-p: 0.9                │                               │
│   └──────────────┬──────────────┘                               │
│                  │ feedback text + score                         │
│                  ▼                                               │
│   ┌─────────────────────────────┐                               │
│   │   CodeBERT-base Encoder     │                               │
│   │   125M params               │                               │
│   └──────────────┬──────────────┘                               │
│                  │ [CLS] embedding                               │
│                  ▼                                               │
│   ┌─────────────────────────────┐                               │
│   │   3 Specialized Reward Heads│                               │
│   │   Consistency   → score     │                               │
│   │   Agreement     → score     │                               │
│   │   Usefulness    → score     │                               │
│   └─────────────────────────────┘                               │
│                                                                  │
│   Loss = Feedback-weighted BCE + Synthetic feedback loss         │
│   Label smoothing: 0.1                                           │
│                                                                  │
│   Note: Policy (CodeGPT) is frozen throughout.                  │
│         Text metrics are invalid (self-comparison artifact).     │
└──────────────────────────────────────────────────────────────────┘
```

### Component Details

| Component | Spec |
|-----------|------|
| Feedback generator | GPT-2 (117M params) |
| Reward encoder | `microsoft/codebert-base` (125M params) |
| Reward heads | 3× specialized heads (Consistency, Agreement, Usefulness) |
| Feedback loop interval | Every 5 batches |
| Score range | 1–5 per dimension |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 8 |
| Gradient accumulation | 4× |
| Epochs | 30 |
| Feedback weight | 0.3 |
| Label smoothing | 0.1 |
| Frozen encoder layers | 4 |

> **Note on text metrics**: Stage 2B text metrics (BERTScore, ROUGE, BLEU, Ruby, CodeBLEU) are marked `invalid_self_comparison` in the export. Because the policy is frozen, it generates identical outputs every epoch, making text metrics constant and uninformative. Only `val_accuracy_*` metrics are meaningful.

---

## Directory Structure

```
stage2/
├── __init__.py
├── README.md
├── stage2A/
│   ├── config.py                     # Stage 2A configuration
│   ├── train.py                      # Training pipeline
│   ├── contrastive_model.py          # ContrastiveRewardModel
│   ├── data_loader.py                # Stage2ADataLoader
│   ├── metrics.py                    # Stage2AMetricsEvaluator
│   ├── outputs/
│   │   ├── stage2a_results.json      # Full per-epoch results
│   │   ├── stage2a_metrics.csv
│   │   └── stage2a_experiment.log
│   └── artifacts/
│       ├── checkpoint_epoch_1/
│       └── ...
└── stage2B/
    ├── config.py                     # Stage 2B configuration
    ├── train.py                      # Training pipeline
    ├── feedback_generator.py         # GPT2FeedbackGenerator
    ├── reward_model.py               # FeedbackRewardModel
    ├── outputs/
    │   ├── run_log_30epochs.txt      # Per-epoch metrics log
    │   └── reward_model/             # Saved reward model weights
    └── artifacts/
        └── ...
```

---

## How to Run

```bash
# Stage 2A
python run_stage2a.py
python run_stage2a.py --epochs 30 --margin 0.5 --batch-size 8

# Stage 2B
python run_stage2b.py
python run_stage2b.py --epochs 30 --batch-size 8 --feedback-weight 0.3
```
