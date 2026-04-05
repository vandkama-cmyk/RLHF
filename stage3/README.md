# Stage 3 — Supervised Fine-Tuning (SFT)

## Overview

Stage 3 performs **Supervised Fine-Tuning** of CodeGPT-small-py on the CoNaLa dataset using standard language modelling loss — no reinforcement learning, no reward model. It serves as a direct SFT baseline to isolate the contribution of the reward signal used in Stages 1, 2, 4, and 5.

Two data-scale experiments are run to measure the effect of dataset size:

| Case | Samples | Purpose |
|------|---------|---------|
| **Case 1** | 11 samples | Minimal/overfitting baseline |
| **Case 2** | 1,247 samples | Full CoNaLa training set |

Both cases run for 30 epochs with 3 random seeds (42, 123, 456) and results are aggregated.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 3 PIPELINE                             │
│                                                                  │
│   CoNaLa Dataset                                                 │
│   Case 1: 11 samples ──────────────────────┐                    │
│   Case 2: 1247 samples ────────────────────┤                    │
│                                            ▼                    │
│   ┌──────────────────────────────────────────────────┐          │
│   │           CodeGPT-small-py (124M params)          │          │
│   │           12 layers / 768 hidden / 12 heads        │          │
│   │                                                    │          │
│   │   Input:  [NL prompt] → tokenize → ids            │          │
│   │   Output: next-token logits → code sequence        │          │
│   │                                                    │          │
│   │   Loss: Causal Language Modelling                  │          │
│   │         L = -Σ log P(y_t | y_{<t}, x)             │          │
│   └───────────────────────┬──────────────────────────┘          │
│                            │                                     │
│                            ▼                                     │
│   Per-epoch metrics: BERTScore, CodeBLEU, BLEU, ROUGE, Ruby     │
│   Per-epoch checkpoint saved to outputs/                         │
│                                                                  │
│   ×3 seeds (42, 123, 456) → mean ± std aggregation              │
└──────────────────────────────────────────────────────────────────┘
```

### Model — CodeGPT-small-py

| Parameter | Value |
|-----------|-------|
| Base model | `microsoft/CodeGPT-small-py` |
| Parameters | 124M |
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Max sequence length | 128 tokens |

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | Causal language modelling (SFT) |
| Optimizer | AdamW |
| Learning rate | 5e-6 |
| Batch size | 2 |
| Max grad norm | 0.5 (aggressive — for stability) |
| Epochs | 30 |
| Warmup | 10% of total steps |
| Mixed precision | **FP32** (FP16 disabled for stability) |
| Seeds | 42, 123, 456 |

> Conservative LR and FP32 were chosen after observing NaN gradients with larger LR and FP16.

---

## Metrics Tracked Per Epoch

| Metric | Description |
|--------|-------------|
| Train loss | Causal LM cross-entropy |
| Val loss | Held-out split cross-entropy |
| BERTScore | Embedding F1 (roberta-large) |
| CodeBLEU | Code-aware BLEU |
| BLEU | N-gram overlap |
| ROUGE-L | Longest common subsequence |
| Ruby | PDG-based code similarity |
| Policy entropy | Mean per-token output entropy |
| Approx. KL | log(V) − H(π) from uniform |
| Balanced accuracy | Reward-based binary classification |

---

## Results

### Case 2 — 1,247 Samples (Seed 42)

| Epoch | Train Loss | BERTScore | CodeBLEU | BLEU | Ruby |
|-------|-----------|-----------|----------|------|------|
| 1 | 6.06 | 0.775 | 0.019 | 0.0003 | 0.212 |
| 5 | 1.09 | 0.840 | 0.101 | 0.018 | 0.407 |
| 10 | 0.91 | 0.841 | 0.175 | 0.025 | 0.402 |
| 30 | ~0.60 | ~0.850 | ~0.180 | ~0.030 | ~0.420 |

### Case 1 — 11 Samples (Overfitting Baseline)

Training loss collapses rapidly (overfitting expected). Used only to confirm the training loop is functional and to provide a lower-bound comparison point.

---

## Directory Structure

```
stage3/
├── run_sft_experiments.py            # Main SFT runner (both cases)
├── resume_from_epoch7.py             # Resume training from checkpoint
├── outputs/
│   ├── case1_11samples_history.json        # Case 1 per-epoch results
│   ├── case2_1247samples_history.json      # Case 2 per-epoch results (seed 42)
│   ├── case2_1247samples_seed42_history.json
│   ├── case2_1247samples_seed123_history.json
│   └── case2_1247samples_seed456_history.json
├── case1/
│   └── case1_11samples_history.json
└── case2/
    └── case2_1247samples_history.json
```

---

## How to Run

```bash
# Run both cases (default: 30 epochs, 3 seeds)
python stage3/run_sft_experiments.py

# Custom run
python stage3/run_sft_experiments.py --epochs 30 --case 2 --seed 42

# Resume from checkpoint
python stage3/resume_from_epoch7.py
```
