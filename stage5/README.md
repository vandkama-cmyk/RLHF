# Stage 5 — Advanced LLM-Based Classifiers

## Overview

Stage 5 implements three progressively enhanced **transformer-based classifiers** that replace the lightweight MLP heads of Stage 4 with full cross-attention architectures and — in Stage 5C — real LLM-generated feedback for supervision.

| Sub-stage | Name | Key addition over previous |
|-----------|------|---------------------------|
| **5A** | Base LLM Classifier | GraphCodeBERT encoder + attention pooling |
| **5B** | Enhanced v2 | Cross-attention Q↔A, contrastive aux loss, EMA |
| **5C** | Enhanced v3 | Real LLM feedback (GPT-4 / Claude), confidence weighting |

Each sub-stage trains on 3 seeds (42, 123, 456) with results aggregated as mean ± std.

> **Excel mapping**: Stage 5A (Excel) = Stage 5B (JSON v2). Stage 5B (Excel) = Stage 5C (JSON v3). Stage 5A JSON (v1) is omitted from analysis (4 epochs, val_loss diverged).

---

## Stage 5A — Base LLM Classifier (JSON v1)

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 5A ARCHITECTURE                        │
│                                                                  │
│   (question_text, answer_text)                                   │
│          │                                                       │
│          ▼                                                       │
│   ┌───────────────────────┐                                      │
│   │  GraphCodeBERT-base   │  (Frozen or partial)                 │
│   │  125M params          │                                      │
│   └──────────┬────────────┘                                      │
│              │ sequence output [B, L, 768]                       │
│              ▼                                                   │
│   ┌───────────────────────┐                                      │
│   │   Attention Pooling   │  → single 768-d vector               │
│   └──────────┬────────────┘                                      │
│              │                                                   │
│              ▼                                                   │
│   ┌───────────────────────────────────────┐                      │
│   │         Classification Heads          │                      │
│   │   Consistent → 1    Correct → 1       │                      │
│   │   Useful → 1                          │                      │
│   └───────────────────────────────────────┘                      │
│                                                                  │
│   Loss: BCE per head                                             │
│   Status: 4 epochs only — val_loss diverged, excluded from paper │
└──────────────────────────────────────────────────────────────────┘
```

---

## Stage 5B — Enhanced v2 with Cross-Attention (Excel "Stage 5A")

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 5B ARCHITECTURE                        │
│                                                                  │
│   question_text            answer_text                           │
│        │                        │                               │
│        ▼                        ▼                               │
│   ┌──────────┐            ┌──────────┐                          │
│   │CodeBERT  │            │CodeBERT  │  (shared encoder)        │
│   │(partial  │            │(partial  │                          │
│   │ freeze)  │            │ freeze)  │                          │
│   └────┬─────┘            └────┬─────┘                          │
│        │ [B,L,768]              │ [B,L,768]                     │
│        ▼                        ▼                               │
│   ┌──────────┐            ┌──────────┐                          │
│   │Attention │            │Attention │                          │
│   │Pooling   │            │Pooling   │                          │
│   └────┬─────┘            └────┬─────┘                          │
│        │ q_pool 768-d           │ a_pool 768-d                  │
│        └──────────┬─────────────┘                               │
│                   ▼                                             │
│   ┌──────────────────────────────────┐                          │
│   │     Cross-Attention Layer        │                          │
│   │   MultiheadAttention (8 heads)   │                          │
│   │   Q=question, K=V=answer         │                          │
│   │   LayerNorm + FFN residual       │                          │
│   └──────────────┬───────────────────┘                          │
│                  │ interaction 768-d                             │
│                  ▼                                              │
│   ┌──────────────────────────────────┐                          │
│   │  Enhanced Multi-Head Classifier  │                          │
│   │  Shared proj: 1536 → 768         │                          │
│   │  Residual from input             │                          │
│   │  Second layer:  768 → 384        │                          │
│   │  Gate per head                   │                          │
│   │  Consistent / Correct / Useful → 1 each                    │
│   └──────────────────────────────────┘                          │
│                                                                  │
│   Loss = BCE + Contrastive auxiliary loss                        │
│   EMA decay: 0.999                                               │
│   ×3 seeds → aggregate_seeds.py                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Model Details

| Component | Spec |
|-----------|------|
| Encoder | `microsoft/codebert-base` (shared for Q and A) |
| Pooling | Attention-based (not [CLS]) |
| Cross-attention | MultiheadAttention (8 heads, dim=768) |
| Shared projection | Linear(1536→768) + residual |
| Head layers | Linear(768→384) + gate + Linear(384→1) per head |
| EMA | Enabled, decay=0.999 |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Gradient accumulation | 4× |
| Epochs | 30 |
| Dropout | 0.3 |
| Mixed precision | AMP |
| EMA | Enabled (0.999) |
| Seeds | 42, 123, 456 |

### Results (Seed 42)

| Epoch | Val Acc (Cons) | Val Acc (Corr) | Val Acc (Useful) | Val Loss |
|-------|----------------|----------------|------------------|----------|
| 1 | 0.68 | 0.64 | 0.61 | 0.65 |
| 10 | 0.74 | 0.70 | 0.67 | 0.58 |
| 20 | 0.76 | 0.72 | 0.70 | 0.55 |
| 30 | 0.77 | 0.73 | 0.71 | 0.54 |

---

## Stage 5C — Enhanced v3 with Real LLM Feedback (Excel "Stage 5B")

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 5C ARCHITECTURE                        │
│                                                                  │
│   (question, code) pair                                          │
│        │                                                        │
│        ├──────────────────────────────────────────────┐         │
│        │                                              ▼         │
│        │                               ┌─────────────────────┐  │
│        │                               │  LLM Feedback API   │  │
│        │                               │  GPT-4 / Claude /   │  │
│        │                               │  HuggingFace local  │  │
│        │                               │                     │  │
│        │                               │  Structured prompt: │  │
│        │                               │  → consistency: 1-5 │  │
│        │                               │  → correctness: 1-5 │  │
│        │                               │  → usefulness: 1-5  │  │
│        │                               │  → confidence: 0-1  │  │
│        │                               │  (cached to disk)   │  │
│        │                               └──────────┬──────────┘  │
│        │                                          │ feedback     │
│        │                                          │ + confidence │
│        ▼                                          ▼              │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │      Same encoder + cross-attention + enhanced head      │  │
│   │      as Stage 5B (all v2 improvements retained)          │  │
│   └───────────────────────────┬──────────────────────────────┘  │
│                               │                                 │
│                               ▼                                 │
│   Loss = BCE (confidence-weighted) + Contrastive                │
│   Label smoothing: 0.1                                           │
│   Frozen encoder layers: 6                                       │
│   EMA: Enabled (0.999)                                           │
│   ×3 seeds → aggregate_seeds.py                                  │
└──────────────────────────────────────────────────────────────────┘
```

### LLM Integration

| Setting | Value |
|---------|-------|
| Supported providers | OpenAI (GPT-3.5/4), Anthropic (Claude), HuggingFace, Local |
| Feedback format | Structured JSON with scores + reasoning |
| Score range | 1–5 per dimension |
| Confidence range | 0–1 |
| Feedback caching | `llm_feedback_cache.json` |
| Training weighting | Confidence-weighted loss |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Gradient accumulation | 4× |
| Epochs | 30 |
| Dropout | 0.3 |
| Label smoothing | 0.1 |
| Frozen encoder layers | 6 |
| Mixed precision | AMP |
| EMA | Enabled (0.999) |
| Seeds | 42, 123, 456 |

### Results (Mean ± Std, 3 Seeds, Final Epoch)

| Metric | Value |
|--------|-------|
| Consistent Accuracy | 0.78 ± 0.012 |
| Correct Accuracy | 0.74 ± 0.015 |
| Useful Accuracy | 0.72 ± 0.018 |
| Val Loss | 0.52 ± 0.03 |
| BERTScore | 0.56 ± 0.02 |
| CodeBLEU | 0.42 ± 0.03 |

---

## Directory Structure

```
stage5/
├── aggregate_seeds.py                     # Multi-seed aggregation script
├── stage5A/                               # Base LLM classifier (v1, excluded)
│   ├── model.py
│   ├── train.py
│   ├── integrated_system.py
│   └── artifacts/
│       └── training_history_llm.json      # 4 epochs, diverged
├── stage5B/                               # Enhanced v2 (Excel "Stage 5A")
│   ├── model.py                           # Cross-attention classifier
│   ├── train.py
│   ├── compare_methods.py
│   ├── integrated_system.py
│   └── artifacts/
│       ├── training_history_llm_v2.json   # Seed 42
│       ├── multiseed_summary_training_history_llm_v2.json
│       ├── seed_123/
│       │   └── training_history_llm_v2.json
│       └── seed_456/
│           └── training_history_llm_v2.json
└── stage5C/                               # Enhanced v3 (Excel "Stage 5B")
    ├── config.py
    ├── model.py                           # Full enhanced head
    ├── train.py
    ├── llm_feedback.py                    # LLM feedback generation + caching
    ├── metrics.py
    ├── integrated_system.py
    ├── test_basic.py
    └── artifacts/
        ├── training_history_llm_v3.json   # Seed 42
        ├── multiseed_summary_training_history_llm_v3.json
        ├── seed_123/
        │   └── training_history_llm_v3.json
        └── seed_456/
            └── training_history_llm_v3.json
```

---

## How to Run

```bash
# Stage 5B (enhanced v2, all seeds)
python stage5/stage5B/train.py --seed 42
python stage5/stage5B/train.py --seed 123
python stage5/stage5B/train.py --seed 456

# Stage 5C (real LLM feedback, all seeds)
python stage5/stage5C/train.py --seed 42 --llm-provider openai
python stage5/stage5C/train.py --seed 123
python stage5/stage5C/train.py --seed 456

# Aggregate multi-seed results
python stage5/aggregate_seeds.py
```
