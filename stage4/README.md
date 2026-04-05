# Stage 4 — Multi-Head MLP Classifiers

## Overview

Stage 4 trains three independent **lightweight MLP classifiers** on top of frozen CodeBERT embeddings. Each classifier predicts one aspect of code quality from pairwise human-feedback data. The CodeBERT encoder is frozen; only the MLP heads (~3M parameters) are trained. This makes Stage 4 fast and low-resource compared to full fine-tuning.

| Sub-stage | Classifier | Predicts |
|-----------|-----------|---------|
| **4A** | Consistent | Is the code grammatically/stylistically consistent? |
| **4B** | Correct | Is the code semantically correct for the intent? |
| **4C** | Useful | Is the code practically useful for the intent? |

Each classifier is trained with **3 random seeds** (42, 123, 456) and results are aggregated as mean ± std.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                     STAGE 4 ARCHITECTURE                          │
│                                                                   │
│   Input: (question_text, answer_text) pair                        │
│          │                                                        │
│          ▼                                                        │
│   ┌────────────────────────────────────────────────────┐         │
│   │          CodeBERT-base Encoder (FROZEN)             │         │
│   │          125M params — not updated                  │         │
│   └───┬────────────────────────────────────────┬────────┘         │
│       │ question [CLS] 768-d                    │ answer [CLS] 768-d│
│       ▼                                         ▼                 │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │                 Feature Vector Assembly                    │   │
│   │   q_emb (768) ┐                                           │   │
│   │   a_emb (768) ┤                                           │   │
│   │   |q-a| (768) ┤ → concat → 3,146-d feature vector        │   │
│   │   q⊙a  (768) ┤                                           │   │
│   │   extra (74)  ┘                                           │   │
│   └───────────────────────┬───────────────────────────────────┘   │
│                           │                                       │
│                           ▼                                       │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │              MLP Classifier (~3M params, trainable)        │   │
│   │                                                            │   │
│   │   Linear(3146 → 768) → LayerNorm → ReLU → Dropout(0.2)   │   │
│   │   Linear(768 → 768)  → LayerNorm → ReLU → Dropout(0.2)   │   │
│   │   Linear(768 → 1)    → sigmoid → binary prediction        │   │
│   └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│   Loss: Binary Cross-Entropy per head                             │
│   Labels: from pairwise human feedback (−2/−1/+1/+2 → 0/1)       │
│                                                                   │
│   ×3 seeds → mean ± std aggregation (aggregate_seeds.py)         │
└───────────────────────────────────────────────────────────────────┘
```

### Feature Vector Composition (3,146-d)

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| `q_emb` | 768 | Question [CLS] embedding from CodeBERT |
| `a_emb` | 768 | Answer [CLS] embedding from CodeBERT |
| `\|q − a\|` | 768 | Absolute element-wise difference |
| `q ⊙ a` | 768 | Element-wise product |
| Extra features | 74 | Lexical overlap, length ratios, etc. |
| **Total** | **3,146** | |

### MLP Head

| Layer | In → Out | Activation |
|-------|----------|-----------|
| FC 1 | 3,146 → 768 | LayerNorm → ReLU → Dropout(0.2) |
| FC 2 | 768 → 768 | LayerNorm → ReLU → Dropout(0.2) |
| Output | 768 → 1 | Sigmoid |

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-3 |
| Batch size | 16 |
| Epochs | 20–30 |
| Dropout | 0.2–0.4 |
| Early stopping patience | 5 epochs |
| Mixed precision | AMP (FP16) |
| Seeds | 42, 123, 456 |

---

## Metrics Tracked Per Epoch

| Metric | Description |
|--------|-------------|
| Train loss | BCE |
| Val loss | BCE on held-out split |
| Val accuracy | Binary classification accuracy |
| Precision / Recall / F1 | Per-class metrics |
| Balanced accuracy | Average per-class accuracy |
| BERTScore | Embedding F1 (for generated outputs) |
| CodeBLEU / BLEU / ROUGE / Ruby | Text generation quality |

---

## Results

### Per-Classifier (Seed 42, Best Epoch)

| Classifier | Val Accuracy | Val F1 | Val BERTScore |
|-----------|-------------|--------|---------------|
| 4A Consistent | 0.683 | 0.567 | 0.547 |
| 4B Correct | 0.671 | 0.551 | 0.538 |
| 4C Useful | 0.658 | 0.534 | 0.530 |

### Multi-Seed Summary (Mean ± Std, Final Epoch)

| Classifier | Val Accuracy |
|-----------|-------------|
| 4A Consistent | 0.683 ± 0.015 |
| 4B Correct | 0.671 ± 0.018 |
| 4C Useful | 0.658 ± 0.012 |

### Stage 4A Training Progression (Seed 42)

| Epoch | Train Loss | Val Loss | Val Acc | Val F1 |
|-------|-----------|----------|---------|--------|
| 1 | 0.945 | 0.881 | 0.634 | 0.494 |
| 5 | 0.884 | 0.864 | 0.646 | 0.527 |
| 10 | 0.882 | 0.860 | 0.659 | 0.545 |
| 20 | 0.858 | 0.857 | 0.683 | 0.567 |

---

## Directory Structure

```
stage4/
├── model.py                              # MultiHeadClassifier (MLP)
├── embedding.py                          # CodeBERT embedding extractor
├── dataset.py                            # Data loading + feature assembly
├── train_fixed.py                        # Main training script
├── integrated_system_fixed.py            # End-to-end inference pipeline
├── analyze_training.py                   # Training curve analysis
├── aggregate_seeds.py                    # Multi-seed result aggregation
├── improved_artifacts/
│   └── plot_training_history.py
├── stage4A_consist/
│   ├── train_consistent.py
│   ├── integrated_system_consistent.py
│   └── artifacts/
│       ├── training_history_consistent.json      # Seed 42 results
│       ├── multiseed_summary_training_history_consistent.json
│       ├── seed_123/
│       │   └── training_history_consistent.json
│       └── seed_456/
│           └── training_history_consistent.json
├── stage4B_corct/
│   ├── train_correct.py
│   ├── integrated_system_correct.py
│   └── artifacts/
│       ├── training_history_correct.json
│       ├── multiseed_summary_training_history_correct.json
│       └── seed_*/
└── stage4C_useful/
    ├── train_useful.py
    ├── integrated_system_useful.py
    └── artifacts/
        ├── training_history_useful.json
        ├── multiseed_summary_training_history_useful.json
        └── seed_*/
```

---

## How to Run

```bash
# Train all three classifiers (all seeds)
python stage4/train_fixed.py --classifier consistent --seed 42
python stage4/train_fixed.py --classifier correct    --seed 42
python stage4/train_fixed.py --classifier useful     --seed 42

# Aggregate multi-seed results
python stage4/aggregate_seeds.py
```
