# Stage 4A - MLP-Based Reward Model

## Overview

Stage 4A utilizes a **lightweight Multi-Layer Perceptron (MLP)** trained on 3,146-dimensional feature vectors to act as a reward model. This demonstrates that a compact architecture can effectively capture human or synthetic judgment variance while maintaining high inference speeds.

---

## Model / Architecture

**CodeBERT Embeddings + MLP Classifier**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Stage 4A: MLP-Based Reward Model Architecture             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: (Question, Answer) pair                                         │
│           │                                                              │
│           ▼                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              EMBEDDING LAYER (Frozen)                               │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────┐                           │ │
│  │  │      CodeBERT-base Encoder          │                           │ │
│  │  │      (microsoft/codebert-base)      │                           │ │
│  │  │      - Frozen (no training)         │                           │ │
│  │  │      - 768-dim embeddings           │                           │ │
│  │  └─────────────────────────────────────┘                           │ │
│  │           │                        │                                │ │
│  │           ▼                        ▼                                │ │
│  │    Question Embedding       Answer Embedding                       │ │
│  │    [batch, 768]             [batch, 768]                           │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              FEATURE ENGINEERING                                    │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │  Features = concat([Q_emb, A_emb, |Q-A|, Q*A])                     │ │
│  │                                                                     │ │
│  │  ┌───────────┬───────────┬───────────┬───────────┐                │ │
│  │  │  Q_emb    │  A_emb    │  |Q - A|  │  Q * A    │                │ │
│  │  │  768 dim  │  768 dim  │  768 dim  │  768 dim  │                │ │
│  │  └───────────┴───────────┴───────────┴───────────┘                │ │
│  │                                                                     │ │
│  │  Total: 768 × 4 = 3,072 dimensions                                 │ │
│  │  (+ 74 additional features = 3,146 total)                          │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              MULTI-HEAD MLP CLASSIFIER                              │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────┐                           │ │
│  │  │       Shared Encoder (Body)         │                           │ │
│  │  │       - Linear(3146 → 768)          │                           │ │
│  │  │       - LayerNorm + ReLU + Dropout  │                           │ │
│  │  │       - Linear(768 → 768)           │                           │ │
│  │  │       - LayerNorm + ReLU + Dropout  │                           │ │
│  │  └─────────────────────────────────────┘                           │ │
│  │           │                                                         │ │
│  │     ┌─────┴─────┬───────────┐                                      │ │
│  │     ▼           ▼           ▼                                      │ │
│  │  ┌──────┐  ┌──────┐  ┌──────┐                                      │ │
│  │  │Consis│  │Correc│  │Useful│                                      │ │
│  │  │ Head │  │ Head │  │ Head │                                      │ │
│  │  │768→1 │  │768→1 │  │768→1 │                                      │ │
│  │  └──────┘  └──────┘  └──────┘                                      │ │
│  │                                                                     │ │
│  │  Output: [Consistent, Correct, Useful] scores                      │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Model + Training Information

### Feature Vector Composition

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Question Embedding | 768 | CodeBERT [CLS] token |
| Answer Embedding | 768 | CodeBERT [CLS] token |
| Absolute Difference | 768 | \|Q_emb - A_emb\| |
| Element-wise Product | 768 | Q_emb * A_emb |
| Additional Features | 74 | Text statistics, code features |
| **Total** | **3,146** | Full feature vector |

### MLP Architecture

| Layer | Input → Output | Activation |
|-------|----------------|------------|
| Shared Linear 1 | 3,146 → 768 | ReLU |
| LayerNorm + Dropout | 768 | - |
| Shared Linear 2 | 768 → 768 | ReLU |
| LayerNorm + Dropout | 768 | - |
| Head: Consistent | 768 → 1 | Sigmoid |
| Head: Correct | 768 → 1 | Sigmoid |
| Head: Useful | 768 → 1 | Sigmoid |

### Parameter Count

| Component | Parameters |
|-----------|------------|
| Shared Encoder | ~2.4M |
| Linear 1: 3,146 × 768 | 2,416,128 |
| Linear 2: 768 × 768 | 589,824 |
| Classification Heads | ~2.3K |
| 3 × (768 × 1) | 2,304 |
| LayerNorm + Biases | ~3K |
| **Total MLP** | **~3M parameters** |

### Complete System

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| CodeBERT Embeddings | ~125M | Frozen |
| MLP Classifier | **~3M** | **Yes** |
| **Total** | ~128M | ~3M trainable |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 0.01 |
| Batch Size | 32 |
| Epochs | 20 |
| Dropout | 0.2 |
| Loss Function | BCE (per head) |

### Training Data

| Source | Samples |
|--------|---------|
| Evaluation datasets | ~3,000 |
| Human feedback | ~500 |
| Synthetic labels | ~2,000 |

---

## Key Advantages

### 1. Lightweight Architecture
- Only **~3M trainable parameters** (vs 125M+ in full fine-tuning)
- Fast training: minutes vs hours
- Low memory footprint

### 2. High Inference Speed
- Feature extraction: ~10ms per sample
- MLP forward pass: <1ms per sample
- Suitable for real-time reward computation

### 3. Interpretable Features
- Explicit feature engineering captures:
  - Semantic similarity (cosine via product)
  - Semantic difference (absolute difference)
  - Individual representations (concatenation)

### 4. Multi-Task Learning
- Shared encoder learns common representations
- Independent heads specialize for each criterion
- Reduces overfitting through shared learning

---

## Stage 4B Extension: Markov Chain Reward Aggregation

The MLP model can be extended with a **Markov chain** to model dependencies between criteria:

```
                    ┌──────────────┐
                    │  Consistent  │
                    └──────┬───────┘
                           │ P(C→A)
                    ┌──────▼───────┐
                    │  Agreement   │
                    └──────┬───────┘
                           │ P(A→U)
                    ┌──────▼───────┐
                    │  Usefulness  │
                    └──────────────┘

Transition Matrix:
     To:    C      A      U
From:
  C     [ 0.6    0.3    0.1 ]
  A     [ 0.2    0.5    0.3 ]
  U     [ 0.1    0.2    0.7 ]
```

The Markov chain computes a **holistic reward** by:
1. Starting with individual head predictions
2. Propagating through transition probabilities
3. Computing steady-state distribution
4. Returning weighted combination as final reward

---

## Files

| File | Description |
|------|-------------|
| `model.py` | MultiHeadClassifier MLP implementation |
| `embedding.py` | CodeBERT embedding extraction |
| `dataset.py` | Dataset loading and feature engineering |
| `train_improved.py` | Training pipeline |
| `integrated_system.py` | End-to-end system |
| `consist_mlp/` | Consistency-focused variant |
| `corct_mlp/` | Correctness-focused variant |
| `useful_mlp/` | Usefulness-focused variant |

---

## Results

| Metric | MLP Reward Model |
|--------|------------------|
| Training Accuracy | ~85% |
| Validation Accuracy | ~80% |
| Inference Speed | <1ms/sample |
| Memory Usage | ~500MB |

**Conclusion:** A lightweight MLP can effectively serve as a reward model, capturing human judgment patterns while maintaining computational efficiency.
