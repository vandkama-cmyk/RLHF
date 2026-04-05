# Stage 5A & 5B - Fine-tuned CodeBERT Classifier with LLM Feedback

## Overview

Stage 5 advances the reward model through two complementary approaches:

- **Stage 5A**: Fine-tunes CodeBERT using both supervised objectives and **contrastive loss** to optimize the embedding space for code quality assessment.

- **Stage 5B**: Integrates **real-time LLM feedback** from a high-quality language model acting as an expert code reviewer.

---

## Stage 5A: Fine-tuned CodeBERT Classifier

### Model / Architecture

**CodeBERT + Contrastive Learning + Multi-Head Classification**

```
┌─────────────────────────────────────────────────────────────────────────┐
│           Stage 5A: Fine-tuned CodeBERT Classifier Architecture         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: (Question, Answer) pair                                         │
│           │                                                              │
│           ▼                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              CodeBERT-base ENCODER (Fine-tuned)                     │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────┐                           │ │
│  │  │      CodeBERT-base                  │                           │ │
│  │  │      (microsoft/codebert-base)      │                           │ │
│  │  │      - 12 Transformer layers        │                           │ │
│  │  │      - 768 hidden size              │                           │ │
│  │  │      - Partially frozen (6 layers)  │                           │ │
│  │  │      - ~125M parameters             │                           │ │
│  │  └─────────────────────────────────────┘                           │ │
│  │           │                                                         │ │
│  │           ▼                                                         │ │
│  │  Mean Pooling / [CLS] Token                                        │ │
│  │  [batch, 768]                                                      │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                    ┌─────────┴─────────┐                                │
│                    │                   │                                 │
│                    ▼                   ▼                                 │
│  ┌─────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │   CLASSIFICATION HEAD   │  │   CONTRASTIVE HEAD (v2+)             │ │
│  ├─────────────────────────┤  ├──────────────────────────────────────┤ │
│  │                         │  │                                       │ │
│  │  Linear(768 → 384)      │  │  Cross-Attention Layer               │ │
│  │  LayerNorm + GELU       │  │  - Q/A interaction                   │ │
│  │  Dropout(0.3)           │  │                                       │ │
│  │                         │  │  Projection Head                     │ │
│  │  ┌─────┬─────┬─────┐   │  │  - Linear(768 → 256)                 │ │
│  │  │ C   │ Co  │ U   │   │  │  - L2 Normalization                  │ │
│  │  │384→1│384→1│384→1│   │  │                                       │ │
│  │  └─────┴─────┴─────┘   │  │  Output: 256-dim embedding           │ │
│  │                         │  │  for contrastive loss                │ │
│  │  Output: 3 scores       │  │                                       │ │
│  │  (Consistent, Correct,  │  └──────────────────────────────────────┘ │
│  │   Useful)               │                                           │
│  │                         │                                           │
│  └─────────────────────────┘                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Model + Training Information (Stage 5A)

#### Parameter Count

| Component | Parameters |
|-----------|------------|
| CodeBERT-base Encoder | ~125M |
| Shared Projection | ~295K |
| Classification Heads | ~1.2K |
| Contrastive Head (v2) | ~200K |
| **Total** | **~125.5M parameters** |
| **Trainable** | ~63M (6 layers + heads) |

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Batch Size | 8 |
| Epochs | 20 |
| Frozen Encoder Layers | 6 (bottom) |
| Dropout | 0.3 |
| Label Smoothing | 0.1 |

#### Loss Functions

**Combined Loss:**
```
L_total = α × L_classification + β × L_contrastive
```

Where:
- `L_classification` = BCE loss for each head
- `L_contrastive` = InfoNCE loss for embedding alignment
- Default: α = 0.7, β = 0.3

---

## Stage 5B: Real-time LLM Feedback Loop

### Architecture

**CodeBERT Classifier + External LLM Evaluator**

```
┌─────────────────────────────────────────────────────────────────────────┐
│           Stage 5B: Real-time LLM Feedback Loop Architecture            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              EXTERNAL LLM EVALUATOR                                 │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │  Supported Providers:                                              │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │ │
│  │  │  OpenAI     │  │ HuggingFace │  │   Local     │                │ │
│  │  │  GPT-4      │  │ Inference   │  │Transformers │                │ │
│  │  │  ~1.7T*     │  │ API         │  │ DialoGPT    │                │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │ │
│  │           │              │               │                         │ │
│  │           └──────────────┴───────────────┘                         │ │
│  │                          │                                          │ │
│  │                          ▼                                          │ │
│  │  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │                 Feedback Generation                          │   │ │
│  │  │  Input: (Question, Code Answer)                              │   │ │
│  │  │  Output: {consistent, correct, useful, explanation,          │   │ │
│  │  │           confidence} scores                                  │   │ │
│  │  └─────────────────────────────────────────────────────────────┘   │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              │ LLM Feedback Scores                      │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              FEEDBACK-DRIVEN TRAINING                               │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │  Training Signal = α × Human_Labels + (1-α) × LLM_Feedback        │ │
│  │                                                                     │ │
│  │  Where α is determined by LLM confidence:                          │ │
│  │  - High confidence (>0.8): Use mostly LLM feedback                 │ │
│  │  - Low confidence (<0.4): Prefer human labels                      │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              CodeBERT CLASSIFIER (Training)                        │ │
│  │              Same architecture as Stage 5A                         │ │
│  │              ~125.5M parameters                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Model + Training Information (Stage 5B)

#### LLM Providers

| Provider | Model | Parameters* | Latency |
|----------|-------|-------------|---------|
| OpenAI | GPT-4 | ~1.7T | ~2-3s |
| OpenAI | GPT-3.5 | ~175B | ~1s |
| HuggingFace | Various | 117M-13B | 1-5s |
| Local | DialoGPT | 117M-762M | <1s |

*Estimated, not officially confirmed

#### Feedback Schema

```json
{
    "consistent": float,    // -2 to +2 scale
    "correct": float,       // -2 to +2 scale
    "useful": float,        // -2 to +2 scale
    "explanation": string,  // Text justification
    "confidence": float     // 0 to 1 scale
}
```

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| LLM Provider | Configurable |
| Temperature | 0.7 |
| Max Tokens | 1000 |
| Batch Processing | 5 samples parallel |
| Cache | Enabled |
| Fallback | Heuristic evaluation |

### Combined System Parameters

| Component | Parameters |
|-----------|------------|
| CodeBERT Classifier | ~125.5M |
| External LLM (GPT-4) | ~1.7T* |
| **Local Trainable** | **~63M** |

---

## Version Comparison

### ClassifLLM Versions

| Version | Key Feature | Complexity |
|---------|-------------|------------|
| v1 | Basic CodeBERT + heads | Simple |
| v2 | + Cross-attention + Contrastive | Medium |
| v3 | + Real-time LLM feedback | Advanced |

### Architectural Differences

| Aspect | v1 | v2 | v3 |
|--------|-----|-----|-----|
| Encoder | CodeBERT | CodeBERT | CodeBERT |
| Pooling | Mean/CLS | Attention | Attention |
| Q-A Interaction | Concat | Cross-Attention | Cross-Attention |
| Contrastive Loss | No | Yes | Yes |
| External LLM | No | No | **Yes** |
| Training Signal | Human only | Human only | Human + LLM |

---

## Key Findings

### Stage 5A Results

| Metric | v1 | v2 (Contrastive) |
|--------|-----|------------------|
| Validation Accuracy | ~78% | ~85% |
| Embedding Quality | Moderate | High |
| Discrimination | Good | Excellent |

### Stage 5B Results

| Metric | Value | Note |
|--------|-------|------|
| Validation Accuracy | ~95%+ | Near-perfect |
| LLM Agreement | ~90% | High correlation |

**Warning:** Risk of **overfitting to LLM style**
- Model may learn to mimic LLM's specific evaluation patterns
- May not generalize to different human evaluators
- Requires careful validation on held-out human data

---

## Files

| File | Description |
|------|-------------|
| `model.py` | Base LLMClassifier implementation |
| `train.py` | Training pipeline |
| `v2/model.py` | Enhanced with contrastive learning |
| `v2/integrated_system.py` | Full v2 training system |
| `v3/llm_feedback.py` | LLM feedback generation |
| `v3/train.py` | LLM-augmented training |
| `v3/config.py` | Configuration for v3 |

---

## Conclusion

**Stage 5A** demonstrates that contrastive learning significantly improves the embedding space for code quality assessment.

**Stage 5B** achieves near-perfect validation accuracy using LLM feedback, but highlights the potential risk of overfitting to synthetic evaluator patterns.
