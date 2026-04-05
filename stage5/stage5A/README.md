# Stage 5A & 5B: Fine-tuned CodeBERT Classifier with LLM Feedback

## Overview

Stage 5 advances the reward model through two complementary approaches:

- **Stage 5A**: Fine-tunes **CodeBERT** using supervised objectives and **contrastive loss** to optimize the embedding space for code quality assessment.

- **Stage 5B**: Integrates **real-time LLM feedback** from a high-quality language model (GPT-4) acting as an expert code reviewer.

---

## Model Summary

### Stage 5A: Fine-tuned CodeBERT Classifier

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| CodeBERT-base Encoder | ~125M | ~63M (top 6 layers) |
| Shared Projection | ~295K | Yes |
| Classification Heads | ~1.2K | Yes |
| Contrastive Head (v2) | ~200K | Yes |
| **Total** | **~125.5M** | **~63M** |

### Stage 5B: LLM Feedback Loop

| Component | Description |
|-----------|-------------|
| CodeBERT Classifier | Same as 5A |
| External LLM | GPT-4 / HuggingFace / Local |
| Training Signal | Human Labels + LLM Feedback |

## Training Summary

| Parameter | Stage 5A | Stage 5B |
|-----------|----------|----------|
| Learning Rate | 2e-5 | 2e-5 |
| Batch Size | 8 | 8 |
| Epochs | 20 | 20 |
| Contrastive Loss | Yes | Yes |
| LLM Feedback | No | **Yes** |

> **[See ARCHITECTURE.md](./ARCHITECTURE.md)** for detailed architecture diagrams and version comparison.

---

## Versions

| Version | Key Feature | Location |
|---------|-------------|----------|
| v1 | Basic CodeBERT + heads | `./` |
| v2 | + Cross-attention + Contrastive | `v2/` |
| v3 | + Real-time LLM feedback | `v3/` |

---

## Running the Training

### Stage 5A (v1/v2)

```bash
# Activate conda environment
conda activate rlhfenv

# Run v2 training with contrastive loss
python clasifLLM/v2/train.py
```

### Stage 5B (v3 with LLM Feedback)

```bash
# Set up API key (for OpenAI)
export OPENAI_API_KEY="your-key-here"

# Run v3 training with LLM feedback
python clasifLLM/v3/train.py
```

---

## Key Files

| File | Description |
|------|-------------|
| `model.py` | Base LLMClassifier |
| `train.py` | Basic training pipeline |
| `v2/model.py` | Enhanced with contrastive learning |
| `v2/integrated_system.py` | Full v2 system |
| `v3/llm_feedback.py` | LLM feedback generation |
| `v3/train.py` | LLM-augmented training |

---

## Results

### Stage 5A

| Metric | v1 | v2 (Contrastive) |
|--------|-----|------------------|
| Validation Accuracy | ~78% | ~85% |

### Stage 5B

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~95%+ |
| LLM Agreement | ~90% |

**Warning:** Stage 5B shows risk of overfitting to LLM evaluation style.
