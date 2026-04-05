# RLHF for Code Generation — Version 1

Reinforcement Learning from Human Feedback pipeline for Python code generation on the [CoNaLa](https://conala-corpus.github.io/) dataset. The pipeline progresses through five stages, from a baseline RLHF setup to advanced LLM-supervised classifiers.

---

## Pipeline Overview

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                         RLHF PIPELINE — VERSION 1                                 │
│                                                                                   │
│   CoNaLa Dataset                                                                  │
│   (NL intent → Python snippet)                                                    │
│          │                                                                        │
│          ▼                                                                        │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │  STAGE 1 — Baseline RLHF                                                │    │
│   │  CodeGPT-small-py (124M) + CodeBERT reward (125M)                       │    │
│   │  Reward-Weighted NLL · 30 epochs · 3 seeds                              │    │
│   └───────────────────────────────┬─────────────────────────────────────────┘    │
│                                   │  baseline checkpoint                          │
│          ┌────────────────────────┴────────────────────────┐                     │
│          ▼                                                  ▼                     │
│   ┌──────────────────────┐                    ┌────────────────────────────┐     │
│   │  STAGE 2A            │                    │  STAGE 2B                  │     │
│   │  Contrastive Reward  │                    │  GPT-2 Feedback Reward     │     │
│   │  CodeBERT + Max      │                    │  CodeBERT + GPT-2 (117M)   │     │
│   │  Contrastive Loss    │                    │  Feedback-weighted BCE     │     │
│   │  30 epochs           │                    │  30 epochs                 │     │
│   └──────────────────────┘                    └────────────────────────────┘     │
│          │  improved reward signal                         │                      │
│          └────────────────────────┬────────────────────────┘                     │
│                                   ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────┐    │
│   │  STAGE 3 — SFT Baseline                                                 │    │
│   │  CodeGPT-small-py · Causal LM loss (no reward)                          │    │
│   │  Case 1: 11 samples · Case 2: 1,247 samples · 30 epochs · 3 seeds       │    │
│   └───────────────────────────────┬─────────────────────────────────────────┘    │
│                                   │  SFT baseline metrics                         │
│          ┌────────────────────────┴────────────────────────┐                     │
│          ▼                                                  ▼                     │
│   ┌──────────────────────┐                    ┌────────────────────────────┐     │
│   │  STAGE 4             │                    │  STAGE 5                   │     │
│   │  MLP Classifiers     │                    │  LLM Classifiers           │     │
│   │  Frozen CodeBERT +   │                    │  Cross-attention encoder + │     │
│   │  3,146-d features +  │                    │  LLM feedback (5C)         │     │
│   │  2-layer MLP (~3M)   │                    │  5A / 5B / 5C variants     │     │
│   │  4A/4B/4C · 3 seeds  │                    │  30 epochs · 3 seeds       │     │
│   └──────────────────────┘                    └────────────────────────────┘     │
│          │                                              │                         │
│          └────────────────────────┬────────────────────┘                         │
│                                   ▼                                               │
│   export_all_stages_results_by_epoch.py  →  general_all_stages_results_by_epoch.json │
│   eval_unified_all_stages.py             →  eval_unified_results.json            │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage Summary

| Stage | Purpose | Models | Training | Seeds |
|-------|---------|--------|----------|-------|
| [Stage 1](stage1/README.md) | Baseline RLHF (reward-weighted NLL) | CodeGPT-small-py + CodeBERT | 30 epochs | 3 |
| [Stage 2A](stage2/README.md) | Contrastive reward model | CodeBERT + contrastive head | 30 epochs | — |
| [Stage 2B](stage2/README.md) | GPT-2 feedback reward model | CodeBERT + GPT-2 | 30 epochs | — |
| [Stage 3](stage3/README.md) | SFT baseline (no reward) | CodeGPT-small-py | 30 epochs | 3 |
| [Stage 4](stage4/README.md) | Lightweight MLP classifiers (3× heads) | Frozen CodeBERT + MLP (~3M) | 20–30 epochs | 3 |
| [Stage 5](stage5/README.md) | Advanced LLM classifiers (3× variants) | CodeBERT + cross-attention + LLM | 30 epochs | 3 |

---

## Dataset

**CoNaLa** — Mining StackOverflow for natural language / Python code pairs.

| Split | Samples |
|-------|---------|
| Train | 1,247 |
| Test | 500 |

Files: `conala-corpus/conala-train.jsonl`, `conala-corpus/conala-test.jsonl`

---

## Models Used

| Model | Role | Parameters |
|-------|------|-----------|
| `microsoft/CodeGPT-small-py` | Policy (code generator) | 124M |
| `microsoft/codebert-base` | Reward encoder / classifier encoder | 125M |
| `microsoft/graphcodebert-base` | Stage 5A/B/C encoder variant | 125M |
| `gpt2` | Stage 2B feedback generator | 117M |

---

## Metrics

All generative stages are evaluated with:

| Metric | Description |
|--------|-------------|
| **BERTScore** | Embedding-level F1 (roberta-large) |
| **ROUGE-L** | Longest common subsequence overlap |
| **BLEU** | N-gram precision |
| **Ruby** | PDG-based structural code similarity |
| **CodeBLEU** | Code-aware BLEU with syntax/dataflow |

Classification stages (4, 5) additionally report accuracy, F1, precision, and recall per head (Consistent / Correct / Useful).

---

## Key Results

### Generative Stages (CoNaLa test set)

| Stage | BERTScore | CodeBLEU | BLEU | Ruby |
|-------|-----------|----------|------|------|
| Stage 1 (baseline) | ~0.82 | ~0.05 | ~0.01 | ~0.15 |
| Stage 3 Case 2 (SFT, 1247 samples) | ~0.85 | ~0.18 | ~0.03 | ~0.42 |

### Classification Stages (Val Accuracy, Mean ± Std, 3 Seeds)

| Stage | Consistent | Correct | Useful |
|-------|-----------|---------|--------|
| Stage 4 MLP | 0.683 ± 0.015 | 0.671 ± 0.018 | 0.658 ± 0.012 |
| Stage 5B v2 | ~0.77 | ~0.73 | ~0.71 |
| Stage 5C v3 (LLM feedback) | 0.78 ± 0.012 | 0.74 ± 0.015 | 0.72 ± 0.018 |

---

## Project Structure

```
Version_1/
├── README.md                                 # This file
│
├── conala-corpus/
│   ├── conala-train.jsonl                    # Training data (1247 samples)
│   └── conala-test.jsonl                     # Test data (500 samples)
│
├── stage1/                                   # Baseline RLHF
│   └── README.md
├── stage2/                                   # Enhanced reward models
│   ├── stage2A/                              # Contrastive reward
│   ├── stage2B/                              # GPT-2 feedback reward
│   └── README.md
├── stage3/                                   # SFT baseline
│   └── README.md
├── stage4/                                   # MLP classifiers
│   ├── stage4A_consist/
│   ├── stage4B_corct/
│   ├── stage4C_useful/
│   └── README.md
├── stage5/                                   # LLM classifiers
│   ├── stage5A/                              # Base (v1, excluded)
│   ├── stage5B/                              # Enhanced v2 (Excel "Stage 5A")
│   ├── stage5C/                              # Real LLM feedback (Excel "Stage 5B")
│   └── README.md
│
├── modern_rlhf/                              # Shared training utilities
│   ├── config.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── metrics_tracker.py
│   ├── pipeline.py
│   ├── reward_model.py
│   └── trainer.py
│
├── run_stage2a.py                            # Stage 2A entry point
├── run_stage2b.py                            # Stage 2B entry point
├── export_all_stages_results_by_epoch.py     # Combine all results → JSON
├── eval_unified_all_stages.py                # Unified evaluation on test set
├── fill_excel.py                             # Excel report generation
└── general_all_stages_results_by_epoch.json  # Combined per-epoch results
```

---

## Reproducing Results

### Step-by-step

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Stage 1 — baseline RLHF
python stage1/run_stage1.py

# 3. Stage 2A — contrastive reward
python run_stage2a.py

# 4. Stage 2B — GPT-2 feedback reward
python run_stage2b.py

# 5. Stage 3 — SFT baseline
python stage3/run_sft_experiments.py

# 6. Stage 4 — MLP classifiers (run for each classifier + seed)
python stage4/train_fixed.py --classifier consistent --seed 42
python stage4/train_fixed.py --classifier correct    --seed 42
python stage4/train_fixed.py --classifier useful     --seed 42
python stage4/aggregate_seeds.py

# 7. Stage 5B — enhanced LLM classifier
python stage5/stage5B/train.py --seed 42

# 8. Stage 5C — real LLM feedback classifier
python stage5/stage5C/train.py --seed 42 --llm-provider openai

# 9. Aggregate multi-seed results
python stage5/aggregate_seeds.py

# 10. Export all results to a single JSON
python export_all_stages_results_by_epoch.py

# 11. Run unified evaluation on the test set
python eval_unified_all_stages.py
```

### Hardware Requirements

| Variant | VRAM | Notes |
|---------|------|-------|
| Full pipeline | 8+ GB GPU | Recommended |
| Stage 1 light | < 4 GB GPU | Use `run_stage1_light.py` |
| Stage 4 | CPU viable | Frozen encoder, small MLP only |

---

## Notes

- **Stage 2B text metrics** are marked `invalid_self_comparison` — the frozen policy produces identical outputs every epoch, so BERTScore/BLEU/etc. are constant and excluded from analysis.
- **Stage 5A (JSON v1)** diverged after 4 epochs and is excluded from the paper. The Excel "Stage 5A" corresponds to JSON Stage 5B (v2).
- **Multi-seed runs** use seeds 42, 123, 456. Results are aggregated as mean ± std via `aggregate_seeds.py` in stage4/ and stage5/.
- **Authoritative stage1 results**: `stage1/output/stage1_results.json` (31 epochs). `stage1/output_upd/` is an older snapshot kept for reference.
