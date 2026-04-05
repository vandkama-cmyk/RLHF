# Version 2 — Rater Bias Detection & Correction in RLHF

## Problem Statement

Standard RLHF assumes human annotators are reliable. In practice, annotators differ in expertise, strictness, and systematic biases. A critical failure mode: **if raters prefer confident-sounding answers over correct ones**, the reward model learns to hallucinate confidently rather than to be accurate.

Version 2 addresses this with three goals:
1. **Detect rater bias** from behavioral signals (leniency, positional, confidence–correctness decoupling)
2. **Estimate rater expertise** via proxy metrics (metric alignment, internal consistency, agreement with peers)
3. **Train a bias-corrected reward model** that down-weights unreliable raters via per-sample loss weighting

---

## Dataset

| Property | Value |
|---|---|
| Total evaluations | 307 |
| Raters | 11 (2 anonymous, 9 named) |
| Rating dimensions | 3: `consistent`, `correct`, `useful` (scale: −2 to +2) |
| Comparison slider | −2 (strongly prefer Left) to +2 (strongly prefer Right) |
| Shared pairs (≥2 raters) | 38 (12.4% overlap) |
| Training samples | 614 (two answer sides per evaluation) |
| Source domain | CoNaLa code generation, Stack Overflow text-to-code |
| Annotation period | August 2022 |
| Models evaluated | CodeGen, GPT-Neo, Copilot, BART, T5 (fine-tuned + vanilla) |

**Class balance in `correct` dimension:** 20% positive (80% negative) — severe imbalance, as raters tended to rate generated code as incorrect.

---

## Pipeline

The pipeline runs sequentially via `run_all.py`. Each stage depends on the previous stage's outputs.

```
run_all.py
  ├── stage1_eda/parse_and_explore.py
  ├── stage2_bias_detection/detect_bias.py
  ├── stage3_expertise/estimate_expertise.py
  ├── stage4_reward_model/train_weighted.py
  └── stage5_report/generate_report.py
```

### Stage 1 — Data Parsing & EDA

Parses 307 JSON evaluation files and 25 CSV datasets. Computes 13 per-rater statistics including mean rating, standard deviation, skewness, and per-dimension means.

**Key output:** `stage1_eda/results/rater_stats.csv`

Notable rater statistics:
| Rater | N evals | Mean rating |
|---|---|---|
| Anonymous | 151 | −0.150 |
| Artem | 39 | −1.611 |
| Anonymous01 | 29 | +0.167 |
| Fedor | 4 | −1.375 |

---

### Stage 2 — Rater Bias Detection

Five independent detection mechanisms:

| Mechanism | Method | Threshold |
|---|---|---|
| **Leniency bias** | Z-score of mean rating across all raters | \|z\| > 1.5 |
| **Positional bias** | Chi-square test on comparison slider distribution | p < 0.05 |
| **Confidence–correctness decoupling** | Spearman r between ratings and auto-quality proxy | \|r\| < 0.15 AND p < 0.10 |
| **Dimension inconsistency** | Logical coherence (e.g., correct=−2 but useful=+2) | mean_incon > 0.1 |
| **Inter-rater agreement** | Cohen's Kappa + Krippendorff's Alpha on shared pairs | reliability check only |

**Auto-quality proxy** (used in decoupling test): type-token ratio + length score of the answer text — a weak but automated signal for answer quality.

**Detected biases:**

| Rater | Bias type | Signal |
|---|---|---|
| Artem | STRICT (deflated ratings) | z = −2.007 |
| Artem | Positional RIGHT | χ² = 26.31, p < 0.001 |
| Fedor | STRICT (deflated ratings) | z = −1.585 |
| Vadim | Positional NEUTRAL EXTREME | χ² = 11.64, p = 0.003 |

No raters were flagged for confidence–correctness decoupling or dimension inconsistency.

**Inter-rater agreement:** Krippendorff's alpha was nullified — 12.4% overlap is below the 15% threshold for reliable agreement estimation.

---

### Stage 3 — Expertise Proxy Estimation

Three proxy signals combined into a composite reliability score:

| Signal | Weight | Description |
|---|---|---|
| **Metric alignment** (α) | 0.50 | Spearman r of rater's `correct` ratings vs auto-quality proxy |
| **Internal consistency** (β) | 0.25 | Logical coherence + cross-dimension correlation |
| **Agreement with majority** (γ) | 0.25 | Cosine similarity vs leave-one-out peer mean (shared pairs only) |

**Uncertainty shrinkage:** raters with fewer than 20 evaluations are shrunk toward the mean: `score_shrunk = score × n / (n + 20)`.

**Softmax temperature scaling** (T = 0.05) amplifies differences between raters ~20×.

**Final rater weights (softmax output):**

| Rater | Weight | Notes |
|---|---|---|
| Anonymous | 0.9398 | Dominant rater, high alignment |
| Artem | 0.0245 | STRICT + POSITIONAL bias |
| Anonymous01 | 0.0196 | |
| Sergey Kovalchuk | 0.0090 | |
| TypingCat | 0.0032 | |
| XXZ | 0.0019 | |
| Vadim | 0.0010 | POSITIONAL NEUTRAL bias |
| Anastasia | 0.0006 | |
| Arthur | 0.0001 | Only 4 evals |
| Gorbatovski | 0.0001 | Only 4 evals |
| Fedor | ~0.0000 | STRICT bias + lowest alignment |

> **Important distinction:** Reliability weight ≠ calibration. A rater can be internally consistent (high consistency score) while still being miscalibrated (low alignment score — preferring confident-sounding wrong answers). Artem has consistency=0.976 but alignment=0.428, meaning he is logically coherent but systematically deflates and favors rightward answers. He is penalized via lower alignment weight.

---

### Stage 4 — Bias-Corrected Reward Model Training

**Architecture:**
- Encoder: CodeBERT (`microsoft/codebert-base`), **frozen**
- Feature vector: [Q_emb; A_emb; |Q−A|; Q⊙A] → 3072-dim
- Shared projection: 3072 → 512 (2 layers, LayerNorm, ReLU, Dropout=0.3)
- 3 task heads: `consistent`, `correct`, `useful` (binary classification each)

**Training:**
- Optimizer: AdamW (lr=2e-4, weight_decay=0.01)
- Scheduler: CosineAnnealingLR (T_max=20)
- Loss: Focal BCE (γ=2.0) with class-rebalancing `pos_weight`
- Batch size: 16, max 20 epochs, early stopping (patience=5)
- Train/val split: 80/20 stratified on `correct_label` → 491 train, 123 val

**Key difference — weighted vs baseline:**

| Mode | Per-sample weight |
|---|---|
| **Weighted** | rater's softmax weight (capped at 0.15 to prevent Anonymous from dominating) |
| **Baseline** | uniform 1/11 (all raters treated equally) |

**Results:**

| Metric | Weighted | Baseline | Delta |
|---|---|---|---|
| Overall accuracy (best epoch) | **55.28%** | 48.78% | **+6.50%** |
| Positive class accuracy | 90.16% | 22.95% | +67.21% |
| Negative class accuracy | 9.68% | 83.87% | −74.19% |
| Best epoch | 1 | 2 | — |
| Val loss (best) | 0.628 | 0.608 | — |

**Statistical test:** McNemar's test on 123 validation samples — χ² = 0.184, **p = 0.668** (not significant).

The weighted model favors the positive class while the baseline favors the negative class. This reversal is partly a product of Anonymous's high weight (94%) combined with Anonymous's relatively more positive ratings — so the weighted model may be learning Anonymous's preferences rather than ground truth correctness.

---

### Stage 5 — Report Synthesis

Generates `results_summary.json`, `master_rater_summary.csv`, and visualization plots. All intermediate results are merged into a single master table with bias signals, reliability scores, and weights per rater.

---

## Results Summary

### What Was Achieved

**Bias detection (fully functional):**
- 5 detection mechanisms implemented and tested
- Identified 3 biased raters with 4 bias instances
- Biased raters are correctly down-weighted in training (Fedor ≈ 0, Artem = 0.025)
- Multi-signal expertise proxy is principled and interpretable

**Training pipeline:**
- Per-sample weighted loss is correctly implemented
- Focal loss + class rebalancing handles severe class imbalance
- Bias-corrected model shows directional improvement (+6.5%)

### What Was Not Achieved

**Statistical significance:**
- McNemar's test p = 0.668 — the 6.5% improvement is within noise at n=614
- Early stopping at epoch 1–2 indicates overfitting on 123-sample validation set

**Inter-rater agreement:**
- 12.4% overlap < 15% threshold — Krippendorff's alpha is statistically unreliable
- Dawid-Skene EM could not run (requires ≥5 shared items per rater pair)

**Confidence–correctness decoupling:**
- No raters were flagged — the auto-quality proxy (type-token ratio + length) is too weak to detect this bias reliably

**Generalizability:**
- 11 raters, 307 evaluations is too small for robust conclusions
- Weight distribution is highly concentrated (Anonymous = 94%) — effectively single-rater training

---

## Limitations

| Limitation | Impact |
|---|---|
| 12.4% inter-rater overlap | Agreement metrics unreliable; Dawid-Skene disabled |
| 307 evaluations total | Statistical power insufficient for significance tests |
| Anonymous = 49% of all evals | Weight cap (0.15) partially mitigates but distorts training distribution |
| Frozen CodeBERT | No fine-tuning on domain — embeddings not optimized for code quality |
| Auto-quality proxy is weak | Cannot reliably detect confidence–correctness decoupling |
| Early stopping at epoch 1–2 | Model does not converge; per-class trade-off is unstable |

---

## Research Improvement Plan

The core methodology is sound. The limitations are primarily data and scale issues. Below is a prioritized plan to make the research conclusions robust.

### Priority 1 — Fix the Data (Prerequisite for Everything Else)

**1. Redesign annotation protocol for overlap**
Each question-answer pair should be rated by **3–5 raters** (not just 1). Target at least 50 shared items and 15%+ overlap rate. This enables Krippendorff's alpha, Dawid-Skene, and reliable bias detection for all raters.

**2. Scale to 1000+ evaluations**
Current 307 evaluations → 614 training samples is insufficient for McNemar's significance at p<0.05. With 1000+ evaluations and forced overlap, most statistical tests become reliable.

**3. Seed gold calibration items**
Include 10–20 items where the correct answer is unambiguous (e.g., code that passes/fails a unit test). This allows direct measurement of rater calibration rather than inferring it from proxy signals.

### Priority 2 — Better Bias Detection

**4. Enable Dawid-Skene EM**
With sufficient shared items (≥5 per rater pair), the EM algorithm for probabilistic error rate estimation can run. This is more principled than the current z-score + chi-square approach.

**5. Replace auto-quality proxy**
Use a code execution engine (e.g., run generated Python code against test cases from CoNaLa) to get a hard correctness signal. Replace the weak type-token ratio proxy with pass/fail rates. This directly enables confidence–correctness decoupling detection.

**6. Ablation study on bias signals**
Train separate models with each bias signal included/excluded to quantify each signal's contribution to weight improvement. Currently all 5 signals are used together without individual evaluation.

### Priority 3 — Better Model

**7. Fine-tune the encoder**
Unfreeze CodeBERT (or switch to CodeT5+) and fine-tune with a low learning rate. Frozen embeddings are not optimized for the correctness signal in code quality annotation.

**8. Cross-validation instead of single split**
Replace the 80/20 single split with 5-fold or 10-fold cross-validation. With only 123 validation samples, a single split has high variance. k-fold gives more reliable accuracy estimates.

**9. Bayesian reward model**
Add uncertainty estimates per sample (e.g., MC Dropout or ensemble). Samples with high uncertainty are candidates for re-annotation rather than being weighted down silently.

### Priority 4 — Calibration Anchor

**10. LLM-as-judge calibration baseline**
Run GPT-4 or Claude on the same question-answer pairs with a structured rubric. Use LLM ratings as a reference to measure human rater calibration. Raters whose ratings diverge significantly from LLM-judge on gold items are likely miscalibrated (not just strict or lenient).

This addresses the core problem statement directly: if a rater prefers confident-sounding answers over correct ones, LLM-judge (prompted to check correctness, not style) will diverge from them on the gold calibration items.

---

## File Structure

```
Version_2/
├── run_all.py                          # Pipeline orchestrator
├── results_summary.json                # Top-level findings
│
├── stage1_eda/
│   ├── parse_and_explore.py
│   └── results/
│       ├── evaluations_parsed.csv      # 307 rows, 24 columns
│       ├── rater_stats.csv             # 11 raters, 13 statistics each
│       └── eda_plots/                  # 5 visualizations
│
├── stage2_bias_detection/
│   ├── detect_bias.py
│   └── results/
│       ├── bias_report.json            # All 5 mechanisms
│       ├── leniency_bias.csv
│       ├── positional_bias.csv
│       ├── decoupling_bias.csv
│       ├── inconsistency_scores.csv
│       └── bias_plots/                 # 2 visualizations
│
├── stage3_expertise/
│   ├── estimate_expertise.py
│   └── results/
│       ├── rater_weights.json          # Weights + component scores
│       └── rater_weights.csv
│
├── stage4_reward_model/
│   ├── train_weighted.py
│   └── results/
│       ├── training_results.json       # Accuracy, best epoch, per-class metrics
│       ├── history_weighted.csv        # 5 epochs
│       ├── history_baseline.csv        # 6 epochs
│       ├── val_predictions_weighted.csv
│       ├── val_predictions_baseline.csv
│       └── checkpoints/
│           ├── best_weighted.pt
│           └── best_baseline.pt
│
├── stage5_report/
│   ├── generate_report.py
│   └── results/
│       ├── master_rater_summary.csv    # All signals + weights merged
│       └── *.png                       # Visualizations
│
└── data/
    ├── evaluation_results_server/      # 307 JSON evaluation files
    └── datasets_for_eval/              # 25 CSV files (CoNaLa + SO)
```

---

## How to Run

```bash
# Run full pipeline
python run_all.py

# Or run individual stages
python stage1_eda/parse_and_explore.py
python stage2_bias_detection/detect_bias.py
python stage3_expertise/estimate_expertise.py
python stage4_reward_model/train_weighted.py
python stage5_report/generate_report.py
```

Each stage reads from the previous stage's `results/` directory. If a stage fails, the pipeline uses fallback data where possible.

---

## Key Takeaway

The **bias detection methodology works** — 3 biased raters were identified with interpretable, statistically grounded signals, and their labels are correctly down-weighted in training. The accuracy improvement (+6.5%) is directionally correct but not statistically significant on this dataset size. The methodology needs to be validated on a larger dataset with proper inter-rater overlap design before claiming accuracy gains.

The primary contribution of Version 2 is the **end-to-end pipeline for rater reliability estimation**, not the final accuracy number.
