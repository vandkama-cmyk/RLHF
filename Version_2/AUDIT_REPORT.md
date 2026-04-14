# RLHF Version 2 — Comprehensive Code & Results Audit

**Date:** 2026-04-14  
**Scope:** All 7 stages + orchestrator + results_summary.json  

---

## CRITICAL BUGS (will produce wrong results)

### 1. ROUGE self-comparison → dead code that misleads  
**File:** `stage2_bias_detection/detect_bias.py`, lines 362–363  

```python
df["auto_quality_L"] = [rouge1_f1(str(ans), str(ans)) for ans in df["answer_L"]]
df["auto_quality_R"] = [rouge1_f1(str(ans), str(ans)) for ans in df["answer_R"]]
```

The function computes ROUGE-1 of each answer **against itself** — always returns 1.0. This is immediately overwritten by the `answer_quality_proxy` function on lines 391–393, so functionally it's just wasted computation. But the comment on line 361 says "Computing ROUGE-1 proxy for auto quality", which will mislead anyone reading logs into thinking ROUGE is used. The entire `try: rouge_score` block (lines 347–367) should be removed.


### 2. Misaligned dimension correlations (data scramble)  
**File:** `stage1_eda/parse_and_explore.py`, lines 265–271  

```python
cols = {
    "consistent": pd.concat([df["consistent_L"], df["consistent_R"]]).dropna(),
    "correct":    pd.concat([df["correct_L"],    df["correct_R"]]).dropna(),
    "useful":     pd.concat([df["useful_L"],     df["useful_R"]]).dropna(),
}
dim_df = pd.DataFrame({k: v.reset_index(drop=True) for k, v in cols.items()}).dropna()
```

Each dimension is concatenated L+R, then **independently** `.dropna()`. If `consistent_L` is NaN on row 5 but `correct_L` is not, the independent `.dropna()` removes different rows for each dimension. After `reset_index(drop=True)`, the remaining rows are stitched together positionally — so row `i` in `consistent` now comes from a different evaluation than row `i` in `correct`. **All reported dimension correlations are between misaligned observations.** The correlation matrix and scatter plots are untrustworthy.

**Fix:** Concat L+R keeping all 3 dimensions together per observation, then dropna once across all 3.


### 3. Same misalignment in Stage 3 consistency scoring  
**File:** `stage3_expertise/estimate_expertise.py`, lines 183–187  

```python
cons = pd.concat([grp["consistent_L"], grp["consistent_R"]]).dropna().reset_index(drop=True)
corr = pd.concat([grp["correct_L"],    grp["correct_R"]]).dropna().reset_index(drop=True)
min_len = min(len(cons), len(corr))
r, _ = scipy_stats.spearmanr(cons[:min_len], corr[:min_len])
```

Same independent-dropna bug. The Spearman correlation `r(consistent, correct)` per rater is computed on misaligned data. This feeds into the **consistency_score** which feeds into the **final rater weights**. The entire rater weighting system is built on potentially corrupted correlations.

Same issue on lines 193–196 for `useful ↔ correct`.

**Fix:** Build a single DataFrame with all 3 columns from each side, drop rows where *any* column is NaN, then compute correlations.

---

## SERIOUS METHODOLOGICAL ISSUES

### 4. Extreme weight concentration: one rater gets 94% of the weight  
**File:** `results_summary.json`  

```
Anonymous:    0.939803  (93.98%)
Artem:        0.024546  (2.45%)
…
Fedor:        0.000044  (0.004%)
```

The combined effect of:
- **Softmax temperature T=0.05** (amplifies score differences by 20×)
- **N_SHRINKAGE pulls toward 0, not neutral** (line 454: `composite * n/(n+20)`)

creates a winner-takes-all dynamic. A rater with n=100 evaluations and composite=0.55 gets shrunk to 0.458; a rater with n=4 and composite=0.55 gets shrunk to 0.092. After division by T=0.05: softmax inputs differ by ~7 points, producing a 1000:1 weight ratio. The "weighted" reward model is effectively **a single-rater model**.

**Fix:** Shrink toward 0.5 (neutral), not toward 0: `composite_adj = 0.5 + (composite - 0.5) * n/(n+N0)`. Also consider T=0.2–0.5 for a gentler sharpening.


### 5. Per-class accuracy catastrophe  
**File:** `results_summary.json`, lines 71–73  

```
delta_positive_class:  +0.6721
delta_negative_class:  -0.7419
```

The weighted model gained +67% on positive class accuracy but **lost 74% on negative class accuracy**. This is not "improved accuracy" — the model simply shifted its decision boundary. The overall +6.5% delta is an artifact of class imbalance: if the dataset is ~60% positive, predicting "always positive" gets ~60% accuracy but 0% negative-class accuracy. The weighted model moved toward this degenerate solution.

McNemar's test correctly identifies this as non-significant (p=0.668), yet `results_summary.json` line 83 says "Bias-corrected weighting **improved** reward model accuracy" — this is misleading.


### 6. Focal loss × pos_weight interaction (double class-imbalance correction)  
**File:** `stage4_reward_model/train_weighted.py`, lines 130–144  

```python
bce = F.binary_cross_entropy_with_logits(
    logits, targets, pos_weight=pos_weight, reduction="none"
)
focal_weight = (1 - p_t) ** gamma
return focal_weight * bce
```

`pos_weight` inside `binary_cross_entropy_with_logits` already up-weights positive samples by `n_neg/n_pos`. The focal term `(1-p_t)^gamma` then further modulates the loss. For hard positive examples (low p_t), both mechanisms boost the loss simultaneously. This double correction can over-emphasize the minority positive class, explaining the per-class accuracy flip in issue #5.

**Fix:** Use only one of: focal loss OR pos_weight, not both. If using focal loss, apply class-balanced focal loss (alpha-balanced focal) instead of stacking pos_weight underneath.


### 7. Positional bias chi-square test uses wrong null hypothesis  
**File:** `stage2_bias_detection/detect_bias.py`, lines 258–261  

```python
expected = total / 3.0
chi2, pval = scipy_stats.chisquare(observed, f_exp=[expected, expected, expected])
```

The null hypothesis assumes raters should uniformly choose left/neutral/right with equal probability (33% each). In code comparison, true quality differences exist — it's natural for raters to rarely choose "neutral". This uniform null inflates chi-square statistics and produces false-positive "positional bias" detections for raters who simply have strong preferences on genuinely different code quality.

**Better null:** Use the dataset-wide distribution of slider values as the expected frequencies, or test left vs. right only (excluding neutral), where uniform IS reasonable under no positional bias.


### 8. Quality proxy is too crude for code generation  
**Files:** stages 2, 3, 6, 7 (identical function defined 4 times)  

```python
diversity = len(set(tokens)) / n
length_score = n / 50.0 if n <= 50 else max(0.0, 1.0 - (n - 50) / 200.0)
return 0.5 * diversity + 0.5 * length_score
```

This proxy combines type-token ratio with an inverted-U length heuristic. For code generation:
- **Type-token ratio is misleading** — correct code reuses variable names, imports, keywords. High diversity might indicate rambling natural-language filler, not correctness.
- **The 50-token peak is arbitrary** — many correct solutions are longer than 50 tokens.
- This proxy is then used to compute "metric alignment" which is the **most heavily weighted signal** (α=0.50 in Stage 3), yet it has no demonstrated correlation with actual code correctness.

All downstream metrics built on this proxy (confidence-correctness decoupling, metric alignment, quality alignment in Stage 7) inherit this weakness.

---

## DATA INTEGRITY ISSUES

### 9. "Anonymous" rater identity assumption  
**File:** `stage1_eda/parse_and_explore.py`, lines 69–73  

```python
if name == "Anonymous" and ip:
    rater_id = f"Anonymous_{ip}"
else:
    rater_id = name
```

Different IPs for "Anonymous" are treated as different raters. But: (a) a single person on a corporate network may have rotating IPs (DHCP), creating phantom raters; (b) multiple people on the same VPN exit node merge into one rater. Without additional signals (user-agent, timing patterns), IP-based deduplication is unreliable.

In the results, the single rater "Anonymous" (IP-deduplicated or not) has **93.98% of the reliability weight** — the entire pipeline's conclusions hinge on whether this rater identity is correct.


### 10. Data quality: duplicate evaluation spotted  
**File:** `res.md`, line 17  

> "pair 65716394_65716394 where Sergey Kovalchuk rated the same pair twice with slider = −1 and +2"

A rater evaluated the **same question-pair twice** with contradictory slider values (−1 vs +2). This is not deduplicated or flagged in Stage 1 parsing. Such contradictory evaluations:
- Inflate inter-rater disagreement metrics
- Inject noise into labels
- Artificially boost "inconsistency" scores for that rater

The pair_key `65716394_65716394` also suggests L and R are the **same question ID** — possible data collection bug.

---

## CODE QUALITY ISSUES

### 11. `answer_quality_proxy` duplicated 4 times  
**Files:** `stage2/detect_bias.py`, `stage3/estimate_expertise.py`, `stage6/lr_raters_experiment.py`, `stage7/pairs_vs_individual.py`

The identical function is copy-pasted across 4 files. If the formula changes, all 4 must be updated manually. Should be factored into a shared `utils.py`.


### 12. `torch.load` without `weights_only=True`  
**File:** `stage4_reward_model/train_weighted.py`, line 584  

```python
model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
```

PyTorch 2.6+ issues a `FutureWarning` and will eventually require `weights_only=True` for safe deserialization. Not a functional bug now, but will break in future PyTorch versions.


### 13. Inconsistency detection rules too restrictive  
**File:** `stage2_bias_detection/detect_bias.py`, lines 459–470  

The rules check only exact extremes: `corr <= -2 AND use >= 2`. On a -2 to +2 scale, this only catches the absolute worst cases. A rater giving `correct=-1, useful=+2` (incorrect but very useful?) is not flagged. The denominator is always 3, making the inconsistency score very coarse (possible values: 0, 0.33, 0.67, 1.0).


### 14. `eval_epoch` ignores sample_weights by design (but confusingly)  
**File:** `stage4_reward_model/train_weighted.py`, line 396  

```python
for features, labels, _weights in loader:
```

The underscore-prefixed `_weights` signals intentional discard. This is technically correct (validation should be unweighted for unbiased accuracy estimation), but creates an asymmetry with `train_epoch` that could confuse readers. Should be documented with a comment.


### 15. `warnings.filterwarnings("ignore")` used globally  
**Files:** stages 2, 4, 5, 6, 7  

Suppressing all warnings globally hides important diagnostics (convergence failures, deprecation warnings, numerical issues). Especially problematic in Stage 4 where class-imbalance warnings are explicitly emitted (line 482) but could be swallowed by the global filter set earlier.


### 16. No random seed for jitter in correlation plot  
**File:** `stage1_eda/parse_and_explore.py`, line 282  

```python
jitter = np.random.uniform(-0.15, 0.15, size=len(dim_df))
```

No seed → plots are non-reproducible across runs. Minor issue but inconsistent with the careful seeding in Stage 4.


### 17. Krippendorff alpha O(n²) implementation  
**Files:** `stage2/detect_bias.py` lines 117–126, `stage7/pairs_vs_individual.py` lines 126–137  

```python
for i in range(n_all):
    for j in range(i + 1, n_all):
        d_exp += (all_vals[i] - all_vals[j]) ** 2
```

Nested loop over all pairs is O(n²). For 614 values this is ~188k iterations — fine now. But the expected disagreement can be computed in O(n) using the variance formula: `D_exp = Var(all_vals) * 2`. Not a correctness bug, but a performance trap if the dataset grows.


---

## RESULTS INTERPRETATION ISSUES

### 18. improvement_delta rounded inconsistently  
**File:** `results_summary.json` line 70 vs. `stage5_report/generate_report.py` line 428  

The delta is reported as `0.065` (3 decimal places) but the underlying values are `0.5528 - 0.4878 = 0.0650`. The rounding is fine, but the interpretation says "improved" while McNemar's p=0.668 says "not significant". These are contradictory. The summary should lead with the statistical test result.


### 19. Krippendorff alpha set to null but overlap_pct computed differently in Stage 5  
**File:** `stage5_report/generate_report.py`, lines 516–519  

```python
"overlap_pct": round(
    100 * bias_report["inter_rater_agreement"]["n_shared_pairs"]
    / max(int(master_df["n_evaluations"].sum()), 1), 1
),
```

Here `n_evaluations` = 307 total evaluations. So overlap_pct = 38/307 = 12.4%. But Stage 2 computes it as `n_shared_pairs / len(df)` where `len(df)` = 307 (same). The value matches, but the denominator semantics differ: Stage 2 uses evaluation count, Stage 5 uses `master_df["n_evaluations"].sum()` which sums per-rater counts. These happen to be equal here, but would diverge if a rater appeared in multiple rows of master_df.


### 20. res.md conclusions are correct but not propagated  
**File:** `res.md`  

The manual analysis correctly identifies that only 1/4 tests support the pair-comparison hypothesis. However, this conclusion is not reflected in `results_summary.json` — Stage 7 results are entirely absent from the summary. Stage 6 results are also absent.

---

## SUMMARY TABLE

| # | Severity | Stage | Issue |
|---|----------|-------|-------|
| 1 | Critical | 2 | ROUGE self-comparison (dead code / misleading logs) |
| 2 | Critical | 1 | Dimension correlation computed on misaligned data |
| 3 | Critical | 3 | Consistency score computed on misaligned data → wrong weights |
| 4 | Serious | 3 | 94% weight concentration on one rater (broken shrinkage) |
| 5 | Serious | 4/5 | Per-class accuracy flip masked by aggregate metric |
| 6 | Serious | 4 | Focal loss × pos_weight double correction |
| 7 | Serious | 2 | Chi-square null hypothesis (uniform) is wrong for this task |
| 8 | Serious | all | Quality proxy too crude for code generation |
| 9 | Data | 1 | IP-based rater identity unreliable |
| 10 | Data | 1 | Duplicate contradictory evaluations not deduplicated |
| 11 | Code | all | Quality proxy function duplicated 4 times |
| 12 | Code | 4 | torch.load missing weights_only=True |
| 13 | Code | 2 | Inconsistency rules catch only extreme combinations |
| 14 | Code | 4 | eval_epoch silently discards weights |
| 15 | Code | all | Global warning suppression |
| 16 | Code | 1 | Non-reproducible jitter in plots |
| 17 | Code | 2,7 | O(n²) Krippendorff alpha |
| 18 | Report | 5 | "Improved" framing contradicts non-significant p-value |
| 19 | Report | 5 | overlap_pct computed with fragile denominator |
| 20 | Report | 5 | Stage 6 and 7 results absent from summary JSON |
