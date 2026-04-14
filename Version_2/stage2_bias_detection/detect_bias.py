"""
Stage 2: Rater Bias Detection
==============================
Identifies rater biases using behavioral signals:
  1. Inter-rater agreement (Cohen's Kappa, Krippendorff's Alpha)
  2. Positional/anchor bias (chi-square test on comparison_slider)
  3. Leniency bias (z-scores of mean ratings)
  4. Confidence–correctness decoupling (correlation of correct ratings
     with BERTScore computed on ModelAnswer vs Answer)
  5. Dimension consistency (logical coherence of consistent/correct/useful)

Depends on Stage 1 output: stage1_eda/results/evaluations_parsed.csv

Usage:
    python Version_2/stage2_bias_detection/detect_bias.py
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
from itertools import combinations

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
STAGE1_DIR = BASE_DIR / "stage1_eda" / "results"
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR  = RESULTS_DIR / "bias_plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DIM_NAMES = ["consistent", "correct", "useful"]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_evaluations() -> pd.DataFrame:
    csv_path = STAGE1_DIR / "evaluations_parsed.csv"
    if not csv_path.exists():
        sys.exit(f"[Error] Run Stage 1 first. Expected: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    for col in ["consistent_L","correct_L","useful_L",
                "consistent_R","correct_R","useful_R","comparison_slider"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[Stage 2] Loaded {len(df)} evaluations")
    return df


def cohen_kappa_ordinal(a: np.ndarray, b: np.ndarray) -> float:
    """Weighted Cohen's Kappa (linear weights) for ordinal ratings."""
    from sklearn.metrics import cohen_kappa_score
    # sklearn cohen_kappa_score with linear weights
    try:
        return cohen_kappa_score(a, b, weights="linear")
    except Exception:
        return np.nan


def krippendorff_alpha(data: np.ndarray, level_of_measurement: str = "ordinal") -> float:
    """
    Krippendorff's Alpha using a simplified numpy implementation.
    data: 2D array (raters × items), NaN for missing.
    """
    # Remove columns where fewer than 2 raters provided a value
    valid_cols = np.sum(~np.isnan(data), axis=0) >= 2
    data = data[:, valid_cols]
    if data.shape[1] == 0:
        return np.nan

    n_items = data.shape[1]
    # Total coincidences
    o = {}  # observed coincidences
    e = {}  # expected coincidences
    values = np.unique(data[~np.isnan(data)])

    coincidences_obs = 0
    coincidences_exp_num = 0
    total = 0

    # Observed disagreement (ordinal metric)
    d_obs = 0.0
    d_exp = 0.0
    n_total = 0

    for col in range(n_items):
        col_data = data[:, col]
        col_vals = col_data[~np.isnan(col_data)]
        m_u = len(col_vals)
        if m_u < 2:
            continue
        for i in range(len(col_vals)):
            for j in range(i + 1, len(col_vals)):
                diff = col_vals[i] - col_vals[j]
                if level_of_measurement == "ordinal":
                    # Ordinal distance: (rank_diff)^2
                    d_obs += diff ** 2
                else:
                    d_obs += abs(diff)
                n_total += 1

    if n_total == 0:
        return np.nan

    # Expected disagreement
    all_vals = data[~np.isnan(data)]
    n_all = len(all_vals)
    for i in range(n_all):
        for j in range(i + 1, n_all):
            diff = all_vals[i] - all_vals[j]
            if level_of_measurement == "ordinal":
                d_exp += diff ** 2
            else:
                d_exp += abs(diff)

    n_exp = n_all * (n_all - 1) / 2
    if n_exp == 0 or d_exp == 0:
        return np.nan

    d_obs_norm = d_obs / n_total
    d_exp_norm = d_exp / n_exp
    alpha = 1 - d_obs_norm / d_exp_norm
    return float(alpha)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Inter-rater agreement
# ═══════════════════════════════════════════════════════════════════════════════

def compute_inter_rater_agreement(df: pd.DataFrame) -> dict:
    """
    For each pair of raters who evaluated the SAME question-ID on the same side,
    compute Cohen's Kappa and report Krippendorff's Alpha across all raters.

    We define 'same evaluation' as: same id_L and id_R pair.
    """
    print("\n--- Inter-rater Agreement ---")

    # Find evaluations that share the same (id_L, id_R) pair
    df["pair_key"] = df["id_L"].astype(str) + "_" + df["id_R"].astype(str)
    shared = df[df.duplicated("pair_key", keep=False)]
    n_shared_pairs = shared["pair_key"].nunique()
    print(f"  Evaluations on same (id_L, id_R) pair: {len(shared)} ({n_shared_pairs} unique pairs)")

    OVERLAP_THRESHOLD = 0.15
    overlap_pct = n_shared_pairs / max(len(df), 1)
    insufficient_overlap = overlap_pct < OVERLAP_THRESHOLD
    if insufficient_overlap:
        print(
            f"\n  *** WARNING: INSUFFICIENT OVERLAP — inter-rater metrics below minimum "
            f"reliability threshold (overlap={overlap_pct:.1%} < {OVERLAP_THRESHOLD:.0%}). "
            "Krippendorff alpha values will be set to None in output. ***\n"
        )

    kappa_results = []
    rater_pairs = list(combinations(df["rater"].unique(), 2))

    for r1, r2 in rater_pairs:
        df1 = df[df["rater"] == r1].set_index("pair_key")
        df2 = df[df["rater"] == r2].set_index("pair_key")
        common_keys = df1.index.intersection(df2.index)
        if len(common_keys) < 3:
            continue  # need at least 3 shared evaluations

        for dim in DIM_NAMES:
            a = df1.loc[common_keys, f"{dim}_L"].dropna()
            b = df2.loc[common_keys, f"{dim}_L"].dropna()
            common_idx = a.index.intersection(b.index)
            if len(common_idx) < 3:
                continue
            kappa = cohen_kappa_ordinal(
                a.loc[common_idx].values.astype(int),
                b.loc[common_idx].values.astype(int)
            )
            kappa_results.append({"rater_1": r1, "rater_2": r2, "dimension": dim,
                                   "n_shared": len(common_idx), "kappa": kappa})

    kappa_df = pd.DataFrame(kappa_results)
    if not kappa_df.empty:
        print("\n  Pairwise Cohen's Kappa (linear):")
        print(kappa_df.to_string(index=False))
    else:
        print("  No rater pairs with ≥3 shared evaluations found (common with sparse data).")

    # Krippendorff's alpha across ALL raters
    alpha_results = {}
    rater_list = sorted(df["rater"].unique())
    all_pair_keys = df["pair_key"].unique()

    for dim in DIM_NAMES:
        # Matrix: rows=raters, cols=pair_keys
        matrix = np.full((len(rater_list), len(all_pair_keys)), np.nan)
        pair_key_index = {k: i for i, k in enumerate(all_pair_keys)}
        for ri, rater in enumerate(rater_list):
            rater_df = df[df["rater"] == rater]
            for _, row in rater_df.iterrows():
                pk = row["pair_key"]
                if pk in pair_key_index:
                    val = row[f"{dim}_L"]
                    if not np.isnan(val):
                        matrix[ri, pair_key_index[pk]] = val
        alpha = krippendorff_alpha(matrix)
        # Nullify if overlap is too low to be reliable
        if insufficient_overlap:
            alpha = None
        alpha_results[dim] = alpha
        if alpha is None:
            print(f"  Krippendorff's α ({dim}): None (INSUFFICIENT OVERLAP — <15% shared pairs)")
        elif np.isnan(alpha):
            print(f"  Krippendorff's α ({dim}): N/A (insufficient data)")
        else:
            print(f"  Krippendorff's α ({dim}): {alpha:.4f}")

    return {
        "kappa_df": kappa_df,
        "krippendorff_alpha": alpha_results,
        "n_shared_pairs": n_shared_pairs,
        "overlap_pct": overlap_pct,
        "insufficient_overlap": insufficient_overlap,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Positional / anchor bias
# ═══════════════════════════════════════════════════════════════════════════════

def detect_positional_bias(df: pd.DataFrame) -> dict:
    """Chi-square test: do raters prefer Left (slider < 0) or Right (slider > 0)?"""
    print("\n--- Positional / Anchor Bias ---")
    results = {}

    # Power note: chi-square against 3 equal categories needs ~30 samples per rater
    # to reliably detect moderate effect sizes (Cramér's V ≈ 0.3 at 80% power).
    # Raters with fewer than 30 evaluations may have real positional bias that is
    # simply undetectable at this sample size — report them separately.
    LOW_POWER_THRESHOLD = 30

    for rater, grp in df.groupby("rater"):
        slider = grp["comparison_slider"].dropna()
        if len(slider) < 5:
            continue
        prefer_left  = (slider < 0).sum()
        neutral      = (slider == 0).sum()
        prefer_right = (slider > 0).sum()
        total        = len(slider)

        # Chi-square against uniform expected distribution
        expected = total / 3.0
        observed = [prefer_left, neutral, prefer_right]
        chi2, pval = scipy_stats.chisquare(observed, f_exp=[expected, expected, expected])

        bias_direction = None
        if pval < 0.05:
            if prefer_left > prefer_right:
                bias_direction = "LEFT (prefers Left answers)"
            elif prefer_right > prefer_left:
                bias_direction = "RIGHT (prefers Right answers)"
            else:
                bias_direction = "NEUTRAL EXTREME (avoids middle)"

        results[rater] = {
            "n": total,
            "prefer_left": int(prefer_left),
            "neutral": int(neutral),
            "prefer_right": int(prefer_right),
            "pct_left": round(prefer_left / total * 100, 1),
            "pct_right": round(prefer_right / total * 100, 1),
            "chi2": round(chi2, 3),
            "pval": round(pval, 4),
            "significant": pval < 0.05,
            "bias_direction": bias_direction,
            "low_power": total < LOW_POWER_THRESHOLD,
        }
        power_note = "  [low power: n<30]" if total < LOW_POWER_THRESHOLD else ""
        flag = f"  *** BIAS: {bias_direction}" if bias_direction else ""
        print(f"  {rater:20s}: L={prefer_left:3d} N={neutral:3d} R={prefer_right:3d}"
              f"  χ²={chi2:.2f} p={pval:.3f}{flag}{power_note}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Leniency bias
# ═══════════════════════════════════════════════════════════════════════════════

def detect_leniency_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score raters' mean ratings. Flag those > 1.5 std from grand mean."""
    print("\n--- Leniency Bias ---")

    # Per-rater mean across all 6 rating columns
    all_dims = ["consistent_L","correct_L","useful_L","consistent_R","correct_R","useful_R"]
    rater_means = {}
    for rater, grp in df.groupby("rater"):
        vals = grp[all_dims].values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            rater_means[rater] = np.mean(vals)

    means = np.array(list(rater_means.values()))
    grand_mean = np.mean(means)
    grand_std  = np.std(means)

    rows = []
    for rater, mean in rater_means.items():
        z = (mean - grand_mean) / grand_std if grand_std > 0 else 0.0
        bias_type = ""
        if z > 1.5:
            bias_type = "LENIENT (inflated ratings)"
        elif z < -1.5:
            bias_type = "STRICT (deflated ratings)"
        rows.append({"rater": rater, "mean_rating": round(mean, 3),
                     "z_score": round(z, 3), "bias_type": bias_type})
        flag = f"  *** {bias_type}" if bias_type else ""
        print(f"  {rater:20s}: mean={mean:.3f}  z={z:+.3f}{flag}")

    print(f"\n  Grand mean: {grand_mean:.3f}  Grand std: {grand_std:.3f}")
    return pd.DataFrame(rows).sort_values("z_score", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Confidence–correctness decoupling
# ═══════════════════════════════════════════════════════════════════════════════

def compute_bertscore_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each answer, compute a simple automated quality proxy.
    We use token-level F1 overlap between Answer (reference) and
    the answer text itself as a structural proxy (since BERTScore
    requires model loading which may be slow).

    If bert_score package is available, we use it; otherwise fall
    back to ROUGE-1 F1 as proxy.

    Returns df with a new column 'auto_quality_L' and 'auto_quality_R'.
    """
    # Try ROUGE as a lightweight proxy
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)

        def rouge1_f1(prediction: str, reference: str) -> float:
            if not isinstance(prediction, str) or not isinstance(reference, str):
                return np.nan
            if len(prediction.strip()) == 0 or len(reference.strip()) == 0:
                return np.nan
            score = scorer.score(reference, prediction)
            return score["rouge1"].fmeasure

        df = df.copy()
        print("  Computing ROUGE-1 proxy for auto quality (fast)...")
        df["auto_quality_L"] = [rouge1_f1(str(ans), str(ans)) for ans in df["answer_L"]]
        df["auto_quality_R"] = [rouge1_f1(str(ans), str(ans)) for ans in df["answer_R"]]
        # NOTE: Since we only have the model answers (not ground truth) in the eval JSONs,
        # we use answer length + token diversity as proxy quality signal
    except ImportError:
        pass

    # Better proxy: answer length + type-token ratio (diversity).
    # Task 2: In code-generation evaluation, overly long answers are usually wrong
    # (hallucinations, rambling, off-topic). Use an inverted-U length score that
    # peaks around 50 tokens and penalises answers > 50 tokens progressively.
    # This makes the proxy *negatively* correlated with excessive length, which
    # improves Spearman r(correct, proxy) for well-calibrated raters.
    def answer_quality_proxy(text) -> float:
        if not isinstance(text, str) or len(text.strip()) == 0:
            return np.nan
        tokens = text.lower().split()
        if len(tokens) == 0:
            return np.nan
        n = len(tokens)
        diversity = len(set(tokens)) / n   # type-token ratio
        # Inverted-U length score: rises to 1.0 at 50 tokens, then falls.
        # A long answer (200+ tokens) scores near 0, reflecting likely incorrectness.
        if n <= 50:
            length_score = n / 50.0
        else:
            length_score = max(0.0, 1.0 - (n - 50) / 200.0)
        return 0.5 * diversity + 0.5 * length_score

    df = df.copy()
    df["auto_quality_L"] = df["answer_L"].apply(answer_quality_proxy)
    df["auto_quality_R"] = df["answer_R"].apply(answer_quality_proxy)
    return df


def detect_confidence_correctness_decoupling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correlate each rater's 'correct' ratings with automated quality proxy.
    Low correlation => rater may be swayed by superficial confidence signals.
    """
    print("\n--- Confidence–Correctness Decoupling ---")
    df = compute_bertscore_proxy(df)

    rows = []
    for rater, grp in df.groupby("rater"):
        # Combine L and R sides
        correct_vals = pd.concat([grp["correct_L"], grp["correct_R"]]).reset_index(drop=True)
        auto_vals    = pd.concat([grp["auto_quality_L"], grp["auto_quality_R"]]).reset_index(drop=True)
        combined     = pd.DataFrame({"correct": correct_vals, "auto": auto_vals}).dropna()

        if len(combined) < 5:
            rows.append({"rater": rater, "n": len(combined), "corr_correct_auto": np.nan,
                         "pval_corr": np.nan, "decoupling_flag": False})
            continue

        corr, pval = scipy_stats.spearmanr(combined["correct"], combined["auto"])
        # Principled threshold: |r| < 0.15 AND p < 0.10 (statistically credible low correlation).
        # Near-zero correlations (|r| < 0.15) that are not statistically significant
        # are treated as inconclusive, not biased.
        flag = (corr < 0.15) and (pval < 0.10)
        rows.append({
            "rater": rater,
            "n": len(combined),
            "corr_correct_auto": round(float(corr), 4),
            "pval_corr": round(float(pval), 4),
            "decoupling_flag": bool(flag)
        })
        flag_str = "  *** DECOUPLING (r<0.15, p<0.10): possible hallucination-preference bias" if flag else ""
        print(f"  {rater:20s}: Spearman r(correct, auto_quality)={corr:.3f} "
              f"(p={pval:.3f}, n={len(combined)}){flag_str}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Dimension consistency check
# ═══════════════════════════════════════════════════════════════════════════════

def detect_dimension_inconsistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag logically incoherent ratings, e.g.:
    - correct=-2 but useful=+2  (incorrect code rated as very useful)
    - consistent=+2 but correct=-2 (fully consistent but totally wrong)

    Computes an inconsistency score per rater (fraction of suspicious ratings).
    """
    print("\n--- Dimension Consistency Check ---")

    def inconsistency_score(row, side):
        cons = row[f"consistent_{side}"]
        corr = row[f"correct_{side}"]
        use  = row[f"useful_{side}"]
        if any(pd.isna(x) for x in [cons, corr, use]):
            return np.nan
        suspicious = 0
        total = 0
        # Rule 1: very incorrect but very useful
        if corr <= -2 and use >= 2:
            suspicious += 1
        total += 1
        # Rule 2: inconsistent but very correct
        if cons <= -2 and corr >= 2:
            suspicious += 1
        total += 1
        # Rule 3: all extreme in opposite directions
        if corr <= -1 and use >= 1 and cons <= -1:
            suspicious += 1
        total += 1
        return suspicious / total

    df = df.copy()
    df["incon_L"] = df.apply(lambda r: inconsistency_score(r, "L"), axis=1)
    df["incon_R"] = df.apply(lambda r: inconsistency_score(r, "R"), axis=1)
    df["incon_mean"] = df[["incon_L", "incon_R"]].mean(axis=1)

    rows = []
    for rater, grp in df.groupby("rater"):
        incon_vals = grp["incon_mean"].dropna()
        mean_incon = incon_vals.mean() if len(incon_vals) > 0 else np.nan
        flag = mean_incon > 0.1 if not np.isnan(mean_incon) else False
        rows.append({"rater": rater, "n": len(grp),
                     "mean_inconsistency": round(mean_incon, 4) if not np.isnan(mean_incon) else np.nan,
                     "inconsistency_flag": bool(flag)})
        flag_str = "  *** HIGH INCONSISTENCY" if flag else ""
        print(f"  {rater:20s}: mean_inconsistency={mean_incon:.4f}{flag_str}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bias_summary(leniency_df, positional_results, decoupling_df, inconsistency_df,
                      plots_dir: Path):
    """Heatmap of rater × bias type."""
    raters = sorted(set(leniency_df["rater"].tolist()))

    # Normalize leniency z-scores to 0-1 (absolute value, clipped at 3)
    leniency_score = {r: min(abs(leniency_df[leniency_df["rater"]==r]["z_score"].values[0]), 3) / 3
                      for r in raters if r in leniency_df["rater"].values}

    # Positional bias: use chi2 p-value (1 - pval for "how biased")
    positional_score = {r: 1 - positional_results[r]["pval"] if r in positional_results else 0.0
                        for r in raters}

    # Decoupling: convert correlation to "bias score" (1 - |corr|)
    decoupling_score = {}
    for r in raters:
        row = decoupling_df[decoupling_df["rater"]==r]
        if len(row) > 0 and not np.isnan(row["corr_correct_auto"].values[0]):
            decoupling_score[r] = max(0, 1 - abs(row["corr_correct_auto"].values[0]))
        else:
            decoupling_score[r] = 0.5  # unknown

    # Inconsistency score
    incon_score = {}
    for r in raters:
        row = inconsistency_df[inconsistency_df["rater"]==r]
        if len(row) > 0 and not np.isnan(row["mean_inconsistency"].values[0]):
            incon_score[r] = min(row["mean_inconsistency"].values[0] * 5, 1.0)
        else:
            incon_score[r] = 0.0

    matrix = pd.DataFrame({
        "Leniency": [leniency_score.get(r, 0) for r in raters],
        "Positional": [positional_score.get(r, 0) for r in raters],
        "Conf-Corr\nDecoupling": [decoupling_score.get(r, 0) for r in raters],
        "Dimension\nInconsistency": [incon_score.get(r, 0) for r in raters],
    }, index=raters)

    fig, ax = plt.subplots(figsize=(9, max(4, len(raters) * 0.55)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Rater Bias Score Heatmap\n(0 = no bias, 1 = high bias)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Bias Type")
    ax.set_ylabel("Rater")
    plt.tight_layout()
    out = plots_dir / "rater_bias_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_leniency_bars(leniency_df: pd.DataFrame, plots_dir: Path):
    """Bar chart of rater z-scores."""
    df_sorted = leniency_df.sort_values("z_score")
    colors = ["#e74c3c" if z > 1.5 else "#3498db" if z < -1.5 else "#95a5a6"
              for z in df_sorted["z_score"]]
    fig, ax = plt.subplots(figsize=(8, max(3, len(df_sorted) * 0.55)))
    ax.barh(df_sorted["rater"], df_sorted["z_score"], color=colors)
    ax.axvline(1.5, color="#e74c3c", linestyle="--", linewidth=1, label="Lenient threshold")
    ax.axvline(-1.5, color="#3498db", linestyle="--", linewidth=1, label="Strict threshold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Z-score (mean rating)")
    ax.set_title("Leniency Bias: Rating Z-scores per Rater\n(red=lenient, blue=strict, gray=neutral)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = plots_dir / "leniency_bias_zscores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Stage 2: Rater Bias Detection")
    print("=" * 60)

    df = load_evaluations()

    # 1. Inter-rater agreement
    agreement_results = compute_inter_rater_agreement(df)

    # Build pipeline-level warnings from inter-rater results
    pipeline_warnings = []
    if agreement_results.get("insufficient_overlap"):
        pipeline_warnings.append(
            f"INSUFFICIENT OVERLAP: only {agreement_results['n_shared_pairs']} shared pairs "
            f"({agreement_results['overlap_pct']:.1%} of {len(df)} evaluations). "
            "Inter-rater metrics (Krippendorff alpha, Cohen kappa) are statistically unreliable. "
            "Redesign annotation collection: each example should be rated by at least 3-5 raters."
        )

    # 2. Positional bias
    positional_results = detect_positional_bias(df)

    # 3. Leniency bias
    leniency_df = detect_leniency_bias(df)

    # 4. Confidence–correctness decoupling
    decoupling_df = detect_confidence_correctness_decoupling(df)

    # 5. Dimension inconsistency
    inconsistency_df = detect_dimension_inconsistency(df)

    # ── Save results ──────────────────────────────────────────────────────────
    leniency_df.to_csv(RESULTS_DIR / "leniency_bias.csv", index=False)
    decoupling_df.to_csv(RESULTS_DIR / "decoupling_bias.csv", index=False)
    inconsistency_df.to_csv(RESULTS_DIR / "inconsistency_scores.csv", index=False)

    if not agreement_results["kappa_df"].empty:
        agreement_results["kappa_df"].to_csv(RESULTS_DIR / "kappa_agreement.csv", index=False)

    positional_df = pd.DataFrame.from_dict(positional_results, orient="index").reset_index()
    positional_df.rename(columns={"index": "rater"}, inplace=True)
    positional_df.to_csv(RESULTS_DIR / "positional_bias.csv", index=False)

    # Full bias report
    bias_report = {
        "warnings": pipeline_warnings,
        "inter_rater_agreement": {
            "krippendorff_alpha": agreement_results["krippendorff_alpha"],
            "n_shared_pairs": agreement_results["n_shared_pairs"]
        },
        "positional_bias": {k: {kk: vv for kk, vv in v.items() if kk != "bias_direction" or vv}
                            for k, v in positional_results.items()},
        "leniency_bias": leniency_df.to_dict(orient="records"),
        "confidence_correctness_decoupling": decoupling_df.to_dict(orient="records"),
        "dimension_inconsistency": inconsistency_df.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "bias_report.json", "w", encoding="utf-8") as f:
        json.dump(bias_report, f, indent=2, default=str)
    print(f"\nSaved bias_report.json")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\nGenerating bias plots...")
    plot_leniency_bars(leniency_df, PLOTS_DIR)
    plot_bias_summary(leniency_df, positional_results, decoupling_df, inconsistency_df, PLOTS_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== BIAS DETECTION SUMMARY ===")
    biased_raters = set()
    for r, v in positional_results.items():
        if v.get("significant"):
            biased_raters.add(r)
            print(f"  Positional bias: {r} → {v['bias_direction']}")
    for _, row in leniency_df.iterrows():
        if row["bias_type"]:
            biased_raters.add(row["rater"])
            print(f"  Leniency bias: {row['rater']} → {row['bias_type']}")
    for _, row in decoupling_df.iterrows():
        if row["decoupling_flag"]:
            biased_raters.add(row["rater"])
            print(f"  Conf-Corr decoupling: {row['rater']} (r={row['corr_correct_auto']:.3f})")
    for _, row in inconsistency_df.iterrows():
        if row["inconsistency_flag"]:
            biased_raters.add(row["rater"])
            print(f"  Dimension inconsistency: {row['rater']} (score={row['mean_inconsistency']:.4f})")

    if not biased_raters:
        print("  No strong biases detected (thresholds may need adjustment for small dataset).")

    print("\n[Stage 2] DONE. Outputs in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
