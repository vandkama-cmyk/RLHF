"""
Stage 3: Expertise Proxy Estimation & Rater Weighting
=======================================================
Since no explicit expertise metadata exists, reliability is estimated
from behavioral proxy signals:

  1. Metric alignment score: Spearman correlation of rater's 'correct'
     ratings with automated quality proxy (from Stage 2)
  2. Consistency score: within-rater logical coherence (inverse of
     dimension inconsistency score from Stage 2)
  3. Agreement-with-majority score: cosine similarity of a rater's
     rating vector against the leave-one-out mean of all other raters
     on the same question pairs
  4. Optional Dawid-Skene: EM-based probabilistic rater reliability
     estimation (falls back gracefully if insufficient overlap)

Final weight: w_r = softmax(α * align + β * consistency + γ * agreement)
Configurable α=0.5, β=0.25, γ=0.25

Depends on:
  - stage1_eda/results/evaluations_parsed.csv
  - stage2_bias_detection/results/decoupling_bias.csv
  - stage2_bias_detection/results/inconsistency_scores.csv

Usage:
    python Version_2/stage3_expertise/estimate_expertise.py
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.special import softmax as scipy_softmax

# Shared quality proxy (Bug #11 fix)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import answer_quality_proxy

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
STAGE1_DIR  = BASE_DIR / "stage1_eda" / "results"
STAGE2_DIR  = BASE_DIR / "stage2_bias_detection" / "results"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Weights for combining proxy signals
ALPHA = 0.50   # metric alignment (most important: reflects domain expertise)
BETA  = 0.25   # consistency (logical coherence of ratings)
GAMMA = 0.25   # agreement with majority

# Bug #4 fix: temperature raised from 0.05 → 0.20.
# T=0.05 amplifies composite-score differences by 20×, which combined with
# shrinkage-toward-0 (see below) created a winner-takes-all dynamic where a
# single rater captured 94% of the total weight.  T=0.20 still meaningfully
# sharpens the softmax (≈5× amplification) without extreme concentration.
SOFTMAX_TEMPERATURE = 0.20

# Bug #4 fix: N_SHRINKAGE_N0 shrinks toward the NEUTRAL value (0.5), not toward 0.
# The original formula  composite * n/(n+N0)  pulled low-n raters toward 0,
# which maps to extremely low softmax inputs regardless of their actual scores.
# The corrected formula is applied in compute_rater_weights():
#   composite_adj = 0.5 + (composite - 0.5) * n / (n + N_SHRINKAGE_N0)
# A rater with composite=0.55 and n=4 is now shrunk to ~0.51 (closer to neutral)
# rather than 0.09 (almost certainly penalised to near-zero weight).
N_SHRINKAGE_N0 = 20.0


# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════

def load_rater_n_counts() -> dict:
    """Per-rater evaluation counts from Stage 1 (for uncertainty shrinkage)."""
    path = STAGE1_DIR / "rater_stats.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "rater" not in df.columns or "n_evaluations" not in df.columns:
        return {}
    return {str(row["rater"]): int(row["n_evaluations"]) for _, row in df.iterrows()}


def load_data():
    eval_path = STAGE1_DIR / "evaluations_parsed.csv"
    if not eval_path.exists():
        sys.exit(f"[Error] Run Stage 1 first. Expected: {eval_path}")
    eval_df = pd.read_csv(eval_path, low_memory=False)

    decoupling_path = STAGE2_DIR / "decoupling_bias.csv"
    inconsistency_path = STAGE2_DIR / "inconsistency_scores.csv"

    decoupling_df = pd.read_csv(decoupling_path) if decoupling_path.exists() else pd.DataFrame()
    inconsistency_df = pd.read_csv(inconsistency_path) if inconsistency_path.exists() else pd.DataFrame()

    for col in ["consistent_L","correct_L","useful_L","consistent_R","correct_R","useful_R","comparison_slider"]:
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

    print(f"[Stage 3] Loaded {len(eval_df)} evaluations, {eval_df['rater'].nunique()} raters")
    return eval_df, decoupling_df, inconsistency_df


# ═══════════════════════════════════════════════════════════════════════════════
# Signal 1: Metric Alignment Score
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metric_alignment(eval_df: pd.DataFrame, decoupling_df: pd.DataFrame) -> dict:
    """
    Use Spearman correlation between rater's correct ratings and auto quality proxy
    as computed in Stage 2. Normalize to [0, 1].

    If decoupling data not available, recompute the proxy inline.
    """
    print("\n--- Signal 1: Metric Alignment ---")

    if not decoupling_df.empty and "corr_correct_auto" in decoupling_df.columns:
        # Use precomputed correlation from Stage 2
        scores = {}
        for _, row in decoupling_df.iterrows():
            corr = row["corr_correct_auto"]
            if pd.isna(corr):
                scores[row["rater"]] = 0.5   # neutral for unknown
            else:
                # Normalize: corr in [-1, 1] → [0, 1]
                scores[row["rater"]] = (float(corr) + 1) / 2
        print("  Using precomputed correlations from Stage 2")
    else:
        # Recompute inline using the shared proxy from utils.py (Bug #11 fix)
        eval_df = eval_df.copy()
        eval_df["auto_quality_L"] = eval_df["answer_L"].apply(answer_quality_proxy)
        eval_df["auto_quality_R"] = eval_df["answer_R"].apply(answer_quality_proxy)

        from scipy import stats as scipy_stats
        scores = {}
        for rater, grp in eval_df.groupby("rater"):
            correct_vals = pd.concat([grp["correct_L"], grp["correct_R"]]).reset_index(drop=True)
            auto_vals    = pd.concat([grp["auto_quality_L"], grp["auto_quality_R"]]).reset_index(drop=True)
            combined     = pd.DataFrame({"correct": correct_vals, "auto": auto_vals}).dropna()
            if len(combined) < 5:
                scores[rater] = 0.5
                continue
            corr, _ = scipy_stats.spearmanr(combined["correct"], combined["auto"])
            scores[rater] = (float(corr) + 1) / 2

    for r, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {r:20s}: alignment_score={s:.4f}")
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Signal 2: Consistency Score
# ═══════════════════════════════════════════════════════════════════════════════

def compute_consistency_score(eval_df: pd.DataFrame, inconsistency_df: pd.DataFrame) -> dict:
    """
    Consistency = 1 - mean_inconsistency_score (from Stage 2).
    Also adds within-rater correlation between correct and consistent ratings
    (an expert should find that correct and consistent correlate strongly).
    """
    print("\n--- Signal 2: Consistency ---")

    from scipy import stats as scipy_stats

    scores = {}
    for rater, grp in eval_df.groupby("rater"):
        # Logical coherence from Stage 2
        incon_row = inconsistency_df[inconsistency_df["rater"]==rater] if not inconsistency_df.empty else pd.DataFrame()
        if len(incon_row) > 0 and not pd.isna(incon_row["mean_inconsistency"].values[0]):
            coherence = max(0.0, 1.0 - incon_row["mean_inconsistency"].values[0] * 5)
        else:
            coherence = 0.5  # neutral

        # Bug #3 fix: build a single aligned DataFrame before computing correlations.
        # The original code called dropna() independently on each dimension and then
        # sliced with min_len — after reset_index the remaining rows came from
        # different evaluations, producing correlations on misaligned observations.
        # Fix: concatenate all three columns together (L then R), drop rows where
        # ANY column is NaN so every observation is aligned.
        left_3  = grp[["consistent_L", "correct_L", "useful_L"]].rename(
            columns={"consistent_L": "cons", "correct_L": "corr", "useful_L": "use"})
        right_3 = grp[["consistent_R", "correct_R", "useful_R"]].rename(
            columns={"consistent_R": "cons", "correct_R": "corr", "useful_R": "use"})
        dim_df  = pd.concat([left_3, right_3], ignore_index=True).dropna()

        if len(dim_df) >= 5:
            r, _  = scipy_stats.spearmanr(dim_df["cons"], dim_df["corr"])
            r2, _ = scipy_stats.spearmanr(dim_df["use"],  dim_df["corr"])
            dim_corr_score  = (float(r)  + 1) / 2
            use_corr_score  = (float(r2) + 1) / 2
        else:
            dim_corr_score = 0.5
            use_corr_score = 0.5

        # Combine: 50% coherence, 30% cons↔corr, 20% use↔corr
        score = 0.5 * coherence + 0.3 * dim_corr_score + 0.2 * use_corr_score
        scores[rater] = float(score)
        print(f"  {rater:20s}: consistency_score={score:.4f} "
              f"(coherence={coherence:.2f}, dim_corr={dim_corr_score:.2f}, use_corr={use_corr_score:.2f})")

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Signal 3: Agreement-with-majority Score
# ═══════════════════════════════════════════════════════════════════════════════

def compute_majority_agreement(eval_df: pd.DataFrame) -> dict:
    """
    For each rater, compute cosine similarity of their rating vectors
    against the leave-one-out mean of all other raters on the same (id_L, id_R) pairs.
    """
    print("\n--- Signal 3: Majority Agreement ---")

    eval_df = eval_df.copy()
    eval_df["pair_key"] = eval_df["id_L"].astype(str) + "_" + eval_df["id_R"].astype(str)
    rating_cols = ["consistent_L","correct_L","useful_L","consistent_R","correct_R","useful_R"]

    # Build: pair_key → {rater: rating_vector}
    pair_ratings = {}
    for _, row in eval_df.iterrows():
        pk = row["pair_key"]
        rater = row["rater"]
        vec = [row[c] for c in rating_cols]
        if any(pd.isna(v) for v in vec):
            continue
        if pk not in pair_ratings:
            pair_ratings[pk] = {}
        pair_ratings[pk][rater] = np.array(vec, dtype=float)

    # Only pairs with ≥2 raters contribute to agreement
    shared_pairs = {pk: v for pk, v in pair_ratings.items() if len(v) >= 2}
    print(f"  Pairs with ≥2 raters: {len(shared_pairs)}")

    def cosine_sim(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    rater_agreement = {r: [] for r in eval_df["rater"].unique()}

    for pk, rater_vecs in shared_pairs.items():
        raters = list(rater_vecs.keys())
        for rater in raters:
            other_vecs = [rater_vecs[r] for r in raters if r != rater]
            if not other_vecs:
                continue
            majority_vec = np.mean(other_vecs, axis=0)
            sim = cosine_sim(rater_vecs[rater], majority_vec)
            rater_agreement[rater].append(sim)

    scores = {}
    for rater, sims in rater_agreement.items():
        n = len(sims)
        if n >= 1:
            raw_cos = float(np.mean(sims))
            # Normalize from [-1,1] (cosine) to [0,1]
            score = (raw_cos + 1) / 2
            scores[rater] = score
            print(f"  {rater:20s}: agreement_score={score:.4f} (n_shared_pairs={n})")
        else:
            # No overlapping pairs with another rater — undefined (do not use neutral+remap bug)
            scores[rater] = float("nan")
            print(f"  {rater:20s}: agreement_score=nan (n_shared_pairs=0)")

    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# Optional: Dawid-Skene EM
# ═══════════════════════════════════════════════════════════════════════════════

def dawid_skene_em(eval_df: pd.DataFrame, n_classes: int = 5, max_iter: int = 50,
                   tol: float = 1e-4) -> dict:
    """
    Simplified Dawid-Skene EM for ordinal labels.
    Estimates per-rater error rates as a proxy for reliability.
    Returns reliability score in [0, 1] per rater.

    Uses 'correct_L' as the primary reliability signal.
    """
    print("\n--- Optional: Dawid-Skene EM ---")

    eval_df = eval_df.copy()
    eval_df["pair_key"] = eval_df["id_L"].astype(str) + "_" + eval_df["id_R"].astype(str)

    # Only use pairs rated by ≥2 raters
    pair_counts = eval_df.groupby("pair_key")["rater"].count()
    shared_keys = pair_counts[pair_counts >= 2].index.tolist()

    if len(shared_keys) < 5:
        print(f"  Insufficient overlap ({len(shared_keys)} shared pairs). "
              f"Skipping Dawid-Skene.")
        return {}

    shared_df = eval_df[eval_df["pair_key"].isin(shared_keys)].copy()
    # Discretize ratings -2..+2 → 0..4
    shared_df["label"] = shared_df["correct_L"].apply(
        lambda x: int(round(x)) + 2 if not pd.isna(x) else None
    )
    shared_df = shared_df.dropna(subset=["label"])
    shared_df["label"] = shared_df["label"].astype(int).clip(0, n_classes - 1)

    raters = sorted(shared_df["rater"].unique())
    items  = sorted(shared_df["pair_key"].unique())
    rater_idx = {r: i for i, r in enumerate(raters)}
    item_idx  = {it: i for i, it in enumerate(items)}

    n_raters = len(raters)
    n_items  = len(items)

    # Initialize: class prevalence uniform
    class_prior = np.ones(n_classes) / n_classes
    # Item-class distribution: uniform
    T = np.ones((n_items, n_classes)) / n_classes

    # Error matrices per rater: n_raters × n_classes × n_classes
    # pi[r, j, k] = P(rater r assigns k | true label j)
    pi = np.ones((n_raters, n_classes, n_classes)) / n_classes

    # Build annotation dict: item → [(rater_idx, label)]
    annotations = {i: [] for i in range(n_items)}
    for _, row in shared_df.iterrows():
        ii = item_idx[row["pair_key"]]
        ri = rater_idx[row["rater"]]
        annotations[ii].append((ri, int(row["label"])))

    prev_log_lik = -np.inf
    for iteration in range(max_iter):
        # E-step: update T (item-class posteriors)
        T_new = np.zeros((n_items, n_classes))
        for ii in range(n_items):
            for j in range(n_classes):
                log_p = np.log(class_prior[j] + 1e-10)
                for (ri, k) in annotations[ii]:
                    log_p += np.log(pi[ri, j, k] + 1e-10)
                T_new[ii, j] = log_p
            # Normalize via log-sum-exp
            T_new[ii] -= T_new[ii].max()
            T_new[ii] = np.exp(T_new[ii])
            T_new[ii] /= T_new[ii].sum()
        T = T_new

        # M-step: update class prior
        class_prior = T.mean(axis=0)
        class_prior /= class_prior.sum()

        # M-step: update error matrices
        pi_new = np.zeros_like(pi)
        for ii in range(n_items):
            for (ri, k) in annotations[ii]:
                for j in range(n_classes):
                    pi_new[ri, j, k] += T[ii, j]
        # Normalize
        for ri in range(n_raters):
            for j in range(n_classes):
                row_sum = pi_new[ri, j].sum()
                pi[ri, j] = (pi_new[ri, j] + 1e-6) / (row_sum + n_classes * 1e-6)

        # Convergence check
        log_lik = 0.0
        for ii in range(n_items):
            for j in range(n_classes):
                for (ri, k) in annotations[ii]:
                    log_lik += T[ii, j] * np.log(pi[ri, j, k] + 1e-10)
        if abs(log_lik - prev_log_lik) < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break
        prev_log_lik = log_lik

    # Compute reliability = trace(pi[r]) / n_classes (diagonal = accuracy)
    reliability = {}
    for ri, rater in enumerate(raters):
        trace = np.trace(pi[ri]) / n_classes
        reliability[rater] = float(trace)
        print(f"  {rater:20s}: D-S reliability={trace:.4f}")

    vals = np.array(list(reliability.values()), dtype=float)
    uniform_ref = 1.0 / n_classes
    if len(vals) >= 2:
        if np.std(vals) < 1e-7 or (np.max(vals) - np.min(vals) < 1e-6):
            print("  Dawid–Skene omitted: degenerate (near-identical reliabilities).")
            return {}
        if np.all(np.abs(vals - uniform_ref) < 0.02):
            print("  Dawid–Skene omitted: uniform confusion / insufficient signal.")
            return {}

    return reliability


# ═══════════════════════════════════════════════════════════════════════════════
# Combine signals into final weights
# ═══════════════════════════════════════════════════════════════════════════════

def _weighted_signal_composite(a: float, c: float, g: float, d: float,
                               use_ds: bool, alpha: float, beta: float,
                               gamma: float, delta: float) -> float:
    """Renormalize weights over signals that are non-NaN (agreement may be undefined)."""
    if use_ds:
        parts = [(alpha, a), (beta, c), (gamma, g), (delta, d)]
    else:
        parts = [(alpha, a), (beta, c), (gamma, g)]
    valid = []
    for w, v in parts:
        if v is None or (isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v))):
            continue
        valid.append((w, float(v)))
    if not valid:
        return 0.5
    s = sum(w for w, _ in valid)
    return sum(w * v for w, v in valid) / s


def compute_rater_weights(eval_df: pd.DataFrame, align_scores: dict,
                          consistency_scores: dict, agreement_scores: dict,
                          ds_reliability: dict,
                          n_eval_by_rater: dict = None) -> pd.DataFrame:
    """
    Combine proxy signals into final reliability weight.
    w_r = softmax(α*align + β*consistency + γ*agreement [+ δ*DS_reliability])
    """
    print("\n--- Computing Final Rater Weights ---")

    raters = sorted(eval_df["rater"].unique())
    n_eval_by_rater = dict(n_eval_by_rater or {})
    if not n_eval_by_rater:
        n_eval_by_rater = {str(r): int(n) for r, n in eval_df.groupby("rater").size().items()}
    use_ds = bool(ds_reliability)
    alpha, beta, gamma = ALPHA, BETA, GAMMA
    delta = 0.0

    if use_ds:
        # Redistribute 20% to DS
        alpha, beta, gamma, delta = 0.35, 0.20, 0.20, 0.25

    rows = []
    raw_scores = []
    for rater in raters:
        a = align_scores.get(rater, 0.5)
        c = consistency_scores.get(rater, 0.5)
        g = agreement_scores.get(rater, float("nan"))
        d = ds_reliability.get(rater, 0.5) if use_ds else float("nan")
        composite = _weighted_signal_composite(a, c, g, d, use_ds, alpha, beta, gamma, delta)
        n_ev = max(int(n_eval_by_rater.get(rater, 1)), 1)
        # Bug #4 fix: shrink toward 0.5 (neutral), not toward 0.
        # Old formula: composite * n/(n+N0)  → pulled low-n raters to ~0,
        # causing a ~1000:1 weight ratio after softmax (winner-takes-all).
        # New formula: 0.5 + (composite - 0.5) * n/(n+N0)  → low-n raters
        # converge to the neutral midpoint regardless of their raw composite.
        composite = 0.5 + (composite - 0.5) * (n_ev / (n_ev + N_SHRINKAGE_N0))
        agr_out = round(float(g), 4) if not (isinstance(g, float) and np.isnan(g)) else np.nan
        rows.append({"rater": rater,
                     "n_evaluations": n_ev,
                     "align_score": round(a, 4),
                     "consistency_score": round(c, 4),
                     "agreement_score": agr_out,
                     "ds_reliability": round(d, 4) if use_ds and not (isinstance(d, float) and np.isnan(d)) else None,
                     "composite_score": round(composite, 4)})
        raw_scores.append(composite)

    # Temperature-scaled softmax: dividing by T sharpens the distribution.
    # Without temperature, composite scores in [0.49, 0.60] give near-uniform
    # weights (max/uniform ≈ 1.06×).  T=0.05 gives max/uniform ≈ 2–4×,
    # which meaningfully down-weights biased raters without being extreme.
    weights = scipy_softmax(np.array(raw_scores) / SOFTMAX_TEMPERATURE)
    for i, row in enumerate(rows):
        row["weight"] = round(float(weights[i]), 6)

    weights_df = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)

    print("\nFinal rater weights:")
    print(weights_df[["rater", "composite_score", "weight"]].to_string(index=False))
    print(f"\nWeight range: [{np.nanmin(weights):.4f}, {np.nanmax(weights):.4f}]")
    print(f"Uniform weight would be: {1/len(raters):.4f}")
    print(f"Max/uniform ratio: {weights.max() / (1/len(raters)):.2f}x")

    return weights_df


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_weights(weights_df: pd.DataFrame, results_dir: Path):
    """Bar chart of rater weights with signal breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(weights_df) * 0.6)))

    # Left: final weights
    df_sorted = weights_df.sort_values("weight")
    uniform = 1 / len(weights_df)
    colors = ["#e74c3c" if w < uniform * 0.7 else
              "#2ecc71" if w > uniform * 1.3 else "#3498db"
              for w in df_sorted["weight"]]
    ax1.barh(df_sorted["rater"], df_sorted["weight"], color=colors)
    ax1.axvline(uniform, color="gray", linestyle="--", linewidth=1.2, label=f"Uniform ({uniform:.3f})")
    ax1.set_xlabel("Final Weight")
    ax1.set_title("Rater Reliability Weights\n(green=high, red=low, blue=neutral)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)

    # Right: stacked signal breakdown
    signal_cols = ["align_score", "consistency_score", "agreement_score"]
    available = [c for c in signal_cols if c in weights_df.columns]
    df_s = weights_df.set_index("rater")[available].apply(pd.to_numeric, errors="coerce").fillna(0.5)
    df_s.plot(kind="barh", stacked=False, ax=ax2, colormap="Set2")
    ax2.set_xlabel("Score [0-1]")
    ax2.set_title("Signal Breakdown per Rater", fontsize=11, fontweight="bold")
    ax2.legend(title="Signal", fontsize=8)

    plt.tight_layout()
    out = results_dir / "rater_weights.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_weight_distribution(weights_df: pd.DataFrame, results_dir: Path):
    """Pie chart of rater weight allocation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights_df)))
    wedges, texts, autotexts = ax.pie(
        weights_df["weight"],
        labels=weights_df["rater"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=140
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Rater Weight Allocation\n(bias-corrected reliability weighting)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = results_dir / "rater_weight_pie.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Stage 3: Expertise Proxy Estimation & Rater Weighting")
    print("=" * 60)

    eval_df, decoupling_df, inconsistency_df = load_data()
    n_eval_by_rater = load_rater_n_counts()

    # Compute proxy signals
    align_scores       = compute_metric_alignment(eval_df, decoupling_df)
    consistency_scores = compute_consistency_score(eval_df, inconsistency_df)
    agreement_scores   = compute_majority_agreement(eval_df)

    # Optional Dawid-Skene
    ds_reliability = dawid_skene_em(eval_df)

    # Combine into final weights
    weights_df = compute_rater_weights(eval_df, align_scores, consistency_scores,
                                       agreement_scores, ds_reliability,
                                       n_eval_by_rater=n_eval_by_rater)

    # Save
    weights_df.to_csv(RESULTS_DIR / "rater_weights.csv", index=False)

    def _json_safe(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v

    weights_json = {}
    for _, row in weights_df.iterrows():
        weights_json[row["rater"]] = {
            "weight": row["weight"],
            "composite_score": row["composite_score"],
            "align_score": row["align_score"],
            "consistency_score": row["consistency_score"],
            "agreement_score": _json_safe(row["agreement_score"]),
            "n_evaluations": int(row["n_evaluations"]) if "n_evaluations" in row and not pd.isna(row["n_evaluations"]) else None,
        }
    with open(RESULTS_DIR / "rater_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_json, f, indent=2)

    print(f"\nSaved rater_weights.csv and rater_weights.json")

    # Visualizations
    print("\nGenerating plots...")
    plot_weights(weights_df, RESULTS_DIR)
    plot_weight_distribution(weights_df, RESULTS_DIR)

    print("\n[Stage 3] DONE. Outputs in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
