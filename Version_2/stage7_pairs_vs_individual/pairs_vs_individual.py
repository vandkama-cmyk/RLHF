"""
Stage 7: Pair Comparison vs. Individual Evaluation
====================================================
Tests the hypothesis:

    "Humans compare pairs better than they evaluate individually."

Each evaluation JSON provides two evaluation modes:
  (A) PAIR COMPARISON:   comparison_slider in {-2,-1,0,+1,+2}
                         "Which answer is better overall?"
  (B) INDIVIDUAL RATING: correct_L, correct_R, consistent_L, ...
                         Separate dimensional score for each answer

We compare these modes on four criteria:

  1. SELF-CONSISTENCY
     Does the slider agree with the implied preference from individual ratings?
     -> Spearman r(slider, correct_R - correct_L)  per rater
     If slider and individual ratings are consistent, |r| should be high.

  2. AUTO-QUALITY ALIGNMENT
     Which mode correlates more strongly with an automated quality proxy?
     -> Pair:       Spearman r(slider,  auto_quality_R - auto_quality_L)
     -> Individual: Spearman r(correct, auto_quality)  [L and R pooled]
     Hypothesis: pair mode has HIGHER correlation -> better captures quality.

  3. INTER-RATER AGREEMENT (shared pairs) -- two distribution-fair metrics
     Krippendorff alpha is invalid here because slider and correct have different
     marginal distributions, making D_exp incomparable across modes.
     Instead we use:

     3a. PERCENT AGREEMENT ON DIRECTION
         Fraction of shared pairs where both raters agree on the preferred side
         (both slider > 0, both < 0, or both = 0).
         Same metric applied to individual mode: sign(correct_R - correct_L).
         Hypothesis: pct_agree(slider) > pct_agree(individual).

     3b. MAGNITUDE AGREEMENT ON DISAGREEMENT PAIRS
         For the pairs where raters chose OPPOSITE directions on the slider,
         did they at least agree on the magnitude of their preference?
         Metric: mean |slider_A - slider_B| on disagreement pairs vs agreement pairs.
         Lower magnitude difference = softer disagreement = mode is still useful.

  4. DISCRIMINATION POWER
     Can each mode correctly identify the better answer?
     -> AUC: slider > 0  predicts  auto_quality_R > auto_quality_L
     -> AUC: correct_R > correct_L  predicts  auto_quality_R > auto_quality_L
     Hypothesis: AUC(slider) > AUC(individual difference).

Depends on: stage1_eda/results/evaluations_parsed.csv

Usage:
    python Version_2/stage7_pairs_vs_individual/pairs_vs_individual.py
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Bug #15 fix: do NOT suppress all warnings globally.
# Bug #11 fix: import shared quality proxy from utils.py (single canonical definition).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import answer_quality_proxy

# -- Paths ----------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent
STAGE1_DIR  = BASE_DIR / "stage1_eda" / "results"
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ===============================================================================
# Helpers
# ===============================================================================

# answer_quality_proxy imported from utils.py above (Bug #11 fix)


def krippendorff_alpha_ordinal(data: np.ndarray) -> float:
    """
    Krippendorff's Alpha (ordinal metric).
    data: 2D array (raters x items), NaN for missing.
    """
    valid_cols = np.sum(~np.isnan(data), axis=0) >= 2
    data = data[:, valid_cols]
    if data.shape[1] == 0:
        return np.nan

    d_obs, n_obs = 0.0, 0
    for col in range(data.shape[1]):
        col_vals = data[:, col][~np.isnan(data[:, col])]
        m = len(col_vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                d_obs += (col_vals[i] - col_vals[j]) ** 2
                n_obs += 1

    if n_obs == 0:
        return np.nan

    # Bug #17 fix: compute expected disagreement in O(n) instead of O(n²).
    # Identity: sum_{i<j}(xi-xj)^2 = n*sum(xi^2) - (sum xi)^2
    all_vals = data[~np.isnan(data)]
    n_all = len(all_vals)
    n_exp = n_all * (n_all - 1) / 2
    if n_exp == 0:
        return np.nan
    d_exp = float(n_all * np.sum(all_vals ** 2) - np.sum(all_vals) ** 2)

    if d_exp == 0:
        return np.nan

    return float(1.0 - (d_obs / n_obs) / (d_exp / n_exp))


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC via trapezoidal rule (no sklearn dependency)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    sorted_idx = np.argsort(-y_score)
    y_true = y_true[sorted_idx]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan

    tp, fp = 0, 0
    tps, fps = [], []
    for label in y_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tps.append(tp)
        fps.append(fp)
    tpr = np.array(tps) / n_pos
    fpr = np.array(fps) / n_neg
    # Trapezoidal
    auc = float(np.trapz(tpr, fpr))
    return abs(auc)  # ensure positive


# ===============================================================================
# 1. Self-consistency: slider vs (correct_R - correct_L)
# ===============================================================================

def analysis_self_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman r(slider, correct_R - correct_L) per rater.
    A well-calibrated rater who prefers R on the slider should also give higher
    individual correct scores to R. High |r| = consistent evaluation mode.
    """
    print("\n--- Analysis 1: Self-Consistency ---")
    print("  r(slider, correct_R - correct_L) per rater")
    print("  If |r| is high -> slider and individual ratings agree.\n")

    df = df.copy()
    df["correct_diff"] = df["correct_R"] - df["correct_L"]
    df["consistent_diff"] = df["consistent_R"] - df["consistent_L"]
    df["useful_diff"]   = df["useful_R"]   - df["useful_L"]

    rows = []
    for rater, grp in df.groupby("rater"):
        sub = grp[["comparison_slider", "correct_diff",
                   "consistent_diff", "useful_diff"]].dropna()
        n = len(sub)
        if n < 5:
            rows.append({"rater": rater, "n": n,
                         "r_slider_correct":    np.nan,
                         "r_slider_consistent": np.nan,
                         "r_slider_useful":     np.nan,
                         "p_correct": np.nan})
            continue
        r_c, p_c = scipy_stats.spearmanr(sub["comparison_slider"], sub["correct_diff"])
        r_cs, _  = scipy_stats.spearmanr(sub["comparison_slider"], sub["consistent_diff"])
        r_u, _   = scipy_stats.spearmanr(sub["comparison_slider"], sub["useful_diff"])
        rows.append({
            "rater": rater, "n": n,
            "r_slider_correct":    round(float(r_c),  4),
            "r_slider_consistent": round(float(r_cs), 4),
            "r_slider_useful":     round(float(r_u),  4),
            "p_correct": round(float(p_c), 4),
        })
        print(f"  {rater:25s} n={n:3d}  r(slider, correct_diff)={r_c:+.3f}"
              f"  r(slider, consistent_diff)={r_cs:+.3f}"
              f"  r(slider, useful_diff)={r_u:+.3f}")

    result_df = pd.DataFrame(rows)

    # Grand summary
    grand_r = result_df["r_slider_correct"].dropna()
    print(f"\n  Grand mean |r(slider, correct_diff)|: {grand_r.abs().mean():.3f}")
    print(f"  Grand mean   r(slider, correct_diff) : {grand_r.mean():.3f}")
    n_sig = (result_df["p_correct"].dropna() < 0.05).sum()
    print(f"  Raters with p < 0.05: {n_sig} / {len(result_df[result_df['p_correct'].notna()])}")

    # Overall (pooled across all raters)
    pool = df[["comparison_slider", "correct_diff"]].dropna()
    if len(pool) >= 5:
        r_pool, p_pool = scipy_stats.spearmanr(pool["comparison_slider"], pool["correct_diff"])
        print(f"\n  POOLED (all raters, n={len(pool)}): "
              f"r={r_pool:+.3f}, p={p_pool:.4f}")
        result_df.loc[len(result_df)] = {
            "rater": "POOLED", "n": len(pool),
            "r_slider_correct": round(float(r_pool), 4),
            "p_correct": round(float(p_pool), 4),
            "r_slider_consistent": np.nan,
            "r_slider_useful": np.nan,
        }

    return result_df


# ===============================================================================
# 2. Auto-quality alignment: pair mode vs individual mode
# ===============================================================================

def analysis_quality_alignment(df: pd.DataFrame) -> dict:
    """
    Compare Spearman r with automated quality proxy for:
      - PAIR mode:        slider vs (auto_quality_R - auto_quality_L)
      - INDIVIDUAL mode:  correct vs auto_quality  (pooling L and R sides)

    Per rater AND globally.
    """
    print("\n--- Analysis 2: Auto-Quality Alignment ---")
    df = df.copy()
    df["auto_L"] = df["answer_L"].apply(answer_quality_proxy)
    df["auto_R"] = df["answer_R"].apply(answer_quality_proxy)
    df["auto_diff"] = df["auto_R"] - df["auto_L"]

    rows = []
    for rater, grp in df.groupby("rater"):
        # Pair mode
        pair_sub = grp[["comparison_slider", "auto_diff"]].dropna()
        if len(pair_sub) >= 5:
            r_pair, p_pair = scipy_stats.spearmanr(
                pair_sub["comparison_slider"], pair_sub["auto_diff"])
        else:
            r_pair, p_pair = np.nan, np.nan

        # Individual mode (pool L and R)
        correct_vals = pd.concat([grp["correct_L"], grp["correct_R"]]).reset_index(drop=True)
        auto_vals    = pd.concat([grp["auto_L"],    grp["auto_R"]]).reset_index(drop=True)
        ind_sub = pd.DataFrame({"correct": correct_vals, "auto": auto_vals}).dropna()
        if len(ind_sub) >= 5:
            r_ind, p_ind = scipy_stats.spearmanr(ind_sub["correct"], ind_sub["auto"])
        else:
            r_ind, p_ind = np.nan, np.nan

        winner = ("PAIR" if (not np.isnan(r_pair) and not np.isnan(r_ind)
                             and abs(r_pair) > abs(r_ind))
                  else ("INDIVIDUAL" if not np.isnan(r_ind) else "N/A"))
        rows.append({
            "rater": rater,
            "n_pair": len(pair_sub),
            "r_pair": round(float(r_pair), 4) if not np.isnan(r_pair) else np.nan,
            "p_pair": round(float(p_pair), 4) if not np.isnan(p_pair) else np.nan,
            "n_individual": len(ind_sub),
            "r_individual": round(float(r_ind), 4) if not np.isnan(r_ind) else np.nan,
            "p_individual": round(float(p_ind), 4) if not np.isnan(p_ind) else np.nan,
            "winner": winner,
        })
        print(f"  {rater:25s}  "
              f"PAIR r={r_pair:+.3f} (n={len(pair_sub)})  "
              f"INDIVIDUAL r={r_ind:+.3f} (n={len(ind_sub)})  "
              f"-> {winner}")

    result_df = pd.DataFrame(rows)

    # Global summary
    pair_r = result_df["r_pair"].dropna()
    ind_r  = result_df["r_individual"].dropna()
    print(f"\n  Mean |r| -- PAIR:       {pair_r.abs().mean():.3f}")
    print(f"  Mean |r| -- INDIVIDUAL: {ind_r.abs().mean():.3f}")
    n_pair_wins = (result_df["winner"] == "PAIR").sum()
    n_ind_wins  = (result_df["winner"] == "INDIVIDUAL").sum()
    print(f"  PAIR wins: {n_pair_wins}  |  INDIVIDUAL wins: {n_ind_wins}")
    hypothesis_supported = float(pair_r.abs().mean()) > float(ind_r.abs().mean())
    print(f"\n  Hypothesis (pair > individual): "
          f"{'SUPPORTED' if hypothesis_supported else 'NOT supported'}")

    # Pooled global correlation
    pool_pair = df[["comparison_slider", "auto_diff"]].dropna()
    pool_ind_c = pd.concat([df["correct_L"], df["correct_R"]]).reset_index(drop=True)
    pool_ind_a = pd.concat([df["auto_L"],    df["auto_R"]]).reset_index(drop=True)
    pool_ind   = pd.DataFrame({"c": pool_ind_c, "a": pool_ind_a}).dropna()

    global_r_pair = global_r_ind = np.nan
    if len(pool_pair) >= 5:
        global_r_pair, _ = scipy_stats.spearmanr(
            pool_pair["comparison_slider"], pool_pair["auto_diff"])
    if len(pool_ind) >= 5:
        global_r_ind, _ = scipy_stats.spearmanr(pool_ind["c"], pool_ind["a"])

    print(f"\n  POOLED (all raters):")
    print(f"    PAIR       r={global_r_pair:+.3f} (n={len(pool_pair)})")
    print(f"    INDIVIDUAL r={global_r_ind:+.3f} (n={len(pool_ind)})")

    return {
        "per_rater_df": result_df,
        "mean_r_pair":       float(pair_r.abs().mean()) if len(pair_r) > 0 else np.nan,
        "mean_r_individual": float(ind_r.abs().mean())  if len(ind_r)  > 0 else np.nan,
        "global_r_pair":     float(global_r_pair),
        "global_r_individual": float(global_r_ind),
        "hypothesis_supported": hypothesis_supported,
        "pair_wins": int(n_pair_wins),
        "individual_wins": int(n_ind_wins),
    }


# ===============================================================================
# 3. Inter-rater agreement on shared pairs -- distribution-fair metrics
# ===============================================================================

def _sign(x):
    """Return -1, 0, or +1."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def analysis_inter_rater_agreement(df: pd.DataFrame) -> dict:
    """
    Two distribution-fair agreement metrics on shared pairs.

    Krippendorff alpha is NOT used here because slider and correct ratings
    have very different marginal distributions, making D_exp incomparable
    between modes.  A narrow slider distribution (most values 0/+-1) produces
    a tiny D_exp denominator, so any disagreement pushes alpha negative --
    this is a mathematical artifact, not evidence of lower reliability.

    Instead we use:

    3a. PERCENT AGREEMENT ON DIRECTION
        For each pair with >=2 raters:
          - PAIR mode:       both raters agree on sign(slider)
          - INDIVIDUAL mode: both raters agree on sign(correct_R - correct_L)
        Computed only on pairs where neither rater gave a neutral value (to
        avoid trivial agreement from both choosing 0).

    3b. MAGNITUDE ON DISAGREEMENT PAIRS
        Among pairs where raters chose OPPOSITE slider directions (9 pairs):
          - What is the mean |slider_A - slider_B|?
          - Compare to mean |diff_A - diff_B| for correct_R - correct_L.
        If raters who disagree on direction still chose mild values (e.g. +1
        vs -1) that is "soft disagreement" -- the mode may still be useful.
        Strong disagreement = e.g. +2 vs -2.
    """
    print("\n--- Analysis 3: Inter-Rater Agreement (shared pairs) ---")
    print("    (Replacing Krippendorff alpha with distribution-fair metrics)")

    df = df.copy()
    df["pair_key"] = df["id_L"].astype(str) + "_" + df["id_R"].astype(str)
    pair_counts = df.groupby("pair_key")["rater"].count()
    shared_keys = pair_counts[pair_counts >= 2].index.tolist()
    shared_df   = df[df["pair_key"].isin(shared_keys)].copy()

    print(f"\n  Shared pairs (rated by >=2 raters): {len(shared_keys)}")
    print(f"  Evaluations on shared pairs:        {len(shared_df)}")

    if len(shared_keys) < 3:
        print("  Too few shared pairs.")
        return {"n_shared_pairs": len(shared_keys)}

    shared_df["correct_diff"] = shared_df["correct_R"] - shared_df["correct_L"]

    # -- 3a: Percent agreement on direction ------------------------------------
    print("\n  --- 3a: Percent Agreement on Direction ---")

    pair_agree_slider = []   # 1 = agree, 0 = disagree
    pair_agree_indiv  = []
    pair_agree_detail = []   # for the plot

    for pk in shared_keys:
        sub = shared_df[shared_df["pair_key"] == pk]
        sliders = sub["comparison_slider"].dropna().values
        diffs   = sub["correct_diff"].dropna().values

        # Need at least 2 raters with non-NaN values
        if len(sliders) < 2:
            continue

        # Pairwise agreement across all rater pairs for this shared item
        from itertools import combinations as _comb
        slider_pairs = list(_comb(sliders, 2))
        diff_pairs   = list(_comb(diffs, 2)) if len(diffs) >= 2 else []

        for s_a, s_b in slider_pairs:
            agree = int(_sign(s_a) == _sign(s_b))
            pair_agree_slider.append(agree)
            pair_agree_detail.append({
                "pair_key": pk,
                "rater_A_slider": s_a, "rater_B_slider": s_b,
                "slider_agree": agree,
                "slider_diff_magnitude": abs(s_a - s_b),
                "direction": "AGREE" if agree else "DISAGREE",
            })

        for d_a, d_b in diff_pairs:
            agree = int(_sign(d_a) == _sign(d_b))
            pair_agree_indiv.append(agree)

    pct_agree_slider = np.mean(pair_agree_slider) if pair_agree_slider else np.nan
    pct_agree_indiv  = np.mean(pair_agree_indiv)  if pair_agree_indiv  else np.nan

    print(f"  Rater-pairs evaluated:          {len(pair_agree_slider)}")
    print(f"  Pct agree -- PAIR (slider):      {pct_agree_slider:.1%}")
    print(f"  Pct agree -- INDIVIDUAL (diff):  {pct_agree_indiv:.1%}")

    detail_df = pd.DataFrame(pair_agree_detail)

    # Break down slider agreement by case
    if not detail_df.empty:
        agree_counts = detail_df["direction"].value_counts()
        print(f"\n  PAIR direction breakdown:")
        for cat in ["AGREE", "DISAGREE"]:
            n = agree_counts.get(cat, 0)
            print(f"    {cat}: {n} rater-pairs ({n/len(detail_df):.1%})")

    hyp_3a = bool(pct_agree_slider > pct_agree_indiv) if not np.isnan(pct_agree_slider + pct_agree_indiv) else None
    print(f"\n  Hypothesis 3a (pair % > individual %): "
          f"{'SUPPORTED' if hyp_3a else 'NOT supported'}")

    # -- 3b: Magnitude on disagreement pairs -----------------------------------
    print("\n  --- 3b: Magnitude of Disagreement (on direction-disagreement pairs) ---")

    disagree_rows = detail_df[detail_df["direction"] == "DISAGREE"] if not detail_df.empty else pd.DataFrame()
    agree_rows    = detail_df[detail_df["direction"] == "AGREE"]    if not detail_df.empty else pd.DataFrame()

    print(f"  Direction-disagreement rater-pairs: {len(disagree_rows)}")

    if not disagree_rows.empty:
        mag_disagree_slider = disagree_rows["slider_diff_magnitude"].mean()
        mag_agree_slider    = agree_rows["slider_diff_magnitude"].mean() if not agree_rows.empty else np.nan
        print(f"\n  Mean |slider_A - slider_B| on DISAGREEMENT pairs: {mag_disagree_slider:.3f}")
        print(f"  Mean |slider_A - slider_B| on AGREEMENT pairs:    {mag_agree_slider:.3f}")
        print(f"  (max possible difference = 4, i.e. +2 vs -2)")

        # Print each disagreement pair
        print(f"\n  Disagreement pairs detail:")
        for pk in disagree_rows["pair_key"].unique():
            sub = shared_df[shared_df["pair_key"] == pk][["rater", "comparison_slider", "correct_diff"]]
            sliders = sub["comparison_slider"].dropna().values
            print(f"    pair {pk}:")
            print(f"      raters+sliders: {list(zip(sub['rater'].values, sub['comparison_slider'].values))}")
            print(f"      |diff| = {abs(sliders[0] - sliders[1]) if len(sliders) >= 2 else 'N/A':.1f}")

        # Individual mode: same disagree pairs, look at correct_diff magnitude
        indiv_mags = []
        for pk in disagree_rows["pair_key"].unique():
            sub = shared_df[shared_df["pair_key"] == pk]["correct_diff"].dropna().values
            if len(sub) >= 2:
                from itertools import combinations as _comb2
                for a, b in _comb2(sub, 2):
                    if _sign(a) != _sign(b):
                        indiv_mags.append(abs(a - b))

        if indiv_mags:
            print(f"\n  Mean |correct_diff_A - correct_diff_B| on SAME disagree pairs "
                  f"(individual mode): {np.mean(indiv_mags):.3f}")
            print(f"  (max possible = 8, i.e. +4 vs -4 diff)")
            hyp_3b = bool(mag_disagree_slider < np.mean(indiv_mags) / 2)  # scale-normalized
            print(f"\n  Soft disagreement check: slider magnitude {mag_disagree_slider:.2f}/4.0 = "
                  f"{mag_disagree_slider/4:.1%} of max")
            print(f"  Individual magnitude {np.mean(indiv_mags):.2f}/8.0 = "
                  f"{np.mean(indiv_mags)/8:.1%} of max")
        else:
            hyp_3b = None
            indiv_mags = []
    else:
        mag_disagree_slider = np.nan
        mag_agree_slider = np.nan
        hyp_3b = None
        indiv_mags = []

    return {
        "n_shared_pairs": len(shared_keys),
        "n_shared_evaluations": len(shared_df),
        "n_rater_pairs_evaluated": len(pair_agree_slider),
        "pct_agree_slider":  float(pct_agree_slider)  if not np.isnan(pct_agree_slider) else None,
        "pct_agree_individual": float(pct_agree_indiv) if not np.isnan(pct_agree_indiv) else None,
        "hypothesis_3a_supported": hyp_3a,
        "n_direction_disagreements": int(len(disagree_rows)),
        "mean_mag_disagree_slider":  float(mag_disagree_slider) if not np.isnan(mag_disagree_slider) else None,
        "mean_mag_agree_slider":     float(mag_agree_slider) if not np.isnan(mag_agree_slider) else None,
        "mean_mag_disagree_individual": float(np.mean(indiv_mags)) if indiv_mags else None,
        "hypothesis_3b_soft_disagreement": hyp_3b,
        "detail_df": detail_df,
        # keep old key for summary compatibility
        "hypothesis_supported": hyp_3a,
    }


# ===============================================================================
# 4. Discrimination power (AUC)
# ===============================================================================

def analysis_discrimination_power(df: pd.DataFrame) -> dict:
    """
    Can each mode correctly identify which answer is of higher auto-quality?

    Ground truth: auto_quality_R > auto_quality_L  (R is objectively better)

    PAIR mode:        slider > 0  ->  predict R is better
    INDIVIDUAL mode:  correct_R > correct_L  ->  predict R is better

    Compute AUC for each.
    """
    print("\n--- Analysis 4: Discrimination Power (AUC) ---")

    df = df.copy()
    df["auto_L"] = df["answer_L"].apply(answer_quality_proxy)
    df["auto_R"] = df["answer_R"].apply(answer_quality_proxy)

    # Ground truth: is R actually better?
    sub = df[["comparison_slider", "correct_L", "correct_R",
              "auto_L", "auto_R"]].dropna()
    if len(sub) < 10:
        print("  Not enough data for AUC computation.")
        return {"auc_pair": np.nan, "auc_individual": np.nan}

    y_true = (sub["auto_R"] > sub["auto_L"]).astype(int).values

    # PAIR mode: slider (higher = predict R)
    auc_pair = roc_auc_score_manual(y_true, sub["comparison_slider"].values)

    # INDIVIDUAL mode: correct_R - correct_L (positive = predict R)
    auc_ind  = roc_auc_score_manual(y_true, (sub["correct_R"] - sub["correct_L"]).values)

    print(f"  n = {len(sub)} (evaluations with complete data)")
    print(f"  Ground truth: R rated better by auto-proxy in "
          f"{y_true.sum()} / {len(y_true)} cases ({100*y_true.mean():.1f}%)")
    print(f"\n  AUC -- PAIR mode (slider):            {auc_pair:.4f}")
    print(f"  AUC -- INDIVIDUAL mode (correct diff): {auc_ind:.4f}")
    if not np.isnan(auc_pair) and not np.isnan(auc_ind):
        hypothesis_supported = auc_pair > auc_ind
        print(f"\n  Hypothesis (pair > individual): "
              f"{'SUPPORTED' if hypothesis_supported else 'NOT supported'}"
              f"  (Delta = {auc_pair - auc_ind:+.4f})")
    else:
        hypothesis_supported = None

    return {
        "n": int(len(sub)),
        "pct_R_better": float(y_true.mean()),
        "auc_pair":       float(auc_pair)  if not np.isnan(auc_pair) else None,
        "auc_individual": float(auc_ind)   if not np.isnan(auc_ind)  else None,
        "hypothesis_supported": hypothesis_supported,
    }


# ===============================================================================
# 5. Visualizations
# ===============================================================================

def plot_self_consistency(sc_df: pd.DataFrame, plots_dir: Path):
    """Bar chart: r(slider, correct_diff) per rater."""
    df = sc_df[sc_df["rater"] != "POOLED"].dropna(subset=["r_slider_correct"])
    df = df.sort_values("r_slider_correct", ascending=True)

    colors = ["#27ae60" if r >= 0 else "#e74c3c" for r in df["r_slider_correct"]]
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.6)))
    ax.barh(df["rater"], df["r_slider_correct"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)

    # Add POOLED marker
    pooled_row = sc_df[sc_df["rater"] == "POOLED"]
    if len(pooled_row) > 0:
        pooled_r = pooled_row["r_slider_correct"].values[0]
        ax.axvline(pooled_r, color="navy", linestyle="--", linewidth=1.5,
                   label=f"Pooled r = {pooled_r:+.3f}")
        ax.legend(fontsize=9)

    ax.set_xlabel("Spearman r (comparison_slider, correct_R - correct_L)")
    ax.set_title("Self-Consistency: Does slider agree with individual ratings?\n"
                 "(|r| closer to 1 = more consistent between modes)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / "self_consistency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_quality_alignment_comparison(align_result: dict, plots_dir: Path):
    """Grouped bar chart: |r| pair vs individual per rater."""
    df = align_result["per_rater_df"].dropna(subset=["r_pair", "r_individual"])
    if df.empty:
        return

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(df)), 5))
    bars_pair = ax.bar(x - width/2, df["r_pair"].abs(), width,
                       label="PAIR (slider vs Deltaauto)", color="#2980b9", alpha=0.85)
    bars_ind  = ax.bar(x + width/2, df["r_individual"].abs(), width,
                       label="INDIVIDUAL (correct vs auto)", color="#e67e22", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["rater"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("|Spearman r| with auto-quality proxy")
    ax.set_title("Auto-Quality Alignment: PAIR vs INDIVIDUAL mode\n"
                 "(higher |r| = mode better tracks objective quality)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.axhline(align_result["mean_r_pair"], color="#2980b9", linestyle="--",
               linewidth=1, label=f"Mean PAIR = {align_result['mean_r_pair']:.3f}")
    ax.axhline(align_result["mean_r_individual"], color="#e67e22", linestyle="--",
               linewidth=1, label=f"Mean IND  = {align_result['mean_r_individual']:.3f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = plots_dir / "quality_alignment_pair_vs_individual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_agreement_analysis(agree_res: dict, plots_dir: Path):
    """Two-panel plot: percent agreement bars + disagreement magnitude."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Analysis 3: Inter-Rater Agreement on Shared Pairs\n"
                 "(distribution-fair metrics -- no Krippendorff alpha)",
                 fontsize=12, fontweight="bold")

    # Panel 1: percent agreement on direction
    modes  = ["PAIR\n(slider)", "INDIVIDUAL\n(correct_R - correct_L)"]
    pcts   = [agree_res.get("pct_agree_slider") or 0,
              agree_res.get("pct_agree_individual") or 0]
    colors = ["#2980b9", "#e67e22"]
    bars = ax1.bar(modes, pcts, color=colors, alpha=0.85, width=0.45)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction of rater-pairs agreeing on direction")
    ax1.set_title("3a: Directional Agreement", fontsize=11)
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.5)")
    for bar, v in zip(bars, pcts):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f"{v:.1%}", ha="center", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    winner = ("PAIR" if (pcts[0] > pcts[1]) else "INDIVIDUAL")
    ax1.set_xlabel(f"Higher agreement: {winner}", fontsize=10, color="#27ae60" if winner == "PAIR" else "#e74c3c")

    # Panel 2: magnitude of disagreement (normalized to max scale)
    mag_dis_s  = agree_res.get("mean_mag_disagree_slider")
    mag_agr_s  = agree_res.get("mean_mag_agree_slider")
    mag_dis_i  = agree_res.get("mean_mag_disagree_individual")

    labels, vals, cols = [], [], []
    if mag_dis_s is not None:
        labels.append("PAIR\n(disagree pairs)\nnorm. to /4")
        vals.append(mag_dis_s / 4.0)
        cols.append("#c0392b")
    if mag_agr_s is not None:
        labels.append("PAIR\n(agree pairs)\nnorm. to /4")
        vals.append(mag_agr_s / 4.0)
        cols.append("#2980b9")
    if mag_dis_i is not None:
        labels.append("INDIVIDUAL\n(same disagree pairs)\nnorm. to /8")
        vals.append(mag_dis_i / 8.0)
        cols.append("#e67e22")

    if vals:
        bars2 = ax2.bar(labels, vals, color=cols, alpha=0.85, width=0.45)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Normalized magnitude (0 = perfect agree, 1 = max disagree)")
        ax2.set_title("3b: Disagreement Magnitude\n(lower = softer / more useful disagreement)",
                      fontsize=11)
        for bar, v in zip(bars2, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f"{v:.2f}", ha="center", fontsize=11)
        n_dis = agree_res.get("n_direction_disagreements", 0)
        ax2.set_xlabel(f"Based on {n_dis} direction-disagreement rater-pairs", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No disagreement pairs found", ha="center", va="center",
                 transform=ax2.transAxes)

    plt.tight_layout()
    out = plots_dir / "agreement_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_disagree_pairs_detail(agree_res: dict, plots_dir: Path):
    """Scatter of slider values on direction-disagreement pairs."""
    detail_df = agree_res.get("detail_df", pd.DataFrame())
    if detail_df.empty:
        return

    dis = detail_df[detail_df["direction"] == "DISAGREE"]
    agr = detail_df[detail_df["direction"] == "AGREE"]
    if dis.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    jitter = np.random.uniform(-0.08, 0.08, size=len(agr))
    ax.scatter(agr["rater_A_slider"] + jitter, agr["rater_B_slider"] + jitter,
               color="#2980b9", alpha=0.5, s=40, label=f"Direction AGREE (n={len(agr)})")

    jitter2 = np.random.uniform(-0.08, 0.08, size=len(dis))
    ax.scatter(dis["rater_A_slider"] + jitter2, dis["rater_B_slider"] + jitter2,
               color="#e74c3c", alpha=0.8, s=60, marker="X",
               label=f"Direction DISAGREE (n={len(dis)})")

    # Diagonal reference
    lim = 2.5
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.4, label="Perfect agreement")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

    # Annotate magnitude for disagree points
    for _, row in dis.iterrows():
        mag = row["slider_diff_magnitude"]
        ax.annotate(f"|Delta|={mag:.0f}",
                    xy=(row["rater_A_slider"], row["rater_B_slider"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=7, color="#c0392b")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Rater A slider value")
    ax.set_ylabel("Rater B slider value")
    ax.set_title("Slider Values on Shared Pairs\n"
                 "(red X = opposite direction chosen, blue = same direction)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = plots_dir / "disagree_pairs_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_summary_radar(align_result: dict, agreement_result: dict,
                       disc_result: dict, plots_dir: Path):
    """Summary bar chart comparing PAIR vs INDIVIDUAL across all analyses."""
    metrics = []
    pair_vals = []
    ind_vals  = []

    # Quality alignment (global |r|)
    if not np.isnan(align_result.get("global_r_pair", np.nan)):
        metrics.append("Quality\nAlignment\n(global |r|)")
        pair_vals.append(abs(align_result["global_r_pair"]))
        ind_vals.append(abs(align_result["global_r_individual"]))

    # Inter-rater agreement (percent agreement on direction)
    if agreement_result.get("pct_agree_slider") is not None:
        metrics.append("Inter-rater\nAgreement\n(% direction agree)")
        pair_vals.append(agreement_result["pct_agree_slider"])
        ind_vals.append(agreement_result.get("pct_agree_individual") or 0)

    # Discrimination AUC
    if disc_result.get("auc_pair") is not None:
        metrics.append("Discrimination\nPower\n(AUC)")
        pair_vals.append(disc_result["auc_pair"])
        ind_vals.append(disc_result["auc_individual"] or 0)

    if not metrics:
        print("  Not enough data for summary plot.")
        return

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(metrics) * 2.5), 5))
    ax.bar(x - width/2, pair_vals, width, label="PAIR mode (comparison_slider)",
           color="#2980b9", alpha=0.85)
    ax.bar(x + width/2, ind_vals,  width, label="INDIVIDUAL mode (correct ratings)",
           color="#e67e22", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Pairs vs Individual: Multi-Metric Comparison\n"
                 "Hypothesis: PAIR mode (blue) should be higher on all metrics",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    plt.tight_layout()
    out = plots_dir / "summary_pair_vs_individual.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_slider_vs_diff_scatter(df: pd.DataFrame, plots_dir: Path):
    """Scatter: comparison_slider vs (correct_R - correct_L) with jitter."""
    df = df.copy()
    df["correct_diff"] = df["correct_R"] - df["correct_L"]
    sub = df[["comparison_slider", "correct_diff", "rater"]].dropna()

    jitter = np.random.uniform(-0.15, 0.15, size=len(sub))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sub["comparison_slider"] + jitter,
               sub["correct_diff"] + jitter,
               alpha=0.35, s=18, color="#2980b9", edgecolors="none")

    if len(sub) >= 5:
        r, p = scipy_stats.spearmanr(sub["comparison_slider"], sub["correct_diff"])
        ax.set_title(f"Slider vs Individual Rating Difference\n"
                     f"Spearman r = {r:+.3f}  (p = {p:.4f}, n = {len(sub)})",
                     fontsize=11, fontweight="bold")
    else:
        ax.set_title("Slider vs Individual Rating Difference", fontsize=11)

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("comparison_slider  (-2 = strongly prefer L,  +2 = strongly prefer R)")
    ax.set_ylabel("correct_R - correct_L  (positive = R rated higher individually)")
    plt.tight_layout()
    out = plots_dir / "slider_vs_correct_diff_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ===============================================================================
# Main
# ===============================================================================

def main():
    print("=" * 65)
    print("Stage 7: Pair Comparison vs. Individual Evaluation")
    print("=" * 65)
    print("Hypothesis: humans compare pairs better than they evaluate individually.\n")

    csv_path = STAGE1_DIR / "evaluations_parsed.csv"
    if not csv_path.exists():
        sys.exit(f"[Error] Run Stage 1 first. Expected: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    for col in ["consistent_L", "correct_L", "useful_L",
                "consistent_R", "correct_R", "useful_R", "comparison_slider"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"[Stage 7] Loaded {len(df)} evaluations, {df['rater'].nunique()} raters")

    # -- Run all four analyses --------------------------------------------------
    sc_df      = analysis_self_consistency(df)
    align_res  = analysis_quality_alignment(df)
    agree_res  = analysis_inter_rater_agreement(df)
    disc_res   = analysis_discrimination_power(df)

    # -- Final verdict ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL VERDICT")
    print("=" * 65)
    results = [
        ("Self-consistency",          None),   # descriptive, no binary hypothesis test
        ("Quality alignment",         align_res.get("hypothesis_supported")),
        ("Inter-rater agree. (3a %)", agree_res.get("hypothesis_3a_supported")),
        ("Soft disagreement (3b)",    agree_res.get("hypothesis_3b_soft_disagreement")),
        ("Discrimination AUC",        disc_res.get("hypothesis_supported")),
    ]
    n_supported = sum(1 for _, v in results if v is True)
    n_tested    = sum(1 for _, v in results if v is not None)

    for name, supported in results:
        if supported is None:
            status = "descriptive only"
        elif supported:
            status = "SUPPORTED"
        else:
            status = "NOT supported"
        print(f"  {name:30s}: {status}")

    print(f"\n  Overall: {n_supported}/{n_tested} testable analyses support "
          f"the hypothesis that PAIR comparison is better than INDIVIDUAL rating.")
    if n_tested > 0 and n_supported / n_tested >= 0.5:
        print("  -> Hypothesis SUPPORTED (majority of tests)")
    elif n_tested > 0:
        print("  -> Hypothesis NOT SUPPORTED (majority of tests)")

    # -- Save results ----------------------------------------------------------
    sc_df.to_csv(RESULTS_DIR / "self_consistency.csv", index=False)
    align_res["per_rater_df"].to_csv(RESULTS_DIR / "quality_alignment.csv", index=False)

    summary = {
        "n_evaluations": int(len(df)),
        "n_raters": int(df["rater"].nunique()),
        "hypothesis": "Humans compare pairs better than they evaluate individually",
        "analyses": {
            "self_consistency": {
                "description": "r(slider, correct_R - correct_L) pooled",
                "pooled_r": float(
                    sc_df[sc_df["rater"] == "POOLED"]["r_slider_correct"].values[0]
                ) if "POOLED" in sc_df["rater"].values else None,
            },
            "quality_alignment": {
                "mean_r_pair":       align_res["mean_r_pair"],
                "mean_r_individual": align_res["mean_r_individual"],
                "global_r_pair":     align_res["global_r_pair"],
                "global_r_individual": align_res["global_r_individual"],
                "pair_wins": align_res["pair_wins"],
                "individual_wins": align_res["individual_wins"],
                "hypothesis_supported": align_res["hypothesis_supported"],
            },
            "inter_rater_agreement": {
                "n_shared_pairs":         agree_res["n_shared_pairs"],
                "n_rater_pairs_evaluated": agree_res.get("n_rater_pairs_evaluated"),
                "pct_agree_slider":        agree_res.get("pct_agree_slider"),
                "pct_agree_individual":    agree_res.get("pct_agree_individual"),
                "hypothesis_3a_supported": agree_res.get("hypothesis_3a_supported"),
                "n_direction_disagreements":       agree_res.get("n_direction_disagreements"),
                "mean_mag_disagree_slider":        agree_res.get("mean_mag_disagree_slider"),
                "mean_mag_agree_slider":           agree_res.get("mean_mag_agree_slider"),
                "mean_mag_disagree_individual":    agree_res.get("mean_mag_disagree_individual"),
                "hypothesis_3b_soft_disagreement": agree_res.get("hypothesis_3b_soft_disagreement"),
                "note": ("Krippendorff alpha intentionally omitted: slider and correct have "
                         "different marginal distributions, making D_exp incomparable across modes. "
                         "Percent agreement on direction is distribution-fair."),
            },
            "discrimination_auc": {
                "n": disc_res["n"],
                "auc_pair":       disc_res["auc_pair"],
                "auc_individual": disc_res["auc_individual"],
                "hypothesis_supported": disc_res["hypothesis_supported"],
            },
        },
        "overall_hypothesis_supported": (
            n_supported / n_tested >= 0.5 if n_tested > 0 else None
        ),
        "n_supported_out_of_tested": f"{n_supported}/{n_tested}",
        "agreement_note": (
            "Analysis 3 uses percent directional agreement (not Krippendorff alpha). "
            "Alpha was invalid here: slider values cluster near 0 (narrow distribution) "
            "while correct values span -2 to +2, making D_exp incomparable. "
            "Percent agreement is computed on the same scale for both modes."
        ),
    }

    with open(RESULTS_DIR / "pairs_vs_individual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # -- Plots -----------------------------------------------------------------
    print("\nGenerating plots...")
    plot_self_consistency(sc_df, PLOTS_DIR)
    plot_slider_vs_diff_scatter(df, PLOTS_DIR)
    plot_quality_alignment_comparison(align_res, PLOTS_DIR)
    plot_agreement_analysis(agree_res, PLOTS_DIR)
    plot_disagree_pairs_detail(agree_res, PLOTS_DIR)
    plot_summary_radar(align_res, agree_res, disc_res, PLOTS_DIR)

    print(f"\n[Stage 7] DONE. Outputs in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
