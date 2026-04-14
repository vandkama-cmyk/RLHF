"""
Stage 6: L / R Sides as Separate Virtual Raters Experiment
===========================================================
Each evaluation JSON contains two sides (Left = L, Right = R).
Normally we treat both as one evaluator's output.

This experiment asks: what if we treat L-side ratings and R-side ratings
as coming from *distinct virtual raters*?

  original_rater  → "Rater_L"  rating answer_L with (consistent_L, correct_L, useful_L)
                    "Rater_R"  rating answer_R with (consistent_R, correct_R, useful_R)

Result: 307 evaluations × 2 = **614 total evaluations** (instead of 307).

Research questions:
  1. Do L-virtual-raters and R-virtual-raters show systematic leniency differences?
     (Is one side of a pair systematically rated higher — positional effect?)
  2. With doubled data, which bias signals strengthen?
  3. Spearman r(correct, auto_quality) per virtual rater — is L-side better calibrated?
  4. Dimension consistency: are L-ratings more internally consistent than R-ratings?

Hypothesis check: treating L and R as independent raters is an approximation
(within a pair, ratings are *relative*). But the expanded dataset gives more
statistical power for bias detection.

Depends on: stage1_eda/results/evaluations_parsed.csv

Usage:
    python Version_2/stage6_lr_as_raters/lr_raters_experiment.py
"""

import json
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

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
STAGE1_DIR  = BASE_DIR / "stage1_eda" / "results"
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR   = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Build the doubled dataset (614 rows)
# ═══════════════════════════════════════════════════════════════════════════════

def build_lr_dataset(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand 307-evaluation DataFrame into 614-row virtual-rater DataFrame.

    Each original evaluation produces two rows:
      side=L: virtual_rater = rater + '_L', ratings from consistent_L/correct_L/useful_L
      side=R: virtual_rater = rater + '_R', ratings from consistent_R/correct_R/useful_R
    """
    rows = []
    for _, row in eval_df.iterrows():
        base = {
            "original_rater": row["rater"],
            "file":           row["file"],
            "datetime":       row["datetime"],
            "ip":             row["ip"],
            "pair_key":       str(row.get("id_L", "")) + "_" + str(row.get("id_R", "")),
        }
        # Left side
        rows.append({
            **base,
            "virtual_rater": row["rater"] + "_L",
            "side":           "L",
            "answer":         row.get("answer_L", ""),
            "question":       row.get("question_L", ""),
            "model_tag":      row.get("model_L", ""),
            "consistent":     row.get("consistent_L", np.nan),
            "correct":        row.get("correct_L", np.nan),
            "useful":         row.get("useful_L", np.nan),
            "comparison_slider": row.get("comparison_slider", np.nan),
        })
        # Right side
        rows.append({
            **base,
            "virtual_rater": row["rater"] + "_R",
            "side":           "R",
            "answer":         row.get("answer_R", ""),
            "question":       row.get("question_R", ""),
            "model_tag":      row.get("model_R", ""),
            "consistent":     row.get("consistent_R", np.nan),
            "correct":        row.get("correct_R", np.nan),
            "useful":         row.get("useful_R", np.nan),
            "comparison_slider": row.get("comparison_slider", np.nan),
        })
    lr_df = pd.DataFrame(rows)
    for col in ["consistent", "correct", "useful", "comparison_slider"]:
        lr_df[col] = pd.to_numeric(lr_df[col], errors="coerce")
    return lr_df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Auto quality proxy (inverted-U — long answers penalised)
# ═══════════════════════════════════════════════════════════════════════════════

def answer_quality_proxy(text) -> float:
    """
    Lightweight quality proxy: type-token diversity + inverted-U length score.
    Long answers (code generation) tend to be wrong → penalised beyond 50 tokens.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan
    tokens = text.lower().split()
    if len(tokens) == 0:
        return np.nan
    n = len(tokens)
    diversity = len(set(tokens)) / n
    if n <= 50:
        length_score = n / 50.0
    else:
        length_score = max(0.0, 1.0 - (n - 50) / 200.0)
    return 0.5 * diversity + 0.5 * length_score


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Per-virtual-rater statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_virtual_rater_stats(lr_df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats per virtual rater (original_rater × side)."""
    rows = []
    for vr, grp in lr_df.groupby("virtual_rater"):
        side = grp["side"].iloc[0]
        orig = grp["original_rater"].iloc[0]
        all_ratings = grp[["consistent", "correct", "useful"]].values.flatten()
        all_ratings = all_ratings[~np.isnan(all_ratings)]
        rows.append({
            "virtual_rater":    vr,
            "original_rater":   orig,
            "side":             side,
            "n_evaluations":    len(grp),
            "mean_rating":      np.mean(all_ratings) if len(all_ratings) else np.nan,
            "std_rating":       np.std(all_ratings)  if len(all_ratings) else np.nan,
            "mean_consistent":  grp["consistent"].dropna().mean(),
            "mean_correct":     grp["correct"].dropna().mean(),
            "mean_useful":      grp["useful"].dropna().mean(),
        })
    return pd.DataFrame(rows).sort_values("original_rater").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Leniency bias: L vs R comparison
# ═══════════════════════════════════════════════════════════════════════════════

def detect_leniency_lr(lr_df: pd.DataFrame) -> dict:
    """
    Compare mean ratings of L-side virtual raters vs R-side virtual raters.
    A systematic difference reveals positional bias in how the left vs right
    option is rated absolutely (not comparatively).
    """
    print("\n--- Leniency: L-side vs R-side ---")

    l_means, r_means = [], []
    per_orig = {}
    for orig, grp in lr_df.groupby("original_rater"):
        l_grp = grp[grp["side"] == "L"]
        r_grp = grp[grp["side"] == "R"]
        l_vals = l_grp[["consistent", "correct", "useful"]].values.flatten()
        r_vals = r_grp[["consistent", "correct", "useful"]].values.flatten()
        l_mean = np.nanmean(l_vals) if len(l_vals) > 0 else np.nan
        r_mean = np.nanmean(r_vals) if len(r_vals) > 0 else np.nan
        per_orig[orig] = {"l_mean": l_mean, "r_mean": r_mean,
                          "diff_lr": l_mean - r_mean if not np.isnan(l_mean + r_mean) else np.nan}
        if not np.isnan(l_mean):
            l_means.append(l_mean)
        if not np.isnan(r_mean):
            r_means.append(r_mean)
        print(f"  {orig:25s}: L_mean={l_mean:+.3f}  R_mean={r_mean:+.3f}  "
              f"diff(L-R)={l_mean-r_mean:+.3f}")

    grand_l = np.nanmean(l_means)
    grand_r = np.nanmean(r_means)
    print(f"\n  Grand L mean: {grand_l:.3f}")
    print(f"  Grand R mean: {grand_r:.3f}")
    print(f"  Grand diff (L-R): {grand_l - grand_r:+.3f}")

    # Paired t-test: is there a systematic difference?
    diffs = [v["diff_lr"] for v in per_orig.values() if not np.isnan(v["diff_lr"])]
    if len(diffs) >= 3:
        t_stat, p_val = scipy_stats.ttest_1samp(diffs, 0.0)
        print(f"\n  Paired t-test (H0: diff=0): t={t_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            direction = "L rated HIGHER" if grand_l > grand_r else "R rated HIGHER"
            print(f"  *** SIGNIFICANT positional asymmetry: {direction} ***")
        else:
            print(f"  No significant positional asymmetry (p={p_val:.4f})")
    else:
        t_stat, p_val = np.nan, np.nan
        print("  Not enough raters for t-test.")

    return {
        "per_original_rater": per_orig,
        "grand_l_mean":  grand_l,
        "grand_r_mean":  grand_r,
        "grand_diff_lr": grand_l - grand_r,
        "t_stat":        t_stat,
        "p_value":       p_val,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Quality-proxy alignment per virtual rater
# ═══════════════════════════════════════════════════════════════════════════════

def compute_proxy_alignment(lr_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each virtual rater, compute Spearman r(correct, auto_quality_proxy).
    Compare L-side vs R-side alignment scores.
    """
    print("\n--- Auto-Quality Proxy Alignment (per virtual rater) ---")
    lr_df = lr_df.copy()
    lr_df["auto_quality"] = lr_df["answer"].apply(answer_quality_proxy)

    rows = []
    for vr, grp in lr_df.groupby("virtual_rater"):
        combined = grp[["correct", "auto_quality"]].dropna()
        if len(combined) < 5:
            rows.append({"virtual_rater": vr, "side": grp["side"].iloc[0],
                         "original_rater": grp["original_rater"].iloc[0],
                         "n": len(combined), "spearman_r": np.nan, "p_value": np.nan})
            continue
        r, p = scipy_stats.spearmanr(combined["correct"], combined["auto_quality"])
        rows.append({"virtual_rater": vr, "side": grp["side"].iloc[0],
                     "original_rater": grp["original_rater"].iloc[0],
                     "n": len(combined),
                     "spearman_r": round(float(r), 4),
                     "p_value": round(float(p), 4)})
        print(f"  {vr:30s}: Spearman r={r:+.3f}  (p={p:.3f}, n={len(combined)})")

    align_df = pd.DataFrame(rows)

    # Aggregate: L side vs R side
    l_r = align_df[align_df["side"] == "L"]["spearman_r"].dropna()
    r_r = align_df[align_df["side"] == "R"]["spearman_r"].dropna()
    print(f"\n  Mean Spearman r — L-side: {l_r.mean():.3f}  |  R-side: {r_r.mean():.3f}")

    return align_df


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Dimension consistency (within-virtual-rater)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dimension_consistency(lr_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each virtual rater, compute Spearman r(consistent, correct) and
    r(useful, correct). Experts should show positive correlation between dimensions.
    """
    print("\n--- Dimension Consistency (per virtual rater) ---")
    rows = []
    for vr, grp in lr_df.groupby("virtual_rater"):
        cons = grp["consistent"].dropna().reset_index(drop=True)
        corr = grp["correct"].dropna().reset_index(drop=True)
        use  = grp["useful"].dropna().reset_index(drop=True)

        n = min(len(cons), len(corr))
        if n >= 5:
            r_cc, _ = scipy_stats.spearmanr(cons[:n], corr[:n])
        else:
            r_cc = np.nan

        n2 = min(len(use), len(corr))
        if n2 >= 5:
            r_uc, _ = scipy_stats.spearmanr(use[:n2], corr[:n2])
        else:
            r_uc = np.nan

        rows.append({
            "virtual_rater":  vr,
            "side":           grp["side"].iloc[0],
            "original_rater": grp["original_rater"].iloc[0],
            "n_evals":        len(grp),
            "r_consistent_correct": round(float(r_cc), 4) if not np.isnan(r_cc) else np.nan,
            "r_useful_correct":     round(float(r_uc), 4) if not np.isnan(r_uc) else np.nan,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lr_rating_distributions(lr_df: pd.DataFrame, plots_dir: Path):
    """Violin plots of correct / consistent / useful for L vs R sides."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Rating Distributions: L-side vs R-side (614 virtual evaluations)",
                 fontsize=13, fontweight="bold")

    for ax, dim in zip(axes, ["consistent", "correct", "useful"]):
        sns.violinplot(data=lr_df, x="side", y=dim, hue="side",
                       palette={"L": "#3498db", "R": "#e74c3c"},
                       ax=ax, inner="box", cut=0, legend=False)
        ax.set_title(dim.capitalize(), fontsize=11)
        ax.set_xlabel("Side (virtual rater)")
        ax.set_ylabel("Rating (−2 to +2)" if ax == axes[0] else "")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    out = plots_dir / "lr_rating_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_lr_alignment(align_df: pd.DataFrame, plots_dir: Path):
    """Bar chart: Spearman r per virtual rater, colored L vs R."""
    df = align_df.dropna(subset=["spearman_r"]).sort_values("spearman_r", ascending=True)
    colors = ["#3498db" if s == "L" else "#e74c3c" for s in df["side"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.45)))
    ax.barh(df["virtual_rater"], df["spearman_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.15, color="gray", linestyle="--", linewidth=1,
               label="Threshold r=0.15 (meaningful alignment)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="L-side virtual rater"),
        Patch(facecolor="#e74c3c", label="R-side virtual rater"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel("Spearman r (correct vs auto_quality)")
    ax.set_title("Auto-Quality Alignment per Virtual Rater\n"
                 "(positive r = rater's 'correct' score tracks quality proxy)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / "lr_alignment_spearman.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_lr_leniency_diff(leniency_result: dict, plots_dir: Path):
    """Bar chart of L-mean minus R-mean per original rater."""
    per_orig = leniency_result["per_original_rater"]
    raters = sorted(per_orig.keys())
    diffs = [per_orig[r]["diff_lr"] for r in raters]

    fig, ax = plt.subplots(figsize=(9, max(3, len(raters) * 0.6)))
    colors = ["#27ae60" if d > 0 else "#e74c3c" for d in diffs]
    ax.barh(raters, diffs, color=colors)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("L-side mean − R-side mean rating")
    ax.set_title("Positional Asymmetry per Rater (L minus R mean)\n"
                 "(green = L rated higher, red = R rated higher)",
                 fontsize=11, fontweight="bold")
    grand_diff = leniency_result["grand_diff_lr"]
    ax.axvline(grand_diff, color="gray", linestyle="--", linewidth=1.5,
               label=f"Grand diff = {grand_diff:+.3f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = plots_dir / "lr_positional_asymmetry.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_virtual_rater_stats(stats_df: pd.DataFrame, plots_dir: Path):
    """Heatmap: virtual rater × mean_correct / mean_consistent / mean_useful."""
    heat = stats_df.set_index("virtual_rater")[["mean_consistent", "mean_correct", "mean_useful"]]
    heat.columns = ["Consistent", "Correct", "Useful"]
    heat = heat.astype(float)

    fig, ax = plt.subplots(figsize=(6, max(5, len(stats_df) * 0.35)))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-2, vmax=2, linewidths=0.4, ax=ax,
                cbar_kws={"label": "Mean rating"})
    ax.set_title("Mean Ratings per Virtual Rater (L/R split)\n"
                 "(614 evaluations)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Virtual rater")
    plt.tight_layout()
    out = plots_dir / "virtual_rater_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Stage 6: L/R Sides as Separate Virtual Raters Experiment")
    print("=" * 65)

    # Load stage 1 output
    csv_path = STAGE1_DIR / "evaluations_parsed.csv"
    if not csv_path.exists():
        sys.exit(f"[Error] Run Stage 1 first. Expected: {csv_path}")
    eval_df = pd.read_csv(csv_path, low_memory=False)
    for col in ["consistent_L", "correct_L", "useful_L",
                "consistent_R", "correct_R", "useful_R", "comparison_slider"]:
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")
    print(f"[Stage 6] Loaded {len(eval_df)} evaluations, "
          f"{eval_df['rater'].nunique()} raters")

    # Build doubled dataset
    lr_df = build_lr_dataset(eval_df)
    print(f"[Stage 6] Doubled dataset: {len(lr_df)} virtual evaluations "
          f"({lr_df['virtual_rater'].nunique()} virtual raters)")

    # Statistics
    stats_df = compute_virtual_rater_stats(lr_df)
    print("\nVirtual rater statistics:")
    print(stats_df[["virtual_rater", "side", "n_evaluations",
                    "mean_rating", "mean_correct"]].to_string(index=False))

    # Leniency: L vs R
    leniency_result = detect_leniency_lr(lr_df)

    # Quality proxy alignment
    align_df = compute_proxy_alignment(lr_df)

    # Dimension consistency
    dim_con_df = compute_dimension_consistency(lr_df)
    print("\nDimension consistency (r_consistent_correct | r_useful_correct):")
    print(dim_con_df[["virtual_rater", "r_consistent_correct",
                      "r_useful_correct"]].to_string(index=False))

    # Summary: compare L-side vs R-side globally
    print("\n=== L vs R SUMMARY ===")
    l_align = align_df[align_df["side"] == "L"]["spearman_r"].dropna()
    r_align = align_df[align_df["side"] == "R"]["spearman_r"].dropna()
    l_consist = dim_con_df[dim_con_df["side"] == "L"]["r_consistent_correct"].dropna()
    r_consist = dim_con_df[dim_con_df["side"] == "R"]["r_consistent_correct"].dropna()
    print(f"  Mean quality alignment  — L: {l_align.mean():+.3f}  R: {r_align.mean():+.3f}")
    print(f"  Mean dim consistency    — L: {l_consist.mean():+.3f}  R: {r_consist.mean():+.3f}")
    print(f"  Positional asymmetry (L-R mean rating): {leniency_result['grand_diff_lr']:+.3f}")
    if not np.isnan(leniency_result["p_value"]):
        print(f"  Paired t-test p={leniency_result['p_value']:.4f} "
              f"({'significant' if leniency_result['p_value'] < 0.05 else 'not significant'})")

    # Save results
    stats_df.to_csv(RESULTS_DIR / "virtual_rater_stats.csv", index=False)
    align_df.to_csv(RESULTS_DIR / "virtual_rater_alignment.csv", index=False)
    dim_con_df.to_csv(RESULTS_DIR / "virtual_rater_consistency.csv", index=False)
    lr_df.to_csv(RESULTS_DIR / "lr_expanded_dataset.csv", index=False)

    summary = {
        "total_evaluations": int(len(lr_df)),
        "total_virtual_raters": int(lr_df["virtual_rater"].nunique()),
        "original_evaluations": int(len(eval_df)),
        "original_raters": int(eval_df["rater"].nunique()),
        "grand_l_mean": float(leniency_result["grand_l_mean"]),
        "grand_r_mean": float(leniency_result["grand_r_mean"]),
        "grand_diff_lr": float(leniency_result["grand_diff_lr"]),
        "t_stat_lr": float(leniency_result["t_stat"]) if not np.isnan(leniency_result["t_stat"]) else None,
        "p_value_lr": float(leniency_result["p_value"]) if not np.isnan(leniency_result["p_value"]) else None,
        "significant_positional_asymmetry": (
            bool(leniency_result["p_value"] < 0.05)
            if not np.isnan(leniency_result["p_value"]) else None
        ),
        "mean_alignment_L": float(l_align.mean()) if len(l_align) > 0 else None,
        "mean_alignment_R": float(r_align.mean()) if len(r_align) > 0 else None,
        "mean_dim_consistency_L": float(l_consist.mean()) if len(l_consist) > 0 else None,
        "mean_dim_consistency_R": float(r_consist.mean()) if len(r_consist) > 0 else None,
    }
    with open(RESULTS_DIR / "lr_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Visualizations
    print("\nGenerating plots...")
    plot_lr_rating_distributions(lr_df, PLOTS_DIR)
    plot_lr_alignment(align_df, PLOTS_DIR)
    plot_lr_leniency_diff(leniency_result, PLOTS_DIR)
    plot_virtual_rater_stats(stats_df, PLOTS_DIR)

    print(f"\n[Stage 6] DONE. Outputs in: {RESULTS_DIR}")
    print(f"  Key result: 307 → 614 evaluations, {lr_df['virtual_rater'].nunique()} virtual raters")


if __name__ == "__main__":
    main()
