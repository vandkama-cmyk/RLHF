"""
Stage 5: Analysis Report & Visualization
==========================================
Aggregates outputs from all previous stages and produces:
  1. Master summary table (rater × all bias types × final weight)
  2. Side-by-side weighted vs baseline model comparison
  3. Rater reliability interpretation (who to trust, who to discount)
  4. Summary JSON: Version_2/results_summary.json

Depends on:
  - stage1_eda/results/rater_stats.csv
  - stage2_bias_detection/results/bias_report.json
  - stage3_expertise/results/rater_weights.json
  - stage4_reward_model/results/training_results.json

Usage:
    python Version_2/stage5_report/generate_report.py
"""

import json
import math
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2

# Bug #15 fix: do not suppress all warnings globally.

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
STAGE1_DIR  = BASE_DIR / "stage1_eda"  / "results"
STAGE2_DIR  = BASE_DIR / "stage2_bias_detection" / "results"
STAGE3_DIR  = BASE_DIR / "stage3_expertise" / "results"
STAGE4_DIR  = BASE_DIR / "stage4_reward_model" / "results"
STAGE6_DIR  = BASE_DIR / "stage6_lr_as_raters" / "results"
STAGE7_DIR  = BASE_DIR / "stage7_pairs_vs_individual" / "results"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Load all stage outputs
# ═══════════════════════════════════════════════════════════════════════════════

def load_stage_outputs():
    data = {}

    # Stage 1
    rater_stats_path = STAGE1_DIR / "rater_stats.csv"
    if rater_stats_path.exists():
        data["rater_stats"] = pd.read_csv(rater_stats_path)
    else:
        print("[Warning] rater_stats.csv not found. Run Stage 1 first.")
        data["rater_stats"] = pd.DataFrame()

    # Stage 2
    bias_report_path = STAGE2_DIR / "bias_report.json"
    if bias_report_path.exists():
        with open(bias_report_path) as f:
            data["bias_report"] = json.load(f)
    else:
        print("[Warning] bias_report.json not found. Run Stage 2 first.")
        data["bias_report"] = {}

    leniency_path = STAGE2_DIR / "leniency_bias.csv"
    if leniency_path.exists():
        data["leniency_df"] = pd.read_csv(leniency_path)
    else:
        data["leniency_df"] = pd.DataFrame()

    positional_path = STAGE2_DIR / "positional_bias.csv"
    if positional_path.exists():
        data["positional_df"] = pd.read_csv(positional_path)
    else:
        data["positional_df"] = pd.DataFrame()

    decoupling_path = STAGE2_DIR / "decoupling_bias.csv"
    if decoupling_path.exists():
        data["decoupling_df"] = pd.read_csv(decoupling_path)
    else:
        data["decoupling_df"] = pd.DataFrame()

    inconsistency_path = STAGE2_DIR / "inconsistency_scores.csv"
    if inconsistency_path.exists():
        data["inconsistency_df"] = pd.read_csv(inconsistency_path)
    else:
        data["inconsistency_df"] = pd.DataFrame()

    # Stage 3
    weights_path = STAGE3_DIR / "rater_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            data["rater_weights"] = json.load(f)
    else:
        print("[Warning] rater_weights.json not found. Run Stage 3 first.")
        data["rater_weights"] = {}

    # Stage 4
    training_path = STAGE4_DIR / "training_results.json"
    if training_path.exists():
        with open(training_path) as f:
            data["training_results"] = json.load(f)
    else:
        print("[Warning] training_results.json not found. Run Stage 4 first.")
        data["training_results"] = {}

    # Stage 4 history CSVs
    data["histories"] = {}
    for mode in ["weighted", "baseline"]:
        hist_path = STAGE4_DIR / f"history_{mode}.csv"
        if hist_path.exists():
            data["histories"][mode] = pd.read_csv(hist_path)

    # Stage 6 — L/R as virtual raters experiment (Bug #20 fix)
    s6_path = STAGE6_DIR / "lr_experiment_summary.json"
    if s6_path.exists():
        with open(s6_path) as f:
            data["stage6_summary"] = json.load(f)
    else:
        data["stage6_summary"] = {}

    # Stage 7 — pair vs individual comparison (Bug #20 fix)
    s7_path = STAGE7_DIR / "pairs_vs_individual_summary.json"
    if s7_path.exists():
        with open(s7_path) as f:
            data["stage7_summary"] = json.load(f)
    else:
        data["stage7_summary"] = {}

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Build master summary table
# ═══════════════════════════════════════════════════════════════════════════════

def build_master_table(data: dict) -> pd.DataFrame:
    """Create rater × (all signals) master table."""
    rater_stats  = data.get("rater_stats", pd.DataFrame())
    leniency_df  = data.get("leniency_df", pd.DataFrame())
    positional_df = data.get("positional_df", pd.DataFrame())
    decoupling_df = data.get("decoupling_df", pd.DataFrame())
    inconsistency_df = data.get("inconsistency_df", pd.DataFrame())
    rater_weights = data.get("rater_weights", {})
    training_results = data.get("training_results", {})
    rwft = training_results.get("rater_weights_for_training") or {}
    effective_map = rwft.get("effective_rater_weight") or {}

    # Collect all raters
    raters = set()
    for df in [rater_stats, leniency_df, positional_df, decoupling_df, inconsistency_df]:
        if not df.empty and "rater" in df.columns:
            raters.update(df["rater"].tolist())
    raters.update(rater_weights.keys())
    raters = sorted(raters)

    rows = []
    for rater in raters:
        row = {"rater": rater}

        # Stage 1: basic stats
        if not rater_stats.empty and "rater" in rater_stats.columns:
            rs = rater_stats[rater_stats["rater"] == rater]
            if len(rs) > 0:
                row["n_evaluations"] = rs["n_evaluations"].values[0]
                row["mean_rating"]   = round(rs["mean_rating"].values[0], 3) if "mean_rating" in rs else None
                row["std_rating"]    = round(rs["std_rating"].values[0], 3) if "std_rating" in rs else None

        # Stage 2: leniency
        if not leniency_df.empty and "rater" in leniency_df.columns:
            lr = leniency_df[leniency_df["rater"] == rater]
            if len(lr) > 0:
                row["z_score"]    = round(float(lr["z_score"].values[0]), 3)
                row["bias_type"]  = lr["bias_type"].values[0]

        # Stage 2: positional bias
        if not positional_df.empty and "rater" in positional_df.columns:
            pr = positional_df[positional_df["rater"] == rater]
            if len(pr) > 0:
                row["positional_pval"]  = round(float(pr["pval"].values[0]), 4) if "pval" in pr else None
                row["positional_sig"]   = bool(pr["significant"].values[0]) if "significant" in pr else None
                row["positional_dir"]   = pr["bias_direction"].values[0] if "bias_direction" in pr else None

        # Stage 2: confidence-correctness decoupling
        if not decoupling_df.empty and "rater" in decoupling_df.columns:
            dr = decoupling_df[decoupling_df["rater"] == rater]
            if len(dr) > 0:
                row["corr_correct_auto"] = round(float(dr["corr_correct_auto"].values[0]), 4) if "corr_correct_auto" in dr else None
                row["decoupling_flag"]   = bool(dr["decoupling_flag"].values[0]) if "decoupling_flag" in dr else None

        # Stage 2: dimension inconsistency
        if not inconsistency_df.empty and "rater" in inconsistency_df.columns:
            ir = inconsistency_df[inconsistency_df["rater"] == rater]
            if len(ir) > 0:
                row["inconsistency_score"] = round(float(ir["mean_inconsistency"].values[0]), 4) if "mean_inconsistency" in ir else None
                row["inconsistency_flag"]  = bool(ir["inconsistency_flag"].values[0]) if "inconsistency_flag" in ir else None

        # Stage 3: softmax reliability weight (pre-cap)
        if rater in rater_weights:
            rw = rater_weights[rater]
            row["reliability_weight_stage3"] = round(rw["weight"], 6) if isinstance(rw, dict) else round(float(rw), 6)
            if isinstance(rw, dict):
                row["align_score"]       = rw.get("align_score")
                row["consistency_score"] = rw.get("consistency_score")
                row["agreement_score"]   = rw.get("agreement_score")

        # Stage 4: per-rater weight after MAX_WEIGHT_CAP (used in weighted training loss)
        if rater in effective_map:
            row["effective_weight_stage4"] = round(float(effective_map[rater]), 6)

        rows.append(row)

    master_df = pd.DataFrame(rows)
    sort_cols = (
        ["effective_weight_stage4", "reliability_weight_stage3"]
        if "effective_weight_stage4" in master_df.columns
        else ["reliability_weight_stage3"]
    )
    if "reliability_weight_stage3" not in master_df.columns:
        return master_df
    master_df = master_df.sort_values(
        sort_cols[0] if len(sort_cols) == 1 else sort_cols,
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    return master_df


# ═══════════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_master_heatmap(master_df: pd.DataFrame, results_dir: Path):
    """Comprehensive rater × signal heatmap."""
    weight_col = "effective_weight_stage4" if "effective_weight_stage4" in master_df.columns else "reliability_weight_stage3"
    numeric_cols = ["z_score", "corr_correct_auto", "inconsistency_score", weight_col]
    avail_cols = [c for c in numeric_cols if c in master_df.columns]
    if not avail_cols:
        return

    heat = master_df.set_index("rater")[avail_cols].astype(float)
    # Normalize each column to [0,1] for comparable visualization
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    col_labels = {
        "z_score": "Leniency\n(z-score)",
        "corr_correct_auto": "Metric\nAlignment",
        "inconsistency_score": "Dimension\nInconsistency",
        "reliability_weight_stage3": "Weight\n(Stage 3)",
        "effective_weight_stage4": "Weight\n(Stage 4 cap)",
    }

    fig, ax = plt.subplots(figsize=(max(8, len(avail_cols) * 2), max(4, len(master_df) * 0.7)))
    heat_norm.columns = [col_labels.get(c, c) for c in heat_norm.columns]
    sns.heatmap(heat_norm, annot=heat.round(3), fmt=".3f",
                cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Normalized score"})
    ax.set_title("Rater Signals Master Heatmap\n(normalized to [0,1])",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Rater")
    plt.tight_layout()
    out = results_dir / "master_rater_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_model_comparison(data: dict, results_dir: Path):
    """Side-by-side comparison of weighted vs baseline model training curves."""
    histories = data.get("histories", {})
    if not histories:
        print("  No training history available. Skipping model comparison plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Stage 4: Weighted vs Baseline Reward Model\n(Bias-Corrected Label Training)",
                 fontsize=13, fontweight="bold")

    plot_configs = [
        ("val_correct_acc", "Val Correct Accuracy", axes[0, 0]),
        ("val_consistent_acc", "Val Consistent Accuracy", axes[0, 1]),
        ("val_useful_acc", "Val Useful Accuracy", axes[1, 0]),
        ("val_reward_gap", "Reward Gap", axes[1, 1]),
    ]
    colors = {"weighted": "#2ecc71", "baseline": "#e74c3c"}

    for metric, title, ax in plot_configs:
        for mode, hist_df in histories.items():
            if metric in hist_df.columns:
                vals = hist_df[metric].values
                epochs = range(1, len(vals) + 1)
                # Handle NaN for reward gap
                mask = ~pd.isna(vals)
                valid_e = [e for e, m in zip(epochs, mask) if m]
                valid_v = vals[mask]
                if len(valid_v) > 0:
                    ax.plot(valid_e, valid_v, label=mode,
                            color=colors.get(mode, "gray"), linewidth=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = results_dir / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_reliability_interpretation(master_df: pd.DataFrame, results_dir: Path):
    """Visual guide to rater reliability tiers."""
    wcol = "effective_weight_stage4" if "effective_weight_stage4" in master_df.columns else "reliability_weight_stage3"
    if wcol not in master_df.columns:
        return

    uniform = 1 / len(master_df) if len(master_df) > 0 else 0.1
    df = master_df.copy().sort_values(wcol, ascending=True)

    # Color by tier relative to uniform
    def tier_color(w):
        if w > uniform * 1.3:
            return "#27ae60"   # high reliability
        elif w < uniform * 0.7:
            return "#e74c3c"   # low reliability
        else:
            return "#3498db"   # near uniform

    colors = [tier_color(w) for w in df[wcol]]

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.65)))
    bars = ax.barh(df["rater"], df[wcol], color=colors)
    ax.axvline(uniform, color="black", linestyle="--", linewidth=1.5,
               label=f"Uniform ({uniform:.3f})")

    # Annotate with weight value
    for bar, w in zip(bars, df[wcol]):
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=9)

    # Legend patches
    patches = [
        mpatches.Patch(color="#27ae60", label="High reliability (>1.3× uniform)"),
        mpatches.Patch(color="#3498db", label="Near-uniform reliability"),
        mpatches.Patch(color="#e74c3c", label="Low reliability (<0.7× uniform)"),
    ]
    ax.legend(handles=patches + [plt.Line2D([0], [0], color="black", linestyle="--",
                                             label=f"Uniform ({uniform:.3f})")],
              fontsize=9, loc="lower right")
    ax.set_xlabel("Reliability Weight (Stage 4 capped when available, else Stage 3)")
    ax.set_title("Rater Reliability Weights\n(Bias-Corrected Proxy Estimation)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = results_dir / "reliability_interpretation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# McNemar's test
# ═══════════════════════════════════════════════════════════════════════════════

def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray,
                 labels: np.ndarray, threshold: float = 0.0) -> dict:
    """
    McNemar's test comparing two binary classifiers on the same validation examples.
    preds_a: logit array from model A (e.g. weighted).
    preds_b: logit array from model B (e.g. baseline).
    labels:  ground-truth binary labels.
    Returns dict with statistic, p_value, significant (p < 0.05), and contingency counts.
    """
    bin_a = (preds_a > threshold).astype(int)
    bin_b = (preds_b > threshold).astype(int)
    correct_a = (bin_a == labels.astype(int))
    correct_b = (bin_b == labels.astype(int))
    # b01: B correct, A wrong; b10: A correct, B wrong
    b01 = int(((~correct_a) & correct_b).sum())
    b10 = int((correct_a & (~correct_b)).sum())
    n_discordant = b01 + b10
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b01": 0, "b10": 0,
                "n_discordant": 0, "significant": False,
                "note": "No discordant pairs — models agree on every example"}
    # McNemar's chi-square with continuity correction
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = float(1 - chi2.cdf(stat, df=1))
    return {
        "statistic":    round(float(stat), 4),
        "p_value":      round(p_value, 4),
        "b01":          b01,
        "b10":          b10,
        "n_discordant": n_discordant,
        "significant":  p_value < 0.05,
        "note":         "McNemar's test with continuity correction (chi2, df=1)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Generate summary JSON
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary_json(master_df: pd.DataFrame, data: dict) -> dict:
    """Create results_summary.json with all key findings."""
    training = data.get("training_results", {})
    bias_report = data.get("bias_report", {})
    rater_weights = data.get("rater_weights", {})
    stage6_summary = data.get("stage6_summary", {})   # Bug #20 fix
    stage7_summary = data.get("stage7_summary", {})   # Bug #20 fix

    # Identify biased raters.
    # Note: bias_type="" (empty string) means no leniency bias — only non-empty strings are real flags.
    # A rater can appear in both biased_raters and top_reliability_raters: this is expected because
    # *reliability* (consistency of ratings) is distinct from *calibration* (alignment with ground truth).
    # A highly consistent but miscalibrated rater will score high on consistency but low on metric alignment.
    biased_raters = []
    if not data.get("leniency_df", pd.DataFrame()).empty:
        for _, row in data["leniency_df"].iterrows():
            bt = str(row.get("bias_type", "") or "").strip()
            if bt and bt.lower() != "nan":
                biased_raters.append({"rater": row["rater"], "bias_type": bt})

    if not data.get("positional_df", pd.DataFrame()).empty:
        for _, row in data["positional_df"].iterrows():
            if row.get("significant"):
                biased_raters.append({"rater": row["rater"],
                                       "bias_type": f"Positional: {row.get('bias_direction', 'unknown')}"})

    if not data.get("decoupling_df", pd.DataFrame()).empty:
        for _, row in data["decoupling_df"].iterrows():
            if row.get("decoupling_flag"):
                biased_raters.append({"rater": row["rater"],
                                       "bias_type": f"Confidence-Correctness decoupling (r={row['corr_correct_auto']:.3f})"})

    # Model comparison
    model_comparison = {}
    if "weighted" in training and "baseline" in training:
        w = training["weighted"]
        b = training["baseline"]
        delta         = round(w["best_val_acc"] - b["best_val_acc"], 4)
        delta_pos     = round(w.get("best_correct_acc_pos", float("nan")) -
                              b.get("best_correct_acc_pos", float("nan")), 4)
        delta_neg     = round(w.get("best_correct_acc_neg", float("nan")) -
                              b.get("best_correct_acc_neg", float("nan")), 4)
        within_noise  = abs(delta) <= 0.01

        # McNemar's test using per-sample validation predictions saved by Stage 4
        mcnemar_result = {}
        w_pred_path = STAGE4_DIR / "val_predictions_weighted.csv"
        b_pred_path = STAGE4_DIR / "val_predictions_baseline.csv"
        if w_pred_path.exists() and b_pred_path.exists():
            w_pred_df = pd.read_csv(w_pred_path)
            b_pred_df = pd.read_csv(b_pred_path)
            if len(w_pred_df) == len(b_pred_df) and len(w_pred_df) > 0:
                mcnemar_result = mcnemar_test(
                    w_pred_df["correct_logit"].values,
                    b_pred_df["correct_logit"].values,
                    w_pred_df["correct_label"].values,
                )
            else:
                mcnemar_result = {"note": "Prediction files have mismatched lengths — skipped"}
        else:
            mcnemar_result = {"note": "val_predictions_*.csv not found — re-run Stage 4 to generate"}

        # Build significance interpretation using McNemar's p-value when available
        p_val = mcnemar_result.get("p_value")
        is_sig = mcnemar_result.get("significant", False)
        sig_str = (f"McNemar's test p={p_val} ({'significant' if is_sig else 'not significant'})"
                   if p_val is not None else "no formal significance test performed")

        # Bug #5/#18 fix: lead with significance result; detect and flag class-accuracy flip.
        # The old code said "improved" whenever delta > 0.005, regardless of whether
        # McNemar's p-value was significant and regardless of whether the gain was an
        # artefact of a decision-boundary shift that helped one class at the expense of
        # the other.  A large positive delta_pos paired with a large negative delta_neg
        # is the signature of a degenerate shift (e.g. toward always-predict-positive).
        is_class_flip = (
            not math.isnan(delta_pos) and not math.isnan(delta_neg)
            and delta_pos > 0.1 and delta_neg < -0.1
        )
        if is_class_flip:
            interpretation = (
                f"Class-accuracy flip detected: weighted model gained on positive class "
                f"({delta_pos:+.3f}) but lost on negative class ({delta_neg:+.3f}). "
                f"The overall Δacc ({delta:+.4f}) likely reflects a decision-boundary shift "
                f"rather than genuine improvement — the model moved toward predicting the "
                f"majority class more often.  {sig_str}."
            )
        elif within_noise:
            interpretation = (
                f"No practically meaningful accuracy difference (|Δ| ≤ 0.01, {sig_str}), "
                "consistent with dataset size limitations (n=614). "
                "Per-class breakdown may reveal differential effects hidden by aggregated accuracy."
            )
        elif delta > 0.005:
            if is_sig:
                interpretation = (
                    f"Bias-corrected weighting improved reward model accuracy "
                    f"(Δ={delta:+.4f}, statistically significant). {sig_str}."
                )
            else:
                interpretation = (
                    f"Weighted model shows higher accuracy (Δ={delta:+.4f}) but the "
                    f"difference is NOT statistically significant. {sig_str}. "
                    "Dataset size (n=614) is too small to confirm this improvement; "
                    "treat as indicative only."
                )
        else:
            interpretation = (
                f"Weighted model underperformed baseline ({sig_str}) — check per-class "
                "breakdown and consider whether high-weight raters have systematically "
                "harder examples."
            )

        model_comparison = {
            "weighted_best_val_acc":   w["best_val_acc"],
            "baseline_best_val_acc":   b["best_val_acc"],
            "improvement_delta":       delta,
            "delta_positive_class":    delta_pos,
            "delta_negative_class":    delta_neg,
            "within_noise_threshold":  within_noise,
            "mcnemar_test":            mcnemar_result,
            "interpretation":          interpretation,
        }

        # Critical warnings: surface zero positive-class accuracy prominently
        critical_warnings = []
        for model_label, res in [("weighted", w), ("baseline", b)]:
            pos_acc = res.get("best_correct_acc_pos", float("nan"))
            try:
                pos_acc_float = float(pos_acc)
            except (TypeError, ValueError):
                pos_acc_float = float("nan")
            if not math.isnan(pos_acc_float) and pos_acc_float < 0.05:
                critical_warnings.append(
                    f"CRITICAL: {model_label} model positive-class accuracy = {pos_acc_float:.3f} "
                    "(near zero). Model is predicting almost exclusively negative class. "
                    "Check label threshold, pos_weight, and class balance."
                )
        model_comparison["critical_warnings"] = critical_warnings
        if critical_warnings:
            print("\n  !!! CRITICAL WARNINGS !!!")
            for w_msg in critical_warnings:
                print(f"  {w_msg}")

    # Top/bottom raters by weight (Stage 4 effective when present)
    if not master_df.empty and "reliability_weight_stage3" in master_df.columns:
        wcols = ["rater", "reliability_weight_stage3"]
        if "effective_weight_stage4" in master_df.columns:
            wcols.append("effective_weight_stage4")
        top_raters    = master_df.head(3)[wcols].to_dict(orient="records")
        bottom_raters = master_df.tail(3)[wcols].to_dict(orient="records")
    else:
        top_raters = bottom_raters = []

    summary = {
        "project": "RLHF Version 2 - Rater Bias Investigation",
        "n_raters": int(master_df["rater"].nunique()) if "rater" in master_df.columns else 0,
        "n_evaluations": int(master_df["n_evaluations"].sum()) if "n_evaluations" in master_df.columns else 0,
        "inter_rater_agreement": {
            **bias_report.get("inter_rater_agreement", {}),
            **({
                "overlap_pct": round(
                    100 * bias_report["inter_rater_agreement"]["n_shared_pairs"]
                    / max(int(master_df["n_evaluations"].sum()) if "n_evaluations" in master_df.columns else 1, 1),
                    1
                ),
                "reliability_note": (
                    "WARNING: overlap too low (<15%) for reliable agreement estimates — "
                    "Krippendorff alpha and rater rankings should be treated as indicative only."
                    if (bias_report.get("inter_rater_agreement", {}).get("n_shared_pairs", 0) /
                        max(int(master_df["n_evaluations"].sum()) if "n_evaluations" in master_df.columns else 1, 1)) < 0.15
                    else "Agreement metrics computed on sufficient shared pairs."
                ),
            } if bias_report.get("inter_rater_agreement") else {})
        },
        "biased_raters_detected": biased_raters,
        "top_reliability_raters":    top_raters,
        "bottom_reliability_raters": bottom_raters,
        "model_comparison": model_comparison,
        "methodology_note": (
            "Reliability weight is NOT the same as calibration. "
            "A rater can be internally consistent (high consistency_score) while still being "
            "miscalibrated (low align_score, e.g. preferring confident-sounding wrong answers). "
            "Bias detection flags directional/leniency/decoupling issues independently. "
            "Raters appearing in both biased_raters_detected and top_reliability_raters are "
            "consistent-but-miscalibrated — their labels are penalised via lower alignment weight. "
            "reliability_weight_stage3 is the softmax output from Stage 3; effective_weight_stage4 "
            "is the same weight after per-rater capping used in Stage 4 weighted training."
        ),
        "rater_weight_columns": {
            "stage3_softmax": "reliability_weight_stage3",
            "stage4_capped_for_training": "effective_weight_stage4",
        },
        "inter_rater_agreement_note": (
            "Krippendorff alpha values should be interpreted cautiously: only 38 shared pairs "
            "across 11 raters (~12% overlap). Values reflect genuine disagreement but confidence "
            "intervals are wide at this sample size."
        ),
        # Bug #20 fix: include Stage 6 and Stage 7 results that were previously absent.
        "stage6_lr_as_raters": stage6_summary if stage6_summary else {
            "note": "Stage 6 results not available — run stage6_lr_as_raters/lr_raters_experiment.py"
        },
        "stage7_pairs_vs_individual": stage7_summary if stage7_summary else {
            "note": "Stage 7 results not available — run stage7_pairs_vs_individual/pairs_vs_individual.py"
        },
        "key_findings": [],
    }

    # Auto-generate key findings
    findings = []
    if biased_raters:
        unique_biased = sorted(set(r["rater"] for r in biased_raters))
        n_detections  = len(biased_raters)
        # A rater may have multiple bias types (e.g. Artem: STRICT + Positional).
        # Report unique rater count AND total detection count to avoid confusion.
        findings.append(
            f"{len(unique_biased)} rater(s) with {n_detections} bias detection(s) "
            f"({', '.join(unique_biased)}). "
            f"Note: one rater may have multiple bias types."
        )
    else:
        findings.append("No strong biases detected (dataset may be too small for definitive conclusions).")

    if top_raters and bottom_raters:
        tr0 = top_raters[0]
        br = bottom_raters[-1]
        w3 = tr0.get("reliability_weight_stage3")
        w4 = tr0.get("effective_weight_stage4")
        if w4 is not None:
            findings.append(
                f"Highest reliability: {tr0['rater']} "
                f"(Stage3={w3:.4f}, Stage4-capped={w4:.4f})"
            )
        else:
            findings.append(f"Highest reliability: {tr0['rater']} (Stage3 weight={w3:.4f})")
        w3b = br.get("reliability_weight_stage3")
        w4b = br.get("effective_weight_stage4")
        if w4b is not None:
            findings.append(
                f"Lowest reliability: {br['rater']} "
                f"(Stage3={w3b:.4f}, Stage4-capped={w4b:.4f})"
            )
        else:
            findings.append(f"Lowest reliability: {br['rater']} (Stage3 weight={w3b:.4f})")

    if model_comparison:
        findings.append(f"Weighted reward model vs baseline: dacc={model_comparison['improvement_delta']:+.4f}")

    findings.append("Rater expertise proxy: based on alignment with automated metrics, "
                    "dimension consistency, and majority agreement.")

    # Stage 6 finding
    if stage6_summary:
        positional_sig = stage6_summary.get("positional_asymmetry_significant")
        if positional_sig is not None:
            findings.append(
                "Stage 6 (L/R as virtual raters): positional asymmetry "
                + ("SIGNIFICANT" if positional_sig else "not significant")
                + f" (p={stage6_summary.get('positional_pval', 'N/A')})."
            )

    # Stage 7 finding
    if stage7_summary:
        verdict = stage7_summary.get("hypothesis_verdict", "")
        n_supported = stage7_summary.get("n_tests_supported", "?")
        n_total = stage7_summary.get("n_tests_total", "?")
        if verdict:
            findings.append(
                f"Stage 7 (pairs vs individual): hypothesis {verdict} "
                f"({n_supported}/{n_total} tests support pair-comparison superiority)."
            )

    summary["key_findings"] = findings

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Stage 5: Analysis Report & Visualization")
    print("=" * 60)

    # Load all outputs
    data = load_stage_outputs()

    # Build master table
    print("\nBuilding master rater summary table...")
    master_df = build_master_table(data)
    master_df.to_csv(RESULTS_DIR / "master_rater_summary.csv", index=False)
    print("\nMaster Rater Summary:")
    cols = [c for c in ["rater", "n_evaluations", "z_score",
                         "reliability_weight_stage3", "effective_weight_stage4",
                         "bias_type", "decoupling_flag", "inconsistency_flag"]
            if c in master_df.columns]
    print(master_df[cols].to_string(index=False))

    # Visualizations
    print("\nGenerating report plots...")
    plot_master_heatmap(master_df, RESULTS_DIR)
    plot_model_comparison(data, RESULTS_DIR)
    plot_reliability_interpretation(master_df, RESULTS_DIR)

    # Summary JSON
    summary = generate_summary_json(master_df, data)
    summary_path = BASE_DIR / "results_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved results_summary.json -> {summary_path}")

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    for finding in summary.get("key_findings", []):
        print(f"  - {finding}")

    if summary.get("model_comparison"):
        mc = summary["model_comparison"]
        print(f"\n=== MODEL COMPARISON ===")
        print(f"  Weighted accuracy:      {mc.get('weighted_best_val_acc', 'N/A')}")
        print(f"  Baseline accuracy:      {mc.get('baseline_best_val_acc', 'N/A')}")
        print(f"  dacc (overall):         {mc.get('improvement_delta', 'N/A')}")
        if mc.get("delta_positive_class") is not None:
            print(f"  dacc (positive class):  {mc.get('delta_positive_class', 'N/A')}")
        if mc.get("delta_negative_class") is not None:
            print(f"  dacc (negative class):  {mc.get('delta_negative_class', 'N/A')}")
        if mc.get("within_noise_threshold"):
            print(f"  Within-noise flag:      True  (|delta| <= 0.01)")
        print(f"  Interpretation:     {mc.get('interpretation', '')}")

    print("\n[Stage 5] DONE. Outputs in:", RESULTS_DIR)
    print(f"         Summary JSON:  {summary_path}")


if __name__ == "__main__":
    main()
