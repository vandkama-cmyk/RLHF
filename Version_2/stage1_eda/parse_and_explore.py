"""
Stage 1: Data Parsing & Exploratory Analysis
=============================================
Parses all 307 JSON evaluation files from evaluation_results_server/,
joins with dataset CSVs on ID, computes per-rater statistics,
and produces visualizations of rating distributions.

Usage:
    python Version_2/stage1_eda/parse_and_explore.py
"""

import json
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent        # Version_2/
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = DATA_DIR / "evaluation_results_server"
DATASET_DIR = DATA_DIR / "datasets_for_eval"
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = RESULTS_DIR / "eda_plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Rating columns ─────────────────────────────────────────────────────────────
RATING_COLS = ["consistent_L", "correct_L", "useful_L",
               "consistent_R", "correct_R", "useful_R",
               "comparison_slider"]
DIM_COLS_L = ["consistent_L", "correct_L", "useful_L"]
DIM_COLS_R = ["consistent_R", "correct_R", "useful_R"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Parse evaluation JSONs
# ═══════════════════════════════════════════════════════════════════════════════

def parse_eval_files(eval_dir: Path) -> pd.DataFrame:
    """Load all evaluation JSONs into a flat DataFrame (one row per evaluation)."""
    records = []
    for fpath in sorted(eval_dir.glob("*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"  WARNING: Could not read {fpath.name}: {e}")
            continue

        questions = d.get("questions_df", [])
        if len(questions) < 2:
            continue  # malformed

        q_L = questions[0]
        q_R = questions[1]

        # Task 1: Anonymous raters with different IP addresses are treated as
        # DIFFERENT raters. A person who entered "Anonymous" and a different IP
        # is a distinct evaluator, not the same person (40 unique IPs found in data).
        # Named raters (e.g. "Artem", "Anonymous01") keep their name as-is.
        name = d.get("name_input", "Unknown").strip()
        ip   = d.get("address", "").strip()
        if name == "Anonymous" and ip:
            rater_id = f"Anonymous_{ip}"
        else:
            rater_id = name

        record = {
            "file": fpath.name,
            "rater": rater_id,
            "datetime": d.get("datetime", ""),
            "ip": ip,
            # ratings
            "comparison_slider": d.get("comparison_slider", np.nan),
            "consistent_L": d.get("consistent_L", np.nan),
            "correct_L":    d.get("correct_L",    np.nan),
            "useful_L":     d.get("useful_L",     np.nan),
            "consistent_R": d.get("consistent_R", np.nan),
            "correct_R":    d.get("correct_R",    np.nan),
            "useful_R":     d.get("useful_R",     np.nan),
            # left answer metadata
            "id_L":         q_L.get("ID", np.nan),
            "index_L":      q_L.get("index", np.nan),
            "csv_L":        q_L.get("CSV_PATH", ""),
            "model_L":      q_L.get("MODEL_TAG", ""),
            "code_fmt_L":   q_L.get("CODE_FORMATTING", False),
            "question_L":   q_L.get("Question", ""),
            "answer_L":     q_L.get("Answer", ""),
            # right answer metadata
            "id_R":         q_R.get("ID", np.nan),
            "index_R":      q_R.get("index", np.nan),
            "csv_R":        q_R.get("CSV_PATH", ""),
            "model_R":      q_R.get("MODEL_TAG", ""),
            "code_fmt_R":   q_R.get("CODE_FORMATTING", False),
            "question_R":   q_R.get("Question", ""),
            "answer_R":     q_R.get("Answer", ""),
        }
        records.append(record)

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in RATING_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bug #10 fix: detect and remove duplicate evaluations (same rater, same
    # question-pair rated more than once).  Keeping contradictory duplicates
    # inflates inter-rater disagreement metrics and injects noise into labels.
    # Strategy: for each (rater, id_L, id_R) group keep only the LATEST
    # evaluation by datetime (or the first row if datetime is missing) so that
    # the most recent judgement is used and the contradiction is flagged.
    pair_key = df["id_L"].astype(str) + "_" + df["id_R"].astype(str)
    dup_mask = df.duplicated(subset=["rater", "id_L", "id_R"], keep=False)
    n_dup = dup_mask.sum()
    if n_dup > 0:
        print(f"  WARNING: {n_dup} rows share the same (rater, id_L, id_R). "
              "Keeping the latest evaluation per duplicate group.")
        df = (df.sort_values("datetime", na_position="first")
                .drop_duplicates(subset=["rater", "id_L", "id_R"], keep="last")
                .reset_index(drop=True))
        print(f"  After deduplication: {len(df)} evaluations remain.")

    print(f"[Stage 1] Parsed {len(df)} evaluations from {eval_dir}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load dataset CSVs
# ═══════════════════════════════════════════════════════════════════════════════

def load_datasets(dataset_dir: Path) -> pd.DataFrame:
    """Load all CSV dataset files, keeping ID + ModelAnswer + Answer columns."""
    frames = []
    for fpath in sorted(dataset_dir.glob("*.csv")):
        try:
            chunk = pd.read_csv(fpath, low_memory=False)
            chunk["source_csv"] = fpath.name
            # Normalize column names
            chunk.columns = [c.strip() for c in chunk.columns]
            # Keep only needed columns where available
            keep = [c for c in ["ID", "Question", "Answer", "ModelAnswer", "MODEL_TAG", "source_csv"] if c in chunk.columns]
            frames.append(chunk[keep])
        except Exception as e:
            print(f"  WARNING: Could not read {fpath.name}: {e}")
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["ID"] = pd.to_numeric(combined["ID"], errors="coerce")
    print(f"[Stage 1] Loaded {len(combined):,} rows from {len(frames)} CSV files")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Per-rater statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rater_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per rater."""
    # All numeric rating values (both L and R sides, per answer)
    # We melt L and R sides to get individual answer ratings
    rows = []
    for rater, grp in df.groupby("rater"):
        # Collect all individual ratings (6 per evaluation: 3 dims × 2 sides)
        all_ratings = grp[RATING_COLS[:-1]].values.flatten()
        all_ratings = all_ratings[~np.isnan(all_ratings)]
        slider_vals = grp["comparison_slider"].dropna().values

        row = {
            "rater": rater,
            "n_evaluations": len(grp),
            "n_unique_ips": grp["ip"].nunique(),
            "mean_rating": np.mean(all_ratings) if len(all_ratings) else np.nan,
            "std_rating":  np.std(all_ratings)  if len(all_ratings) else np.nan,
            "median_rating": np.median(all_ratings) if len(all_ratings) else np.nan,
            "skewness_rating": pd.Series(all_ratings).skew() if len(all_ratings) > 2 else np.nan,
            "mean_slider": np.mean(slider_vals) if len(slider_vals) else np.nan,
            "pct_positive_ratings": np.mean(all_ratings > 0) * 100 if len(all_ratings) else np.nan,
            "pct_negative_ratings": np.mean(all_ratings < 0) * 100 if len(all_ratings) else np.nan,
            "mean_consistent": grp[["consistent_L", "consistent_R"]].values.flatten()[
                ~np.isnan(grp[["consistent_L", "consistent_R"]].values.flatten())].mean()
                if len(grp) else np.nan,
            "mean_correct": grp[["correct_L", "correct_R"]].values.flatten()[
                ~np.isnan(grp[["correct_L", "correct_R"]].values.flatten())].mean()
                if len(grp) else np.nan,
            "mean_useful": grp[["useful_L", "useful_R"]].values.flatten()[
                ~np.isnan(grp[["useful_L", "useful_R"]].values.flatten())].mean()
                if len(grp) else np.nan,
        }
        rows.append(row)

    stats = pd.DataFrame(rows).sort_values("n_evaluations", ascending=False).reset_index(drop=True)
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def _melt_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Melt L and R side ratings into long format: rater, dimension, value."""
    records = []
    for _, row in df.iterrows():
        for dim in ["consistent", "correct", "useful"]:
            records.append({"rater": row["rater"], "dimension": dim, "value": row[f"{dim}_L"]})
            records.append({"rater": row["rater"], "dimension": dim, "value": row[f"{dim}_R"]})
    return pd.DataFrame(records).dropna()


def plot_rating_distributions(df: pd.DataFrame, plots_dir: Path):
    """Violin plots of rating distributions per rater, per dimension."""
    long = _melt_ratings(df)
    rater_order = df.groupby("rater").size().sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Rating Distributions by Rater and Dimension", fontsize=14, fontweight="bold")

    for ax, dim in zip(axes, ["consistent", "correct", "useful"]):
        sub = long[long["dimension"] == dim]
        sns.violinplot(
            data=sub, x="rater", y="value",
            order=rater_order, hue="rater", hue_order=rater_order,
            ax=ax, palette="Set2", inner="box", cut=0, legend=False,
        )
        ax.set_title(dim.capitalize())
        ax.set_xlabel("")
        ax.set_ylabel("Rating (−2 to +2)" if ax == axes[0] else "")
        ax.set_ylim(-2.5, 2.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out = plots_dir / "rating_distributions_by_rater.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_mean_ratings_heatmap(stats: pd.DataFrame, plots_dir: Path):
    """Heatmap: rater × dimension mean rating."""
    heat_data = stats.set_index("rater")[["mean_consistent", "mean_correct", "mean_useful"]]
    heat_data.columns = ["Consistent", "Correct", "Useful"]

    fig, ax = plt.subplots(figsize=(7, max(4, len(stats) * 0.55)))
    sns.heatmap(
        heat_data, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-2, vmax=2, linewidths=0.5, ax=ax
    )
    ax.set_title("Mean Rating per Rater and Dimension", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Rater")
    plt.tight_layout()
    out = plots_dir / "mean_ratings_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_rating_count_bar(stats: pd.DataFrame, plots_dir: Path):
    """Bar chart of number of evaluations per rater."""
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#e74c3c" if "anon" in r.lower() else "#3498db" for r in stats["rater"]]
    ax.barh(stats["rater"], stats["n_evaluations"], color=colors)
    ax.set_xlabel("Number of Evaluations")
    ax.set_title("Evaluations per Rater\n(red = Anonymous, blue = Named)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    for i, v in enumerate(stats["n_evaluations"]):
        ax.text(v + 0.5, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    out = plots_dir / "evaluations_per_rater.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_dimension_correlation(df: pd.DataFrame, plots_dir: Path):
    """Scatter-matrix of consistent / correct / useful (all raters combined)."""
    # Bug #2 fix: keep all 3 dimensions together per observation so that
    # dropna() removes the same rows from every column.  Independent
    # dropna() per column followed by reset_index() stitches together
    # ratings from different evaluations, producing misaligned correlations.
    left = df[["consistent_L", "correct_L", "useful_L"]].rename(
        columns={"consistent_L": "consistent", "correct_L": "correct", "useful_L": "useful"})
    right = df[["consistent_R", "correct_R", "useful_R"]].rename(
        columns={"consistent_R": "consistent", "correct_R": "correct", "useful_R": "useful"})
    dim_df = pd.concat([left, right], ignore_index=True).dropna()

    rng = np.random.RandomState(42)  # Bug #16 fix: seeded RNG for reproducible jitter
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    dims = ["consistent", "correct", "useful"]
    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            ax = axes[i][j]
            if i == j:
                ax.hist(dim_df[d1], bins=9, range=(-2.5, 2.5), color="#5dade2", edgecolor="white")
                ax.set_title(d1.capitalize(), fontsize=10)
            else:
                jitter = rng.uniform(-0.15, 0.15, size=len(dim_df))
                ax.scatter(dim_df[d2] + jitter, dim_df[d1] + jitter, alpha=0.3, s=8, color="#5dade2")
                corr = dim_df[[d1, d2]].corr().iloc[0, 1]
                ax.set_title(f"r={corr:.2f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(d1.capitalize(), fontsize=9)
            if i == 2:
                ax.set_xlabel(d2.capitalize(), fontsize=9)
    fig.suptitle("Dimension Correlation Matrix (all raters)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / "dimension_correlation_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_slider_distribution(df: pd.DataFrame, plots_dir: Path):
    """Distribution of comparison_slider values per rater."""
    rater_order = df.groupby("rater").size().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    # Stacked bar: counts per slider value (−2..+2) per rater
    slider_vals = [-2, -1, 0, 1, 2]
    cmap = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(slider_vals)))
    bottoms = np.zeros(len(rater_order))

    for val, color in zip(slider_vals, cmap):
        counts = [df[(df["rater"] == r) & (df["comparison_slider"] == val)].shape[0]
                  for r in rater_order]
        ax.bar(rater_order, counts, bottom=bottoms, label=str(val), color=color)
        bottoms += np.array(counts, dtype=float)

    ax.set_xlabel("Rater")
    ax.set_ylabel("Count")
    ax.set_title("Comparison Slider Distribution per Rater\n(−2=strongly prefer Left, +2=strongly prefer Right)",
                 fontsize=11, fontweight="bold")
    ax.legend(title="Slider value", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    out = plots_dir / "slider_distribution_per_rater.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Stage 1: Data Parsing & Exploratory Analysis")
    print("=" * 60)

    # --- Parse evaluations ---
    eval_df = parse_eval_files(EVAL_DIR)

    # --- Rater summary ---
    rater_stats = compute_rater_stats(eval_df)
    print("\nPer-rater statistics:")
    print(rater_stats[["rater", "n_evaluations", "mean_rating", "std_rating",
                        "skewness_rating", "mean_correct", "mean_useful"]].to_string(index=False))

    # --- Save CSVs ---
    eval_df.to_csv(RESULTS_DIR / "evaluations_parsed.csv", index=False)
    rater_stats.to_csv(RESULTS_DIR / "rater_stats.csv", index=False)
    print(f"\nSaved evaluations_parsed.csv ({len(eval_df)} rows)")
    print(f"Saved rater_stats.csv ({len(rater_stats)} rows)")

    # --- Load datasets (for Stage 2 join) ---
    datasets_df = load_datasets(DATASET_DIR)
    if not datasets_df.empty:
        datasets_df.to_csv(RESULTS_DIR / "datasets_combined.csv", index=False)
        print(f"Saved datasets_combined.csv ({len(datasets_df):,} rows)")

    # --- Visualizations ---
    print("\nGenerating plots...")
    plot_rating_distributions(eval_df, PLOTS_DIR)
    plot_mean_ratings_heatmap(rater_stats, PLOTS_DIR)
    plot_rating_count_bar(rater_stats, PLOTS_DIR)
    plot_dimension_correlation(eval_df, PLOTS_DIR)
    plot_slider_distribution(eval_df, PLOTS_DIR)

    # --- Basic dataset coverage stats ---
    print("\n--- Dataset coverage ---")
    print(f"  Total evaluations:      {len(eval_df)}")
    print(f"  Unique raters:          {eval_df['rater'].nunique()}")
    print(f"  Unique IDs (left):      {eval_df['id_L'].nunique()}")
    print(f"  Unique IDs (right):     {eval_df['id_R'].nunique()}")
    print(f"  Date range:             {eval_df['datetime'].min()} → {eval_df['datetime'].max()}")
    print(f"  Models seen:            {sorted(eval_df['model_L'].dropna().unique().tolist())}")

    print("\n[Stage 1] DONE. Outputs in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
