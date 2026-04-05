"""
Aggregate multi-seed training results for Stage 4 classifiers.

Reads per-seed training history JSONs from:
    stage4/<classifier>/artifacts/seed_<seed>/training_history_<metric>.json

and writes a summary file:
    stage4/<classifier>/artifacts/multiseed_summary_<metric>.json

containing mean ± std per epoch across all available seeds.

Usage:
    python stage4/aggregate_seeds.py
    python stage4/aggregate_seeds.py --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASSIFIERS = {
    "stage4A_consist": ("artifacts", "training_history_consistent.json"),
    "stage4B_corct":   ("artifacts", "training_history_correct.json"),
    "stage4C_useful":  ("artifacts", "training_history_useful.json"),
}

DEFAULT_SEEDS = [42, 123, 456]


def load_history(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(histories: List[List[Dict]], seeds: List[int]) -> Dict:
    """Aggregate per-epoch metrics across seeds.

    Returns a dict with keys:
      - epochs: sorted list of epoch numbers present in all seeds
      - per_epoch: { epoch_num: { metric: {mean, std, per_seed} } }
      - final_epoch: aggregate of the last epoch only
    """
    # Find common epochs
    all_epoch_sets = [
        {e["epoch"] for e in h if "epoch" in e}
        for h in histories
    ]
    common_epochs = sorted(set.intersection(*all_epoch_sets)) if all_epoch_sets else []

    metric_keys = [
        k for k in (histories[0][0].keys() if histories else [])
        if k != "epoch" and isinstance(histories[0][0].get(k), (int, float, type(None)))
    ]

    per_epoch: Dict[int, Dict] = {}
    for ep in common_epochs:
        ep_data: Dict[str, Dict] = {}
        for k in metric_keys:
            vals = []
            for h, seed in zip(histories, seeds):
                row = next((e for e in h if e.get("epoch") == ep), None)
                if row is not None and row.get(k) is not None:
                    vals.append((seed, float(row[k])))
            if vals:
                seed_vals = [v for _, v in vals]
                ep_data[k] = {
                    "mean": float(np.mean(seed_vals)),
                    "std":  float(np.std(seed_vals)),
                    "per_seed": {str(s): v for s, v in vals},
                }
        per_epoch[ep] = ep_data

    final_epoch_num = common_epochs[-1] if common_epochs else None
    final_agg = per_epoch.get(final_epoch_num, {}) if final_epoch_num else {}

    return {
        "seeds": seeds,
        "epochs": common_epochs,
        "per_epoch": {str(k): v for k, v in per_epoch.items()},
        "final_epoch": final_epoch_num,
        "final_epoch_summary": final_agg,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate Stage 4 multi-seed results")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Seeds to aggregate (default: 42 123 456)")
    args = parser.parse_args()
    seeds = args.seeds

    for clf_dir, (artifacts_subdir, history_filename) in CLASSIFIERS.items():
        print(f"\n{'='*60}")
        print(f"Classifier: {clf_dir}")
        base = REPO_ROOT / "stage4" / clf_dir / artifacts_subdir

        histories: List[List[Dict]] = []
        used_seeds: List[int] = []

        for seed in seeds:
            if seed == 42:
                hist_path = base / history_filename
            else:
                hist_path = base / f"seed_{seed}" / history_filename

            h = load_history(hist_path)
            if h is not None:
                histories.append(h)
                used_seeds.append(seed)
                print(f"  seed={seed}: {len(h)} epochs — {hist_path}")
            else:
                print(f"  seed={seed}: NOT FOUND — {hist_path}")

        if not histories:
            print(f"  No histories found for {clf_dir}. Run training first.")
            continue

        if len(histories) == 1:
            print(f"  Only 1 seed available — mean=value, std=0.0")

        result = aggregate(histories, used_seeds)

        summary_path = base / f"multiseed_summary_{history_filename}"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  Summary saved: {summary_path}")

        # Print final-epoch table
        final = result.get("final_epoch_summary", {})
        if final:
            print(f"\n  Final epoch ({result['final_epoch']}) mean ± std:")
            for k, v in final.items():
                if isinstance(v, dict) and "mean" in v:
                    print(f"    {k:<30} {v['mean']:.4f} ± {v['std']:.4f}")


if __name__ == "__main__":
    main()
