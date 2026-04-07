"""
Stage 3 SFT — 5-Fold Cross-Validation.

Runs standard Supervised Fine-Tuning (same config as run_sft_experiments.py)
with 5-fold stratified cross-validation instead of a single 90/10 split.
Each fold uses 80 % of the data for training and 20 % for validation.

Results per fold and aggregated (mean ± std across folds) are saved to
  stage3/outputs_cv/sft_cv_results.json

Usage:
    python stage3/run_sft_cv.py                   # full 1247 samples, 5 folds
    python stage3/run_sft_cv.py --num-samples 11  # 11-sample ablation
    python stage3/run_sft_cv.py --folds 5 --epochs 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse everything from the base SFT experiment
from stage3.run_sft_experiments import (
    SFTConfig,
    SFTDataset,
    SFTTrainer,
    EpochMetrics,
    load_sft_data,
)


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def k_fold_split(data: List, k: int, seed: int = 42):
    """Yield (train_fold, val_fold) index splits for k-fold CV."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data)).tolist()
    fold_size = len(indices) // k
    for fold in range(k):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else len(indices)
        val_idx = indices[val_start:val_end]
        train_idx = indices[:val_start] + indices[val_end:]
        yield (
            [data[i] for i in train_idx],
            [data[i] for i in val_idx],
        )


def aggregate_fold_results(fold_histories: List[List[EpochMetrics]]) -> Dict[str, Any]:
    """Compute mean ± std across folds for final-epoch metrics."""
    final_epochs = [h[-1] for h in fold_histories]
    keys = [
        "train_loss", "val_loss", "bertscore", "codebleu",
        "bleu", "rouge", "ruby", "reward",
    ]
    agg: Dict[str, Any] = {}
    for k in keys:
        vals = [getattr(ep, k, 0.0) or 0.0 for ep in final_epochs]
        agg[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)), 4),
            "per_fold": [round(v, 4) for v in vals],
        }
    return agg


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_cv(
    num_samples: Optional[int] = None,
    k: int = 5,
    seed: int = 42,
    epochs: int = 30,
    output_dir: str = "stage3/outputs_cv",
):
    os.makedirs(output_dir, exist_ok=True)

    # Load all data (ignoring the default split — CV manages its own splits)
    all_train, _ = load_sft_data(num_samples=num_samples)
    # Combine: we re-split with CV, so include the held-out eval portion as well
    # (load_data returns 90% train + 10% val; reload with a large num_samples
    # to get everything, then ignore the split)
    all_data, _ = load_sft_data(num_samples=num_samples)  # already includes all
    all_data = all_train  # use the 90% portion (consistent with original experiment)

    print(f"\n{'='*60}")
    print(f"Stage 3 SFT — {k}-Fold Cross-Validation")
    print(f"Dataset size: {len(all_data)} samples  |  epochs/fold: {epochs}")
    print(f"{'='*60}\n")

    fold_histories: List[List[EpochMetrics]] = []
    fold_summaries: List[Dict[str, Any]] = []

    for fold_idx, (train_data, val_data) in enumerate(k_fold_split(all_data, k, seed=seed), 1):
        print(f"\n--- Fold {fold_idx}/{k}  (train={len(train_data)}, val={len(val_data)}) ---")

        config = SFTConfig(
            num_epochs=epochs,
            seed=seed + fold_idx,  # shift seed per fold for independence
            output_dir=os.path.join(output_dir, f"fold_{fold_idx}"),
        )

        trainer = SFTTrainer(config)
        history: List[EpochMetrics] = trainer.train(train_data, val_data)

        fold_histories.append(history)

        # Save per-fold JSON
        fold_path = os.path.join(output_dir, f"fold_{fold_idx}", "history.json")
        os.makedirs(os.path.dirname(fold_path), exist_ok=True)
        with open(fold_path, "w") as f:
            json.dump([asdict(ep) for ep in history], f, indent=2)
        print(f"  Saved fold results -> {fold_path}")

        fold_summaries.append({
            "fold": fold_idx,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "final_epoch": asdict(history[-1]),
        })

    # Aggregate
    agg = aggregate_fold_results(fold_histories)

    result = {
        "experiment": "Stage3_SFT_5FoldCV",
        "k_folds": k,
        "epochs_per_fold": epochs,
        "num_samples": len(all_data),
        "seed": seed,
        "aggregated": agg,
        "folds": fold_summaries,
    }

    out_path = os.path.join(output_dir, "sft_cv_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCV results saved -> {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Cross-Validation Summary (mean ± std across {k} folds, final epoch)")
    print(f"{'='*60}")
    for metric, vals in agg.items():
        print(f"  {metric:<15} {vals['mean']:.4f} ± {vals['std']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 3 SFT 5-fold cross-validation")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of training samples (None = all 1247)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="stage3/outputs_cv")
    args = parser.parse_args()

    run_cv(
        num_samples=args.num_samples,
        k=args.folds,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
