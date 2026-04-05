"""
Stage 4: Bias-Corrected Reward Model Training
==============================================
Trains a CodeBERT-based multi-head reward model with rater-reliability
weighted labels. Compares against an unweighted (uniform) baseline.

Architecture:
  CodeBERT-base (frozen) → embeddings (768d)
  build_feature_vector() → [Q, A, |Q-A|, Q*A] (3072d)
  MultiHeadClassifier → {consistent, correct, useful} scores

Label aggregation:
  weighted:   label = Σ(w_r * rating_r) / Σ(w_r)   (binarized at threshold=0)
  baseline:   label = mean of all ratings             (binarized at threshold=0)

Depends on:
  - stage1_eda/results/evaluations_parsed.csv
  - stage3_expertise/results/rater_weights.json
  - Version_1/stage4/model.py  (MultiHeadClassifier + build_feature_vector)

Usage:
    python Version_2/stage4_reward_model/train_weighted.py
    python Version_2/stage4_reward_model/train_weighted.py --mode baseline
    python Version_2/stage4_reward_model/train_weighted.py --mode both
"""

import argparse
import json
import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
V1_STAGE4   = BASE_DIR.parent / "Version_1" / "stage4"   # reuse model.py
STAGE1_DIR  = BASE_DIR / "stage1_eda" / "results"
STAGE3_DIR  = BASE_DIR / "stage3_expertise" / "results"
RESULTS_DIR = Path(__file__).parent / "results"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Add Version_1/stage4 to path to reuse model.py
sys.path.insert(0, str(V1_STAGE4))

# ── Hyperparameters ─────────────────────────────────────────────────────────
EMBEDDING_DIM = 768
HIDDEN_DIM    = 512
DROPOUT       = 0.3
LEARNING_RATE = 2e-4
N_EPOCHS      = 20
BATCH_SIZE    = 16
LABEL_THRESHOLD = 0.0     # binarize: rating >= 0 → positive (neutral counts as positive)
VAL_SPLIT     = 0.2
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Early stopping on validation loss (upper bound = N_EPOCHS)
EARLY_STOP_PATIENCE   = 5
EARLY_STOP_MIN_DELTA  = 1e-4

# Maximum per-sample weight cap.
# Prevents raters with very few evaluations (e.g. Arthur: 4 evals, weight≈0.22)
# from dominating training.  After capping, weights are still normalised by
# their sum inside the loss, so the absolute scale does not matter.
MAX_WEIGHT_CAP = 0.15

USE_FOCAL_LOSS = True   # Focal loss to focus on hard examples and counter class imbalance
FOCAL_GAMMA    = 2.0    # Standard focal loss focusing parameter


# ═══════════════════════════════════════════════════════════════════════════════
# Model import (reuse from Version 1)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from model import MultiHeadClassifier, build_feature_vector
    print("[Stage 4] Imported MultiHeadClassifier from Version_1/stage4/model.py")
except ImportError:
    print("[Stage 4] model.py not found at expected path — using inline fallback definition")

    def build_feature_vector(q_emb: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        if a_emb.dim() == 1:
            a_emb = a_emb.unsqueeze(0)
        diff = torch.abs(q_emb - a_emb)
        prod = q_emb * a_emb
        return torch.cat([q_emb, a_emb, diff, prod], dim=-1)

    class MultiHeadClassifier(nn.Module):
        def __init__(self, embedding_dim: int, hidden_dim: int = 512, dropout: float = 0.3):
            super().__init__()
            input_dim = embedding_dim * 4
            self.shared_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            )
            self.head_consistent = nn.Linear(hidden_dim, 1)
            self.head_correct    = nn.Linear(hidden_dim, 1)
            self.head_useful     = nn.Linear(hidden_dim, 1)

        def forward(self, features: torch.Tensor) -> Dict:
            h = self.shared_encoder(features)
            return {"consistent": self.head_consistent(h).squeeze(-1),
                    "correct":    self.head_correct(h).squeeze(-1),
                    "useful":     self.head_useful(h).squeeze(-1)}


# ═══════════════════════════════════════════════════════════════════════════════
# Focal loss
# ═══════════════════════════════════════════════════════════════════════════════

def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                   pos_weight: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal BCE loss: down-weights easy examples to focus training on hard ones.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    Compatible with pos_weight for class-imbalance correction.
    Returns per-sample loss (unreduced) — caller applies sample weights.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    return focal_weight * bce


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding encoder
# ═══════════════════════════════════════════════════════════════════════════════

class CodeBERTEncoder:
    """Encode text using CodeBERT (microsoft/codebert-base)."""

    def __init__(self, device: str = "cpu"):
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.model = AutoModel.from_pretrained("microsoft/codebert-base")
            self.model.eval()
            self.model.to(device)
            self.device = device
            self.available = True
            print("[Stage 4] CodeBERT loaded successfully")
        except Exception as e:
            print(f"[Stage 4] CodeBERT unavailable ({e}). Using text-hash fallback encoder.")
            self.available = False
            self.device = device

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        if not self.available:
            return self._hash_encode(texts)

        from transformers import AutoTokenizer
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            # CLS token embedding
            emb = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)

    def _hash_encode(self, texts: List[str]) -> torch.Tensor:
        """Deterministic hash-based fallback embedding (reproducible)."""
        import hashlib
        embs = []
        for text in texts:
            text = text or ""
            h = hashlib.md5(text.encode()).hexdigest()
            seed = int(h[:8], 16) % (2**31)
            rng = np.random.RandomState(seed)
            embs.append(rng.randn(EMBEDDING_DIM).astype(np.float32))
        return torch.tensor(np.array(embs))


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class RewardDataset(Dataset):
    """
    Each item: (feature_vector, labels_dict, sample_weight)
    where labels_dict = {consistent, correct, useful} as binary floats,
    and sample_weight is the rater reliability weight for this observation.

    For the baseline (uniform) model, all sample_weights = 1/N.
    For the weighted model, sample_weights = reliability weight of the rater
    who produced this observation — so every individual (q,a,rater) triplet
    contributes proportionally to that rater's reliability, regardless of
    whether the (q,a) pair was rated by one rater or many.
    """

    def __init__(self, features: torch.Tensor,
                 labels: Dict[str, torch.Tensor],
                 sample_weights: torch.Tensor):
        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx],
                {k: v[idx] for k, v in self.labels.items()},
                self.sample_weights[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# Label aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def build_individual_samples(eval_df: pd.DataFrame, rater_weights: Dict[str, float],
                              weighted: bool = True) -> pd.DataFrame:
    """
    Return individual (question, answer, rater, sample_weight, labels) rows —
    one row per answer per evaluation.

    WHY individual rows instead of aggregating:
      For the 93% of (q,a) pairs with only one rater, label aggregation gives
      label_weighted = w_r * rating / w_r = rating — identical to unweighted.
      Per-sample weights in the training loss are the only mechanism that
      actually distinguishes the two models across all 614 training samples.

    For the weighted model: sample_weight = reliability weight of the rater.
    For the baseline model: sample_weight = 1.0 for all (uniform).
    Labels are binarized at LABEL_THRESHOLD (rating >= 0 → positive; neutral included as positive).
    """
    records = []
    n_raters = len(eval_df["rater"].unique())
    uniform_w = 1.0 / n_raters if n_raters > 0 else 1.0

    for _, row in eval_df.iterrows():
        rater = row["rater"]
        w = min(rater_weights.get(rater, uniform_w), MAX_WEIGHT_CAP) if weighted else uniform_w
        for side in ["L", "R"]:
            cons = row[f"consistent_{side}"]
            corr = row[f"correct_{side}"]
            use  = row[f"useful_{side}"]
            if any(pd.isna(v) for v in [cons, corr, use]):
                continue
            records.append({
                "question":         str(row.get(f"question_{side}", "") or ""),
                "answer":           str(row.get(f"answer_{side}", "") or ""),
                "rater":            rater,
                "sample_weight":    w,
                # >= includes neutral (0) as positive; > would map neutral to negative (inflating neg class)
                "consistent_label": float(float(cons) >= LABEL_THRESHOLD),
                "correct_label":    float(float(corr) >= LABEL_THRESHOLD),
                "useful_label":     float(float(use)  >= LABEL_THRESHOLD),
            })

    samples_df = pd.DataFrame(records)
    weights_arr = samples_df["sample_weight"].values

    # Sanity check: weights must have non-trivial variance for weighted mode
    if weighted:
        w_var = np.var(weights_arr)
        assert w_var > 1e-10, (
            f"Sample weight variance = {w_var:.2e} — weights are effectively uniform. "
            "Check that rater_weights.json contains differentiated values."
        )
        print(f"  Weight variance: {w_var:.6f}  "
              f"(min={weights_arr.min():.4f}, max={weights_arr.max():.4f}, "
              f"max/min ratio={weights_arr.max()/weights_arr.min():.2f}x)")
    else:
        print(f"  Uniform weights: {uniform_w:.6f} per sample")

    pos_rate = samples_df[["consistent_label","correct_label","useful_label"]].mean()
    print(f"  Individual samples: {len(samples_df)}  "
          f"(positive rates: {pos_rate.round(3).to_dict()})")
    return samples_df


# ═══════════════════════════════════════════════════════════════════════════════
# Build features
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(samples_df: pd.DataFrame, encoder: CodeBERTEncoder,
                   batch_size: int = 32) -> torch.Tensor:
    """Encode questions and answers, then build combined feature vectors."""
    print(f"  Encoding {len(samples_df)} questions...")
    q_embs = encoder.encode(samples_df["question"].tolist(), batch_size=batch_size)
    print(f"  Encoding {len(samples_df)} answers...")
    a_embs = encoder.encode(samples_df["answer"].tolist(), batch_size=batch_size)
    print(f"  Building feature vectors ({q_embs.shape[1]*4}d)...")
    features = build_feature_vector(q_embs, a_embs)
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# Train / Eval
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: str,
                pos_weight: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Training with per-sample reliability weights and class-balance pos_weight.
    Uses BCEWithLogitsLoss(reduction='none') so each sample's loss is scaled
    by its rater's reliability weight before averaging. This is the mechanism
    that actually differentiates the weighted model from the baseline:
      - Baseline: all sample_weights = uniform (1/N_raters), so scaling is flat.
      - Weighted: biased raters (low weight) contribute less gradient per sample,
        regardless of whether the (q,a) pair was rated by one rater or many.
    pos_weight addresses class imbalance separately from rater reliability:
      pos_weight[dim] = n_neg / n_pos so positive examples are up-weighted in loss.
    """
    model.train()
    # One criterion per dimension to carry per-class pos_weight
    bce_fns = {dim: nn.BCEWithLogitsLoss(pos_weight=pos_weight[dim].to(device),
                                          reduction="none")
               for dim in ["consistent", "correct", "useful"]}
    totals = {"loss": 0.0, "consistent": 0.0, "correct": 0.0, "useful": 0.0}
    n = 0

    for features, labels, sample_weights in loader:
        features       = features.to(device)
        labels         = {k: v.float().to(device) for k, v in labels.items()}
        sample_weights = sample_weights.float().to(device)   # shape: (batch,)

        optimizer.zero_grad()
        preds = model(features)

        # Per-sample, per-dimension loss; weight then average
        loss_per_dim = []
        for dim in ["consistent", "correct", "useful"]:
            if USE_FOCAL_LOSS:
                per_sample_loss = focal_bce_loss(
                    preds[dim], labels[dim],
                    pos_weight=pos_weight[dim].to(device), gamma=FOCAL_GAMMA
                )
            else:
                per_sample_loss = bce_fns[dim](preds[dim], labels[dim])  # (batch,)
            weighted_loss   = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
            loss_per_dim.append(weighted_loss)
        loss = sum(loss_per_dim) / 3.0

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = features.size(0)
        totals["loss"] += loss.item() * bs
        for dim in ["consistent", "correct", "useful"]:
            pred_bin = (preds[dim].detach() > 0).float()
            totals[dim] += (pred_bin == labels[dim]).float().mean().item() * bs
        n += bs

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str,
               pos_weight: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Evaluation with per-class accuracy (positive / negative class separately).
    Aggregated accuracy can mask bias-correction effects that help one class
    at the expense of another — per-class breakdown exposes this.
    pos_weight is used in the val loss so it is consistent with training loss.
    """
    model.eval()
    crit = {dim: nn.BCEWithLogitsLoss(pos_weight=pos_weight[dim].to(device))
            for dim in ["consistent", "correct", "useful"]}
    totals = {"loss": 0.0, "consistent": 0.0, "correct": 0.0, "useful": 0.0}
    n = 0
    # Collect all predictions and labels for post-hoc per-class analysis
    collected: Dict[str, List] = {d: {"preds": [], "labels": []}
                                   for d in ["consistent", "correct", "useful"]}

    for features, labels, _weights in loader:
        features = features.to(device)
        labels = {k: v.float().to(device) for k, v in labels.items()}
        preds = model(features)
        loss = (crit["consistent"](preds["consistent"], labels["consistent"]) +
                crit["correct"](   preds["correct"],    labels["correct"]) +
                crit["useful"](    preds["useful"],     labels["useful"])) / 3.0

        bs = features.size(0)
        totals["loss"] += loss.item() * bs
        for dim in ["consistent", "correct", "useful"]:
            pred_bin = (preds[dim] > 0).float()
            totals[dim] += (pred_bin == labels[dim]).float().mean().item() * bs
            collected[dim]["preds"].extend(preds[dim].cpu().numpy().tolist())
            collected[dim]["labels"].extend(labels[dim].cpu().numpy().tolist())
        n += bs

    metrics = {k: v / n for k, v in totals.items()}

    # Per-class accuracy and reward gap for each dimension
    for dim in ["consistent", "correct", "useful"]:
        preds_arr  = np.array(collected[dim]["preds"])
        labels_arr = np.array(collected[dim]["labels"])
        pred_bin   = (preds_arr > 0).astype(float)
        pos_mask   = labels_arr == 1.0
        neg_mask   = labels_arr == 0.0

        metrics[f"{dim}_acc_pos"] = float(pred_bin[pos_mask].mean()) if pos_mask.sum() > 0 else float("nan")
        metrics[f"{dim}_acc_neg"] = float((pred_bin[neg_mask] == 0).mean()) if neg_mask.sum() > 0 else float("nan")
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            metrics[f"{dim}_reward_gap"] = float(preds_arr[pos_mask].mean() - preds_arr[neg_mask].mean())
        else:
            metrics[f"{dim}_reward_gap"] = float("nan")

    # Primary reward gap on correct dimension (kept for backward compat)
    metrics["reward_gap"] = metrics["correct_reward_gap"]
    return metrics


def run_training(mode: str, eval_df: pd.DataFrame, rater_weights: Dict[str, float],
                 encoder: CodeBERTEncoder) -> Dict:
    """Train one model (weighted or baseline). Returns history dict."""
    weighted = (mode == "weighted")
    print(f"\n{'='*50}")
    print(f"Training: {mode.upper()} reward model")
    print(f"{'='*50}")

    # Build individual (q, a, rater) rows so per-sample weights are active
    samples_df = build_individual_samples(eval_df, rater_weights, weighted=weighted)
    samples_df = samples_df.reset_index(drop=True)

    # Stratified split on correct_label, seeded BEFORE encoding.
    # This ensures:
    #   (a) Equal positive/negative ratio in both train and val.
    #   (b) Split is identical across weighted and baseline runs (same SEED).
    #   (c) Split is stable even if dataset size changes between runs.
    # train_test_split on integer positions (np.arange) so the result is
    # always a plain numpy array of row positions — safe for both pandas .iloc
    # and tensor indexing.
    n_samples = len(samples_df)
    positions = np.arange(n_samples)
    train_pos_arr, val_pos_arr = train_test_split(
        positions,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=samples_df["correct_label"].values,
    )
    train_idx = torch.tensor(train_pos_arr)
    val_idx   = torch.tensor(val_pos_arr)

    # Compute class-balance pos_weight from training labels only (never val).
    # Addresses class imbalance separately from rater reliability weighting:
    #   pos_weight[dim] = n_neg_train / n_pos_train, clamped to [1, 10].
    # This prevents the model from collapsing to "always predict negative"
    # when the positive class is underrepresented (~22% for correct/useful).
    train_df = samples_df.iloc[train_pos_arr]
    pos_weight: Dict[str, torch.Tensor] = {}
    for dim in ["consistent", "correct", "useful"]:
        col   = f"{dim}_label"
        n_pos = int((train_df[col] == 1).sum())
        n_neg = int((train_df[col] == 0).sum())
        ratio = n_neg / max(n_pos, 1)
        pos_weight[dim] = torch.tensor([min(ratio, 30.0)], dtype=torch.float32)
        print(f"  pos_weight[{dim}] = {pos_weight[dim].item():.2f}  "
              f"(train n_pos={n_pos}, n_neg={n_neg})")
        if ratio > 5.0:
            warnings.warn(
                f"[Stage 4] Class imbalance WARNING: {dim} ratio={ratio:.1f}:1 "
                f"(n_neg={n_neg}, n_pos={n_pos}). "
                "Positive class accuracy may be unreliable. Consider focal loss.",
                RuntimeWarning, stacklevel=2
            )

    # Log label balance in each split for reproducibility checks
    train_pos = samples_df.iloc[train_pos_arr]["correct_label"].mean()
    val_pos   = samples_df.iloc[val_pos_arr]["correct_label"].mean()
    print(f"  Stratified split — train correct_pos={train_pos:.3f}, val correct_pos={val_pos:.3f}")

    features   = build_features(samples_df, encoder)
    labels = {
        "consistent": torch.tensor(samples_df["consistent_label"].values, dtype=torch.float32),
        "correct":    torch.tensor(samples_df["correct_label"].values,    dtype=torch.float32),
        "useful":     torch.tensor(samples_df["useful_label"].values,     dtype=torch.float32),
    }
    sample_weights = torch.tensor(samples_df["sample_weight"].values, dtype=torch.float32)

    train_ds = RewardDataset(features[train_idx], {k: v[train_idx] for k, v in labels.items()},
                             sample_weights[train_idx])
    val_ds   = RewardDataset(features[val_idx],   {k: v[val_idx]   for k, v in labels.items()},
                             sample_weights[val_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Model
    feat_dim = features.shape[1]
    model = MultiHeadClassifier(embedding_dim=feat_dim // 4, hidden_dim=HIDDEN_DIM,
                                dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # T_max matches max epochs; if early stopping exits sooner, LR schedule is partial (acceptable).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    history = {"train_loss": [], "val_loss": [], "val_correct_acc": [],
               "val_consistent_acc": [], "val_useful_acc": [], "val_reward_gap": [],
               "val_correct_acc_pos": [], "val_correct_acc_neg": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience_ctr = 0
    stopped_early = False

    print(f"\n  Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Device: {DEVICE}")
    print(f"  Max weight cap: {MAX_WEIGHT_CAP} ({'active' if weighted else 'n/a — baseline'})")
    print(f"  Early stopping: val_loss, patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}")

    for epoch in range(1, N_EPOCHS + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE, pos_weight)
        val_metrics   = eval_epoch(model, val_loader,   DEVICE, pos_weight)
        scheduler.step()

        vloss = val_metrics["loss"]
        val_acc = (val_metrics["correct"] + val_metrics["consistent"] + val_metrics["useful"]) / 3
        if vloss < best_val_loss - EARLY_STOP_MIN_DELTA:
            best_val_loss = vloss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), CKPT_DIR / f"best_{mode}.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                stopped_early = True
                print(f"  Early stopping at epoch {epoch} (no val_loss improvement for {EARLY_STOP_PATIENCE} epochs).")
                break

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_correct_acc"].append(val_metrics["correct"])
        history["val_consistent_acc"].append(val_metrics["consistent"])
        history["val_useful_acc"].append(val_metrics["useful"])
        history["val_reward_gap"].append(val_metrics.get("reward_gap", float("nan")))
        history["val_correct_acc_pos"].append(val_metrics.get("correct_acc_pos", float("nan")))
        history["val_correct_acc_neg"].append(val_metrics.get("correct_acc_neg", float("nan")))

        if epoch % 5 == 0 or epoch == 1:
            gap = val_metrics.get("reward_gap", float("nan"))
            pos = val_metrics.get("correct_acc_pos", float("nan"))
            neg = val_metrics.get("correct_acc_neg", float("nan"))
            print(f"  Epoch {epoch:3d}/{N_EPOCHS} | "
                  f"train_loss={train_metrics['loss']:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f} | "
                  f"correct_acc={val_metrics['correct']:.4f} "
                  f"[pos={pos:.3f} neg={neg:.3f}] | "
                  f"gap={'N/A' if np.isnan(gap) else f'{gap:.4f}'}")

    print(f"\n  Best val loss: {best_val_loss:.4f} at epoch {best_epoch} "
          f"(val_acc snapshot={best_val_acc:.4f})")
    history["best_val_acc"] = best_val_acc
    history["best_epoch"]   = best_epoch
    history["best_val_loss"] = best_val_loss
    history["stopped_early"] = stopped_early
    history["mode"] = mode

    # Threshold tuning on validation set + save per-sample predictions for McNemar's test
    best_ckpt = CKPT_DIR / f"best_{mode}.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        model.eval()
        all_logits = {d: [] for d in ["consistent", "correct", "useful"]}
        all_labels_collected = {d: [] for d in ["consistent", "correct", "useful"]}
        with torch.no_grad():
            for features_b, labels_b, _ in val_loader:
                preds_b = model(features_b.to(DEVICE))
                for d in ["consistent", "correct", "useful"]:
                    all_logits[d].extend(preds_b[d].cpu().numpy().tolist())
                    all_labels_collected[d].extend(labels_b[d].numpy().tolist())

        # Save per-sample predictions for Stage 5 McNemar's test
        val_pred_records = []
        for i in range(len(all_logits["correct"])):
            val_pred_records.append({
                "correct_logit": all_logits["correct"][i],
                "correct_label": all_labels_collected["correct"][i],
            })
        pd.DataFrame(val_pred_records).to_csv(
            RESULTS_DIR / f"val_predictions_{mode}.csv", index=False
        )
        print(f"  Saved val_predictions_{mode}.csv ({len(val_pred_records)} samples)")

        # Grid search over thresholds [-2, 2] to maximize validation F1 per dimension
        best_thresholds = {}
        for d in ["consistent", "correct", "useful"]:
            logits_arr = np.array(all_logits[d])
            labels_arr = np.array(all_labels_collected[d])
            best_f1, best_t = 0.0, 0.0
            for t in np.linspace(-2.0, 2.0, 41):
                preds_bin = (logits_arr > t).astype(float)
                tp = float(((preds_bin == 1) & (labels_arr == 1)).sum())
                fp = float(((preds_bin == 1) & (labels_arr == 0)).sum())
                fn = float(((preds_bin == 0) & (labels_arr == 1)).sum())
                prec = tp / max(tp + fp, 1)
                rec  = tp / max(tp + fn, 1)
                f1   = 2 * prec * rec / max(prec + rec, 1e-8)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            best_thresholds[d] = best_t
            print(f"  [{mode}] Tuned threshold ({d}): {best_t:.2f}  (val F1={best_f1:.3f})")
        history["best_thresholds"] = best_thresholds

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison Plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(histories: List[Dict], results_dir: Path):
    """
    Plot weighted vs baseline training curves.
    2×3 grid: top row = overall metrics, bottom row = per-class accuracy.
    Per-class breakdown can reveal bias-correction helping one class at the
    expense of another — which aggregated accuracy would hide.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Weighted vs Baseline Reward Model Comparison\n"
                 "(top: overall  |  bottom: per-class accuracy on 'correct' dimension)",
                 fontsize=13, fontweight="bold")

    colors       = {"weighted": "#2ecc71", "baseline": "#e74c3c"}
    linestyles   = {"weighted": "-",       "baseline": "--"}
    top_metrics  = [
        ("val_correct_acc",  "Val Correct Accuracy (overall)"),
        ("val_loss",         "Validation Loss"),
        ("val_reward_gap",   "Reward Gap (correct)"),
    ]
    bot_metrics  = [
        ("val_correct_acc_pos", "Positive Class Accuracy\n(model answer rated correct)"),
        ("val_correct_acc_neg", "Negative Class Accuracy\n(model answer rated incorrect)"),
        ("val_useful_acc",      "Useful Accuracy (overall)"),
    ]

    def _plot_series(ax, histories, metric, title):
        for h in histories:
            mode = h["mode"]
            vals = h.get(metric, [])
            if not vals:
                return
            epochs = list(range(1, len(vals) + 1))
            clean = [(e, v) for e, v in zip(epochs, vals) if not (isinstance(v, float) and np.isnan(v))]
            if clean:
                es, vs = zip(*clean)
                ax.plot(es, vs, label=mode, color=colors.get(mode, "gray"),
                        linestyle=linestyles.get(mode, "-"), linewidth=2,
                        marker="o" if len(vs) <= 20 else None, markersize=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    for ax, (metric, title) in zip(axes[0], top_metrics):
        _plot_series(ax, histories, metric, title)
    for ax, (metric, title) in zip(axes[1], bot_metrics):
        _plot_series(ax, histories, metric, title)

    plt.tight_layout()
    out = results_dir / "weighted_vs_baseline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["weighted", "baseline", "both"],
                        default="both", help="Which model to train")
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 4: Bias-Corrected Reward Model Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load data
    eval_path = STAGE1_DIR / "evaluations_parsed.csv"
    if not eval_path.exists():
        sys.exit(f"[Error] Run Stage 1 first. Expected: {eval_path}")
    eval_df = pd.read_csv(eval_path, low_memory=False)
    for col in ["consistent_L","correct_L","useful_L","consistent_R","correct_R","useful_R"]:
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

    # Load rater weights
    weights_path = STAGE3_DIR / "rater_weights.json"
    if weights_path.exists():
        with open(weights_path) as f:
            weights_raw = json.load(f)
        rater_weights = {k: v["weight"] for k, v in weights_raw.items()}
        print(f"\nLoaded rater weights for {len(rater_weights)} raters:")
        for r, w in sorted(rater_weights.items(), key=lambda x: -x[1]):
            print(f"  {r:20s}: {w:.6f}")
    else:
        print("[Warning] rater_weights.json not found. Run Stage 3 first.")
        print("  Using uniform weights for demonstration.")
        rater_weights = {r: 1.0 for r in eval_df["rater"].unique()}

    # Encoder
    encoder = CodeBERTEncoder(device=DEVICE)

    # Train
    modes = []
    if args.mode in ("weighted", "both"):
        modes.append("weighted")
    if args.mode in ("baseline", "both"):
        modes.append("baseline")

    histories = []
    all_results = {}
    for mode in modes:
        h = run_training(mode, eval_df, rater_weights, encoder)
        histories.append(h)
        all_results[mode] = {
            "best_val_acc": h["best_val_acc"],
            "best_epoch":   h["best_epoch"],
            "best_val_loss": h.get("best_val_loss"),
            "stopped_early": h.get("stopped_early", False),
            "final_val_correct_acc":       h["val_correct_acc"][-1],
            "final_val_consistent_acc":    h["val_consistent_acc"][-1],
            "final_val_useful_acc":        h["val_useful_acc"][-1],
            "final_reward_gap":            h["val_reward_gap"][-1],
            "best_correct_acc_pos":        h["val_correct_acc_pos"][h["best_epoch"] - 1],
            "best_correct_acc_neg":        h["val_correct_acc_neg"][h["best_epoch"] - 1],
            "best_thresholds":             h.get("best_thresholds", {}),
        }

    all_results["rater_weights_for_training"] = {
        "max_weight_cap": float(MAX_WEIGHT_CAP),
        "weight_cap_applies_to_weighted_only": True,
        "effective_rater_weight": {
            r: round(min(float(rater_weights[r]), MAX_WEIGHT_CAP), 6) for r in rater_weights
        },
    }

    # Save results
    with open(RESULTS_DIR / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save history CSVs (only epochs actually run)
    for h in histories:
        mode = h["mode"]
        n_ep = len(h["train_loss"])
        hist_df = pd.DataFrame({
            "epoch":              range(1, n_ep + 1),
            "train_loss":         h["train_loss"],
            "val_loss":           h["val_loss"],
            "val_correct_acc":    h["val_correct_acc"],
            "val_consistent_acc": h["val_consistent_acc"],
            "val_useful_acc":     h["val_useful_acc"],
            "val_reward_gap":     h["val_reward_gap"],
            "val_correct_acc_pos": h["val_correct_acc_pos"],
            "val_correct_acc_neg": h["val_correct_acc_neg"],
        })
        hist_df.to_csv(RESULTS_DIR / f"history_{mode}.csv", index=False)

    # Summary
    print("\n=== FINAL COMPARISON ===")
    for mode, res in all_results.items():
        if mode not in ("weighted", "baseline"):
            continue
        pos = res.get("best_correct_acc_pos", float("nan"))
        neg = res.get("best_correct_acc_neg", float("nan"))
        print(f"\n  [{mode.upper()}]")
        print(f"    Best val accuracy (at best val_loss epoch): {res['best_val_acc']:.4f} (epoch {res['best_epoch']})")
        if res.get("stopped_early"):
            print(f"    Stopped early: yes (best val_loss={res.get('best_val_loss')})")
        print(f"    Final val correct acc:       {res['final_val_correct_acc']:.4f}")
        print(f"      — positive class acc:      {pos:.4f}" if not np.isnan(pos) else "")
        print(f"      — negative class acc:      {neg:.4f}" if not np.isnan(neg) else "")
        print(f"    Final reward gap:            {res['final_reward_gap']}")

    if len(histories) == 2:
        w_res = all_results["weighted"]
        b_res = all_results["baseline"]
        diff       = w_res["best_val_acc"] - b_res["best_val_acc"]
        diff_pos   = w_res.get("best_correct_acc_pos", float("nan")) - \
                     b_res.get("best_correct_acc_pos", float("nan"))
        diff_neg   = w_res.get("best_correct_acc_neg", float("nan")) - \
                     b_res.get("best_correct_acc_neg", float("nan"))
        print(f"\n  Weighted vs Baseline:")
        print(f"    dacc (overall):   {diff:+.4f}  "
              f"({'improvement' if diff > 0.001 else 'within noise' if abs(diff) <= 0.01 else 'regression'})")
        if not np.isnan(diff_pos):
            print(f"    dacc (pos class): {diff_pos:+.4f}")
        if not np.isnan(diff_neg):
            print(f"    dacc (neg class): {diff_neg:+.4f}")
        if abs(diff) <= 0.01:
            print("\n  NOTE: |dacc| <= 0.01 is within noise for n=614. "
                  "Report as 'no practically meaningful accuracy difference (heuristic, no formal test performed)' — "
                  "frame contribution around bias detection methodology, not accuracy gain.")
        plot_comparison(histories, RESULTS_DIR)

    print("\n[Stage 4] DONE. Outputs in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
