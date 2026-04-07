"""
Stage 3 Dual-Head SFT — Joint language-modelling + preference classification.

Architecture:
  CodeGPT-small-py (124M)
    ├── Language-model head   (standard causal LM loss, next-token prediction)
    └── Preference head       (Linear(hidden_size → 1) + Sigmoid)
                               binary: does the generated code satisfy human
                               preference? label derived from pairwise_prefs.csv

Loss:
  L = L_lm  +  λ * L_pref
  L_lm   = CrossEntropy over next tokens  (standard SFT objective)
  L_pref = BCE on the [EOS] hidden state  (preference classification)
  λ = 0.5  (preference weight)

The preference labels are derived from `pairwise_prefs.csv`:
  score >= 1  → positive (1)
  score <= -1 → negative (0)
Samples without a preference row keep only L_lm (λ drops to 0 for that batch).

Training details:
  Same SFTConfig as run_sft_experiments.py
  Batch size 2, lr 5e-6, FP32, 30 epochs, AdamW + linear warmup
  Evaluation: same proxy metrics + preference accuracy

Results saved to stage3/outputs_dual_head/dual_head_results.json.

Usage:
    python stage3/run_sft_dual_head.py
    python stage3/run_sft_dual_head.py --epochs 30 --seed 42 --lam 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from stage3.run_sft_experiments import (
    SFTConfig,
    SFTDataset,
    EpochMetrics,
    SimpleMetricsEvaluator,
    load_sft_data,
)

try:
    from modern_rlhf.metrics import ModernMetricsEvaluator, get_metric_float
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dual-head model
# ---------------------------------------------------------------------------

class DualHeadCodeGPT(nn.Module):
    """
    CodeGPT-small-py with an added binary preference classification head.

    The LM head is the existing tied embedding head from the pre-trained model.
    The preference head is a freshly initialised linear probe on the [EOS] token
    hidden state (last non-padding position).
    """

    def __init__(self, base_model: nn.Module, hidden_size: int = 768, dropout: float = 0.1):
        super().__init__()
        self.lm = base_model  # AutoModelForCausalLM instance
        self.pref_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, torch.Tensor]:
        """
        Returns:
            lm_output  — output of the base LM (has .loss if labels given)
            pref_logit — scalar logit per sequence [B]
        """
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        # Last hidden state: [B, T, H]
        last_hidden = out.hidden_states[-1]
        # Use the last non-padding token (EOS position) as the sequence representation
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
        seq_repr = last_hidden[torch.arange(last_hidden.size(0)), seq_lengths]  # [B, H]
        pref_logit = self.pref_head(seq_repr).squeeze(-1)  # [B]
        return out, pref_logit


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_preference_labels(data: List[Dict], prefs_csv: Optional[str] = None) -> Dict[str, int]:
    """
    Build a question → preference_label (0/1) lookup from pairwise_prefs.csv.
    Returns empty dict if the file is not found.
    """
    import pandas as pd

    if prefs_csv is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prefs_csv = os.path.join(root, "datasets_for_training", "pairwise_prefs.csv")

    if not os.path.exists(prefs_csv):
        print(f"[DualHead] pairwise_prefs.csv not found at {prefs_csv}; preference loss disabled.")
        return {}

    df = pd.read_csv(prefs_csv)
    label_map: Dict[str, int] = {}
    for _, row in df.iterrows():
        q = str(row.get("question", "")).strip()
        score = row.get("score", row.get("label", 0))
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue
        if score >= 1:
            label_map[q] = 1
        elif score <= -1:
            label_map[q] = 0
        # scores 0 are ambiguous — excluded
    return label_map


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DualHeadSFTTrainer:

    def __init__(self, config: SFTConfig, lam: float = 0.5):
        self.config = config
        self.lam = lam
        self.device = torch.device(config.device)

        import random as _random
        _random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        print(f"[DualHead] Loading {config.model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, use_fast=False, local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        base_lm = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.float32, local_files_only=True
        )
        hs = getattr(base_lm.config, "hidden_size", getattr(base_lm.config, "n_embd", 768))
        self.model = DualHeadCodeGPT(base_lm, hidden_size=hs).to(self.device)

        if METRICS_AVAILABLE:
            self.metrics_eval = ModernMetricsEvaluator()
        else:
            self.metrics_eval = SimpleMetricsEvaluator()

        self.history: List[EpochMetrics] = []

    # ------------------------------------------------------------------
    def generate_response(self, prompt: str, max_new_tokens: int = 64) -> str:
        inputs = self.tokenizer(
            f"### Question: {prompt}\n### Answer:",
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            ids = self.model.lm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        self.model.train()
        response = self.tokenizer.decode(ids[0], skip_special_tokens=True)
        if "### Answer:" in response:
            response = response.split("### Answer:")[-1].strip()
        return response or "N/A"

    def evaluate(self, eval_data: List[Dict]) -> Dict[str, float]:
        preds, refs = [], []
        for item in eval_data[:50]:
            q = item.get("question", item.get("intent", ""))
            r = item.get("answer", item.get("snippet", item.get("best_answer", "")))
            preds.append(self.generate_response(q))
            refs.append(r)
        if METRICS_AVAILABLE:
            res = self.metrics_eval.compute_all_metrics(preds, refs)
            return {
                "bertscore": get_metric_float(res, "bertscore"),
                "codebleu": get_metric_float(res, "codebleu", "codebleu_proxy"),
                "bleu": get_metric_float(res, "bleu"),
                "rouge": get_metric_float(res, "rouge"),
                "ruby": get_metric_float(res, "ruby", "ruby_like_heuristic"),
            }
        return self.metrics_eval.compute_all(preds, refs)

    # ------------------------------------------------------------------
    def train(
        self,
        train_data: List[Dict],
        eval_data: List[Dict],
        pref_labels: Dict[str, int],
    ) -> List[EpochMetrics]:
        dataset = SFTDataset(train_data, self.tokenizer, self.config.max_length)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = len(loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        pref_criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.config.num_epochs):
            t0 = time.time()
            self.model.train()
            total_lm_loss = total_pref_loss = total_loss = 0.0
            pref_correct = pref_total = 0
            num_batches = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_ids = batch["labels"].to(self.device)
                questions = batch["question"]  # list of strings

                lm_out, pref_logit = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_ids,
                )
                lm_loss = lm_out.loss
                if torch.isnan(lm_loss) or torch.isinf(lm_loss):
                    optimizer.zero_grad()
                    continue

                # Build preference labels for this batch (where available)
                pref_targets = []
                pref_mask = []
                for q in questions:
                    if q in pref_labels:
                        pref_targets.append(float(pref_labels[q]))
                        pref_mask.append(True)
                    else:
                        pref_targets.append(0.0)
                        pref_mask.append(False)

                pref_mask_t = torch.tensor(pref_mask, device=self.device)
                if pref_mask_t.any():
                    pref_tgt = torch.tensor(pref_targets, device=self.device, dtype=torch.float32)
                    p_loss = pref_criterion(pref_logit[pref_mask_t], pref_tgt[pref_mask_t])
                    loss = lm_loss + self.lam * p_loss
                    total_pref_loss += p_loss.item()
                    # Accuracy
                    pred_labels = (torch.sigmoid(pref_logit[pref_mask_t]) > 0.5).long()
                    true_labels = pref_tgt[pref_mask_t].long()
                    pref_correct += (pred_labels == true_labels).sum().item()
                    pref_total += pref_mask_t.sum().item()
                else:
                    loss = lm_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                scheduler.step()

                total_lm_loss += lm_loss.item()
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"lm_loss": f"{lm_loss.item():.4f}"})

            avg_lm_loss = total_lm_loss / max(1, num_batches)
            avg_total_loss = total_loss / max(1, num_batches)
            pref_acc = pref_correct / max(1, pref_total)

            print(f"\nEvaluating epoch {epoch+1}…")
            eval_m = self.evaluate(eval_data)
            reward = float(np.mean(list(eval_m.values())))

            metrics = EpochMetrics(
                epoch=epoch + 1,
                train_loss=round(avg_total_loss, 4),
                val_loss=None,
                reward=round(reward, 4),
                gradient_norm=0.0,
                bertscore=eval_m.get("bertscore", 0.0),
                codebleu=eval_m.get("codebleu", 0.0),
                bleu=eval_m.get("bleu", 0.0),
                rouge=eval_m.get("rouge", 0.0),
                ruby=eval_m.get("ruby", 0.0),
                learning_rate=scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else self.config.learning_rate,
                epoch_time=round(time.time() - t0, 1),
                balanced_accuracy=round(pref_acc, 4),
            )
            self.history.append(metrics)

            print(
                f"  lm_loss={avg_lm_loss:.4f}  pref_acc={pref_acc:.4f}  "
                f"bertscore={metrics.bertscore:.4f}  codebleu={metrics.codebleu:.4f}  "
                f"ruby={metrics.ruby:.4f}"
            )

        return self.history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3 Dual-Head SFT")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of training samples (None = all 1247)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=0.5,
                        help="Preference loss weight λ (default 0.5)")
    parser.add_argument("--output-dir", type=str, default="stage3/outputs_dual_head")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_data, eval_data = load_sft_data(num_samples=args.num_samples)
    pref_labels = load_preference_labels(train_data)

    config = SFTConfig(
        num_epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    trainer = DualHeadSFTTrainer(config, lam=args.lam)
    history = trainer.train(train_data, eval_data, pref_labels)

    out_path = os.path.join(args.output_dir, "dual_head_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "Stage3_DualHead_SFT",
                "num_samples": len(train_data),
                "lam": args.lam,
                "seed": args.seed,
                "history": [asdict(ep) for ep in history],
            },
            f,
            indent=2,
        )
    print(f"\nDual-head results saved -> {out_path}")


if __name__ == "__main__":
    main()
