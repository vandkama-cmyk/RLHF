"""
Stage 5 LoRA — classifier with LoRA-adapted CodeBERT encoder.

Applies Low-Rank Adaptation (LoRA) to the attention layers of the frozen
CodeBERT encoder used in Stage 5B (cross-attention + EMA variant).
Only the LoRA adapter weights are updated; the base CodeBERT weights are frozen.

LoRA config applied to CodeBERT (bert-style):
  task_type      : FEATURE_EXTRACTION
  r              : 8
  lora_alpha     : 16
  target_modules : ["query", "key", "value"]   (BERT self-attention projections)
  lora_dropout   : 0.05
  bias           : "none"

Architecture (same as Stage 5B after LoRA injection):
  CodeBERT (LoRA-adapted, ~0.6M trainable) ->
  AttentionPooling ->
  CrossAttentionLayer (Q=question, K/V=answer) ->
  MLP classification head x 3 (Consistent / Correct / Useful)

Results saved to stage5/stage5_lora/outputs/stage5_lora_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoTokenizer, AutoModel

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from stage5.stage5B.model import AttentionPooling, CrossAttentionLayer
from stage5.stage5B.integrated_system import (
    load_samples,
    normalize_labels,
    generate_synthetic_samples,
    EnhancedCodeQualityDataset,
    collate_fn,
    balanced_split,
)


# ---------------------------------------------------------------------------
# Tokenising wrapper dataset
# ---------------------------------------------------------------------------

class TokenizedDataset(torch.utils.data.Dataset):
    """Wraps EnhancedCodeQualityDataset and tokenises on-the-fly."""

    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        q = s.get("question", "")
        a = s.get("answer", "")
        labels = s.get("labels", {})

        def enc(text):
            return self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

        q_enc = enc(q)
        a_enc = enc(a)
        return {
            "q_input_ids": q_enc["input_ids"].squeeze(0),
            "q_attention_mask": q_enc["attention_mask"].squeeze(0),
            "a_input_ids": a_enc["input_ids"].squeeze(0),
            "a_attention_mask": a_enc["attention_mask"].squeeze(0),
            "consistent": float(labels.get("consistent", 0.5)),
            "correct": float(labels.get("correct", labels.get("agreement", 0.5))),
            "useful": float(labels.get("useful", 0.5)),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Stage5LoRAClassifier(nn.Module):
    """
    CodeBERT encoder with LoRA adapters + cross-attention + 3-head MLP.
    """

    HIDDEN = 768
    NUM_HEADS = 8

    def __init__(
        self,
        encoder_name: str = "microsoft/codebert-base",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft is required for LoRA experiments. "
                "Install with: pip install peft>=0.3.0"
            )

        base_encoder = AutoModel.from_pretrained(encoder_name, local_files_only=True)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.encoder = get_peft_model(base_encoder, lora_config)
        self.encoder.print_trainable_parameters()

        h = self.HIDDEN
        self.attn_pool = AttentionPooling(h)
        self.cross_attn = CrossAttentionLayer(h, num_heads=self.NUM_HEADS, dropout=dropout)

        def _head():
            return nn.Sequential(
                nn.Linear(h * 2, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h, 1),
            )

        self.head_consistent = _head()
        self.head_correct = _head()
        self.head_useful = _head()

    def _encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.attn_pool(out.last_hidden_state, attention_mask)

    def forward(self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask):
        q_emb = self._encode(q_input_ids, q_attention_mask)
        a_emb = self._encode(a_input_ids, a_attention_mask)

        q_seq = q_emb.unsqueeze(1)
        a_seq = a_emb.unsqueeze(1)
        cross = self.cross_attn(q_seq, a_seq).squeeze(1)

        combined = torch.cat([q_emb, cross], dim=-1)

        return {
            "consistent": self.head_consistent(combined).squeeze(-1),
            "correct": self.head_correct(combined).squeeze(-1),
            "useful": self.head_useful(combined).squeeze(-1),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_lora(
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 2e-4,
    encoder_name: str = "microsoft/codebert-base",
    output_dir: str = "stage5/stage5_lora/outputs",
    feedback_dir: str = "clasifNN/feedback",
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Data ---------------------------------------------------------------
    from pathlib import Path
    samples = load_samples(Path(feedback_dir))
    samples = normalize_labels(samples)
    if not samples:
        samples = generate_synthetic_samples()

    n = len(samples)
    split = int(n * 0.85)
    train_samples, val_samples = samples[:split], samples[split:]

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, local_files_only=True)

    train_loader = DataLoader(
        TokenizedDataset(train_samples, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TokenizedDataset(val_samples, tokenizer),
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Train: {len(train_samples)} samples  Val: {len(val_samples)} samples")

    # ---- Model --------------------------------------------------------------
    model = Stage5LoRAClassifier(encoder_name=encoder_name).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            q_ids = batch["q_input_ids"].to(device)
            q_mask = batch["q_attention_mask"].to(device)
            a_ids = batch["a_input_ids"].to(device)
            a_mask = batch["a_attention_mask"].to(device)
            labels = {k: batch[k].float().to(device) for k in ("consistent", "correct", "useful")}

            logits = model(q_ids, q_mask, a_ids, a_mask)
            loss = sum(criterion(logits[k], labels[k]) for k in labels) / 3

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(num_batches, 1)

        # ---- Validation -----
        model.eval()
        val_loss = 0.0
        correct_counts = {"consistent": 0, "correct": 0, "useful": 0}
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                q_ids = batch["q_input_ids"].to(device)
                q_mask = batch["q_attention_mask"].to(device)
                a_ids = batch["a_input_ids"].to(device)
                a_mask = batch["a_attention_mask"].to(device)
                labels = {k: batch[k].float().to(device) for k in ("consistent", "correct", "useful")}

                logits = model(q_ids, q_mask, a_ids, a_mask)
                loss = sum(criterion(logits[k], labels[k]) for k in labels) / 3
                val_loss += loss.item()

                bs = q_ids.size(0)
                total_val += bs
                for k in correct_counts:
                    preds = (torch.sigmoid(logits[k]) > 0.5).long()
                    correct_counts[k] += (preds == labels[k].round().long()).sum().item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        accs = {k: correct_counts[k] / max(total_val, 1) for k in correct_counts}

        rec = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_acc_consistent": round(accs["consistent"], 4),
            "val_acc_correct": round(accs["correct"], 4),
            "val_acc_useful": round(accs["useful"], 4),
            "val_acc_mean": round(sum(accs.values()) / 3, 4),
            "epoch_time": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(
            f"Epoch {epoch:02d} | train={rec['train_loss']:.4f} "
            f"val={rec['val_loss']:.4f} "
            f"acc_c={rec['val_acc_consistent']:.4f} "
            f"acc_k={rec['val_acc_correct']:.4f} "
            f"acc_u={rec['val_acc_useful']:.4f} "
            f"({rec['epoch_time']:.0f}s)"
        )

    out_path = os.path.join(output_dir, f"stage5_lora_seed{seed}_results.json")
    with open(out_path, "w") as f:
        json.dump({"seed": seed, "history": history}, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 5 LoRA classifier")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--encoder", type=str, default="microsoft/codebert-base")
    parser.add_argument("--output-dir", type=str, default="stage5/stage5_lora/outputs")
    parser.add_argument("--feedback-dir", type=str, default="clasifNN/feedback")
    args = parser.parse_args()

    train_lora(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_name=args.encoder,
        output_dir=args.output_dir,
        feedback_dir=args.feedback_dir,
    )


if __name__ == "__main__":
    main()
