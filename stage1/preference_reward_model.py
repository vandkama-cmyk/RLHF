"""
Preference-based reward model (Bradley–Terry / pairwise ranking loss).

Fine-tunes a CodeBERT (or compatible encoder) encoder with a scalar score head:
input = prompt + response, output = scalar quality score.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def load_preference_triples_from_csv(path: str) -> List[Tuple[str, str, str]]:
    """Load (prompt, chosen, rejected) rows from a CSV file.

    Recognized column names (case-insensitive):
    prompt / instruction / question
    chosen / winner / response_a / better
    rejected / loser / response_b / worse
    """
    import csv

    triples: List[Tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return triples
        fields = {name.lower().strip(): name for name in reader.fieldnames}

        def col(*candidates: str) -> Optional[str]:
            for c in candidates:
                if c in fields:
                    return fields[c]
            return None

        pc = col("prompt", "instruction", "question", "input")
        cc = col("chosen", "winner", "response_a", "better", "preferred")
        rc = col("rejected", "loser", "response_b", "worse", "non_preferred")
        if not pc or not cc or not rc:
            logger.warning(
                "CSV %s missing expected columns (prompt/chosen/rejected); got %s",
                path,
                list(reader.fieldnames),
            )
            return triples

        for row in reader:
            p = (row.get(pc) or "").strip()
            c = (row.get(cc) or "").strip()
            r = (row.get(rc) or "").strip()
            if p and c and r:
                triples.append((p, c, r))
    return triples


def build_preference_pairs(
    train_data: Sequence[Any],
    max_samples: int,
    seed: int,
) -> List[Tuple[str, str, str]]:
    """Build (prompt, chosen, rejected) tuples from supervised data.

    chosen = reference response; rejected = degraded variant (truncation / noise).
    """
    import random

    rng = random.Random(seed)
    pairs: List[Tuple[str, str, str]] = []

    for sample in train_data:
        if len(pairs) >= max_samples:
            break
        if isinstance(sample, dict):
            prompt = (sample.get("prompt") or "").strip()
            chosen = (sample.get("reference") or sample.get("response") or "").strip()
        else:
            prompt = (getattr(sample, "prompt", None) or "").strip()
            chosen = (getattr(sample, "reference", None) or getattr(sample, "response", None) or "").strip()

        if not prompt or not chosen:
            continue

        # Simple synthetic "worse" completion: truncated / shuffled line order
        lines = chosen.splitlines()
        if len(lines) > 1:
            rng.shuffle(lines)
            rejected = "\n".join(lines[: max(1, len(lines) // 2)])
        else:
            half = max(1, len(chosen) // 2)
            rejected = chosen[:half]

        if rejected.strip() == chosen.strip():
            rejected = chosen[: max(1, len(chosen) // 3)] + "  # worse"

        pairs.append((prompt, chosen, rejected))

    return pairs


class PreferencePairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        return self.pairs[idx]


class PreferenceRewardModel(nn.Module):
    """Encoder + linear head producing a scalar score for (prompt, response) text."""

    def __init__(
        self,
        encoder_name: str,
        device: torch.device,
        trust_remote_code: bool = True,
        local_files_only: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        hidden = self.encoder.config.hidden_size
        self.score_head = nn.Linear(hidden, 1)

        self.to(device)

    def encode_batch(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    def forward_scores(self, texts: List[str]) -> torch.Tensor:
        """Return shape [B] scores."""
        batch = self.encode_batch(texts)
        out = self.encoder(**batch)
        cls = out.last_hidden_state[:, 0, :]
        return self.score_head(cls).squeeze(-1)

    @torch.no_grad()
    def score_pairs(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        self.eval()
        texts = [f"{p} {r}" for p, r in zip(prompts, responses)]
        return self.forward_scores(texts)


def bradley_terry_loss(
    score_chosen: torch.Tensor,
    score_rejected: torch.Tensor,
) -> torch.Tensor:
    """Bradley–Terry pairwise loss: -log σ(s_chosen - s_rejected)."""
    return -F.logsigmoid(score_chosen - score_rejected).mean()


def train_preference_reward_model(
    model: PreferenceRewardModel,
    pairs: List[Tuple[str, str, str]],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    max_grad_norm: float,
) -> List[float]:
    """Train the reward model; returns epoch mean losses."""
    if not pairs:
        logger.warning("No preference pairs; skipping reward model training.")
        return []

    ds = PreferencePairDataset(pairs)

    def _collate(batch: List[Tuple[str, str, str]]) -> Tuple[List[str], List[str], List[str]]:
        p, c, r = zip(*batch)
        return list(p), list(c), list(r)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=_collate)

    opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.score_head.parameters()), lr=lr)
    losses_out: List[float] = []

    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        n = 0
        for prompt, chosen, rejected in loader:
            texts_w = [f"{p} {c}" for p, c in zip(prompt, chosen)]
            texts_l = [f"{p} {r}" for p, r in zip(prompt, rejected)]

            s_w = model.forward_scores(list(texts_w))
            s_l = model.forward_scores(list(texts_l))
            loss = bradley_terry_loss(s_w, s_l)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder.parameters()) + list(model.score_head.parameters()),
                max_grad_norm,
            )
            opt.step()

            epoch_loss += loss.item()
            n += 1

        mean_l = epoch_loss / max(1, n)
        losses_out.append(mean_l)
        logger.info(f"Preference reward model epoch {ep + 1}/{epochs} loss={mean_l:.4f}")

    return losses_out


def save_reward_model(path: str, model: PreferenceRewardModel) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    enc_id = getattr(
        model.encoder.config,
        "name_or_path",
        getattr(model.encoder.config, "_name_or_path", "microsoft/codebert-base"),
    )
    torch.save(
        {
            "encoder_state": model.encoder.state_dict(),
            "score_head_state": model.score_head.state_dict(),
            "encoder_name": enc_id,
        },
        path,
    )
    logger.info(f"Saved preference reward model to {path}")


def load_reward_checkpoint(
    path: str,
    device: torch.device,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
) -> PreferenceRewardModel:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    name = ckpt.get("encoder_name") or "microsoft/codebert-base"
    model = PreferenceRewardModel(
        name,
        device,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model.encoder.load_state_dict(ckpt["encoder_state"])
    model.score_head.load_state_dict(ckpt["score_head_state"])
    model.eval()
    return model
