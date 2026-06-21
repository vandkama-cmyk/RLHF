"""
BCRM – Bias-Corrected Reward Model for Human-Annotated Code Evaluation
=======================================================================
Dissertation model for RLHF pipeline (vandkama-cmyk/RLHF).

Architecture overview
---------------------
Input : concatenated [code_prompt || generated_code] token sequence
Backbone : CodeBERT-style Transformer encoder (6 layers, lightweight)
Head 1 : MultiDimRewardHead  — 3-axis reward scores
           (Correctness, Consistency, Usefulness)
Head 2 : RaterBiasEstimator  — per-rater bias offset vectors
           (used at training time; detached at inference)
Output : bias-corrected scalar reward  R̂ = w · R_multi − B_rater

Scientific novelty (as discussed in Lecture 04.03.26 and 18.03.26)
--------------------------------------------------------------------
This model is NOT a simple application of an existing neural network.
It introduces:
  1. A unified multi-dimensional reward parameterization that separates
     correctness, consistency, and usefulness into orthogonal axes —
     enabling interpretable reward decomposition (not present in prior work
     on CodeBERT-based reward models).
  2. An end-to-end differentiable rater bias correction layer trained
     jointly with the reward backbone, instead of the standard two-stage
     post-hoc correction (Dawid–Skene / EM fallback).
  3. An R-score reliability metric (from the EMNLP paper) used as an
     auxiliary training signal to weight rater contributions dynamically.

Reference to lecture content
-----------------------------
- Lecture 04.03.26: "нейронная сеть решает две задачи: классификация или
  регрессия" — this model frames reward prediction as regression with a
  structured multi-head output, grounded in a scientific hypothesis
  (as the professor distinguishes from mere empirical black-box use).
- Lecture 18.03.26: scientific novelty requires standing on predecessors'
  shoulders and adding an *increment* — the bias correction layer is
  that increment over CodeBERT-based reward models (e.g. CodeRL).
- Lecture 15.04.26: YOLO-style insight — unify multi-pass evaluation
  into a single forward pass with multiple output heads.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BCRMConfig:
    """Configuration for the Bias-Corrected Reward Model."""

    # Tokenizer / vocabulary
    vocab_size: int = 50265          # CodeBERT / RoBERTa vocabulary
    max_seq_len: int = 512

    # Transformer encoder (lightweight backbone)
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1

    # Reward axes (Head 1)
    reward_axes: List[str] = field(
        default_factory=lambda: ["Correctness", "Consistency", "Usefulness"]
    )
    # Learned axis weights (initialised equal; updated during training)
    axis_weight_init: float = 1.0 / 3.0

    # Rater bias estimation (Head 2)
    num_raters: int = 6              # Artem, Arthur, Anastasia, Fedor, Vadim, TypingCat
    rater_bias_dim: int = 3          # one bias offset per reward axis

    # R-score auxiliary loss weight (reliability signal)
    r_score_loss_weight: float = 0.1

    # KL regularisation (prevents policy collapse, replaces KL(π‖π_ref)
    # with KL-from-Uniform as used in the research pipeline)
    kl_coeff: float = 0.05


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CodeTransformerEncoder(nn.Module):
    """
    Lightweight Transformer encoder backbone.

    Design choice: shared encoder for both code and prompt, separated
    by a special [SEP] token — mirrors CodeBERT's cross-attention pattern
    but is smaller (6 layers, hidden=256) for on-device deployment.
    This is justified by the lecture's emphasis on pruning / dropouts
    (Lecture 15.04.26: Dropout & structured sparsity sections).
    """

    def __init__(self, cfg: BCRMConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=1)
        self.pos_enc = PositionalEncoding(cfg.hidden_dim, cfg.max_seq_len, cfg.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers
        )
        self.layer_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) 1 = keep, 0 = pad

        Returns:
            cls_repr: (B, hidden_dim) — [CLS] token representation
        """
        x = self.pos_enc(self.embed(input_ids))
        key_padding_mask = None
        if attention_mask is not None:
            # TransformerEncoder expects True = ignore (pad)
            key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.layer_norm(x)
        return x[:, 0, :]   # [CLS] pooling


class MultiDimRewardHead(nn.Module):
    """
    Head 1: Decomposes reward into K interpretable axes.

    Scientific novelty: explicit axis separation allows post-hoc
    analysis of which reward dimension drives preference differences,
    directly supporting the EMNLP annotation-bias paper.
    """

    def __init__(self, cfg: BCRMConfig):
        super().__init__()
        K = len(cfg.reward_axes)
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim // 2, 1),
            )
            for _ in range(K)
        ])
        # Learnable axis weights; softmax-normalised during forward
        self.axis_logits = nn.Parameter(
            torch.full((K,), math.log(cfg.axis_weight_init / (1 - cfg.axis_weight_init + 1e-8)))
        )

    def forward(self, cls_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            axis_scores : (B, K) raw scores per axis
            scalar_reward: (B,)  weighted aggregate
        """
        axis_scores = torch.cat(
            [proj(cls_repr) for proj in self.projectors], dim=-1
        )  # (B, K)
        weights = F.softmax(self.axis_logits, dim=0)      # (K,)
        scalar_reward = (axis_scores * weights).sum(dim=-1)  # (B,)
        return axis_scores, scalar_reward


class RaterBiasEstimator(nn.Module):
    """
    Head 2: Estimates per-rater bias offsets.

    Each rater r has a learned bias vector b_r ∈ R^K (one per reward axis).
    During training the network learns to subtract rater-specific
    annotation artefacts from the raw reward signal.

    This implements the 3-part pipeline idea from rater-bias-correction.md
    in a single end-to-end differentiable module.
    """

    def __init__(self, cfg: BCRMConfig):
        super().__init__()
        # Bias embedding: rater_id → bias vector over K axes
        self.bias_embed = nn.Embedding(cfg.num_raters, cfg.rater_bias_dim)
        nn.init.zeros_(self.bias_embed.weight)   # Initialise to no bias

        # Reliability-weighted aggregation projection
        self.reliability_proj = nn.Linear(cfg.rater_bias_dim, 1, bias=False)

    def forward(
        self, rater_ids: torch.Tensor, r_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            rater_ids : (B,) integer rater indices
            r_scores  : (B,) optional reliability scores in [0,1]
                        (from EMNLP R-score metric); if None, weights = uniform

        Returns:
            bias_offset: (B, K) bias to subtract from axis_scores
        """
        biases = self.bias_embed(rater_ids)      # (B, K)
        if r_scores is not None:
            # Scale bias correction by rater reliability
            scale = r_scores.unsqueeze(-1).clamp(0.0, 1.0)
            biases = biases * scale
        return biases


# ---------------------------------------------------------------------------
# Full BCRM Model
# ---------------------------------------------------------------------------

class BCRM(nn.Module):
    """
    Bias-Corrected Reward Model (BCRM).

    Forward pass (training):
        R̂_corrected = w · (R_multi − B_rater)

    Forward pass (inference / RLHF rollout):
        R̂ = w · R_multi   (rater_ids=None → no bias subtraction needed)

    This two-mode design mirrors the insight from Lecture 04.03.26:
    "нейронная сеть как инструмент познания" — the model is used both
    as a measurement instrument (training, with bias estimation) and
    as a scoring tool in the RLHF loop (inference).
    """

    def __init__(self, cfg: BCRMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CodeTransformerEncoder(cfg)
        self.reward_head = MultiDimRewardHead(cfg)
        self.bias_head = RaterBiasEstimator(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rater_ids: Optional[torch.Tensor] = None,
        r_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids      : (B, L)
            attention_mask : (B, L)
            rater_ids      : (B,) — only needed during training
            r_scores       : (B,) — optional reliability weights

        Returns dict with keys:
            'reward'        : (B,) final scalar reward
            'axis_scores'   : (B, K) per-axis rewards
            'bias_offsets'  : (B, K) estimated rater biases (zeros if no rater_ids)
        """
        cls_repr = self.encoder(input_ids, attention_mask)
        axis_scores, _ = self.reward_head(cls_repr)

        # Bias correction
        if rater_ids is not None:
            bias_offsets = self.bias_head(rater_ids, r_scores)
        else:
            bias_offsets = torch.zeros_like(axis_scores)

        # Apply bias correction before aggregating axes
        corrected_axes = axis_scores - bias_offsets
        weights = F.softmax(self.reward_head.axis_logits, dim=0)
        reward = (corrected_axes * weights).sum(dim=-1)

        return {
            "reward": reward,
            "axis_scores": axis_scores,
            "bias_offsets": bias_offsets,
        }

    def compute_loss(
        self,
        chosen_reward: torch.Tensor,
        rejected_reward: torch.Tensor,
        r_scores: Optional[torch.Tensor] = None,
        axis_scores_chosen: Optional[torch.Tensor] = None,
        axis_scores_rejected: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Bradley-Terry preference loss + optional R-score auxiliary loss.

        Main loss: L_BT = -log σ(R_chosen - R_rejected)
        Aux loss : L_R  = ||axis_scores_chosen - axis_scores_rejected||_F
                          weighted by r_scores (reliable raters penalised
                          more for axis inconsistency)

        Reference: Lecture 15.04.26 — функция потерь, которая состоит из
        двух компонентов: потери класса и потери, связанные с локализацией.
        Same two-component loss design applied to reward learning.
        """
        # Primary Bradley-Terry loss
        bt_loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

        total_loss = bt_loss
        loss_dict = {"bt_loss": bt_loss}

        # Auxiliary axis-consistency loss (optional)
        if (
            axis_scores_chosen is not None
            and axis_scores_rejected is not None
            and r_scores is not None
        ):
            axis_diff = (axis_scores_chosen - axis_scores_rejected).pow(2)  # (B, K)
            # Weight by reliability: high-reliability raters should show
            # consistent axis preferences
            reliability_weight = r_scores.unsqueeze(-1)                      # (B, 1)
            aux_loss = (axis_diff * reliability_weight).mean()
            total_loss = bt_loss + self.cfg.r_score_loss_weight * aux_loss
            loss_dict["aux_loss"] = aux_loss

        loss_dict["total_loss"] = total_loss
        return loss_dict

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

class BCRMScorer:
    """
    Thin inference wrapper for use in the RLHF rollout loop.
    Mirrors Stage 5 of the research pipeline (real-time LLM feedback).
    """

    def __init__(self, model: BCRM, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def score(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return self.model(input_ids, attention_mask)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

def _smoke_test():
    cfg = BCRMConfig()
    model = BCRM(cfg)

    print(f"BCRM parameters: {model.num_parameters:,}")
    print(f"Reward axes: {cfg.reward_axes}")
    print(f"Rater bias dim: {cfg.rater_bias_dim} (one offset per axis per rater)")

    B, L = 4, 128
    input_ids = torch.randint(2, cfg.vocab_size, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    rater_ids = torch.randint(0, cfg.num_raters, (B,))
    r_scores = torch.rand(B)

    # Training forward pass
    out = model(input_ids, attention_mask, rater_ids, r_scores)
    print(f"\nTraining forward pass:")
    print(f"  reward shape    : {out['reward'].shape}")
    print(f"  axis_scores     : {out['axis_scores'].shape}")
    print(f"  bias_offsets    : {out['bias_offsets'].shape}")
    print(f"  reward sample   : {out['reward'].detach().numpy().round(3)}")

    # Loss computation
    B2 = B // 2
    chosen = out["reward"][:B2]
    rejected = out["reward"][B2:]
    ax_c = out["axis_scores"][:B2]
    ax_r = out["axis_scores"][B2:]
    losses = model.compute_loss(chosen, rejected, r_scores[:B2], ax_c, ax_r)
    print(f"\nLoss:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Inference (no rater_ids)
    scorer = BCRMScorer(model)
    inf_out = scorer.score(input_ids, attention_mask)
    print(f"\nInference reward (no bias correction): {inf_out['reward'].detach().numpy().round(3)}")
    print("\n✓ BCRM smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
