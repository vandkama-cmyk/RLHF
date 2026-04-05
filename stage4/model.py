"""
Neural classifier architecture for multi-metric reward prediction.
"""

from __future__ import annotations

from typing import Union, Dict

import numpy as np
import torch
import torch.nn as nn

TensorLike = Union[torch.Tensor, np.ndarray]


def _ensure_tensor(value: TensorLike) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.from_numpy(value).float()


def build_feature_vector(question_emb: TensorLike, answer_emb: TensorLike) -> torch.Tensor:
    """Build combined features from question and answer embeddings."""
    question_tensor = _ensure_tensor(question_emb)
    answer_tensor = _ensure_tensor(answer_emb)

    if question_tensor.dim() == 1:
        question_tensor = question_tensor.unsqueeze(0)
    if answer_tensor.dim() == 1:
        answer_tensor = answer_tensor.unsqueeze(0)

    if question_tensor.shape[0] != answer_tensor.shape[0]:
        raise ValueError("Question and answer embeddings must have the same batch size.")

    diff = torch.abs(question_tensor - answer_tensor)
    prod = question_tensor * answer_tensor
    features = torch.cat([question_tensor, answer_tensor, diff, prod], dim=-1)
    return features


class MultiHeadClassifier(nn.Module):
    """
    Multi-task classifier that predicts consistent, correct, and useful scores simultaneously.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 768, dropout: float = 0.2) -> None:
        super().__init__()
        input_dim = embedding_dim * 4
        hidden_dim = max(hidden_dim, embedding_dim)

        # Shared backbone (Body)
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Independent Heads
        self.head_consistent = nn.Linear(hidden_dim, 1)
        self.head_correct = nn.Linear(hidden_dim, 1)
        self.head_useful = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary of logits: {'consistent': ..., 'correct': ..., 'useful': ...}
        """
        shared_repr = self.shared_encoder(features)

        return {
            "consistent": self.head_consistent(shared_repr).squeeze(-1),
            "correct": self.head_correct(shared_repr).squeeze(-1),
            "useful": self.head_useful(shared_repr).squeeze(-1),
        }
