"""
Embedding functionality for ClassifMLP
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class EmbeddingConfig:
    """Configuration for embedding encoder."""

    def __init__(self, device: str = "cpu", embedding_dim: int = 768):
        self.device = device
        self.embedding_dim = embedding_dim


class EmbeddingEncoder:
    """Base class for text embedding encoders."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.embedding_dim = config.embedding_dim

    def encode(self, texts) -> torch.Tensor:
        """Encode texts to embeddings."""
        raise NotImplementedError

    def to(self, device):
        """Move to device."""
        self.device = torch.device(device)
        return self


class DummyEmbeddingEncoder(EmbeddingEncoder):
    """Deterministic embedding encoder for reproducible demos."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._cache: Dict[str, torch.Tensor] = {}

    def encode(self, texts):
        """Return deterministic embeddings derived from text hashes."""
        import numpy as np

        texts_list = list(texts) if not isinstance(texts, torch.Tensor) else [texts]

        if not texts_list:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        embeddings = []
        for text in texts_list:
            normalized = self._normalize_text(text)
            embeddings.append(self._get_embedding(normalized))

        return torch.stack(embeddings, dim=0)

    def _normalize_text(self, text: str) -> str:
        if text is None:
            return ""
        return " ".join(str(text).split()).lower()

    def _get_embedding(self, normalized_text: str) -> torch.Tensor:
        cached = self._cache.get(normalized_text)
        if cached is not None:
            return cached.clone()

        tensor = self._build_embedding(normalized_text)
        self._cache[normalized_text] = tensor
        return tensor.clone()

    def _build_embedding(self, normalized_text: str) -> torch.Tensor:
        import hashlib
        import numpy as np

        digest = hashlib.sha256(normalized_text.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(digest, "little", signed=False)
        rng = np.random.default_rng(seed)
        values = rng.standard_normal(self.embedding_dim, dtype=np.float32)
        return torch.from_numpy(values)
