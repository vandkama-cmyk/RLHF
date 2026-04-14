"""
Shared utilities for RLHF Version 2 pipeline.

Centralises functions used across multiple stages so that changes
propagate automatically to all consumers.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Quality proxy
# ─────────────────────────────────────────────────────────────────────────────

def answer_quality_proxy(text) -> float:
    """
    Lightweight automated quality proxy for code-generation answers.

    Combines type-token ratio (lexical diversity) with an inverted-U length
    score that peaks at 50 tokens and penalises longer answers:

        score = 0.5 * diversity + 0.5 * length_score

    where:
        diversity    = unique_tokens / total_tokens
        length_score = n/50            if n <= 50
                     = max(0, 1-(n-50)/200)  otherwise

    KNOWN LIMITATION (Audit Bug #8):
        - Type-token ratio is misleading for code (correct code reuses
          variable names and keywords).
        - The 50-token peak is arbitrary; many correct solutions exceed it.
        - Downstream metrics built on this proxy (metric alignment,
          confidence-correctness decoupling, quality alignment in Stage 7)
          inherit this weakness.  Treat as a rough heuristic only.

    Returns np.nan for empty or non-string inputs.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.nan
    tokens = text.lower().split()
    if len(tokens) == 0:
        return np.nan
    n = len(tokens)
    diversity = len(set(tokens)) / n
    if n <= 50:
        length_score = n / 50.0
    else:
        length_score = max(0.0, 1.0 - (n - 50) / 200.0)
    return 0.5 * diversity + 0.5 * length_score
