"""
Stage 2A - Contrastive Learning for Enhanced Reward Model
=========================================================

This stage introduces contrastive learning and an improved reward function design
to enhance the discriminative capacity of the reward model.

Key features:
- Maximum contrastive loss function for learning embeddings
- Better separation of similar vs dissimilar samples
- Uses T2C-CoNaLa and T2T-SO SFT datasets
"""

from .config import Stage2AConfig, get_stage2a_config
from .contrastive_model import ContrastiveRewardModel
from .train import Stage2AExperiment

__all__ = [
    'Stage2AConfig',
    'get_stage2a_config', 
    'ContrastiveRewardModel',
    'Stage2AExperiment'
]
