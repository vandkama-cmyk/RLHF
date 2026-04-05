"""
Stage 2B - GPT-2 Feedback Generator for Reward Model Training
=============================================================

This stage uses GPT-2 to generate synthetic feedback scores (Consistency, 
Agreement, Usefulness) for training the reward model on CoNaLa corpus.
"""

from .config import Stage2BConfig, get_stage2b_config
from .feedback_generator import GPT2FeedbackGenerator
from .reward_model import FeedbackRewardModel
from .train import Stage2BExperiment

__all__ = [
    'Stage2BConfig',
    'get_stage2b_config',
    'GPT2FeedbackGenerator',
    'FeedbackRewardModel',
    'Stage2BExperiment',
]
