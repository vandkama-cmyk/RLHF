"""
Stage 1: Baseline RLHF Experiment
================================

Baseline RLHF setup using:
- CodeGPT-small as the policy model
- Preference reward model (Bradley–Terry) on CodeBERT
- Policy training: Reward-Weighted NLL (Stage 1) or PPO (Stage 1B, ``run_stage1b``)

This stage verifies the end-to-end pipeline operates correctly
and produces syntactically valid code.
"""

from .config import Stage1Config, get_stage1_config
from .run_stage1 import Stage1Experiment
from .run_stage1b import Stage1BExperiment

__all__ = ["Stage1Config", "get_stage1_config", "Stage1Experiment", "Stage1BExperiment"]
