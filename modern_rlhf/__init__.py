"""
Modern RLHF Framework for Code Generation
=========================================

A clean, modern implementation of RLHF (Reinforcement Learning from Human Feedback)
specifically designed for code generation tasks with state-of-the-art methods.

Key Features:
- Direct Preference Optimization (DPO) support
- Modern reward modeling with human feedback integration
- Comprehensive evaluation metrics (BERTScore, CodeBLEU, BLEU, ROUGE)
- Efficient training pipeline with GPU optimization
- Clean, modular architecture

Author: Research Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Research Team"

# Import main classes
from .config import ModernRLHFConfig, get_research_config, get_production_config, get_fast_config
from .pipeline import ModernRLHFPipeline
from .metrics import ModernMetricsEvaluator
from .reward_model import ModernRewardModel
from .trainer import ModernRLHFTrainer
from .data_loader import ModernDataLoader

# Make main classes available at package level
__all__ = [
    'ModernRLHFConfig',
    'get_research_config',
    'get_production_config', 
    'get_fast_config',
    'ModernRLHFPipeline',
    'ModernMetricsEvaluator',
    'ModernRewardModel',
    'ModernRLHFTrainer',
    'ModernDataLoader'
]
