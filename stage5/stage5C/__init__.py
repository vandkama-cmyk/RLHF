"""
ClassifLLM v3 - Enhanced LLM-based Classification with Real LLM Feedback

This version uses real LLM API calls to generate human-like feedback for code quality assessment,
instead of synthetic labels. Supports multiple LLM providers (OpenAI, local models, etc.).

Key features:
- Real LLM feedback generation with structured prompts
- Multi-provider LLM support (OpenAI, Anthropic, local models)
- Quality assessment based on actual LLM responses
- Enhanced training pipeline with real feedback integration
"""

from .llm_feedback import LLMFeedbackGenerator
from .integrated_system import train_llm_system_v3

__version__ = "3.0.0"
__all__ = ['LLMFeedbackGenerator', 'train_llm_system_v3']
