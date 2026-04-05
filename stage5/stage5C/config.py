"""
ClassifLLM v3 Configuration
==========================

Configuration presets for different LLM feedback providers and training setups.
"""

from typing import Dict, Any
from .llm_feedback import LLMConfig


def get_openai_config(model: str = "gpt-4", api_key: str = None) -> LLMConfig:
    """Get OpenAI configuration."""
    return LLMConfig(
        provider="openai",
        model_name=model,
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000,
        batch_size=3,  # Be respectful to API limits
        max_retries=3,
        retry_delay=2.0,
        timeout=30
    )


def get_anthropic_config(model: str = "claude-3-sonnet-20240229", api_key: str = None) -> LLMConfig:
    """Get Anthropic configuration."""
    return LLMConfig(
        provider="anthropic",
        model_name=model,
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000,
        batch_size=3,
        max_retries=3,
        retry_delay=2.0,
        timeout=30
    )


def get_huggingface_config(model: str = "microsoft/DialoGPT-medium", api_key: str = None) -> LLMConfig:
    """Get HuggingFace Inference API configuration.

    Note: Requires HF_TOKEN environment variable or api_key parameter.
    Recommended models: GPT-2, DialoGPT, or other conversational models.
    """
    return LLMConfig(
        provider="huggingface",
        model_name=model,
        api_key=api_key,
        temperature=0.8,
        max_tokens=500,
        batch_size=2,  # HF API has rate limits
        max_retries=3,
        retry_delay=1.0,
        timeout=30
    )


def get_local_config(model: str = "microsoft/DialoGPT-small") -> LLMConfig:
    """Get local transformers model configuration.

    Note: Use small conversational models for feedback generation.
    Code understanding models like CodeBERT cannot generate text.
    Recommended models: DialoGPT-small, GPT-2, or other generative models.
    """
    return LLMConfig(
        provider="local",
        model_name=model,
        temperature=0.8,
        max_tokens=500,
        batch_size=1,  # Smaller batch size for local models
        max_retries=1,
        timeout=60
    )


def get_fast_training_config() -> Dict[str, Any]:
    """Fast training configuration for testing."""
    return {
        'batch_size': 8,
        'gradient_accumulation': 2,
        'epochs': 5,
        'learning_rate': 5e-5,
        'freeze_layers': 8,
        'use_amp': True,
        'use_ema': False,
        'use_cross_attention': True,
        'use_contrastive': False,
        'max_feedback_samples': 200,
    }


def get_research_config() -> Dict[str, Any]:
    """Research-grade training configuration."""
    return {
        'batch_size': 4,
        'gradient_accumulation': 4,
        'epochs': 20,
        'learning_rate': 2e-5,
        'freeze_layers': 4,
        'use_amp': True,
        'use_ema': True,
        'use_cross_attention': True,
        'use_contrastive': True,
        'contrastive_weight': 0.1,
        'label_smoothing': 0.1,
        'max_feedback_samples': 2000,
        'use_confidence_weighting': True,
    }


def get_production_config() -> Dict[str, Any]:
    """Production-grade training configuration."""
    return {
        'batch_size': 2,
        'gradient_accumulation': 8,
        'epochs': 30,
        'learning_rate': 1e-5,
        'freeze_layers': 2,
        'use_amp': True,
        'use_ema': True,
        'use_cross_attention': True,
        'use_contrastive': True,
        'contrastive_weight': 0.05,
        'label_smoothing': 0.05,
        'max_feedback_samples': 5000,
        'use_confidence_weighting': True,
        'patience': 8,
    }


def create_training_config(
    llm_config: LLMConfig = None,
    preset: str = "research",
    **overrides
) -> Dict[str, Any]:
    """
    Create a complete training configuration.

    Args:
        llm_config: LLM feedback configuration
        preset: Training preset ('fast', 'research', 'production')
        **overrides: Additional configuration overrides

    Returns:
        Complete configuration dictionary
    """

    # Get preset configuration
    if preset == "fast":
        config = get_fast_training_config()
    elif preset == "research":
        config = get_research_config()
    elif preset == "production":
        config = get_production_config()
    else:
        raise ValueError(f"Unknown preset: {preset}")

    # Add LLM configuration
    if llm_config:
        config['llm_config'] = llm_config
    else:
        config['llm_config'] = None

    # Apply overrides
    config.update(overrides)

    # Add default values for missing keys
    defaults = {
        'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
        'model_type': 'codebert',
        'dropout': 0.3,
        'weight_decay': 0.01,
        'max_length': 256,
        'force_regenerate_feedback': False,
        'output_dir': 'clasifLLM/v3/artifacts',
    }

    for key, value in defaults.items():
        if key not in config:
            config[key] = value

    return config


# Example configurations
EXAMPLE_CONFIGS = {
    'openai_gpt4': {
        'llm_config': get_openai_config("gpt-4"),
        'preset': 'research',
    },

    'openai_gpt35': {
        'llm_config': get_openai_config("gpt-3.5-turbo"),
        'preset': 'research',
    },

    'anthropic_claude': {
        'llm_config': get_anthropic_config(),
        'preset': 'research',
    },

    'local_dialogpt': {
        'llm_config': get_local_config("microsoft/DialoGPT-small"),
        'preset': 'fast',  # Local models are slower
    },

    'huggingface_gpt2': {
        'llm_config': get_huggingface_config("gpt2"),
        'preset': 'research',
    },

    'huggingface_dialogpt': {
        'llm_config': get_huggingface_config("microsoft/DialoGPT-medium"),
        'preset': 'research',
    },

    'synthetic_baseline': {
        'llm_config': None,  # No LLM feedback
        'preset': 'research',
    },
}


def get_example_config(name: str) -> Dict[str, Any]:
    """Get an example configuration by name."""
    if name not in EXAMPLE_CONFIGS:
        available = list(EXAMPLE_CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")

    config_spec = EXAMPLE_CONFIGS[name]
    return create_training_config(**config_spec)
