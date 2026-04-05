"""
ClassifLLM v3 - Training Script with Real LLM Feedback

Enhanced LLM-based code quality classifier with real LLM feedback generation.
Uses actual LLM API calls to generate authentic human-like feedback for training.

Key features:
- Real LLM feedback generation (OpenAI, Anthropic, local models)
- Structured prompts for consistent quality assessment
- Confidence-weighted training
- All v2 enhancements: AMP, cross-attention, contrastive learning, EMA

Supported LLM providers:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- Local transformers models
- HuggingFace models

Usage:
    # With OpenAI GPT-4
    python clasifLLM/v3/train.py --llm-provider openai --api-key YOUR_API_KEY

    # With HuggingFace API
    export HF_TOKEN="your-hf-token"
    python clasifLLM/v3/train.py --llm-provider huggingface --llm-model gpt2

    # With local model
    python clasifLLM/v3/train.py --llm-provider local --llm-model microsoft/DialoGPT-small

    # Without LLM feedback (synthetic baseline)
    python clasifLLM/v3/train.py

    # Full configuration
    python clasifLLM/v3/train.py --llm-provider huggingface --api-key YOUR_HF_TOKEN \\
        --model-type graphcodebert --epochs 20 --batch-size 2 --gradient-accumulation 8
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from stage5.stage5C.integrated_system import train_llm_system_v3, create_parser


def main():
    """Main training function."""
    parser = create_parser()
    args = parser.parse_args()

    print("=" * 70)
    print("  ClassifLLM v3 - Enhanced LLM-based Code Quality Classification")
    print("  With Real LLM Feedback Integration")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Model: {args.model_type}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Frozen layers: {args.freeze_layers}")
    print(f"  Dropout: {args.dropout}")
    print()
    print("LLM Feedback Configuration:")
    if hasattr(args, 'llm_provider') and args.llm_provider:
        print(f"  Provider: {args.llm_provider}")
        print(f"  Model: {getattr(args, 'llm_model', 'gpt-4')}")
        print(f"  Temperature: {getattr(args, 'llm_temperature', 0.7)}")
        print(f"  Force regenerate: {getattr(args, 'force_regenerate_feedback', False)}")
    else:
        print("  No LLM feedback configured (using synthetic)")
    print()
    print("Enhanced Features:")
    print(f"  Cross-attention: {'Yes' if args.use_cross_attention else 'No'}")
    print(f"  Contrastive learning: {'Yes' if args.use_contrastive else 'No'}")
    print(f"  Mixed precision (AMP): {'Yes' if args.use_amp else 'No'}")
    print(f"  EMA: {'Yes' if args.use_ema else 'No'}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Confidence weighting: {'Yes' if getattr(args, 'use_confidence_weighting', True) else 'No'}")
    print("=" * 70)

    # Run training (async)
    import asyncio
    asyncio.run(train_llm_system_v3(args))

    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)
    print()
    print("Output files:")
    print(f"  - {args.output_dir}/training_history_llm_v3.json")
    print(f"  - best_model_llm_v3.pt")
    if hasattr(args, 'llm_provider') and args.llm_provider:
        print(f"  - llm_feedback_cache.json")
    print()
    print("Improvements in v3:")
    print("  [+] Real LLM feedback generation instead of synthetic labels")
    print("  [+] Multi-provider LLM support (OpenAI, Anthropic, local)")
    print("  [+] Structured prompts for consistent quality assessment")
    print("  [+] Confidence-weighted training based on LLM certainty")
    print("  [+] Feedback caching for efficiency")
    print("  [+] All v2 enhancements: cross-attention, contrastive learning, AMP, EMA")
    print("=" * 70)


if __name__ == "__main__":
    main()
