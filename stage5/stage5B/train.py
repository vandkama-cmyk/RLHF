"""
ClassifLLM v2 - Training Script

Enhanced LLM-based code quality classifier with:
- Cross-attention between question and answer
- Contrastive learning auxiliary loss
- Mixed precision training (AMP)
- Gradient accumulation
- Label smoothing
- EMA (Exponential Moving Average)

Supported models:
- codebert: microsoft/codebert-base (recommended)
- graphcodebert: microsoft/graphcodebert-base
- unixcoder: microsoft/unixcoder-base
- codet5: Salesforce/codet5-base

Usage:
    python clasifLLM/v2/train.py --model-type codebert --epochs 15

    # With all enhanced features
    python clasifLLM/v2/train.py --model-type graphcodebert --use-cross-attention --use-contrastive --use-amp

    # Quick test
    python clasifLLM/v2/train.py --epochs 3 --batch-size 4
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import torch
from stage5.stage5B.integrated_system import train_llm_system_v2


def main():
    parser = argparse.ArgumentParser(
        description="ClassifLLM v2 - Enhanced LLM-based Code Quality Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with CodeBERT and all enhancements
  python clasifLLM/v2/train.py --epochs 15

  # Train with GraphCodeBERT
  python clasifLLM/v2/train.py --model-type graphcodebert --freeze-layers 6

  # Disable some features for comparison
  python clasifLLM/v2/train.py --no-cross-attention --no-contrastive

  # Train with smaller batch and more accumulation
  python clasifLLM/v2/train.py --batch-size 4 --gradient-accumulation 4
        """
    )
    
    # Model settings
    parser.add_argument("--model-type", type=str, default="codebert",
                        choices=['codebert', 'graphcodebert', 'unixcoder', 'codet5', 'roberta-base'],
                        help="Pretrained model to use (default: codebert)")
    parser.add_argument("--freeze-layers", type=int, default=4,
                        help="Number of encoder layers to freeze (default: 4)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum sequence length (default: 128 for 4GB GPU)")
    
    # Enhanced features
    parser.add_argument("--use-cross-attention", action="store_true", default=True,
                        help="Use cross-attention between Q and A (default: True)")
    parser.add_argument("--no-cross-attention", action="store_false", dest="use_cross_attention",
                        help="Disable cross-attention")
    parser.add_argument("--use-contrastive", action="store_true", default=True,
                        help="Use contrastive learning (default: True)")
    parser.add_argument("--no-contrastive", action="store_false", dest="use_contrastive",
                        help="Disable contrastive learning")
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="Weight for contrastive loss (default: 0.1)")
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="Use mixed precision training (default: True)")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp",
                        help="Disable mixed precision")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA for model parameters (default: True)")
    parser.add_argument("--no-ema", action="store_false", dest="use_ema",
                        help="Disable EMA")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate (default: 0.999)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default: 0.1)")
    
    # Training settings
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (default: cuda if available)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4 for 4GB GPU)")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs (default: 30)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW (default: 0.01)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Use data augmentation (default: False)")
    
    # Data settings
    parser.add_argument("--feedback-dir", type=str, default="evaluation_results_server",
                        help="Directory with feedback JSON files")
    parser.add_argument("--dataset-dir", type=str, default="stage4/datasets_for_eval",
                        help="Directory with evaluation CSV files")
    parser.add_argument("--output-dir", type=str, default="stage5/stage5B/artifacts",
                        help="Output directory for model and history")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    
    print("=" * 70)
    print("  ClassifLLM v2 - Enhanced LLM-based Code Quality Classification")
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
    print("Enhanced Features:")
    print(f"  Cross-attention: {'yes' if args.use_cross_attention else 'no'}")
    print(f"  Contrastive learning: {'yes' if args.use_contrastive else 'no'}")
    print(f"  Mixed precision (AMP): {'yes' if args.use_amp else 'no'}")
    print(f"  EMA: {'yes' if args.use_ema else 'no'}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print()
    print("Model descriptions:")
    print("  - codebert: General-purpose code understanding (recommended)")
    print("  - graphcodebert: Code understanding with data flow")
    print("  - unixcoder: Unified cross-modal code representation")
    print("  - codet5: Code-aware T5 model")
    print("=" * 70)
    
    # Run training
    train_llm_system_v2(args)
    
    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)
    print()
    print("Output files:")
    print(f"  - {args.output_dir}/training_history_llm_v2.json")
    print(f"  - best_model_llm_v2.pt")
    print()
    print("Improvements in v2:")
    print("  [+] Cross-attention for better Q-A interaction")
    print("  [+] Contrastive learning for better representations")
    print("  [+] Mixed precision for faster training")
    print("  [+] Gradient accumulation for larger effective batch")
    print("  [+] Label smoothing for better generalization")
    print("  [+] EMA for stable predictions")
    print("=" * 70)


if __name__ == "__main__":
    main()

