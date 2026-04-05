"""
ClassifLLM - Training Script

Train LLM-based code quality classifier using pretrained models from HuggingFace.

Supported models:
- codebert: microsoft/codebert-base (recommended)
- graphcodebert: microsoft/graphcodebert-base
- unixcoder: microsoft/unixcoder-base
- codet5: Salesforce/codet5-base
- roberta-base: roberta-base

Usage:
    python clasifLLM/train.py --model-type codebert --epochs 10 --batch-size 8

    # Use GraphCodeBERT with more frozen layers
    python clasifLLM/train.py --model-type graphcodebert --freeze-layers 8

    # Quick test with small batch
    python clasifLLM/train.py --batch-size 4 --epochs 3
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import torch
from stage5.stage5A.integrated_system import train_llm_system


def main():
    parser = argparse.ArgumentParser(
        description="ClassifLLM - LLM-based Code Quality Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with CodeBERT (default)
  python clasifLLM/train.py --epochs 10

  # Train with GraphCodeBERT
  python clasifLLM/train.py --model-type graphcodebert --freeze-layers 8

  # Train with smaller batch size (for limited GPU memory)
  python clasifLLM/train.py --batch-size 4 --epochs 5
        """
    )
    
    # Model settings
    parser.add_argument("--model-type", type=str, default="codebert",
                        choices=['codebert', 'graphcodebert', 'unixcoder', 'codet5', 'roberta-base'],
                        help="Pretrained model to use (default: codebert)")
    parser.add_argument("--freeze-layers", type=int, default=6,
                        help="Number of encoder layers to freeze (default: 6)")
    
    # Training settings
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (default: cuda if available)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8, reduce if OOM)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs (default: 30)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW (default: 0.01)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (default: 3)")
    
    # Data settings
    parser.add_argument("--feedback-dir", type=str, default="evaluation_results_server",
                        help="Directory with feedback JSON files")
    parser.add_argument("--dataset-dir", type=str, default="stage4/datasets_for_eval",
                        help="Directory with evaluation CSV files")
    parser.add_argument("--output-dir", type=str, default="stage5/stage5A/artifacts",
                        help="Output directory for model and history")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ClassifLLM - LLM-based Code Quality Classification")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Model: {args.model_type}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Frozen layers: {args.freeze_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Early stopping patience: {args.patience}")
    print()
    print("Model descriptions:")
    print("  - codebert: General-purpose code understanding (recommended)")
    print("  - graphcodebert: Code understanding with data flow")
    print("  - unixcoder: Unified cross-modal code representation")
    print("  - codet5: Code-aware T5 model")
    print("=" * 70)
    
    # Run training
    train_llm_system(args)
    
    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)
    print()
    print("Output files:")
    print(f"  - {args.output_dir}/training_history_llm.json")
    print(f"  - best_model_llm.pt")
    print()
    print("To analyze results:")
    print(f"  python clasifNN/analyze_training.py --history-path {args.output_dir}/training_history_llm.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

