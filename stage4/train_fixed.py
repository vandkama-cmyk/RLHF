"""
ClassifMLP - Fixed Training Script

This script runs the fixed version of the training pipeline with:
1. Multi-head classification (consistent/correct/useful)
2. Real semantic embeddings (with fallback)
3. Proper prediction-based metrics
4. Deterministic label processing
5. No data leakage verification

Usage:
    python clasifNN/train_fixed.py --epochs 20 --patience 5 --dropout 0.4
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from stage4.integrated_system_fixed import train_integrated_system_fixed


def main():
    parser = argparse.ArgumentParser(description="ClassifMLP - Fixed Training")

    # Core parameters
    parser.add_argument("--device", type=str, 
                        default="cuda" if __import__('torch').cuda.is_available() else "cpu")
    parser.add_argument("--feedback-dir", type=str, default="evaluation_results_server")
    parser.add_argument("--output-dir", type=str, default="clasifNN/improved_artifacts")
    parser.add_argument("--dataset-dir", type=str, default="clasifNN/datasets_for_eval",
                        help="Path to dataset files")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden layer dimension")

    args = parser.parse_args()

    print("=" * 70)
    print("  ClassifMLP Training (FIXED VERSION)")
    print("=" * 70)
    print("\nFixes in this version:")
    print("  [+] Multi-head classification: consistent, correct, useful")
    print("  [+] Real semantic embeddings (sentence-transformers or fallback)")
    print("  [+] Prediction-based metrics (not static dataset comparisons)")
    print("  [+] Deterministic labels (no random assignment)")
    print("  [+] Verified train/val split (no data leakage)")
    print()
    print("Training configuration:")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print("=" * 70)

    # Run training
    train_integrated_system_fixed(args)

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print("\nOutput files:")
    print(f"  - {args.output_dir}/training_history_fixed.json")
    print(f"  - best_model_fixed.pt")
    print("\nTo analyze results:")
    print(f"  python clasifNN/analyze_training.py --history-path {args.output_dir}/training_history_fixed.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

