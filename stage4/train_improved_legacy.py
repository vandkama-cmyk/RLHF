"""
ClassifMLP - Improved Training Script with Anti-Overfitting

Key improvements:
1. High dropout (0.4) for regularization
2. Early stopping (patience=5)
3. Low learning rate (1e-5) for stability
4. Weight decay (1e-3) for L2 regularization
5. LR scheduler for adaptive learning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from stage4.integrated_system_legacy import train_integrated_system
import argparse


def main():
    parser = argparse.ArgumentParser(description="ClassifMLP - Training (Original Overfitting Version)")

    # Core parameters
    parser.add_argument("--device", type=str, default="cuda" if __import__('torch').cuda.is_available() else "cpu")
    parser.add_argument("--feedback-dir", type=str, default="evaluation_results_server")
    parser.add_argument("--output-dir", type=str, default="clasifNN/improved_artifacts")
    parser.add_argument("--dataset-dir", type=str, default="clasifNN/datasets_for_eval",
                        help="Path to origin dataset used for canonical references")

    # Anti-overfitting parameters
    parser.add_argument("--batch-size", type=int, default=16, help="Stable batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs (early stopping may trigger earlier)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Low LR for stability")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.4, help="High dropout for regularization")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Reduced model complexity")

    args = parser.parse_args()

    print("=== ClassifMLP Training (ANTI-OVERFITTING SOLUTION) ===")
    print("=" * 60)
    print("ANTI-OVERFITTING measures applied:")
    print(f"* Dropout: {args.dropout} (prevents overfitting)")
    print(f"* Early stopping patience: {args.patience}")
    print(f"* Learning rate: {args.learning_rate} (stable training)")
    print(f"* Weight decay: {args.weight_decay} (L2 regularization)")
    print(f"* Batch size: {args.batch_size} (gradient stability)")
    print(f"* Hidden dim: {args.hidden_dim} (reduced complexity)")
    print("=" * 60)
    print("Expected result: Code quality metrics will STABILIZE or IMPROVE!")
    print("=" * 60)

    # Run training with overfitting-prone parameters
    train_integrated_system(args)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("Model saved with anti-overfitting techniques.")
    print("Run: python clasifNN/analyze_training.py --history-path clasifNN/improved_artifacts/integrated_training_history.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
