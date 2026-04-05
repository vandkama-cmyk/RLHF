"""
ClassifMLP - Training Script for CORRECT Metric

Trains MLP classifier using the 'correct' metric from human feedback.
"""

import sys
import os
# Add the project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from stage4.stage4B_corct.integrated_system_correct import train_integrated_system
import argparse


def main():
    parser = argparse.ArgumentParser(description="ClassifMLP - Training (CORRECT METRIC)")

    # Core parameters
    parser.add_argument("--device", type=str, default="cuda" if __import__('torch').cuda.is_available() else "cpu")
    parser.add_argument("--feedback-dir", type=str, default="evaluation_results_server")
    parser.add_argument("--output-dir", type=str, default="stage4/stage4B_corct/artifacts")
    parser.add_argument("--dataset-dir", type=str, default="stage4/datasets_for_eval",
                        help="Path to origin dataset used for canonical references")

    # Anti-overfitting parameters
    parser.add_argument("--batch-size", type=int, default=16, help="Stable batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs (early stopping may trigger earlier)")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Low LR for stability")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.4, help="High dropout for regularization")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Reduced model complexity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    print("=" * 60)
    print("=== ClassifMLP Training (CORRECT METRIC) ===")
    print("=" * 60)
    print("This model is trained using the 'correct' metric from human feedback")
    print("=" * 60)
    print("Training parameters:")
    print(f"* Dropout: {args.dropout} (prevents overfitting)")
    print(f"* Early stopping patience: {args.patience}")
    print(f"* Learning rate: {args.learning_rate} (stable training)")
    print(f"* Weight decay: {args.weight_decay} (L2 regularization)")
    print(f"* Batch size: {args.batch_size} (gradient stability)")
    print(f"* Hidden dim: {args.hidden_dim} (reduced complexity)")
    print("=" * 60)

    # Run training
    train_integrated_system(args)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("Model trained on CORRECT metric from human feedback.")
    print("Results saved to: clasifNN/corct_mlp/artifacts/")
    print("=" * 60)


if __name__ == "__main__":
    main()
