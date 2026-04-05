#!/usr/bin/env python3
"""
Simple Training Analysis Script for ClassifMLP

Analyzes training results and displays key metrics.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def analyze_training_history(history_path: str):
    """Analyze training history and display results."""
    print("=== ClassifMLP Training Analysis ===\n")

    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)

    print(f"Total epochs trained: {len(history)}")
    print(f"Early stopping: {'Yes' if len(history) < 20 else 'No'}")
    print()

    # Extract metrics
    epochs = [entry['epoch'] for entry in history]

    # Basic metrics (always present)
    train_losses = [entry['train_loss'] for entry in history]
    val_losses = [entry['val_loss'] for entry in history]

    # Check for detailed accuracy metrics
    has_detailed_acc = 'train_correct_acc' in history[0]
    train_correct_accs = [entry.get('train_correct_acc', 0) for entry in history] if has_detailed_acc else []
    train_useful_accs = [entry.get('train_useful_acc', 0) for entry in history] if has_detailed_acc else []
    train_consistent_accs = [entry.get('train_consistent_acc', 0) for entry in history] if has_detailed_acc else []
    val_correct_accs = [entry.get('val_correct_acc', 0) for entry in history] if has_detailed_acc else []
    val_useful_accs = [entry.get('val_useful_acc', 0) for entry in history] if has_detailed_acc else []
    val_consistent_accs = [entry.get('val_consistent_acc', 0) for entry in history] if has_detailed_acc else []

    # Legacy basic accuracy (if no detailed metrics)
    if not has_detailed_acc:
        train_accs = [entry.get('train_acc', 0) for entry in history]
        val_accs = [entry.get('val_acc', 0) for entry in history]

    # Check for code quality metrics
    has_code_metrics = 'val_bertscore' in history[0]
    def _metric_values(key: str):
        values = []
        for entry in history:
            value = entry.get(key)
            if value is None:
                values.append(float('nan'))
            else:
                values.append(value)
        return values

    if has_code_metrics:
        bert_scores = _metric_values('val_bertscore')
        code_bleu_scores = _metric_values('val_codebleu')
        bleu_scores = _metric_values('val_bleu')
        rouge_scores = _metric_values('val_rouge')
        ruby_scores = _metric_values('val_ruby')

    # Display summary
    print("Training Summary:")
    print(".4f")
    print(".4f")
    print(".3f")
    print(".3f")
    print()

    # Check for stability (no metric degradation)
    if len(val_losses) > 1:
        loss_trend = "Improving" if val_losses[-1] < val_losses[0] else "Degrading"
        print(f"Loss trend: {loss_trend}")

    def _valid_trend(values):
        cleaned = [v for v in values if not np.isnan(v)]
        if len(cleaned) >= 2:
            return cleaned[0], cleaned[-1]
        return None, None

    if has_code_metrics:
        bert_first, bert_last = _valid_trend(bert_scores)
        code_bleu_first, code_bleu_last = _valid_trend(code_bleu_scores)

        if bert_first is not None:
            bert_trend = "Improving" if bert_last >= bert_first else "Degrading"
            print(f"BERTScore trend: {bert_trend}")
        else:
            print("BERTScore trend: insufficient data")

        if code_bleu_first is not None:
            code_bleu_trend = "Improving" if code_bleu_last >= code_bleu_first else "Degrading"
            print(f"CodeBLEU trend: {code_bleu_trend}")
        else:
            print("CodeBLEU trend: insufficient data")

        if bert_first is not None and code_bleu_first is not None:
            print()
            if bert_last >= bert_first and code_bleu_last >= code_bleu_first:
                print("SUCCESS: Anti-overfitting measures working!")
                print("   Code quality metrics are stable/improving")
            else:
                print("WARNING: Metrics still degrading - may need adjustment")
        elif not any(not np.isnan(v) for v in bert_scores + code_bleu_scores):
            print("\nINFO: Code quality metrics missing for this run.")
    else:
        print("INFO: Basic training only (no code quality metrics)")

    print()
    print("Detailed Results:")

    # Display table - show all available metrics
    if has_detailed_acc:
        print(f"{'Epoch':<5} {'Train Loss':<10} {'Val Loss':<10} {'Train Correct':<13} {'Train Useful':<12} {'Val Correct':<11} {'Val Useful':<10}")
        print("-" * 85)

        for entry in history:
            epoch = entry['epoch']
            t_loss = entry['train_loss']
            v_loss = entry['val_loss']
            t_correct = entry.get('train_correct_acc', 0)
            t_useful = entry.get('train_useful_acc', 0)
            v_correct = entry.get('val_correct_acc', 0)
            v_useful = entry.get('val_useful_acc', 0)
            print("4.1f")
    else:
        print(f"{'Epoch':<5} {'Train Loss':<10} {'Val Loss':<10} {'Train Acc':<10} {'Val Acc':<10}")
        print("-" * 55)

        for entry in history:
            epoch = entry['epoch']
            t_loss = entry['train_loss']
            v_loss = entry['val_loss']
            t_acc = entry.get('train_acc', 0)
            v_acc = entry.get('val_acc', 0)
            print("4.1f")

    if has_code_metrics:
        print()
        print("Code Quality Metrics:")
        print(f"{'Epoch':<5} {'BERTScore':<10} {'CodeBLEU':<10} {'BLEU':<10} {'ROUGE':<10} {'RUBY':<10}")
        print("-" * 60)

        for entry in history:
            epoch = entry['epoch']
            bert = entry.get('val_bertscore')
            cbleu = entry.get('val_codebleu')
            bleu = entry.get('val_bleu')
            rouge = entry.get('val_rouge')
            ruby = entry.get('val_ruby')
            def fmt(value):
                return f"{value:.4f}" if value is not None else "N/A"
            print(f"{epoch:<5} {fmt(bert):<10} {fmt(cbleu):<10} {fmt(bleu):<10} {fmt(rouge):<10} {fmt(ruby):<10}")

    # Comprehensive plots for all metrics
    try:
        fig = plt.figure(figsize=(16, 12))
        plot_count = 1

        # 1. Loss Plot
        plt.subplot(2, 3, plot_count)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', marker='o', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Val Loss', marker='s', linewidth=2)
        plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_count += 1

        # 2. Detailed Accuracy Metrics (if available)
        if has_detailed_acc and train_correct_accs:
            plt.subplot(2, 3, plot_count)
            plt.plot(epochs, train_correct_accs, 'b-', label='Train Correct', marker='o', linewidth=2)
            plt.plot(epochs, val_correct_accs, 'b--', label='Val Correct', marker='s', linewidth=2)
            plt.plot(epochs, train_useful_accs, 'g-', label='Train Useful', marker='^', linewidth=2)
            plt.plot(epochs, val_useful_accs, 'g--', label='Val Useful', marker='v', linewidth=2)
            if train_consistent_accs and any(train_consistent_accs):
                plt.plot(epochs, train_consistent_accs, 'orange', label='Train Consistent', marker='D', linewidth=2)
                plt.plot(epochs, val_consistent_accs, 'orange', linestyle='--', label='Val Consistent', marker='*', linewidth=2)
            plt.title('Detailed Accuracy Metrics', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plot_count += 1
        elif not has_detailed_acc:
            # Fallback to basic accuracy
            plt.subplot(2, 3, plot_count)
            plt.plot(epochs, train_accs, 'b-', label='Train Acc', marker='o', linewidth=2)
            plt.plot(epochs, val_accs, 'r-', label='Val Acc', marker='s', linewidth=2)
            plt.title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_count += 1

        # 3. BERTScore and CodeBLEU (Primary Code Metrics)
        if has_code_metrics:
            plt.subplot(2, 3, plot_count)
            plt.plot(epochs, bert_scores, 'g-', label='BERTScore', marker='^', linewidth=2)
            plt.plot(epochs, code_bleu_scores, 'm-', label='CodeBLEU', marker='v', linewidth=2)
            plt.title('Primary Code Quality Metrics', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_count += 1

        # 4. BLEU, ROUGE, RUBY (Additional Code Metrics)
        if has_code_metrics:
            plt.subplot(2, 3, plot_count)
            plt.plot(epochs, bleu_scores, 'c-', label='BLEU', marker='o', linewidth=2)
            plt.plot(epochs, rouge_scores, 'y-', label='ROUGE', marker='s', linewidth=2)
            plt.plot(epochs, ruby_scores, 'purple', label='RUBY', marker='^', linewidth=2)
            plt.title('Additional Code Quality Metrics', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_count += 1

        # 5. Training Accuracy Comparison (if detailed metrics available)
        if has_detailed_acc and train_correct_accs:
            plt.subplot(2, 3, plot_count)
            plt.plot(epochs, train_correct_accs, 'b-', label='Correct', marker='o', linewidth=2)
            plt.plot(epochs, train_useful_accs, 'g-', label='Useful', marker='^', linewidth=2)
            if train_consistent_accs and any(train_consistent_accs):
                plt.plot(epochs, train_consistent_accs, 'orange', label='Consistent', marker='D', linewidth=2)
            plt.title('Training Accuracy Breakdown', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Training Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_count += 1

        # 6. Validation Accuracy Comparison (if detailed metrics available)
        if has_detailed_acc and val_correct_accs:
            plt.subplot(2, 3, plot_count)
            plt.plot(epochs, val_correct_accs, 'b-', label='Correct', marker='s', linewidth=2)
            plt.plot(epochs, val_useful_accs, 'g-', label='Useful', marker='v', linewidth=2)
            if val_consistent_accs and any(val_consistent_accs):
                plt.plot(epochs, val_consistent_accs, 'orange', label='Consistent', marker='*', linewidth=2)
            plt.title('Validation Accuracy Breakdown', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_analysis.png', dpi=150, bbox_inches='tight')
        print("\n[*] Comprehensive plot saved as 'training_analysis.png'")
        print("   - Loss curves over epochs")
        print("   - Detailed accuracy metrics (Correct/Useful/Consistent)" if has_detailed_acc else "   - Basic accuracy metrics")
        print("   - Primary code quality metrics (BERTScore/CodeBLEU)" if has_code_metrics else "")
        print("   - Additional code quality metrics (BLEU/ROUGE/RUBY)" if has_code_metrics else "")
        print("   - Training vs Validation accuracy breakdowns" if has_detailed_acc else "")

    except ImportError:
        print("\nMatplotlib not available for plotting")

    print("\n=== Analysis Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Analyze ClassifMLP training results")
    parser.add_argument("--history-path", type=str, required=True,
                       help="Path to training history JSON file")

    args = parser.parse_args()

    if not Path(args.history_path).exists():
        print(f"❌ Error: History file not found: {args.history_path}")
        return

    analyze_training_history(args.history_path)


if __name__ == "__main__":
    main()
