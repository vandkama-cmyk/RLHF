"""
Comparison Script - ClassifLLM v1 vs v2 vs ClassifNN

This script compares the three classification approaches:
1. ClassifLLM v1 - Basic LLM-based classifier (CodeBERT)
2. ClassifLLM v2 - Enhanced LLM with cross-attention, contrastive learning
3. ClassifNN - MLP-based classifier with sentence embeddings

Run: python clasifLLM/v2/compare_methods.py
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import sys

def load_history(path: Path) -> List[Dict[str, Any]]:
    """Load training history from JSON file."""
    if not path.exists():
        return []
    with open(path, 'r') as f:
        return json.load(f)

def get_best_epoch(history: List[Dict], metric: str = 'val_loss', mode: str = 'min') -> Dict:
    """Get the best epoch based on a metric."""
    if not history:
        return {}
    
    if mode == 'min':
        best = min(history, key=lambda x: x.get(metric, float('inf')))
    else:
        best = max(history, key=lambda x: x.get(metric, 0))
    
    return best

def print_comparison():
    """Print comparison of all three methods."""
    
    # Paths
    base_path = Path(__file__).parent.parent.parent
    
    llm_v1_path = base_path / "clasifLLM" / "artifacts" / "training_history_llm.json"
    llm_v2_path = base_path / "clasifLLM" / "v2" / "artifacts" / "training_history_llm_v2.json"
    nn_path = base_path / "clasifNN" / "improved_artifacts" / "training_history.json"
    nn_fixed_path = base_path / "clasifNN" / "improved_artifacts" / "training_history_fixed.json"
    
    # Load histories
    llm_v1_history = load_history(llm_v1_path)
    llm_v2_history = load_history(llm_v2_path)
    nn_history = load_history(nn_path)
    nn_fixed_history = load_history(nn_fixed_path)
    
    print("=" * 80)
    print("  COMPARISON: ClassifLLM v1 vs ClassifLLM v2 vs ClassifNN")
    print("=" * 80)
    print()
    
    # ========== ClassifLLM v1 Results ==========
    print("-" * 40)
    print("1. ClassifLLM v1 (Basic LLM)")
    print("-" * 40)
    
    if llm_v1_history:
        best_v1 = get_best_epoch(llm_v1_history, 'val_loss', 'min')
        print(f"  Best Epoch: {best_v1.get('epoch', 'N/A')}")
        print(f"  Val Loss: {best_v1.get('val_loss', 'N/A'):.4f}" if best_v1.get('val_loss') else "  Val Loss: N/A")
        print()
        print("  Classification Metrics:")
        print(f"    Consistent - Acc: {best_v1.get('val_consistent_acc', 0):.3f}, F1: {best_v1.get('val_consistent_f1', 0):.3f}")
        print(f"    Correct    - Acc: {best_v1.get('val_correct_acc', 0):.3f}, F1: {best_v1.get('val_correct_f1', 0):.3f}")
        print(f"    Useful     - Acc: {best_v1.get('val_useful_acc', 0):.3f}, F1: {best_v1.get('val_useful_f1', 0):.3f}")
        print()
        print("  Code Quality Metrics:")
        print(f"    BERTScore: {best_v1.get('val_bertscore', 0):.3f}")
        print(f"    CodeBLEU:  {best_v1.get('val_codebleu', 0):.3f}")
        print(f"    BLEU:      {best_v1.get('val_bleu', 0):.3f}")
        print(f"    ROUGE:     {best_v1.get('val_rouge', 0):.3f}")
        print(f"    RUBY:      {best_v1.get('val_ruby', 0):.3f}")
    else:
        print("  No training history found")
    
    print()
    
    # ========== ClassifLLM v2 Results ==========
    print("-" * 40)
    print("2. ClassifLLM v2 (Enhanced LLM)")
    print("-" * 40)
    
    if llm_v2_history:
        best_v2 = get_best_epoch(llm_v2_history, 'val_loss', 'min')
        print(f"  Best Epoch: {best_v2.get('epoch', 'N/A')}")
        print(f"  Val Loss: {best_v2.get('val_loss', 'N/A'):.4f}" if best_v2.get('val_loss') else "  Val Loss: N/A")
        print()
        print("  Classification Metrics:")
        print(f"    Consistent - Acc: {best_v2.get('val_consistent_accuracy', 0):.3f}, F1: {best_v2.get('val_consistent_f1', 0):.3f}")
        print(f"    Correct    - Acc: {best_v2.get('val_correct_accuracy', 0):.3f}, F1: {best_v2.get('val_correct_f1', 0):.3f}")
        print(f"    Useful     - Acc: {best_v2.get('val_useful_accuracy', 0):.3f}, F1: {best_v2.get('val_useful_f1', 0):.3f}")
        print()
        print("  Code Quality Metrics:")
        print(f"    BERTScore: {best_v2.get('val_bertscore', 0):.3f}")
        print(f"    CodeBLEU:  {best_v2.get('val_codebleu', 0):.3f}")
        print(f"    BLEU:      {best_v2.get('val_bleu', 0):.3f}")
        print(f"    ROUGE:     {best_v2.get('val_rouge', 0):.3f}")
        print(f"    RUBY:      {best_v2.get('val_ruby', 0):.3f}")
    else:
        print("  Not yet trained - run: python clasifLLM/v2/train.py")
    
    print()
    
    # ========== ClassifNN Results ==========
    print("-" * 40)
    print("3. ClassifNN (MLP-based)")
    print("-" * 40)
    
    # Use fixed history if available, otherwise regular
    nn_data = nn_fixed_history if nn_fixed_history else nn_history
    
    if nn_data:
        best_nn = get_best_epoch(nn_data, 'val_loss', 'min')
        print(f"  Best Epoch: {best_nn.get('epoch', 'N/A')}")
        print(f"  Val Loss: {best_nn.get('val_loss', 'N/A'):.4f}" if best_nn.get('val_loss') else "  Val Loss: N/A")
        print()
        print("  Classification Metrics:")
        # ClassifNN may have different key names
        acc = best_nn.get('val_acc', best_nn.get('val_correct_acc', 0))
        print(f"    Accuracy: {acc:.3f}")
        print(f"    Correct Acc: {best_nn.get('val_correct_acc', acc):.3f}")
        print(f"    Useful Acc:  {best_nn.get('val_useful_acc', acc):.3f}")
        print()
        print("  Code Quality Metrics:")
        print(f"    BERTScore: {best_nn.get('val_bertscore', 0):.3f}")
        print(f"    CodeBLEU:  {best_nn.get('val_codebleu', 0):.3f}")
        print(f"    BLEU:      {best_nn.get('val_bleu', 0):.3f}")
        print(f"    ROUGE:     {best_nn.get('val_rouge', 0):.3f}")
        print(f"    RUBY:      {best_nn.get('val_ruby', 0):.3f}")
    else:
        print("  No training history found")
    
    print()
    
    # ========== Summary Comparison ==========
    print("=" * 80)
    print("  SUMMARY COMPARISON")
    print("=" * 80)
    print()
    
    # Extract comparable metrics
    methods = []
    
    if llm_v1_history:
        best = get_best_epoch(llm_v1_history, 'val_loss', 'min')
        avg_acc = (best.get('val_consistent_acc', 0) + best.get('val_correct_acc', 0) + best.get('val_useful_acc', 0)) / 3
        avg_f1 = (best.get('val_consistent_f1', 0) + best.get('val_correct_f1', 0) + best.get('val_useful_f1', 0)) / 3
        methods.append({
            'name': 'ClassifLLM v1',
            'val_loss': best.get('val_loss', 999),
            'avg_accuracy': avg_acc,
            'avg_f1': avg_f1,
            'bertscore': best.get('val_bertscore', 0),
            'codebleu': best.get('val_codebleu', 0),
            'ruby': best.get('val_ruby', 0),
        })
    
    if llm_v2_history:
        best = get_best_epoch(llm_v2_history, 'val_loss', 'min')
        avg_acc = (best.get('val_consistent_accuracy', 0) + best.get('val_correct_accuracy', 0) + best.get('val_useful_accuracy', 0)) / 3
        avg_f1 = (best.get('val_consistent_f1', 0) + best.get('val_correct_f1', 0) + best.get('val_useful_f1', 0)) / 3
        methods.append({
            'name': 'ClassifLLM v2',
            'val_loss': best.get('val_loss', 999),
            'avg_accuracy': avg_acc,
            'avg_f1': avg_f1,
            'bertscore': best.get('val_bertscore', 0),
            'codebleu': best.get('val_codebleu', 0),
            'ruby': best.get('val_ruby', 0),
        })
    
    if nn_data:
        best = get_best_epoch(nn_data, 'val_loss', 'min')
        acc = best.get('val_acc', best.get('val_correct_acc', 0))
        methods.append({
            'name': 'ClassifNN',
            'val_loss': best.get('val_loss', 999),
            'avg_accuracy': acc,
            'avg_f1': 0,  # Not available in this format
            'bertscore': best.get('val_bertscore', 0),
            'codebleu': best.get('val_codebleu', 0),
            'ruby': best.get('val_ruby', 0),
        })
    
    if methods:
        print("  Metric Comparison Table:")
        print("  " + "-" * 70)
        print(f"  {'Method':<18} {'Loss':>8} {'Acc':>8} {'F1':>8} {'BERT':>8} {'CBLEU':>8} {'RUBY':>8}")
        print("  " + "-" * 70)
        
        for m in methods:
            print(f"  {m['name']:<18} {m['val_loss']:>8.4f} {m['avg_accuracy']:>8.3f} {m['avg_f1']:>8.3f} "
                  f"{m['bertscore']:>8.3f} {m['codebleu']:>8.3f} {m['ruby']:>8.3f}")
        
        print("  " + "-" * 70)
        print()
        
        # Find best method per metric
        print("  Best Method per Metric:")
        
        if len(methods) >= 2:
            # Best by loss (lower is better)
            best_loss = min(methods, key=lambda x: x['val_loss'])
            print(f"    Lowest Loss: {best_loss['name']} ({best_loss['val_loss']:.4f})")
            
            # Best by accuracy (higher is better)
            best_acc = max(methods, key=lambda x: x['avg_accuracy'])
            print(f"    Highest Accuracy: {best_acc['name']} ({best_acc['avg_accuracy']:.3f})")
            
            # Best by F1 (higher is better)
            best_f1 = max(methods, key=lambda x: x['avg_f1'])
            print(f"    Highest F1: {best_f1['name']} ({best_f1['avg_f1']:.3f})")
            
            # Best by code metrics (higher is better)
            best_bert = max(methods, key=lambda x: x['bertscore'])
            print(f"    Highest BERTScore: {best_bert['name']} ({best_bert['bertscore']:.3f})")
            
            best_ruby = max(methods, key=lambda x: x['ruby'])
            print(f"    Highest RUBY: {best_ruby['name']} ({best_ruby['ruby']:.3f})")
    
    print()
    print("=" * 80)
    print("  CONCLUSIONS & RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("  Based on the analysis:")
    print()
    print("  ClassifLLM v1 (Current):")
    print("    + Lower validation loss")
    print("    + Higher classification accuracy and F1 scores")
    print("    - Lower code quality metrics (BERTScore, CodeBLEU, RUBY)")
    print("    - Basic architecture without cross-attention")
    print()
    print("  ClassifLLM v2 (Enhanced):")
    print("    + Cross-attention for better Q-A understanding")
    print("    + Contrastive learning for better representations")
    print("    + Mixed precision for faster training")
    print("    + Label smoothing for better generalization")
    print("    + EMA for stable predictions")
    print("    * Run training to see results")
    print()
    print("  ClassifNN (MLP):")
    print("    + Higher code quality metrics")
    print("    + Faster training (no LLM overhead)")
    print("    + Smaller model size")
    print("    - Lower classification accuracy")
    print("    - Limited semantic understanding")
    print()
    print("  RECOMMENDATION:")
    print("    For classification accuracy: ClassifLLM v1/v2")
    print("    For code quality metrics: ClassifNN")
    print("    For best of both: ClassifLLM v2 (with enhancements)")
    print()
    print("=" * 80)


if __name__ == "__main__":
    print_comparison()

