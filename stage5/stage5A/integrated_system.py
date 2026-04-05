"""
ClassifLLM - LLM-based Code Quality Classification Pipeline

Complete training pipeline for fine-tuning LLMs on code quality classification.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import hashlib
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

try:
    from .model import LLMClassifier, CodeBERTClassifier
except ImportError:
    from model import LLMClassifier, CodeBERTClassifier


# =============================================================================
# DATASET
# =============================================================================

class CodeQualityDataset(Dataset):
    """Dataset for LLM-based code quality classification."""
    
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        labels = sample.get('labels', {})
        if not labels:
            single_label = float(sample.get('label', 0.5))
            labels = {
                'consistent': single_label,
                'correct': single_label,
                'useful': single_label
            }
        
        return {
            'question': sample['question'],
            'answer': sample['answer'],
            'labels': labels,
            'metadata': sample.get('metadata', {})
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    return {
        'question': [item['question'] for item in batch],
        'answer': [item['answer'] for item in batch],
        'labels': [item['labels'] for item in batch],
        'metadata': [item['metadata'] for item in batch]
    }


# =============================================================================
# DATA LOADING (reuse from clasifNN)
# =============================================================================

def load_samples(feedback_dir: Path, dataset_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load samples with proper label handling."""
    try:
        import sys
        import os
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from stage4.dataset import load_real_datasets

        current_dir = Path.cwd()
        eval_dir = dataset_dir or (current_dir / "stage4" / "datasets_for_eval")
        feedback_dir_full = current_dir / "stage4" / str(feedback_dir).replace("clasifNN/", "").replace("stage4/", "")
        
        if eval_dir.exists() and feedback_dir_full.exists():
            real_samples = load_real_datasets(eval_dir, feedback_dir_full)
            if real_samples:
                print(f"Loaded {len(real_samples)} real samples")
                return normalize_labels(real_samples)
    except Exception as e:
        print(f"Error loading real datasets: {e}")
    
    print("Using synthetic training data")
    return generate_synthetic_samples()


def normalize_labels(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize labels without random assignment."""
    for sample in samples:
        if 'labels' in sample:
            labels = sample['labels']
            for head in ['consistent', 'correct', 'useful']:
                if head in labels:
                    labels[head] = max(0.0, min(1.0, float(labels[head])))
            sample['label'] = labels.get('useful', 0.5)
        elif 'label' not in sample:
            sample['label'] = 0.5
            sample['labels'] = {'consistent': 0.5, 'correct': 0.5, 'useful': 0.5}
        else:
            label = float(sample['label'])
            sample['labels'] = {'consistent': label, 'correct': label, 'useful': label}
    return samples


def generate_synthetic_samples() -> List[Dict[str, Any]]:
    """Generate synthetic training samples."""
    samples = []
    
    good_examples = [
        ("How to sort a list in Python?", "sorted_list = sorted(my_list)", 1.0),
        ("How to read a file?", "with open('file.txt', 'r') as f: content = f.read()", 1.0),
        ("How to use list comprehension?", "squares = [x**2 for x in range(10)]", 1.0),
        ("How to handle exceptions?", "try: risky() except Exception as e: handle(e)", 0.8),
        ("How to define a function?", "def greet(name): return f'Hello {name}'", 1.0),
    ]
    
    bad_examples = [
        ("How to sort a list?", "list.sort()  # modifies original", 0.3),
        ("How to read a file?", "f = open('file'); data = f.read()  # no close", 0.2),
        ("How to handle errors?", "try: code() except: pass  # bare except", 0.1),
        ("How to write code?", "# TODO: implement", 0.0),
        ("How to optimize?", "import *  # bad practice", 0.2),
    ]
    
    for q, a, label in good_examples + bad_examples:
        samples.append({
            'question': q,
            'answer': a,
            'label': label,
            'labels': {'consistent': label, 'correct': label, 'useful': label},
            'metadata': {'source': 'synthetic'}
        })
    
    return samples


def balanced_split(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Create balanced train/val split."""
    positive_idx = [i for i, s in enumerate(samples) if s.get('label', 0.5) >= 0.6]
    negative_idx = [i for i, s in enumerate(samples) if s.get('label', 0.5) <= 0.4]
    neutral_idx = [i for i, s in enumerate(samples) if 0.4 < s.get('label', 0.5) < 0.6]
    
    rng = random.Random(seed)
    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)
    rng.shuffle(neutral_idx)
    
    def split_group(indices):
        n_val = max(1, int(len(indices) * val_ratio)) if indices else 0
        return indices[n_val:], indices[:n_val]
    
    train_pos, val_pos = split_group(positive_idx)
    train_neg, val_neg = split_group(negative_idx)
    train_neu, val_neu = split_group(neutral_idx)
    
    train_indices = train_pos + train_neg + train_neu
    val_indices = val_pos + val_neg + val_neu
    
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    
    print(f"Split: {len(train_indices)} train, {len(val_indices)} val")
    print(f"  Train: {len(train_pos)} pos, {len(train_neg)} neg, {len(train_neu)} neutral")
    print(f"  Val: {len(val_pos)} pos, {len(val_neg)} neg, {len(val_neu)} neutral")
    
    return train_indices, val_indices


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    predictions: Dict[str, List[float]],
    labels: Dict[str, List[float]]
) -> Dict[str, float]:
    """Compute classification metrics."""
    metrics = {}
    
    for head_name in ['consistent', 'correct', 'useful']:
        if head_name not in predictions or head_name not in labels:
            continue
        
        preds = np.array(predictions[head_name])
        true_labels = np.array(labels[head_name])
        
        pred_binary = (preds >= 0.5).astype(int)
        true_binary = (true_labels >= 0.5).astype(int)
        
        accuracy = (pred_binary == true_binary).mean()
        
        tp = ((pred_binary == 1) & (true_binary == 1)).sum()
        fp = ((pred_binary == 1) & (true_binary == 0)).sum()
        fn = ((pred_binary == 0) & (true_binary == 1)).sum()
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        metrics[f'{head_name}_accuracy'] = float(accuracy)
        metrics[f'{head_name}_precision'] = float(precision)
        metrics[f'{head_name}_recall'] = float(recall)
        metrics[f'{head_name}_f1'] = float(f1)
    
    return metrics


def _tokenize_code(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|->|::|[^\s]", text)


def compute_code_quality_metrics(
    answers: List[str],
    references: List[str],
    predictions: Dict[str, List[float]]
) -> Dict[str, float]:
    """Compute code quality metrics weighted by model confidence."""
    if not answers:
        return {}
    
    useful_probs = np.array(predictions.get('useful', [0.5] * len(answers)))
    
    bleu_scores, rouge_scores, bertscore_scores = [], [], []
    codebleu_scores, ruby_scores = [], []
    
    for i, (answer, reference) in enumerate(zip(answers, references)):
        if not answer or not reference or answer.strip() == reference.strip():
            continue
        
        pred_tokens = _tokenize_code(answer)
        ref_tokens = _tokenize_code(reference)
        
        if not pred_tokens or not ref_tokens:
            continue
        
        weight = useful_probs[i] if i < len(useful_probs) else 0.5
        
        # BLEU
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        match = sum(min(pred_counter[t], ref_counter.get(t, 0)) for t in pred_counter)
        bleu = match / len(pred_tokens) if pred_tokens else 0
        bleu_scores.append((bleu, weight))
        
        # ROUGE-L
        m, n = len(pred_tokens), len(ref_tokens)
        if m > 0 and n > 0:
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for ii in range(m - 1, -1, -1):
                for jj in range(n - 1, -1, -1):
                    if pred_tokens[ii] == ref_tokens[jj]:
                        dp[ii][jj] = 1 + dp[ii + 1][jj + 1]
                    else:
                        dp[ii][jj] = max(dp[ii + 1][jj], dp[ii][jj + 1])
            rouge_scores.append((dp[0][0] / n, weight))
        
        # BERTScore (token overlap proxy)
        common = sum(min(pred_counter[t], ref_counter.get(t, 0)) for t in ref_counter)
        precision = common / len(pred_tokens) if pred_tokens else 0
        recall = common / len(ref_tokens) if ref_tokens else 0
        bertscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        bertscore_scores.append((bertscore, weight))
        codebleu_scores.append((bertscore, weight))
        
        # RUBY (syntax + structure + tokens)
        try:
            compile(answer, '<string>', 'exec')
            syntax_valid = 1.0
        except:
            syntax_valid = 0.0
        
        ruby = 0.4 * bertscore + 0.3 * syntax_valid + 0.3 * bleu
        ruby_scores.append((ruby, weight))
    
    def weighted_mean(scores_weights):
        if not scores_weights:
            return 0.0
        total_weight = sum(w for _, w in scores_weights)
        if total_weight == 0:
            return 0.0
        return sum(s * w for s, w in scores_weights) / total_weight
    
    return {
        'bertscore': weighted_mean(bertscore_scores),
        'codebleu': weighted_mean(codebleu_scores),
        'bleu': weighted_mean(bleu_scores),
        'rouge': weighted_mean(rouge_scores),
        'ruby': weighted_mean(ruby_scores)
    }


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class LLMTrainingPipeline:
    """Training pipeline for LLM-based classification."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        model_type = config.get('model_type', 'codebert')
        dropout = config.get('dropout', 0.3)
        freeze_layers = config.get('freeze_layers', 6)
        
        print(f"[Pipeline] Initializing {model_type} model...")
        self.model = LLMClassifier(
            model_name=model_type,
            dropout=dropout,
            freeze_encoder_layers=freeze_layers,
            device=self.device
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10
    ) -> Dict[str, Any]:
        """Train the LLM classifier."""
        print("=" * 60)
        print("=== LLM Classification Training ===")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model: {self.config.get('model_type', 'codebert')}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {self.config.get('learning_rate', 2e-5)}")
        print(f"Batch size: {self.config.get('batch_size', 8)}")
        print("=" * 60)
        
        # Optimizer with different LR for encoder and classifier
        encoder_params = list(self.model.encoder.parameters())
        classifier_params = list(self.model.classifier.parameters())
        
        optimizer = AdamW([
            {'params': encoder_params, 'lr': self.config.get('learning_rate', 2e-5)},
            {'params': classifier_params, 'lr': self.config.get('learning_rate', 2e-5) * 10}
        ], weight_decay=self.config.get('weight_decay', 0.01))
        
        # Learning rate scheduler with warmup
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)
        
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=num_warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps)
        scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 3)
        history = []
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Training
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler)
            
            # Validation
            val_metrics = self._validate_epoch(val_loader)
            
            # Print metrics
            print(f"  Train - Loss: {train_metrics['loss']:.4f}")
            print(f"    Consistent Acc: {train_metrics['consistent_acc']:.3f}, "
                  f"Correct Acc: {train_metrics['correct_acc']:.3f}, "
                  f"Useful Acc: {train_metrics['useful_acc']:.3f}")
            print(f"  Val - Loss: {val_metrics['loss']:.4f}")
            print(f"    Consistent Acc: {val_metrics['consistent_acc']:.3f}, "
                  f"Correct Acc: {val_metrics['correct_acc']:.3f}, "
                  f"Useful Acc: {val_metrics['useful_acc']:.3f}")
            
            if 'consistent_f1' in val_metrics:
                print(f"    Consistent F1: {val_metrics['consistent_f1']:.3f}, "
                      f"Correct F1: {val_metrics['correct_f1']:.3f}, "
                      f"Useful F1: {val_metrics['useful_f1']:.3f}")
            
            if 'bertscore' in val_metrics:
                print(f"    Code Quality: BERTScore={val_metrics['bertscore']:.3f}, "
                      f"CodeBLEU={val_metrics['codebleu']:.3f}, "
                      f"BLEU={val_metrics['bleu']:.3f}, "
                      f"ROUGE={val_metrics['rouge']:.3f}, "
                      f"RUBY={val_metrics['ruby']:.3f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_model("best_model_llm.pt")
            else:
                patience_counter += 1
            
            if patience is not None and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Record history
            history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_consistent_acc': train_metrics['consistent_acc'],
                'train_correct_acc': train_metrics['correct_acc'],
                'train_useful_acc': train_metrics['useful_acc'],
                'val_consistent_acc': val_metrics['consistent_acc'],
                'val_correct_acc': val_metrics['correct_acc'],
                'val_useful_acc': val_metrics['useful_acc'],
                'val_consistent_f1': val_metrics.get('consistent_f1'),
                'val_correct_f1': val_metrics.get('correct_f1'),
                'val_useful_f1': val_metrics.get('useful_f1'),
                'val_bertscore': val_metrics.get('bertscore'),
                'val_codebleu': val_metrics.get('codebleu'),
                'val_bleu': val_metrics.get('bleu'),
                'val_rouge': val_metrics.get('rouge'),
                'val_ruby': val_metrics.get('ruby'),
            })
        
        return {'history': history, 'best_val_loss': best_val_loss}
    
    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        head_correct = {'consistent': 0, 'correct': 0, 'useful': 0}
        total_samples = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            questions = batch['question']
            answers = batch['answer']
            labels_dict = batch['labels']
            
            # Tokenize
            encoded = self.model.encode_text(questions, answers)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(**encoded)
            
            # Compute loss for each head
            loss = 0.0
            for head_name in ['consistent', 'correct', 'useful']:
                head_labels = torch.tensor(
                    [l[head_name] for l in labels_dict],
                    dtype=torch.float32
                ).to(self.device)
                head_labels = (head_labels >= 0.5).float()
                
                head_loss = self.criterion(logits[head_name], head_labels)
                loss += head_loss
                
                preds = (torch.sigmoid(logits[head_name]) >= 0.5).long()
                head_correct[head_name] += (preds == head_labels.long()).sum().item()
            
            loss = loss / 3
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * len(questions)
            total_samples += len(questions)
            
            pbar.set_postfix({'loss': loss.item()})
        
        return {
            'loss': total_loss / total_samples,
            'consistent_acc': head_correct['consistent'] / total_samples,
            'correct_acc': head_correct['correct'] / total_samples,
            'useful_acc': head_correct['useful'] / total_samples,
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = {'consistent': [], 'correct': [], 'useful': []}
        all_labels = {'consistent': [], 'correct': [], 'useful': []}
        all_answers = []
        all_references = []
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                questions = batch['question']
                answers = batch['answer']
                labels_dict = batch['labels']
                metadata = batch.get('metadata', [{}] * len(answers))
                
                encoded = self.model.encode_text(questions, answers)
                logits = self.model(**encoded)
                
                loss = 0.0
                for head_name in ['consistent', 'correct', 'useful']:
                    head_labels = torch.tensor(
                        [l[head_name] for l in labels_dict],
                        dtype=torch.float32
                    ).to(self.device)
                    head_labels_binary = (head_labels >= 0.5).float()
                    
                    head_loss = self.criterion(logits[head_name], head_labels_binary)
                    loss += head_loss
                    
                    probs = torch.sigmoid(logits[head_name]).cpu().numpy()
                    all_preds[head_name].extend(probs.tolist())
                    all_labels[head_name].extend(head_labels_binary.cpu().numpy().tolist())
                
                loss = loss / 3
                total_loss += loss.item() * len(questions)
                total_samples += len(questions)
                
                # Collect for code metrics
                for i, (answer, meta) in enumerate(zip(answers, metadata)):
                    all_answers.append(answer)
                    ref = meta.get('reference_answer') if isinstance(meta, dict) else None
                    # Empty string is filtered by compute_code_quality_metrics.
                    # Using questions[i] (natural language) as a code reference was wrong.
                    all_references.append(ref or '')
        
        # Compute metrics
        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / total_samples
        
        for head in ['consistent', 'correct', 'useful']:
            metrics[f'{head}_acc'] = metrics.get(f'{head}_accuracy', 0.0)
        
        # Code quality metrics
        code_metrics = compute_code_quality_metrics(all_answers, all_references, all_preds)
        metrics.update(code_metrics)
        
        return metrics
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved: {path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_llm_system(args: argparse.Namespace) -> None:
    """Main training function for LLM classifier."""
    print("=" * 70)
    print("  ClassifLLM Training - LLM-based Code Quality Classification")
    print("=" * 70)
    print(f"Model: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Freeze layers: {args.freeze_layers}")
    print("=" * 70)
    
    config = {
        'device': args.device,
        'model_type': args.model_type,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'freeze_layers': args.freeze_layers,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
    }
    
    # Load data
    feedback_dir = Path(args.feedback_dir)
    dataset_dir = Path(args.dataset_dir)
    samples = load_samples(feedback_dir, dataset_dir)
    print(f"Total samples: {len(samples)}")
    
    # Label distribution
    label_dist = Counter([
        'positive' if s['label'] >= 0.6 else
        'negative' if s['label'] <= 0.4 else
        'neutral'
        for s in samples
    ])
    print(f"Label distribution: {dict(label_dist)}")
    
    # Create dataset and split
    dataset = CodeQualityDataset(samples)
    train_indices, val_indices = balanced_split(samples, val_ratio=0.2)
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Train
    pipeline = LLMTrainingPipeline(config)
    results = pipeline.train(train_loader, val_loader, num_epochs=args.epochs)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history_path = output_dir / "training_history_llm.json"
    with history_path.open('w') as f:
        json.dump(results['history'], f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"History saved to: {history_path}")
    print(f"Best model saved to: best_model_llm.pt")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ClassifLLM - LLM-based Code Quality Classifier")
    
    # Model settings
    parser.add_argument("--model-type", type=str, default="codebert",
                        choices=['codebert', 'graphcodebert', 'unixcoder', 'codet5', 'roberta-base'],
                        help="Pretrained model to use")
    parser.add_argument("--freeze-layers", type=int, default=6,
                        help="Number of encoder layers to freeze")
    
    # Training settings
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=3)
    
    # Data settings
    parser.add_argument("--feedback-dir", type=str, default="evaluation_results_server")
    parser.add_argument("--dataset-dir", type=str, default="stage4/datasets_for_eval")
    parser.add_argument("--output-dir", type=str, default="clasifLLM/artifacts")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_llm_system(args)

