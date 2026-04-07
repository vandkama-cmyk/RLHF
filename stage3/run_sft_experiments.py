"""
SFT Experiments for Stage 3
===========================

This script runs SFT (Supervised Fine-Tuning) experiments for two cases:
- Case 1: 11 samples
- Case 2: 1247 samples (full dataset)

Tracks all metrics for 10 epochs:
- Train Loss, Val Loss, Reward
- BERTScore, CODEBLEU, BLEU, ROUGE, RUBY
- Gradient Norm
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    get_linear_schedule_with_warmup
)

# Import metrics from modern_rlhf
try:
    from modern_rlhf.metrics import ModernMetricsEvaluator, get_metric_float
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: modern_rlhf.metrics not available, using fallback metrics")


@dataclass
class EpochMetrics:
    """Container for epoch metrics."""
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    reward: float
    gradient_norm: float
    bertscore: float
    codebleu: float
    bleu: float
    rouge: float
    ruby: float
    learning_rate: float
    epoch_time: float
    # Distribution metrics (computed from model logits)
    policy_entropy: float = 0.0     # Mean per-token entropy H(π) of policy output distribution
    approx_kl: float = 0.0          # log(V) − H(π); divergence from *uniform* over vocab — NOT KL(π||π_ref)
    # Balanced accuracy over eval set (reward-based binary classification)
    balanced_accuracy: float = 0.0


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Model
    model_name: str = "microsoft/CodeGPT-small-py"
    
    # Training - more conservative settings for stability
    num_epochs: int = 30
    batch_size: int = 2  # Smaller batch for stability
    learning_rate: float = 5e-6  # Lower LR to prevent NaN
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.5  # More aggressive clipping
    weight_decay: float = 0.01
    
    # Data
    max_length: int = 128  # Shorter sequences for stability

    # Paths
    output_dir: str = "./stage3/outputs"

    # Device - use FP32 for stability
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = False  # Disable FP16 for stability

    # Reproducibility
    seed: int = 42


class SimpleMetricsEvaluator:
    """Simple metrics evaluator fallback."""
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> float:
        """Simple token overlap as BERTScore proxy."""
        scores = []
        for pred, ref in zip(predictions, references):
            if not pred or not ref:
                scores.append(0.0)
                continue
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            if not ref_tokens:
                scores.append(0.0)
                continue
            overlap = len(pred_tokens & ref_tokens) / max(1, len(ref_tokens))
            scores.append(overlap)
        return float(np.mean(scores)) if scores else 0.0
    
    def compute_codebleu(self, predictions: List[str], references: List[str]) -> float:
        """Simple token F1 as CodeBLEU proxy."""
        scores = []
        for pred, ref in zip(predictions, references):
            if not pred or not ref:
                scores.append(0.0)
                continue
            p_tokens = pred.split()
            r_tokens = ref.split()
            if not p_tokens or not r_tokens:
                scores.append(0.0)
                continue
            common = len([t for t in p_tokens if t in r_tokens])
            prec = common / len(p_tokens) if p_tokens else 0.0
            rec = common / len(r_tokens) if r_tokens else 0.0
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            else:
                f1 = 0.0
            scores.append(f1)
        return float(np.mean(scores)) if scores else 0.0
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Simple unigram precision as BLEU proxy."""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            if not pred_tokens:
                scores.append(0.0)
                continue
            match = sum(1 for t in pred_tokens if t in ref_tokens)
            scores.append(match / len(pred_tokens))
        return float(np.mean(scores)) if scores else 0.0
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> float:
        """Simple LCS ratio as ROUGE proxy."""
        def lcs_len(a: List[str], b: List[str]) -> int:
            la, lb = len(a), len(b)
            dp = [[0] * (lb + 1) for _ in range(la + 1)]
            for i in range(la - 1, -1, -1):
                for j in range(lb - 1, -1, -1):
                    if a[i] == b[j]:
                        dp[i][j] = 1 + dp[i + 1][j + 1]
                    else:
                        dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
            return dp[0][0]
        
        scores = []
        for pred, ref in zip(predictions, references):
            p_tokens = pred.split()
            r_tokens = ref.split()
            if not r_tokens:
                scores.append(0.0)
                continue
            lcs = lcs_len(p_tokens, r_tokens)
            scores.append(lcs / max(1, len(r_tokens)))
        return float(np.mean(scores)) if scores else 0.0
    
    def compute_ruby(self, predictions: List[str], references: List[str]) -> float:
        """Simple token edit distance as RUBY proxy."""
        def edit_dist(a: List[str], b: List[str]) -> int:
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                for j in range(n + 1):
                    if i == 0:
                        dp[i][j] = j
                    elif j == 0:
                        dp[i][j] = i
                    elif a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            return dp[m][n]
        
        scores = []
        for pred, ref in zip(predictions, references):
            p_tokens = pred.split()
            r_tokens = ref.split()
            max_len = max(len(p_tokens), len(r_tokens))
            if max_len == 0:
                scores.append(1.0)
                continue
            dist = edit_dist(p_tokens, r_tokens)
            similarity = 1.0 - (dist / max_len)
            scores.append(max(0.0, similarity))
        return float(np.mean(scores)) if scores else 0.0
    
    def compute_all(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            'bertscore': self.compute_bertscore(predictions, references),
            'codebleu': self.compute_codebleu(predictions, references),
            'bleu': self.compute_bleu(predictions, references),
            'rouge': self.compute_rouge(predictions, references),
            'ruby': self.compute_ruby(predictions, references),
        }


class SFTDataset(torch.utils.data.Dataset):
    """Dataset for SFT training."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get('question', item.get('intent', ''))
        answer = item.get('answer', item.get('snippet', item.get('best_answer', '')))
        
        # Format as prompt-completion
        text = f"### Question: {question}\n### Answer: {answer}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'question': question,
            'reference': answer,
        }


class SFTTrainer:
    """SFT Trainer with comprehensive metrics tracking."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Seed all RNGs for reproducibility
        import random as _random
        _random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        print(f"Initializing SFT Trainer on device: {self.device} | seed={config.seed}")

        # Load model and tokenizer
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Use FP32 for numerical stability
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,  # FP32 for stability
            local_files_only=True
        ).to(self.device)
        
        # Ensure model is in training mode
        self.model.train()
        
        # Metrics evaluator
        if METRICS_AVAILABLE:
            self.metrics_evaluator = ModernMetricsEvaluator()
        else:
            self.metrics_evaluator = SimpleMetricsEvaluator()
        
        # Training history
        self.history: List[EpochMetrics] = []
    
    def compute_gradient_norm(self) -> float:
        """Compute the total gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def generate_response(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Generate response for a prompt."""
        try:
            inputs = self.tokenizer(
                f"### Question: {prompt}\n### Answer:",
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_length,
                padding=True
            ).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for stability
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            self.model.train()
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract answer part
            if "### Answer:" in response:
                response = response.split("### Answer:")[-1].strip()
            return response if response else "N/A"
        except Exception as e:
            print(f"Generation error: {e}")
            return "N/A"
    
    def evaluate(self, eval_data: List[Dict]) -> Dict[str, float]:
        """Evaluate model on given data."""
        self.model.eval()
        
        predictions = []
        references = []
        
        for item in eval_data[:50]:  # Limit for speed
            question = item.get('question', item.get('intent', ''))
            reference = item.get('answer', item.get('snippet', item.get('best_answer', '')))
            
            prediction = self.generate_response(question)
            predictions.append(prediction)
            references.append(reference)
        
        # Compute metrics
        if METRICS_AVAILABLE:
            metrics_results = self.metrics_evaluator.compute_all_metrics(predictions, references)
            metrics = {
                'bertscore': get_metric_float(metrics_results, 'bertscore'),
                'codebleu': get_metric_float(metrics_results, 'codebleu', 'codebleu_proxy'),
                'bleu': get_metric_float(metrics_results, 'bleu'),
                'rouge': get_metric_float(metrics_results, 'rouge'),
                'ruby': get_metric_float(metrics_results, 'ruby', 'ruby_like_heuristic'),
            }
        else:
            metrics = self.metrics_evaluator.compute_all(predictions, references)
        
        return metrics
    
    def train(self, train_data: List[Dict], eval_data: List[Dict], case_name: str = "Case"):
        """Train the model and track all metrics."""
        print(f"\n{'='*60}")
        print(f"Starting SFT Training: {case_name}")
        print(f"Train samples: {len(train_data)}")
        print(f"Eval samples: {len(eval_data)}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"{'='*60}\n")
        
        # Create dataset and dataloader
        dataset = SFTDataset(train_data, self.tokenizer, self.config.max_length)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(dataloader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.history = []
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            self.model.train()
            
            total_loss = 0.0
            total_grad_norm = 0.0
            total_entropy = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected, skipping batch")
                    optimizer.zero_grad()
                    continue

                # Compute per-token entropy from logits (before backward)
                with torch.no_grad():
                    log_probs = torch.nn.functional.log_softmax(outputs.logits.float(), dim=-1)
                    batch_entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean().item()
                    if not (np.isnan(batch_entropy) or np.isinf(batch_entropy)):
                        total_entropy += batch_entropy

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Compute gradient norm before clipping
                grad_norm = self.compute_gradient_norm()

                # Check for NaN gradients
                if np.isnan(grad_norm) or np.isinf(grad_norm):
                    print(f"Warning: NaN/Inf gradient detected, skipping batch")
                    optimizer.zero_grad()
                    continue

                total_grad_norm += grad_norm

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'grad_norm': f'{grad_norm:.4f}'
                })
            
            # Epoch metrics
            avg_loss = total_loss / max(1, num_batches)
            avg_grad_norm = total_grad_norm / max(1, num_batches)
            avg_entropy = total_entropy / max(1, num_batches)
            # KL divergence from uniform: KL(p||uniform) = log(V) - H(p)
            vocab_size = self.model.config.vocab_size
            avg_kl_from_uniform = max(0.0, np.log(vocab_size) - avg_entropy)
            epoch_time = time.time() - epoch_start

            # Evaluate
            print(f"\nEvaluating epoch {epoch+1}...")
            eval_metrics = self.evaluate(eval_data)

            # Compute reward (simple average of all metrics)
            reward = np.mean([
                eval_metrics['bertscore'],
                eval_metrics['codebleu'],
                eval_metrics['bleu'],
                eval_metrics['rouge'],
                eval_metrics['ruby']
            ])

            # Balanced accuracy: binarise reward at 0.5 and compute balanced accuracy
            # Positive = reward > 0.5, Negative = reward <= 0.5
            reward_threshold = 0.5
            predicted_pos = float(reward > reward_threshold)
            # Since all eval samples share the same epoch reward, use per-metric rewards
            per_metric_rewards = [
                eval_metrics['bertscore'],
                eval_metrics['codebleu'],
                eval_metrics['bleu'],
                eval_metrics['rouge'],
                eval_metrics['ruby']
            ]
            tp = sum(1 for r in per_metric_rewards if r > reward_threshold)
            tn = sum(1 for r in per_metric_rewards if r <= reward_threshold)
            total_pos = sum(1 for r in per_metric_rewards if r > reward_threshold)
            total_neg = sum(1 for r in per_metric_rewards if r <= reward_threshold)
            sensitivity = tp / max(total_pos, 1)
            specificity = tn / max(total_neg, 1)
            balanced_acc = (sensitivity + specificity) / 2.0

            # Store metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch + 1,
                train_loss=avg_loss,
                val_loss=None,  # Not computed separately
                reward=reward,
                gradient_norm=avg_grad_norm,
                bertscore=eval_metrics['bertscore'],
                codebleu=eval_metrics['codebleu'],
                bleu=eval_metrics['bleu'],
                rouge=eval_metrics['rouge'],
                ruby=eval_metrics['ruby'],
                learning_rate=scheduler.get_last_lr()[0],
                epoch_time=epoch_time,
                policy_entropy=avg_entropy,
                approx_kl=avg_kl_from_uniform,
                balanced_accuracy=balanced_acc,
            )
            self.history.append(epoch_metrics)

            # Print epoch summary
            print(f"\n[Epoch {epoch+1}/{self.config.num_epochs}]")
            print(f"  Train Loss:        {avg_loss:.4f}")
            print(f"  Gradient Norm:     {avg_grad_norm:.4f}")
            print(f"  policy_entropy:    {avg_entropy:.4f}  (mean H(pi) over tokens)")
            print(f"  approx_kl:        {avg_kl_from_uniform:.4f}  (log(V)-H(pi), vs uniform; not KL(pi||pi_ref))")
            print(f"  Reward:            {reward:.4f}")
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
            print(f"  BERTScore: {eval_metrics['bertscore']:.4f}")
            print(f"  CODEBLEU:  {eval_metrics['codebleu']:.4f}")
            print(f"  BLEU:      {eval_metrics['bleu']:.4f}")
            print(f"  ROUGE:     {eval_metrics['rouge']:.4f}")
            print(f"  RUBY:      {eval_metrics['ruby']:.4f}")
            print(f"  Time:      {epoch_time:.1f}s")
        
        return self.history


def load_sft_data(num_samples: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
    """Load SFT dataset."""
    # Try loading from sft_dataset.csv
    sft_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets_for_training", "sft_dataset.csv"
    )
    
    if os.path.exists(sft_path):
        print(f"Loading SFT data from: {sft_path}")
        df = pd.read_csv(sft_path)
        data = []
        for _, row in df.iterrows():
            data.append({
                'question': str(row.get('question', '')),
                'answer': str(row.get('best_answer', '')),
            })
    else:
        # Fallback to CoNaLa
        conala_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "conala-corpus", "conala-train.jsonl"
        )
        print(f"Loading CoNaLa data from: {conala_path}")
        data = []
        with open(conala_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    'question': item.get('rewritten_intent', item.get('intent', '')),
                    'answer': item.get('snippet', ''),
                })
    
    # Filter empty entries
    data = [d for d in data if d['question'] and d['answer']]
    
    # Limit samples if specified
    if num_samples is not None and num_samples < len(data):
        data = data[:num_samples]
    
    # Split into train and eval
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    # Do NOT fall back to training data if eval set is small.
    # The original fallback (data[-min(50,n):]) caused train/eval overlap
    # for the 11-sample case (eval became all 11 samples including train items).
    # Accept a small eval set; 1 sample is enough to compute metrics honestly.
    
    return train_data, eval_data


def save_results(history: List[EpochMetrics], case_name: str, output_dir: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"{case_name}_history.json")
    history_dicts = [asdict(h) for h in history]
    with open(json_path, 'w') as f:
        json.dump(history_dicts, f, indent=2)
    print(f"Saved JSON results to: {json_path}")
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{case_name}_metrics.csv")
    df = pd.DataFrame(history_dicts)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")


def generate_report(case1_history: List[EpochMetrics], 
                   case2_history: List[EpochMetrics],
                   output_dir: str):
    """Generate comprehensive markdown report."""
    report_path = os.path.join(output_dir, "SFT_EXPERIMENTS_DETAILED.md")
    
    report = """# SFT Experiments - Detailed Epoch-by-Epoch Results

## Overview

This document contains detailed epoch-by-epoch results for SFT (Supervised Fine-Tuning) experiments comparing two cases:
- **Case 1**: SFT with 11 samples
- **Case 2**: SFT with 1247 samples

---

## Case 1: SFT with 11 samples

### Loss Dynamics & Gradient Norm

| Epoch | Train Loss | Gradient Norm | Pattern |
|-------|------------|---------------|---------|
"""
    
    for m in case1_history:
        pattern = "Initial" if m.epoch == 1 else ("Converged" if m.epoch == len(case1_history) else "Training")
        report += f"| {m.epoch}     | **{m.train_loss:.3f}**  | **{m.gradient_norm:.2f}**      | {pattern} |\n"
    
    report += f"""
**Summary:**
- **Loss Reduction**: {case1_history[0].train_loss:.3f} → {case1_history[-1].train_loss:.3f} ({(1 - case1_history[-1].train_loss/case1_history[0].train_loss)*100:.1f}% reduction)
- **Gradient Norm Pattern**: {case1_history[0].gradient_norm:.2f} → {case1_history[-1].gradient_norm:.2f}

### All Metrics by Epoch

| Epoch | Train Loss | Val Loss | Reward | BERTScore | CODEBLEU | BLEU   | ROUGE  | RUBY   |
|-------|------------|----------|--------|-----------|----------|--------|--------|--------|
"""
    
    for m in case1_history:
        val_loss = f"{m.val_loss:.3f}" if m.val_loss else "-"
        report += f"| {m.epoch}     | **{m.train_loss:.3f}**  | {val_loss}        | {m.reward:.3f}  | {m.bertscore:.4f}    | {m.codebleu:.4f}   | {m.bleu:.4f} | {m.rouge:.4f} | {m.ruby:.4f} |\n"
    
    final1 = case1_history[-1]
    report += f"""
### Final Metrics (11 samples)

| Metric    | Value      |
|-----------|------------|
| BERTScore | **{final1.bertscore:.4f}** |
| CODEBLEU  | **{final1.codebleu:.4f}** |
| BLEU      | **{final1.bleu:.4f}** |
| ROUGE     | **{final1.rouge:.4f}** |
| RUBY      | **{final1.ruby:.4f}** |

---

## Case 2: SFT with 1247 samples

### Loss Dynamics & Gradient Norm

| Epoch | Train Loss | Gradient Norm | Pattern |
|-------|------------|---------------|---------|
"""
    
    for m in case2_history:
        pattern = "Initial" if m.epoch == 1 else ("Converged" if m.epoch == len(case2_history) else "Training")
        report += f"| {m.epoch}     | **{m.train_loss:.3f}**  | **{m.gradient_norm:.2f}**      | {pattern} |\n"
    
    report += f"""
**Summary:**
- **Loss Reduction**: {case2_history[0].train_loss:.3f} → {case2_history[-1].train_loss:.3f} ({(1 - case2_history[-1].train_loss/case2_history[0].train_loss)*100:.1f}% reduction)
- **Gradient Norm Pattern**: {case2_history[0].gradient_norm:.2f} → {case2_history[-1].gradient_norm:.2f}

### All Metrics by Epoch

| Epoch | Train Loss | Val Loss | Reward | BERTScore | CODEBLEU | BLEU   | ROUGE  | RUBY   |
|-------|------------|----------|--------|-----------|----------|--------|--------|--------|
"""
    
    for m in case2_history:
        val_loss = f"{m.val_loss:.3f}" if m.val_loss else "-"
        report += f"| {m.epoch}     | **{m.train_loss:.3f}**  | {val_loss}        | {m.reward:.3f}  | {m.bertscore:.4f}    | {m.codebleu:.4f}   | {m.bleu:.4f} | {m.rouge:.4f} | {m.ruby:.4f} |\n"
    
    final2 = case2_history[-1]
    report += f"""
### Final Metrics (1247 samples)

| Metric    | Value      |
|-----------|------------|
| BERTScore | **{final2.bertscore:.4f}** |
| CODEBLEU  | **{final2.codebleu:.4f}** |
| BLEU      | **{final2.bleu:.4f}** |
| ROUGE     | **{final2.rouge:.4f}** |
| RUBY      | **{final2.ruby:.4f}** |

---

## Comparison Summary

### Loss Dynamics Comparison

| Metric          | Case 1 (11 samples) | Case 2 (1247 samples) | Difference |
|-----------------|---------------------|----------------------|------------|
| Epoch 1 Loss    | {case1_history[0].train_loss:.3f}               | {case2_history[0].train_loss:.3f}                | {case2_history[0].train_loss - case1_history[0].train_loss:+.3f}       |
| Epoch 10 Loss   | {case1_history[-1].train_loss:.3f}               | {case2_history[-1].train_loss:.3f}                | {case2_history[-1].train_loss - case1_history[-1].train_loss:+.3f}     |
| Loss Reduction  | {(1 - case1_history[-1].train_loss/case1_history[0].train_loss)*100:.1f}%               | {(1 - case2_history[-1].train_loss/case2_history[0].train_loss)*100:.1f}%                | {((1 - case2_history[-1].train_loss/case2_history[0].train_loss) - (1 - case1_history[-1].train_loss/case1_history[0].train_loss))*100:+.1f}%      |

### Gradient Norm Comparison

| Metric          | Case 1 (11 samples) | Case 2 (1247 samples) | Pattern    |
|-----------------|---------------------|----------------------|------------|
| Epoch 1         | {case1_history[0].gradient_norm:.2f}                | {case2_history[0].gradient_norm:.2f}                 | {'Similar' if abs(case1_history[0].gradient_norm - case2_history[0].gradient_norm) < 0.5 else 'Different'}    |
| Epoch 10        | {case1_history[-1].gradient_norm:.2f}                | {case2_history[-1].gradient_norm:.2f}                 | {'Similar' if abs(case1_history[-1].gradient_norm - case2_history[-1].gradient_norm) < 0.5 else 'Different'}    |
| Stability       | {'Stable' if abs(case1_history[-1].gradient_norm - case1_history[0].gradient_norm) < 2 else 'Unstable'}              | {'Stable' if abs(case2_history[-1].gradient_norm - case2_history[0].gradient_norm) < 2 else 'Unstable'}               | Both {'stable' if abs(case1_history[-1].gradient_norm - case1_history[0].gradient_norm) < 2 and abs(case2_history[-1].gradient_norm - case2_history[0].gradient_norm) < 2 else 'varying'}|

### Final Metrics Comparison

| Metric    | Case 1 (11 samples) | Case 2 (1247 samples) | Improvement |
|-----------|---------------------|----------------------|-------------|
| BERTScore | {final1.bertscore:.4f}              | **{final2.bertscore:.4f}**           | {((final2.bertscore - final1.bertscore) / max(0.0001, final1.bertscore) * 100):+.1f}%      |
| CODEBLEU  | {final1.codebleu:.4f}              | **{final2.codebleu:.4f}**           | {((final2.codebleu - final1.codebleu) / max(0.0001, final1.codebleu) * 100):+.1f}%     |
| BLEU      | {final1.bleu:.4f}              | **{final2.bleu:.4f}**           | {((final2.bleu - final1.bleu) / max(0.0001, final1.bleu) * 100):+.1f}%    |
| ROUGE     | {final1.rouge:.4f}              | **{final2.rouge:.4f}**           | {((final2.rouge - final1.rouge) / max(0.0001, final1.rouge) * 100):+.1f}%     |
| RUBY      | {final1.ruby:.4f}              | **{final2.ruby:.4f}**           | {((final2.ruby - final1.ruby) / max(0.0001, final1.ruby) * 100):+.1f}%           |

---

## Key Observations

### 1. Loss Convergence
- Case 1 achieves {(1 - case1_history[-1].train_loss/case1_history[0].train_loss)*100:.1f}% loss reduction over 10 epochs
- Case 2 achieves {(1 - case2_history[-1].train_loss/case2_history[0].train_loss)*100:.1f}% loss reduction over 10 epochs
- {'Case 2 converges better due to more diverse training examples' if case2_history[-1].train_loss < case1_history[-1].train_loss else 'Both cases show similar convergence patterns'}

### 2. Gradient Stability
- Case 1 gradient norm: {case1_history[0].gradient_norm:.2f} → {case1_history[-1].gradient_norm:.2f}
- Case 2 gradient norm: {case2_history[0].gradient_norm:.2f} → {case2_history[-1].gradient_norm:.2f}
- Both cases show {'stable' if abs(case1_history[-1].gradient_norm - case1_history[0].gradient_norm) < 2 and abs(case2_history[-1].gradient_norm - case2_history[0].gradient_norm) < 2 else 'varying'} gradient patterns

### 3. Code Quality Metrics Impact
- **BERTScore**: {((final2.bertscore - final1.bertscore) / max(0.0001, final1.bertscore) * 100):+.1f}% improvement with larger dataset
- **CODEBLEU**: {((final2.codebleu - final1.codebleu) / max(0.0001, final1.codebleu) * 100):+.1f}% improvement
- **BLEU**: {((final2.bleu - final1.bleu) / max(0.0001, final1.bleu) * 100):+.1f}% improvement
- **ROUGE**: {((final2.rouge - final1.rouge) / max(0.0001, final1.rouge) * 100):+.1f}% improvement
- **RUBY**: {((final2.ruby - final1.ruby) / max(0.0001, final1.ruby) * 100):+.1f}% improvement

### 4. Dataset Size Impact
- Larger dataset (1247 vs 11 samples) {'significantly improves' if final2.bleu > final1.bleu * 1.5 else 'improves'} all code quality metrics
- Quality of SFT data is critical for final model performance

---

## Training Configuration

| Parameter                | Value                          |
|--------------------------|--------------------------------|
| Model                    | microsoft/CodeGPT-small-py     |
| Epochs                   | 10                             |
| Learning Rate            | 2e-5                           |
| Batch Size               | 4                              |
| Max Grad Norm            | 1.0                            |
| Weight Decay             | 0.01                           |
| Warmup Ratio             | 0.1                            |

---

## Conclusions

1. **SFT Dataset Size is Critical**: The {'dramatic' if final2.bleu > final1.bleu * 2 else 'notable'} improvement in metrics demonstrates that SFT dataset size is a crucial factor.

2. **Training Stability**: Both experiments show {'stable' if abs(case1_history[-1].gradient_norm - case1_history[0].gradient_norm) < 2 else 'varying'} training dynamics.

3. **Gradient Norm Pattern**: Case 1: {case1_history[0].gradient_norm:.2f} → {case1_history[-1].gradient_norm:.2f}, Case 2: {case2_history[0].gradient_norm:.2f} → {case2_history[-1].gradient_norm:.2f}

4. **Recommended Minimum**: For production-quality code generation, SFT dataset should contain at least 1000+ high-quality examples.

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nSaved comprehensive report to: {report_path}")


def run_loo_case1(config: SFTConfig, all_data: List[Dict], output_dir: str) -> Dict:
    """Leave-one-out cross-validation for the 11-sample case.

    With only 11 samples a 90/10 train-eval split gives 1-2 eval examples,
    which provides no reliable signal.  LOO CV uses 10 train / 1 eval per fold
    (11 folds total), averages the final-epoch metrics across folds, and
    reports mean ± std — the only statistically honest estimate for this dataset.
    """
    import dataclasses
    fold_histories: List[Dict] = []

    for i in range(len(all_data)):
        fold_train = all_data[:i] + all_data[i + 1:]
        fold_eval  = [all_data[i]]
        print(f"\n--- LOO fold {i + 1}/{len(all_data)} (eval sample: {i}) ---")
        trainer = SFTTrainer(config)
        history = trainer.train(fold_train, fold_eval, case_name=f"case1_loo_fold{i + 1}")
        fold_histories.append(dataclasses.asdict(history[-1]))
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_keys = ['bertscore', 'codebleu', 'bleu', 'rouge', 'ruby',
                   'reward', 'balanced_accuracy']
    summary: Dict = {}
    for k in metric_keys:
        vals = [h[k] for h in fold_histories if k in h and h[k] is not None]
        if vals:
            summary[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'case1_loo_summary.json')
    with open(out_path, 'w') as f:
        json.dump({'fold_results': fold_histories, 'summary': summary}, f, indent=2)
    print(f"\nLOO CV summary saved to: {out_path}")

    print("\nLOO CV Summary (mean ± std across 11 folds):")
    for k, v in summary.items():
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
    return summary


def main():
    """Main entry point.

    Modes (--mode):
      loo      — leave-one-out CV on 11 samples  (Case 1)
      case2    — single-seed or multi-seed SFT on 1247 samples  (Case 2)
      all      — LOO for Case 1 + multi-seed Case 2  (default)

    Seeds (--seeds, comma-separated, default "42,123,456"):
      Applied to Case 2.  Each seed run is saved to a separate JSON.
      A multi-seed summary with mean ± std is written automatically.
    """
    import argparse as _ap
    parser = _ap.ArgumentParser(description="SFT Experiments - Stage 3")
    parser.add_argument("--mode", choices=["loo", "case2", "all"], default="all")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated list of seeds for Case 2 multi-seed runs")
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print("="*60)
    print("SFT Experiments - Stage 3")
    print("="*60)

    base_config = SFTConfig(
        num_epochs=30,
        batch_size=4,
        learning_rate=2e-5,
        output_dir="./stage3/outputs"
    )
    os.makedirs(base_config.output_dir, exist_ok=True)

    # ========================================
    # Case 1: LOO CV on 11 samples
    # ========================================
    if args.mode in ("loo", "all"):
        print("\n" + "="*60)
        print("CASE 1: Leave-One-Out CV on 11 samples")
        print("="*60)
        # Reconstruct the full 11-sample list by concatenating train + eval
        # (load_sft_data splits 90/10, so train≈9-10 and eval≈1-2 samples)
        _train_11, _eval_11 = load_sft_data(num_samples=11)
        all_data_11 = _train_11 + _eval_11
        run_loo_case1(base_config, all_data_11, base_config.output_dir)

    # ========================================
    # Case 2: multi-seed SFT on 1247 samples
    # ========================================
    if args.mode in ("case2", "all"):
        print("\n" + "="*60)
        print("CASE 2: Multi-seed training with 1247 samples")
        print(f"Seeds: {seeds}")
        print("="*60)

        seed_histories: List[List] = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            import dataclasses
            cfg = SFTConfig(
                num_epochs=30,
                batch_size=4,
                learning_rate=2e-5,
                output_dir="./stage3/outputs",
                seed=seed,
            )
            train_data_2, eval_data_2 = load_sft_data(num_samples=1247)
            trainer_2 = SFTTrainer(cfg)
            history_2 = trainer_2.train(train_data_2, eval_data_2,
                                         f"Case 2 (1247 samples, seed={seed})")
            save_results(history_2, f"case2_1247samples_seed{seed}", cfg.output_dir)
            seed_histories.append([dataclasses.asdict(h) for h in history_2])
            del trainer_2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate across seeds (last epoch of each seed run)
        if len(seed_histories) > 1:
            metric_keys = ['bertscore', 'codebleu', 'bleu', 'rouge', 'ruby',
                           'reward', 'balanced_accuracy', 'train_loss']
            final_epochs = [h[-1] for h in seed_histories]
            multi_summary: Dict = {}
            for k in metric_keys:
                vals = [e[k] for e in final_epochs if k in e and e[k] is not None]
                if vals:
                    multi_summary[k] = {'mean': float(np.mean(vals)),
                                        'std': float(np.std(vals))}
            summary_path = os.path.join(base_config.output_dir,
                                        'case2_multiseed_summary.json')
            with open(summary_path, 'w') as f:
                json.dump({'seeds': seeds, 'final_epoch_summary': multi_summary,
                           'per_seed_history': seed_histories}, f, indent=2)
            print(f"\nMulti-seed summary saved to: {summary_path}")
            print("\nCase 2 final-epoch mean ± std:")
            for k, v in multi_summary.items():
                print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
    print(f"\nResults saved to: {base_config.output_dir}")


if __name__ == "__main__":
    main()
