"""
Stage 2B - Training Pipeline
============================

Training pipeline for Stage 2B: GPT-2 feedback-driven reward model.
"""

import os
import sys
import json
import logging
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
import pandas as pd

from .config import Stage2BConfig, get_stage2b_config
from .feedback_generator import GPT2FeedbackGenerator, FeedbackLoss
from .reward_model import FeedbackRewardModel, RewardModelLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _metric_is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return True
    return False


def _fmt_metric_val(x: Any) -> str:
    if _metric_is_missing(x):
        return "n/a"
    return f"{float(x):.4f}"


def _avg_bertscore_codebleu(
    bertscore: Any,
    codebleu: Any,
) -> float:
    """Mean of BERTScore and CodeBLEU using only defined, finite values."""
    vals = [float(v) for v in (bertscore, codebleu) if not _metric_is_missing(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


class Stage2BDataset(Dataset):
    """Dataset for Stage 2B training."""
    
    def __init__(self, questions: List[str], answers: List[str]):
        self.questions = questions
        self.answers = answers
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return {
            'question': self.questions[idx],
            'answer': self.answers[idx]
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    return questions, answers


class Stage2BMetricsEvaluator:
    """Metrics evaluator for Stage 2B."""
    
    def __init__(self):
        self.metrics = {}
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> float:
        """Compute BERTScore."""
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
            return float(F1.mean())
        except:
            return self._fallback_similarity(predictions, references)
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothing = SmoothingFunction().method1
            scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]
                if pred_tokens and ref_tokens[0]:
                    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
                    scores.append(score)
            return np.mean(scores) if scores else 0.0
        except:
            return 0.5
    
    def compute_codebleu(self, predictions: List[str], references: List[str]) -> float:
        """Compute CodeBLEU (simplified)."""
        bleu = self.compute_bleu(predictions, references)
        syntax_score = self._compute_syntax_score(predictions)
        return 0.6 * bleu + 0.4 * syntax_score
    
    def _compute_syntax_score(self, codes: List[str]) -> float:
        """Check syntax validity."""
        valid = 0
        for code in codes:
            try:
                compile(code, '<string>', 'exec')
                valid += 1
            except:
                try:
                    compile(code, '<string>', 'eval')
                    valid += 1
                except:
                    pass
        return valid / len(codes) if codes else 0.0
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> float:
        """Compute ROUGE-L."""
        def lcs_length(x: str, y: str) -> int:
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        scores = []
        for pred, ref in zip(predictions, references):
            lcs = lcs_length(pred, ref)
            if len(pred) > 0 and len(ref) > 0:
                precision = lcs / len(pred)
                recall = lcs / len(ref)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    scores.append(f1)
        return np.mean(scores) if scores else 0.0
    
    def compute_ruby(self, predictions: List[str], references: List[str]) -> float:
        """Compute RUBY (simplified)."""
        return (self.compute_codebleu(predictions, references) + 
                self.compute_rouge(predictions, references)) / 2
    
    def _fallback_similarity(self, predictions: List[str], references: List[str]) -> float:
        """Fallback similarity metric."""
        similarities = []
        for pred, ref in zip(predictions, references):
            pred_set = set(pred.lower().split())
            ref_set = set(ref.lower().split())
            if pred_set or ref_set:
                intersection = len(pred_set & ref_set)
                union = len(pred_set | ref_set)
                similarities.append(intersection / union if union > 0 else 0)
        return np.mean(similarities) if similarities else 0.0
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Compute all metrics."""
        return {
            'bertscore': self.compute_bertscore(predictions, references),
            'bleu': self.compute_bleu(predictions, references),
            'codebleu': self.compute_codebleu(predictions, references),
            'rouge': self.compute_rouge(predictions, references),
            'ruby': self.compute_ruby(predictions, references),
        }


class Stage2BExperiment:
    """Stage 2B Experiment runner."""
    
    def __init__(self, config: Stage2BConfig):
        self.config = config
        self.device = config.hardware.device
        
        # Create output directories
        os.makedirs(config.data.output_path, exist_ok=True)
        os.makedirs(config.data.artifacts_path, exist_ok=True)
        
        # Setup logging
        log_path = os.path.join(config.data.output_path, 'stage2b_experiment.log')
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Initialize components
        self.feedback_generator = None
        self.reward_model = None
        self.metrics_evaluator = Stage2BMetricsEvaluator()
        
        # Training state
        self.epoch_metrics = []
        self.best_score = 0.0
    
    def load_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Load CoNaLa corpus data."""
        logger.info("Loading data...")
        
        questions = []
        answers = []
        
        # Load CoNaLa
        conala_path = self.config.data.conala_path
        if os.path.exists(conala_path):
            with open(conala_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        intent = item.get('rewritten_intent') or item.get('intent', '')
                        snippet = item.get('snippet', '')
                        if intent and snippet:
                            questions.append(intent)
                            answers.append(snippet)
                    except:
                        continue
            logger.info(f"Loaded {len(questions)} samples from CoNaLa")
        
        # Load additional SFT data if available
        sft_path = self.config.data.sft_data_path
        if os.path.exists(sft_path):
            try:
                df = pd.read_csv(sft_path)
                for _, row in df.iterrows():
                    q = row.get('question', '')
                    a = row.get('best_answer', '')
                    if q and a:
                        questions.append(str(q))
                        answers.append(str(a))
                logger.info(f"Added samples from SFT data")
            except:
                pass
        
        # Limit samples
        max_samples = self.config.data.max_train_samples
        if len(questions) > max_samples:
            indices = np.random.choice(len(questions), max_samples, replace=False)
            questions = [questions[i] for i in indices]
            answers = [answers[i] for i in indices]
        
        # Split into train/val
        val_size = int(len(questions) * self.config.data.val_ratio)
        train_questions = questions[val_size:]
        train_answers = answers[val_size:]
        val_questions = questions[:val_size]
        val_answers = answers[:val_size]
        
        logger.info(f"Train: {len(train_questions)}, Val: {len(val_questions)}")
        
        return train_questions, train_answers, val_questions, val_answers
    
    def initialize_models(self):
        """Initialize feedback generator and reward model."""
        logger.info("Initializing models...")
        
        # GPT-2 Feedback Generator
        self.feedback_generator = GPT2FeedbackGenerator(
            model_name=self.config.feedback_generator.model_name,
            device=self.device,
            max_length=self.config.feedback_generator.max_length,
            temperature=self.config.feedback_generator.temperature,
        )
        
        # Reward Model
        self.reward_model = FeedbackRewardModel(
            model_name='codebert',
            hidden_dim=self.config.reward_model.hidden_dim,
            dropout=self.config.reward_model.dropout,
            freeze_encoder_layers=self.config.reward_model.freeze_encoder_layers,
            max_length=self.config.reward_model.max_length,
            device=self.device,
        )
        
        logger.info("Models initialized")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer,
        scheduler,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.reward_model.train()

        total_loss = 0.0
        num_batches = 0
        accuracy_scores: Dict[str, List[float]] = {
            'consistency': [], 'agreement': [], 'usefulness': []
        }

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (questions, answers) in enumerate(pbar):
            # Generate feedback scores using GPT-2
            with torch.no_grad():
                feedback_scores = self.feedback_generator(questions, answers)

            # Forward pass through reward model
            predictions = self.reward_model(questions, answers)

            # Compute loss and per-criterion accuracy
            loss = 0.0
            for criterion in ['consistency', 'agreement', 'usefulness']:
                if criterion in predictions and criterion in feedback_scores:
                    pred = predictions[criterion]
                    target = feedback_scores[criterion].to(self.device)
                    criterion_loss = F.mse_loss(pred, target)
                    loss += criterion_loss
                    # Binary accuracy: pred and target on same side of 0.5 threshold
                    pred_binary = (pred.detach() >= 0.5).float()
                    target_binary = (target >= 0.5).float()
                    acc = (pred_binary == target_binary).float().mean().item()
                    accuracy_scores[criterion].append(acc)

            # Backward pass
            loss = loss / self.config.training.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.reward_model.parameters(),
                    self.config.training.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * self.config.training.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{total_loss / num_batches:.4f}',
            })

        mean_accs = [np.mean(v) for v in accuracy_scores.values() if v]
        return {
            'train_loss': total_loss / num_batches,
            'train_accuracy_consistency': float(np.mean(accuracy_scores['consistency'])) if accuracy_scores['consistency'] else 0.0,
            'train_accuracy_agreement': float(np.mean(accuracy_scores['agreement'])) if accuracy_scores['agreement'] else 0.0,
            'train_accuracy_usefulness': float(np.mean(accuracy_scores['usefulness'])) if accuracy_scores['usefulness'] else 0.0,
            'train_accuracy_mean': float(np.mean(mean_accs)) if mean_accs else 0.0,
        }
    
    def evaluate(
        self,
        val_questions: List[str],
        val_answers: List[str]
    ) -> Dict[str, float]:
        """Evaluate the model."""
        self.reward_model.eval()

        with torch.no_grad():
            predictions = self.reward_model.predict_reward(val_questions, val_answers)
            # Use GPT-2 feedback as ground-truth labels for accuracy
            feedback_scores = self.feedback_generator(val_questions, val_answers)

        # Stage 2B is a reward model (discriminative), not a generator.
        # Text metrics (BERTScore, ROUGE etc.) require generated text vs references —
        # they are not meaningful here and the previous call compared val_answers to
        # itself (self-comparison), producing fraudulent BERTScore=ROUGE=1.0.
        _nan = float("nan")
        metrics = {
            'bertscore': _nan,
            'rouge': _nan,
            'bleu': _nan,
            'ruby': _nan,
            'codebleu': _nan,
        }

        # Average reward scores
        metrics['avg_consistency'] = float(np.mean(predictions['consistency']))
        metrics['avg_agreement'] = float(np.mean(predictions['agreement']))
        metrics['avg_usefulness'] = float(np.mean(predictions['usefulness']))
        metrics['avg_quality'] = float(np.mean(predictions['quality']))

        # Per-criterion validation accuracy vs GPT-2 feedback ground truth
        val_accs = []
        for criterion in ['consistency', 'agreement', 'usefulness']:
            if criterion in predictions and criterion in feedback_scores:
                pred_arr = np.array(predictions[criterion])
                target_arr = feedback_scores[criterion].cpu().numpy()
                pred_binary = (pred_arr >= 0.5).astype(int)
                target_binary = (target_arr >= 0.5).astype(int)
                acc = float(np.mean(pred_binary == target_binary))
                metrics[f'val_accuracy_{criterion}'] = acc
                val_accs.append(acc)
        if val_accs:
            metrics['val_accuracy_mean'] = float(np.mean(val_accs))

        return metrics
    
    def run(self):
        """Run the full experiment."""
        start_time = time.time()
        
        print("=" * 70)
        print("STAGE 2B - GPT-2 FEEDBACK GENERATOR FOR REWARD MODEL")
        print("=" * 70)
        
        # Load data
        train_questions, train_answers, val_questions, val_answers = self.load_data()
        
        # Initialize models
        self.initialize_models()
        
        # Create dataset and dataloader
        train_dataset = Stage2BDataset(train_questions, train_answers)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.hardware.num_workers
        )
        
        # Optimizer and scheduler
        optimizer = AdamW(
            self.reward_model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.training.total_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.training.warmup_ratio
        )
        
        # Initial evaluation
        print("\nInitial evaluation (epoch 0)...")
        metrics = self.evaluate(val_questions[:200], val_answers[:200])
        self.epoch_metrics.append({'epoch': 0, **metrics})
        logger.info(f"Epoch 0 Metrics: {metrics}")
        
        print(f"\nEpoch 0 Metrics:")
        print(f"  BERTScore: {_fmt_metric_val(metrics['bertscore'])}")
        print(f"  CodeBLEU:  {_fmt_metric_val(metrics['codebleu'])}")
        
        # Training loop
        for epoch in range(1, self.config.training.total_epochs + 1):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{self.config.training.total_epochs}")
            print(f"{'=' * 50}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler, epoch)
            
            # Evaluate
            metrics = self.evaluate(val_questions[:200], val_answers[:200])
            metrics.update(train_metrics)
            metrics['epoch'] = epoch
            self.epoch_metrics.append(metrics)
            
            logger.info(f"Epoch {epoch} Metrics: {metrics}")
            
            print(f"\nEpoch {epoch} Metrics:")
            print(f"  Train Loss:      {train_metrics['train_loss']:.4f}")
            print(f"  Train Acc Mean:  {train_metrics.get('train_accuracy_mean', 0.0):.4f}")
            print(f"  Val Acc Mean:    {metrics.get('val_accuracy_mean', 0.0):.4f}")
            print(f"  BERTScore:       {_fmt_metric_val(metrics['bertscore'])}")
            print(f"  CodeBLEU:        {_fmt_metric_val(metrics['codebleu'])}")
            print(f"  ROUGE:           {_fmt_metric_val(metrics['rouge'])}")
            print(f"  RUBY:            {_fmt_metric_val(metrics['ruby'])}")
            
            # Save best model (text metrics are n/a for Stage 2B — fall back to val accuracy)
            avg_score = _avg_bertscore_codebleu(metrics['bertscore'], metrics['codebleu'])
            if math.isnan(avg_score):
                avg_score = float(metrics.get('val_accuracy_mean', 0.0) or 0.0)
            if avg_score > self.best_score:
                self.best_score = avg_score
                checkpoint_path = os.path.join(
                    self.config.data.artifacts_path,
                    f'checkpoint_epoch_{epoch}'
                )
                self.reward_model.save_pretrained(checkpoint_path)
                print(f"  [*] New best model (avg score: {avg_score:.4f})")
        
        # Save final results
        self.save_results()
        
        total_time = (time.time() - start_time) / 60
        
        print("\n" + "=" * 70)
        print("EXPERIMENT COMPLETED")
        print("=" * 70)
        print(f"\nFinal Metrics (Epoch {self.config.training.total_epochs}):")
        final_metrics = self.epoch_metrics[-1]
        print(f"  BERTScore: {_fmt_metric_val(final_metrics['bertscore'])}")
        print(f"  BLEU:      {_fmt_metric_val(final_metrics['bleu'])}")
        print(f"  CodeBLEU:  {_fmt_metric_val(final_metrics['codebleu'])}")
        print(f"  ROUGE:     {_fmt_metric_val(final_metrics['rouge'])}")
        print(f"  RUBY:      {_fmt_metric_val(final_metrics['ruby'])}")
        print(f"\nTotal training time: {total_time:.2f} minutes")
        print("=" * 70)
    
    def save_results(self):
        """Save experiment results."""

        def _json_sanitize(obj: Any) -> Any:
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: _json_sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_sanitize(v) for v in obj]
            return obj

        results = {
            'config': self.config.to_dict(),
            'epoch_metrics': _json_sanitize(self.epoch_metrics),
            'best_score': self.best_score,
        }
        
        results_path = os.path.join(self.config.data.output_path, 'stage2b_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics CSV
        metrics_df = pd.DataFrame(self.epoch_metrics)
        csv_path = os.path.join(self.config.data.output_path, 'stage2b_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {self.config.data.output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 2B: GPT-2 Feedback Reward Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=2000, help='Max training samples')
    
    args = parser.parse_args()
    
    # Create config
    config = get_stage2b_config()
    config.training.total_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.data.max_train_samples = args.max_samples
    
    # Run experiment
    experiment = Stage2BExperiment(config)
    experiment.run()


if __name__ == '__main__':
    main()
