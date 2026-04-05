"""
Stage 2A - Contrastive Learning Training
========================================

Main training script for Stage 2A with:
- Maximum contrastive loss for pair-wise learning
- Enhanced reward model with contrastive projections
- Per-epoch metrics: BERTScore, BLEU, CodeBLEU, ROUGE, RUBY
- 20 epochs training
"""

import os
import sys
import json
import time
import logging
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE = None

from stage2.stage2A.config import Stage2AConfig, get_stage2a_config
from stage2.stage2A.contrastive_model import ContrastiveRewardModel
from stage2.stage2A.data_loader import Stage2ADataLoader
from stage2.stage2A.metrics import Stage2AMetricsEvaluator

logger = logging.getLogger(__name__)


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class Stage2ADataset(Dataset):
    """Dataset for Stage 2A training."""
    
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            'prompt': sample.get('prompt', ''),
            'response': sample.get('response', ''),
            'reference': sample.get('reference', sample.get('response', '')),
            'quality': float(sample.get('quality', 0.5)),
            'source': sample.get('source', 'unknown')
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    return {
        'prompts': [item['prompt'] for item in batch],
        'responses': [item['response'] for item in batch],
        'references': [item['reference'] for item in batch],
        'qualities': torch.tensor([item['quality'] for item in batch], dtype=torch.float32),
        'sources': [item['source'] for item in batch]
    }


class Stage2AExperiment:
    """
    Stage 2A Contrastive Learning Experiment.
    
    Implements contrastive learning with maximum contrastive loss
    for enhanced reward model discriminative capacity.
    """
    
    def __init__(self, config: Optional[Stage2AConfig] = None):
        self.config = config or get_stage2a_config()
        self.device = torch.device(self.config.hardware.device)
        
        # Set random seeds
        self._set_seeds(self.config.seed)
        
        # Initialize components
        self.data_loader = Stage2ADataLoader(
            sft_path=self.config.data.train_data_path,
            extra_jsonl_paths=self.config.data.extra_sft_jsonl_paths,
            max_samples=self.config.data.max_train_samples,
            negative_ratio=self.config.data.negative_ratio,
            seed=self.config.seed,
            eval_dir=self.config.data.eval_data_path,
            feedback_dir=self.config.data.feedback_data_path,
        )
        
        # Metrics evaluator
        self.metrics_evaluator = Stage2AMetricsEvaluator()

        # Policy model for per-epoch text quality inference.
        # Stage 2A trains only the reward model; the policy is loaded for
        # inference so text metrics reflect generated output quality rather
        # than the fixed validation set responses.
        self._policy_model = None
        self._policy_tokenizer = None
        self._load_policy_model()

        # Results storage
        self.epoch_results = []
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Stage 2A Experiment initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Contrastive margin: {self.config.contrastive.margin}")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.config.data.output_path
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_dir, 'stage2a_experiment.log'))
            ]
        )
    
    def _load_policy_model(self):
        """Load the policy model (CodeGPT) for per-epoch inference."""
        policy_name = self.config.model.policy_model_name
        if not policy_name:
            return
        try:
            logger.info(f"Loading policy model for inference: {policy_name}")
            self._policy_tokenizer = AutoTokenizer.from_pretrained(
                policy_name, use_fast=True
            )
            if self._policy_tokenizer.pad_token is None:
                self._policy_tokenizer.pad_token = self._policy_tokenizer.eos_token
                self._policy_tokenizer.pad_token_id = self._policy_tokenizer.eos_token_id
            self._policy_model = AutoModelForCausalLM.from_pretrained(
                policy_name, torch_dtype=torch.float32
            ).to(self.device)
            self._policy_model.eval()
            logger.info(
                "Policy model loaded (inference only — not updated during Stage 2A). "
                "Text metrics will be constant across epochs because the policy is frozen."
            )
        except Exception as e:
            logger.warning(
                f"Could not load policy model '{policy_name}': {e}. "
                "Text metrics will fall back to stored val responses."
            )

    def _generate_policy_responses(
        self, prompts: List[str], max_new_tokens: int = 64
    ) -> List[str]:
        """Generate responses from the (frozen) policy model for each prompt.

        Since Stage 2A does not update the policy, these outputs are identical
        across epochs and represent the pre-trained baseline quality.
        """
        if self._policy_model is None:
            # Fallback: return the stored prompts as a proxy — same as before.
            return list(prompts)

        responses: List[str] = []
        for prompt in prompts:
            try:
                enc = self._policy_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.model.max_length,
                    padding=True,
                ).to(self.device)
                with torch.no_grad():
                    out_ids = self._policy_model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self._policy_tokenizer.pad_token_id,
                        eos_token_id=self._policy_tokenizer.eos_token_id,
                    )
                # Decode only the newly generated tokens.
                new_ids = out_ids[0][enc["input_ids"].shape[1]:]
                text = self._policy_tokenizer.decode(new_ids, skip_special_tokens=True)
                responses.append(text.strip() or "<empty>")
            except Exception as e:
                logger.debug(f"Policy generation failed: {e}")
                responses.append("<generation_error>")
        return responses

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare training data."""
        logger.info("Loading data...")
        
        # Load training and validation samples
        train_samples, val_samples = self.data_loader.load_training_data(
            val_ratio=self.config.data.val_ratio
        )
        
        logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
        
        # Create datasets
        train_dataset = Stage2ADataset(train_samples)
        val_dataset = Stage2ADataset(val_samples)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.hardware.num_workers,
            pin_memory=self.config.hardware.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.evaluation.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def initialize_model(self) -> ContrastiveRewardModel:
        """Initialize the contrastive reward model."""
        logger.info("Initializing contrastive reward model...")
        
        model = ContrastiveRewardModel(
            model_name=self.config.model.base_model_name,
            projection_dim=self.config.model.projection_dim,
            margin=self.config.contrastive.margin,
            temperature=self.config.contrastive.temperature,
            dropout=self.config.model.dropout,
            freeze_encoder_layers=self.config.model.freeze_encoder_layers,
            max_length=self.config.model.max_length,
            device=str(self.device)
        )
        
        return model
    
    def train_epoch(
        self,
        model: ContrastiveRewardModel,
        train_loader: DataLoader,
        optimizer,
        scheduler,
        scaler,
        ema: Optional[EMA],
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_reward_loss = 0.0
        total_samples = 0
        
        grad_accum = self.config.training.gradient_accumulation_steps
        use_amp = self.config.training.use_amp and self.device.type == 'cuda'
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            prompts = batch['prompts']
            responses = batch['responses']
            qualities = batch['qualities'].to(self.device)
            
            # Forward pass with AMP
            with autocast(AMP_DEVICE or 'cuda', enabled=use_amp):
                outputs = model(prompts, responses, return_projections=True)
                
                # Reward loss (BCE with quality labels)
                # Use quality directly as soft label for better signal
                reward_loss = 0.0
                for head in ['consistent', 'correct', 'useful']:
                    head_logits = outputs[head]
                    # Use quality as soft target (0.0 for negative, 1.0 for positive)
                    # This provides cleaner supervision than hard threshold
                    target = qualities.clone()
                    head_loss = F.binary_cross_entropy_with_logits(head_logits, target)
                    reward_loss += head_loss
                reward_loss = reward_loss / 3
                
                # Contrastive loss
                # Create pairs within batch: high quality vs low quality
                prompt_proj = outputs['prompt_projection']
                response_proj = outputs['response_projection']
                
                # Maximum contrastive loss
                # Similar: both high or both low quality
                # Dissimilar: one high, one low
                batch_size = qualities.size(0)
                
                if batch_size > 1:
                    # Create all pairwise combinations
                    pair_labels = []
                    emb1_list = []
                    emb2_list = []
                    
                    for i in range(batch_size):
                        for j in range(i + 1, batch_size):
                            emb1_list.append(response_proj[i])
                            emb2_list.append(response_proj[j])
                            
                            # Similar (0) if both above or both below 0.5
                            # Dissimilar (1) if one above and one below
                            q_i = qualities[i].item()
                            q_j = qualities[j].item()
                            
                            same_quality = (q_i >= 0.5) == (q_j >= 0.5)
                            pair_labels.append(0 if same_quality else 1)
                    
                    if emb1_list:
                        emb1 = torch.stack(emb1_list)
                        emb2 = torch.stack(emb2_list)
                        pair_labels_tensor = torch.tensor(pair_labels, dtype=torch.float32, device=self.device)
                        
                        contrastive_losses = model.compute_contrastive_loss(
                            emb1, emb2, pair_labels_tensor, use_infonce=True
                        )
                        contrastive_loss = contrastive_losses['total_contrastive']
                    else:
                        contrastive_loss = torch.tensor(0.0, device=self.device)
                else:
                    contrastive_loss = torch.tensor(0.0, device=self.device)
                
                # Combined loss
                loss = (
                    self.config.contrastive.reward_weight * reward_loss +
                    self.config.contrastive.contrastive_weight * contrastive_loss
                )
                loss = loss / grad_accum
            
            # Backward pass
            if use_amp and scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % grad_accum == 0:
                if use_amp and scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                if ema:
                    ema.update()
            
            # Track metrics
            total_loss += loss.item() * grad_accum * len(prompts)
            total_contrastive_loss += contrastive_loss.item() * len(prompts)
            total_reward_loss += reward_loss.item() * len(prompts)
            total_samples += len(prompts)
            
            pbar.set_postfix({
                'loss': f'{loss.item() * grad_accum:.4f}',
                'contrast': f'{contrastive_loss.item():.4f}',
                'reward': f'{reward_loss.item():.4f}'
            })
            
            # Memory cleanup
            del outputs, loss, contrastive_loss, reward_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return {
            'loss': total_loss / total_samples,
            'contrastive_loss': total_contrastive_loss / total_samples,
            'reward_loss': total_reward_loss / total_samples
        }
    
    def evaluate_epoch(
        self,
        model: ContrastiveRewardModel,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Evaluate model and compute all metrics."""
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        total_correct = {'consistent': 0, 'correct': 0, 'useful': 0}
        total_counts = {'consistent': 0, 'correct': 0, 'useful': 0}
        pos_scores = []
        neg_scores = []
        all_responses = []
        all_references = []
        
        use_amp = self.config.training.use_amp and self.device.type == 'cuda'
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                prompts = batch['prompts']
                responses = batch['responses']
                qualities = batch['qualities'].to(self.device)
                
                with autocast(AMP_DEVICE or 'cuda', enabled=use_amp):
                    outputs = model(prompts, responses, return_projections=False)
                    
                    # Compute validation loss
                    loss = 0.0
                    for head in ['consistent', 'correct', 'useful']:
                        target = (qualities >= 0.5).float()
                        head_loss = F.binary_cross_entropy_with_logits(outputs[head], target)
                        loss += head_loss
                    loss = loss / 3
                
                total_loss += loss.item() * len(prompts)
                total_samples += len(prompts)
                
                # Accuracy per head
                with torch.no_grad():
                    targets = (qualities >= 0.5).float()
                    for head in ['consistent', 'correct', 'useful']:
                        probs = torch.sigmoid(outputs[head])
                        preds = (probs >= 0.5).float()
                        total_correct[head] += (preds == targets).sum().item()
                        total_counts[head] += targets.numel()
                
                    # Generate policy outputs for text metric evaluation.
                    # The policy is not updated in Stage 2A, so these values
                    # are constant across epochs (pre-trained baseline quality).
                    generated = self._generate_policy_responses(prompts)
                    all_responses.extend(generated)
                    all_references.extend(batch['references'])

                    # Track mean reward score by class
                    mean_reward = (
                        torch.sigmoid(outputs['consistent']) +
                        torch.sigmoid(outputs['correct']) +
                        torch.sigmoid(outputs['useful'])
                    ) / 3.0
                    pos_mask = targets >= 0.5
                    neg_mask = targets < 0.5
                    if pos_mask.any():
                        pos_scores.append(mean_reward[pos_mask].mean().item())
                    if neg_mask.any():
                        neg_scores.append(mean_reward[neg_mask].mean().item())
        
        # Compute embedding-based metrics
        embedding_metrics = {}
        if all_responses and all_references:
            try:
                raw = self.metrics_evaluator.compute_all_metrics(all_responses, all_references)
                summary = self.metrics_evaluator.get_summary(raw)
                embedding_metrics = {
                    'bertscore': summary.get('bertscore', 0.0),
                    'bleu': summary.get('bleu', 0.0),
                    'codebleu': summary.get('codebleu', 0.0),
                    'rouge': summary.get('rouge', 0.0),
                    'ruby': summary.get('ruby', 0.0),
                }
            except Exception as e:
                logger.warning(f"Embedding metrics failed: {e}")
                embedding_metrics = {'bertscore': 0.0, 'bleu': 0.0, 'codebleu': 0.0, 'rouge': 0.0, 'ruby': 0.0}

        # Build epoch metrics
        epoch_metrics = {
            'epoch': epoch,
            'val_loss': total_loss / total_samples,
            'reward_acc_consistent': total_correct['consistent'] / max(1, total_counts['consistent']),
            'reward_acc_correct': total_correct['correct'] / max(1, total_counts['correct']),
            'reward_acc_useful': total_correct['useful'] / max(1, total_counts['useful']),
        }
        epoch_metrics['reward_acc_mean'] = (
            epoch_metrics['reward_acc_consistent'] +
            epoch_metrics['reward_acc_correct'] +
            epoch_metrics['reward_acc_useful']
        ) / 3.0
        epoch_metrics.update(embedding_metrics)
        if pos_scores:
            epoch_metrics['reward_pos_mean'] = float(np.mean(pos_scores))
        if neg_scores:
            epoch_metrics['reward_neg_mean'] = float(np.mean(neg_scores))
            if pos_scores:
                epoch_metrics['reward_gap'] = float(np.mean(pos_scores) - np.mean(neg_scores))
        
        logger.info(f"Epoch {epoch} Metrics:")
        logger.info(f"  Val Loss:     {epoch_metrics['val_loss']:.4f}")
        logger.info(f"  Reward Acc:   {epoch_metrics['reward_acc_mean']:.4f}")
        if 'reward_gap' in epoch_metrics:
            logger.info(f"  Reward Gap:   {epoch_metrics['reward_gap']:.4f}")
        
        return epoch_metrics
    
    def run(self) -> Dict[str, Any]:
        """Run the full Stage 2A experiment for 20 epochs."""
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("STAGE 2A - CONTRASTIVE LEARNING FOR ENHANCED REWARD MODEL")
        print("=" * 70)
        print(f"Base Model: {self.config.model.base_model_name}")
        print(f"Contrastive Margin: {self.config.contrastive.margin}")
        print(f"Training Epochs: {self.config.training.total_epochs}")
        print(f"Device: {self.device}")
        print("=" * 70)
        
        try:
            # Load data
            train_loader, val_loader = self.load_data()
            
            # Initialize model
            model = self.initialize_model()
            
            # Setup optimizer with separate learning rates for encoder vs heads
            # Encoder uses lower LR (fine-tuning), heads use higher LR (training from scratch)
            encoder_params = []
            head_params = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'encoder' in name or 'embeddings' in name:
                    encoder_params.append(param)
                else:
                    head_params.append(param)
            
            optimizer = AdamW([
                {'params': encoder_params, 'lr': self.config.training.learning_rate},
                {'params': head_params, 'lr': self.config.training.head_learning_rate}
            ], weight_decay=self.config.training.weight_decay)
            
            logger.info(f"Optimizer: encoder LR={self.config.training.learning_rate}, "
                       f"head LR={self.config.training.head_learning_rate}")
            
            # Setup scheduler
            num_training_steps = (
                len(train_loader) * self.config.training.total_epochs //
                self.config.training.gradient_accumulation_steps
            )
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.training.learning_rate * 2,
                total_steps=num_training_steps,
                pct_start=self.config.training.warmup_ratio,
                anneal_strategy='cos'
            )
            
            # Setup AMP scaler
            use_amp = self.config.training.use_amp and self.device.type == 'cuda'
            scaler = GradScaler(AMP_DEVICE) if use_amp else None
            
            # Setup EMA
            ema = EMA(model, decay=self.config.training.ema_decay) if self.config.training.use_ema else None
            
            # Initial evaluation (epoch 0)
            print("\nInitial evaluation (epoch 0)...")
            epoch_metrics = self.evaluate_epoch(model, val_loader, epoch=0)
            self.epoch_results.append(epoch_metrics)
            
            best_score = 0.0
            patience_counter = 0
            
            # Training loop for 20 epochs
            for epoch in range(1, self.config.training.total_epochs + 1):
                print(f"\n{'='*50}")
                print(f"Epoch {epoch}/{self.config.training.total_epochs}")
                print('='*50)
                
                # Train
                train_metrics = self.train_epoch(
                    model, train_loader, optimizer, scheduler, scaler, ema, epoch
                )
                self.training_history.append({
                    'epoch': epoch,
                    **train_metrics
                })
                
                # Evaluate with EMA if enabled
                if ema:
                    ema.apply_shadow()
                
                epoch_metrics = self.evaluate_epoch(model, val_loader, epoch)
                self.epoch_results.append(epoch_metrics)
                
                if ema:
                    ema.restore()
                
                # Check for improvement
                current_score = epoch_metrics['reward_acc_mean']
                
                if current_score > best_score:
                    best_score = current_score
                    patience_counter = 0
                    self._save_checkpoint(model, epoch)
                    print(f"  [*] New best model (avg score: {current_score:.4f})")
                else:
                    patience_counter += 1
                
                # Early stopping (disabled by default for 20 epochs)
                if self.config.training.patience and patience_counter >= self.config.training.patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
            
            # Save final results
            total_time = time.time() - start_time
            results = self._save_results(total_time)
            
            print("\n" + "=" * 70)
            print("EXPERIMENT COMPLETED")
            print("=" * 70)
            self._print_results_table()
            
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_checkpoint(self, model: ContrastiveRewardModel, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(
            self.config.data.artifacts_path,
            f"checkpoint_epoch_{epoch}"
        )
        model.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _save_results(self, total_time: float) -> Dict[str, Any]:
        """Save experiment results."""
        results = {
            'experiment_name': self.config.experiment_name,
            'config': self.config.to_dict(),
            'epoch_results': self.epoch_results,
            'training_history': self.training_history,
            'total_time_seconds': total_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON
        results_path = os.path.join(self.config.data.output_path, 'stage2a_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
        # Save metrics CSV
        self._save_metrics_csv()
        
        return results
    
    def _save_metrics_csv(self):
        """Save metrics as CSV table."""
        csv_path = os.path.join(self.config.data.output_path, 'stage2a_metrics.csv')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'reward_acc_mean',
                'reward_acc_consistent',
                'reward_acc_correct',
                'reward_acc_useful',
                'reward_pos_mean',
                'reward_neg_mean',
                'reward_gap',
                'val_loss',
                'bertscore',
                'bleu',
                'codebleu',
                'rouge',
                'ruby',
            ])

            for result in self.epoch_results:
                writer.writerow([
                    result['epoch'],
                    f"{result.get('reward_acc_mean', 0):.4f}",
                    f"{result.get('reward_acc_consistent', 0):.4f}",
                    f"{result.get('reward_acc_correct', 0):.4f}",
                    f"{result.get('reward_acc_useful', 0):.4f}",
                    f"{result.get('reward_pos_mean', 0):.4f}",
                    f"{result.get('reward_neg_mean', 0):.4f}",
                    f"{result.get('reward_gap', 0):.4f}",
                    f"{result.get('val_loss', 0):.4f}",
                    f"{result.get('bertscore', 0):.4f}",
                    f"{result.get('bleu', 0):.4f}",
                    f"{result.get('codebleu', 0):.4f}",
                    f"{result.get('rouge', 0):.4f}",
                    f"{result.get('ruby', 0):.4f}",
                ])
        
        logger.info(f"Metrics CSV saved to {csv_path}")
    
    def _print_results_table(self):
        """Print results table to console."""
        print("\nMetrics by Epoch:")
        print("-" * 80)
        print(f"{'Epoch':>6} {'RewardAcc':>10} {'Gap':>10} {'ValLoss':>10}")
        print("-" * 80)
        
        for result in self.epoch_results:
            print(f"{result['epoch']:>6} "
                  f"{result.get('reward_acc_mean', 0):>10.4f} "
                  f"{result.get('reward_gap', 0):>10.4f} "
                  f"{result.get('val_loss', 0):>10.4f}")
        
        print("-" * 80)


def main():
    """Run Stage 2A experiment."""
    config = get_stage2a_config()
    
    config.training.total_epochs = 30
    config.training.patience = None  # Disable early stopping for full 20 epochs
    
    experiment = Stage2AExperiment(config)
    results = experiment.run()
    
    print("\nExperiment completed successfully!")
    print(f"Results saved to: {config.data.output_path}")
    
    return results


if __name__ == "__main__":
    main()
