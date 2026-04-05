"""
ClassifLLM v3 - Enhanced LLM-based Code Quality Classification with Real LLM Feedback

This version integrates real LLM feedback generation for authentic human-like evaluation
of code quality, while maintaining all the technical enhancements from v2.

Key features:
- Real LLM API integration for feedback generation
- Multiple LLM providers support
- Confidence-weighted training
- Feedback caching and reuse
- All v2 enhancements: AMP, cross-attention, contrastive learning, etc.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import random
import re
import hashlib
import copy
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE = 'cuda'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE = None
from tqdm import tqdm

try:
    from .model import EnhancedLLMClassifier
except ImportError:
    from model import EnhancedLLMClassifier

try:
    from .llm_feedback import LLMFeedbackGenerator, LLMConfig, create_default_config
except ImportError:
    from llm_feedback import LLMFeedbackGenerator, LLMConfig, create_default_config

try:
    from .metrics import compute_code_quality_metrics
except ImportError:
    from metrics import compute_code_quality_metrics

# =============================================================================
# LABEL SMOOTHING LOSS
# =============================================================================

class LabelSmoothingBCELoss(nn.Module):
    """Binary Cross Entropy with Label Smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch] - raw logits
            targets: [batch] - binary targets (0 or 1)
        """
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets_smooth)


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss between query and positive pairs.
        Uses in-batch negatives.

        Args:
            query: [batch, dim] - normalized question embeddings
            positive: [batch, dim] - normalized answer embeddings
            labels: [batch] - quality labels to weight loss (optional)
        """
        batch_size = query.size(0)

        # Compute similarity matrix
        similarity = torch.mm(query, positive.t()) / self.temperature  # [batch, batch]

        # Diagonal elements are positive pairs
        targets = torch.arange(batch_size, device=query.device)

        # Cross entropy loss (diagonal should be highest)
        loss = F.cross_entropy(similarity, targets)

        # Weight by label quality if provided
        if labels is not None:
            # High quality pairs should contribute more
            weights = 0.5 + 0.5 * labels.mean()
            loss = loss * weights

        return loss


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

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
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# DATASET
# =============================================================================

class EnhancedCodeQualityDataset(Dataset):
    """Enhanced dataset with LLM feedback support."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        augment: bool = False,
        use_confidence_weighting: bool = True
    ):
        self.samples = samples
        self.augment = augment
        self.use_confidence_weighting = use_confidence_weighting

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

        question = sample['question']
        answer = sample['answer']

        # Simple augmentation - add noise to low-quality samples
        if self.augment and labels.get('useful', 0.5) < 0.3:
            # Slight perturbation for negative samples
            if random.random() < 0.2:
                answer = self._perturb_code(answer)

        result = {
            'question': question,
            'answer': answer,
            'labels': labels,
            'metadata': sample.get('metadata', {})
        }

        # Add confidence weighting if available
        if self.use_confidence_weighting and 'feedback' in sample:
            confidence = sample['feedback'].get('confidence', 1.0)
            result['confidence'] = confidence
            # Weight labels by confidence
            for key in labels:
                labels[key] = labels[key] * confidence + 0.5 * (1 - confidence)

        return result

    def _perturb_code(self, code: str) -> str:
        """Simple code perturbation for augmentation."""
        # Basic perturbations
        perturbations = [
            lambda x: x.replace(' ', '  '),  # Add extra spaces
            lambda x: x.replace('=', ' = '),  # Add spaces around equals
            lambda x: x.lower(),  # Convert to lowercase
        ]

        if random.random() < 0.5:
            perturb_func = random.choice(perturbations)
            try:
                return perturb_func(code)
            except:
                pass

        return code


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    return {
        'question': [item['question'] for item in batch],
        'answer': [item['answer'] for item in batch],
        'labels': [item['labels'] for item in batch],
        'metadata': [item['metadata'] for item in batch],
        'confidence': [item.get('confidence', 1.0) for item in batch]
    }


# =============================================================================
# DATA LOADING WITH LLM FEEDBACK
# =============================================================================

async def load_samples_with_llm_feedback(
    feedback_dir: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
    llm_config: Optional[LLMConfig] = None,
    force_regenerate: bool = False,
    max_samples: int = 1000
) -> List[Dict[str, Any]]:
    """
    Load samples and generate real LLM feedback.

    Args:
        feedback_dir: Directory with existing feedback files
        dataset_dir: Directory with evaluation datasets
        llm_config: LLM configuration for feedback generation
        force_regenerate: Whether to regenerate all feedback
        max_samples: Maximum number of samples to process

    Returns:
        Samples with real LLM feedback
    """

    # First try to load existing real datasets
    samples = []
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from stage4.dataset import load_real_datasets

        current_dir = Path.cwd()
        eval_dir = dataset_dir or (current_dir / "stage4" / "datasets_for_eval")
        feedback_dir_full = current_dir / "stage4" / str(feedback_dir).replace("clasifNN/", "").replace("stage4/", "")

        if eval_dir.exists() and feedback_dir_full.exists():
            real_samples = load_real_datasets(eval_dir, feedback_dir_full)
            if real_samples:
                samples = normalize_labels(real_samples)
                print(f"Loaded {len(samples)} existing real samples")
    except Exception as e:
        print(f"Error loading real datasets: {e}")

    # If no real samples or force regenerate, use synthetic base
    if not samples or force_regenerate:
        print("Generating base samples...")
        samples = generate_base_samples(max_samples)

    # Generate LLM feedback if config provided
    if llm_config:
        print(f"Generating real LLM feedback using {llm_config.provider}...")

        # Initialize feedback generator
        generator = LLMFeedbackGenerator(llm_config)

        # Load existing cache if available
        cache_path = Path("llm_feedback_cache.json")
        if cache_path.exists() and not force_regenerate:
            generator.load_cache(str(cache_path))

        # Limit samples for feedback generation
        feedback_samples = samples[:max_samples]

        # Generate feedback (async)
        import asyncio
        feedback_samples = await generator.generate_feedback_batch(
            feedback_samples,
            use_cache=True,
            show_progress=True
        )

        # Save cache
        generator.save_cache(str(cache_path))

        # Update samples
        samples = feedback_samples + samples[max_samples:]

        print(f"Generated LLM feedback for {len(feedback_samples)} samples")

    return samples


_LOG = logging.getLogger(__name__)
STAGE5C_ROOT = Path(__file__).resolve().parent


def _load_reference_pairs_file(path: Path) -> Dict[str, str]:
    """JSONL with keys question/prompt and reference/canonical_solution/reference_answer."""
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = (row.get("question") or row.get("prompt") or "").strip()
            r = (
                row.get("reference")
                or row.get("canonical_solution")
                or row.get("reference_answer")
                or row.get("code")
                or ""
            ).strip()
            if q:
                out[q] = r
    return out


def _try_load_public_benchmark_rows(n: int) -> List[Dict[str, Any]]:
    """Prefer HuggingFace ``openai_humaneval`` when ``datasets`` is installed."""
    rows: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("openai_humaneval", split="test")
        for row in ds:
            if len(rows) >= n:
                break
            ref = row["canonical_solution"]
            rows.append(
                {
                    "question": row["prompt"],
                    "answer": ref,
                    "metadata": {
                        "source": "humaneval",
                        "task_id": row["task_id"],
                        "reference_answer": ref,
                    },
                }
            )
    except Exception as e:
        _LOG.info("Benchmark HumanEval not available (%s); using built-in templates.", e)
    return rows


def generate_base_samples(n: int = 1000) -> List[Dict[str, Any]]:
    """Generate base samples without feedback for LLM evaluation."""

    bench = _try_load_public_benchmark_rows(n)
    if bench:
        _LOG.info("Using %d HumanEval samples from `datasets` (openai_humaneval).", len(bench))
        return bench[:n]

    samples = []

    # Code quality examples across different categories
    # Format: (question, canonical_answer, alternative_answers, difficulty)
    categories = {
        'data_structures': [
            ('How to create a list in Python?', 'my_list = []', 
             ['list()', 'new_list = list()', 'data = []'], 'basic'),
            ('How to create a dictionary?', 'my_dict = {}', 
             ['dict()', 'new_dict = dict()', 'data = {}'], 'basic'),
            ('How to iterate over a dictionary?', 'for key, value in my_dict.items():', 
             ['for k, v in d.items(): pass', 'for key in my_dict: val = my_dict[key]'], 'intermediate'),
            ('How to sort a list of tuples by second element?', 'sorted_list = sorted(my_list, key=lambda x: x[1])', 
             ['result = sorted(data, key=lambda t: t[1])', 'my_list.sort(key=lambda x: x[1])'], 'advanced'),
        ],
        'file_operations': [
            ('How to read a text file?', 'with open("file.txt", "r") as f: content = f.read()', 
             ['f = open("file.txt"); data = f.read(); f.close()', 'content = open("file.txt").read()'], 'basic'),
            ('How to write to a file?', 'with open("file.txt", "w") as f: f.write("content")', 
             ['f = open("file.txt", "w"); f.write("text"); f.close()', 'open("out.txt", "w").write("data")'], 'basic'),
            ('How to read JSON file?', 'import json; with open("file.json") as f: data = json.load(f)', 
             ['import json; data = json.loads(open("file.json").read())', 'from json import load; data = load(open("f.json"))'], 'intermediate'),
            ('How to handle file encoding?', 'with open("file.txt", encoding="utf-8") as f: content = f.read()', 
             ['open("file.txt", "r", encoding="utf-8").read()', 'codecs.open("file.txt", encoding="utf-8").read()'], 'advanced'),
        ],
        'error_handling': [
            ('How to catch exceptions?', 'try: code() except Exception as e: print(e)', 
             ['try: func() except: pass', 'try: run() except Exception: handle()'], 'basic'),
            ('How to handle specific exceptions?', 'try: code() except ValueError: handle_value_error()', 
             ['try: x = int(s) except ValueError: x = 0', 'try: parse() except (ValueError, TypeError): fallback()'], 'intermediate'),
            ('How to use finally block?', 'try: code() except: pass finally: cleanup()', 
             ['try: process() finally: close()', 'try: run() except Error: log() finally: done()'], 'intermediate'),
            ('How to create custom exceptions?', 'class MyError(Exception): pass', 
             ['class CustomError(Exception): def __init__(self, msg): super().__init__(msg)', 'class AppError(RuntimeError): pass'], 'advanced'),
        ],
        'algorithms': [
            ('How to find maximum in list?', 'max_value = max(my_list)', 
             ['result = max(data)', 'largest = max(numbers)'], 'basic'),
            ('How to implement binary search?', 'def binary_search(arr, target): left, right = 0, len(arr)-1', 
             ['def bsearch(a, x): l, r = 0, len(a); return bisect.bisect_left(a, x)', 'import bisect; idx = bisect.bisect_left(arr, val)'], 'advanced'),
            ('How to reverse a string?', 'reversed_string = my_string[::-1]', 
             ['result = "".join(reversed(s))', 'rev = s[::-1]'], 'basic'),
            ('How to check if string is palindrome?', 'is_palindrome = my_string == my_string[::-1]', 
             ['def is_pal(s): return s == s[::-1]', 'palindrome = s.lower() == s.lower()[::-1]'], 'intermediate'),
        ]
    }

    # Generate samples from categories
    samples_per_category = n // len(categories)

    for category, examples in categories.items():
        for question, canonical_answer, alternatives, difficulty in examples:
            # Create samples with canonical answer as reference
            for i in range(max(1, samples_per_category // len(examples))):
                # Use canonical answer as reference, alternate between canonical and alternatives
                if i == 0:
                    # First sample: use canonical answer
                    answer = canonical_answer
                    reference = alternatives[0] if alternatives else canonical_answer
                elif i <= len(alternatives):
                    # Use alternative answers
                    answer = alternatives[(i - 1) % len(alternatives)]
                    reference = canonical_answer
                else:
                    # Create variations
                    base_alt = alternatives[(i - 1) % len(alternatives)] if alternatives else canonical_answer
                    answer = base_alt + f'  # variant {i}'
                    reference = canonical_answer

                sample = {
                    'question': question if i == 0 else question.replace('How to', 'How do I'),
                    'answer': answer,
                    'metadata': {
                        'category': category,
                        'difficulty': difficulty,
                        'variation': i,
                        'source': 'generated',
                        'reference_answer': reference  # Add reference for metrics
                    }
                }
                samples.append(sample)

    # Fill remaining samples with more variations
    while len(samples) < n:
        base_sample = random.choice(samples[:100])  # Use first 100 as templates
        base_answer = base_sample['answer']
        base_ref = base_sample['metadata'].get('reference_answer', base_answer)
        
        # Create a variation that differs from reference
        variation_answer = base_answer.replace('=', ' = ').strip() + '  # modified'
        
        variation = {
            'question': base_sample['question'].replace('How', 'What is the way'),
            'answer': variation_answer,
            'metadata': {
                **base_sample['metadata'],
                'variation': 'extra',
                'reference_answer': base_ref  # Keep original reference
            }
        }
        samples.append(variation)

    # Trim to exact size
    samples = samples[:n]

    print(f"Generated {len(samples)} base samples with reference answers")
    return samples


def normalize_labels(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize labels deterministically."""
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


def balanced_split(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Create balanced train/val split."""
    # For LLM feedback, we don't have predefined labels, so use random split
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)

    n_val = max(1, int(len(samples) * val_ratio))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    print(f"Split: {len(train_indices)} train, {len(val_indices)} val")
    return train_indices, val_indices


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class EnhancedLLMTrainingPipeline:
    """Enhanced training pipeline with LLM feedback support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self._reference_lookup: Dict[str, str] = {}
        _refp = config.get("reference_lookup_path")
        if _refp and os.path.isfile(_refp):
            self._reference_lookup = _load_reference_pairs_file(Path(_refp))
            _LOG.info("Loaded %d question→reference pairs from %s", len(self._reference_lookup), _refp)

        # Model initialization
        model_type = config.get('model_type', 'codebert')
        dropout = config.get('dropout', 0.3)
        freeze_layers = config.get('freeze_layers', 4)

        print(f"[Pipeline] Initializing Enhanced {model_type} model...")
        self.model = EnhancedLLMClassifier(
            model_name=model_type,
            dropout=dropout,
            freeze_encoder_layers=freeze_layers,
            use_cross_attention=config.get('use_cross_attention', True),
            use_contrastive=config.get('use_contrastive', True),
            max_length=config.get('max_length', 256),
            device=self.device
        )

        # Losses
        self.criterion = LabelSmoothingBCELoss(smoothing=config.get('label_smoothing', 0.1))
        self.contrastive_criterion = InfoNCELoss(temperature=config.get('temperature', 0.07))

        # AMP scaler
        self.use_amp = config.get('use_amp', True) and self.device == 'cuda'
        if self.use_amp:
            if AMP_DEVICE:
                self.scaler = GradScaler(AMP_DEVICE)
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # EMA
        self.use_ema = config.get('use_ema', True)
        self.ema = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30
    ) -> Dict[str, Any]:
        """Train with enhanced techniques."""
        print("=" * 70)
        print("=== Enhanced LLM Classification Training (v3) ===")
        print("=== With Real LLM Feedback Integration ===")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Model: {self.config.get('model_type', 'codebert')}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {self.config.get('learning_rate', 2e-5)}")
        print(f"Batch size: {self.config.get('batch_size', 8)}")
        print(f"Gradient accumulation: {self.config.get('gradient_accumulation', 2)}")
        print(f"Mixed precision (AMP): {self.use_amp}")
        print(f"Cross-attention: {self.config.get('use_cross_attention', True)}")
        print(f"Contrastive learning: {self.config.get('use_contrastive', True)}")
        print(f"EMA: {self.use_ema}")
        print("=" * 70)

        # Optimizer with layer-wise learning rates
        encoder_params = list(self.model.encoder.parameters())
        classifier_params = (
            list(self.model.classifier.parameters()) +
            list(self.model.question_pooling.parameters()) +
            list(self.model.answer_pooling.parameters())
        )

        if self.model.use_cross_attention:
            classifier_params += (
                list(self.model.q_to_a_attention.parameters()) +
                list(self.model.a_to_q_attention.parameters())
            )

        if self.model.use_contrastive:
            classifier_params += list(self.model.contrastive_head.parameters())

        base_lr = self.config.get('learning_rate', 2e-5)
        optimizer = AdamW([
            {'params': encoder_params, 'lr': base_lr},
            {'params': classifier_params, 'lr': base_lr * 10}
        ], weight_decay=self.config.get('weight_decay', 0.01))

        # OneCycle scheduler
        num_training_steps = len(train_loader) * num_epochs // self.config.get('gradient_accumulation', 2)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[base_lr * 2, base_lr * 20],
            total_steps=num_training_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )

        # EMA
        if self.use_ema:
            self.ema = EMA(self.model, decay=self.config.get('ema_decay', 0.999))

        best_val_f1 = 0.0
        patience_counter = 0
        patience = self.config.get('patience', 5)
        history = []

        gradient_accumulation_steps = self.config.get('gradient_accumulation', 2)

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            # Training
            train_metrics = self._train_epoch(
                train_loader,
                optimizer,
                scheduler,
                gradient_accumulation_steps
            )

            # Validation with EMA if enabled
            if self.use_ema and self.ema:
                self.ema.apply_shadow()

            val_metrics = self._validate_epoch(val_loader)

            if self.use_ema and self.ema:
                self.ema.restore()

            # Print metrics
            self._print_epoch_metrics(epoch, train_metrics, val_metrics)

            # Early stopping based on average F1
            avg_f1 = (
                val_metrics.get('consistent_f1', 0) +
                val_metrics.get('correct_f1', 0) +
                val_metrics.get('useful_f1', 0)
            ) / 3

            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                patience_counter = 0
                self.save_model("best_model_llm_v3.pt")
                print(f"  [NEW BEST] New best model (F1: {avg_f1:.4f})")
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
                **{f'train_{k}': v for k, v in train_metrics.items() if k != 'loss'},
                **{f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'},
            })

        return {'history': history, 'best_val_f1': best_val_f1}

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        scheduler,
        gradient_accumulation_steps: int
    ) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation and AMP."""
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_contrastive_loss = 0.0
        head_correct = {'consistent': 0, 'correct': 0, 'useful': 0}
        total_samples = 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            questions = batch['question']
            answers = batch['answer']
            labels_dict = batch['labels']
            confidences = batch.get('confidence', [1.0] * len(questions))

            # Tokenize
            encoded = self.model.encode_qa_pair(questions, answers)
            encoded = {k: v for k, v in encoded.items() if v is not None}

            # AMP forward pass
            with autocast(AMP_DEVICE or 'cuda', enabled=self.use_amp):
                logits = self.model(
                    **encoded,
                    return_contrastive=self.model.use_contrastive
                )

                # Classification loss with confidence weighting
                cls_loss = 0.0
                for head_name in ['consistent', 'correct', 'useful']:
                    head_labels = torch.tensor(
                        [l[head_name] for l in labels_dict],
                        dtype=torch.float32
                    ).to(self.device)
                    head_labels_binary = (head_labels >= 0.5).float()

                    head_loss = self.criterion(logits[head_name], head_labels_binary)
                    cls_loss += head_loss

                    preds = (torch.sigmoid(logits[head_name]) >= 0.5).long()
                    head_correct[head_name] += (preds == head_labels_binary.long()).sum().item()

                cls_loss = cls_loss / 3

                # Contrastive loss
                contrastive_loss = torch.tensor(0.0, device=self.device)
                if self.model.use_contrastive and 'q_contrastive' in logits:
                    useful_labels = torch.tensor(
                        [l['useful'] for l in labels_dict],
                        dtype=torch.float32
                    ).to(self.device)
                    contrastive_loss = self.contrastive_criterion(
                        logits['q_contrastive'],
                        logits['a_contrastive'],
                        useful_labels
                    )

                # Total loss
                contrastive_weight = self.config.get('contrastive_weight', 0.1)
                loss = cls_loss + contrastive_weight * contrastive_loss
                loss = loss / gradient_accumulation_steps

            # Backward
            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                # EMA update
                if self.use_ema and self.ema:
                    self.ema.update()

            total_loss += loss.item() * gradient_accumulation_steps * len(questions)
            total_cls_loss += cls_loss.item() * len(questions)
            total_contrastive_loss += contrastive_loss.item() * len(questions)
            total_samples += len(questions)

            pbar.set_postfix({
                'loss': loss.item() * gradient_accumulation_steps,
                'cls': cls_loss.item(),
                'cont': contrastive_loss.item()
            })

            # Memory cleanup for low-memory GPUs
            del logits, loss, cls_loss, contrastive_loss, encoded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            'loss': total_loss / total_samples,
            'cls_loss': total_cls_loss / total_samples,
            'contrastive_loss': total_contrastive_loss / total_samples,
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

                encoded = self.model.encode_qa_pair(questions, answers)
                encoded = {k: v for k, v in encoded.items() if v is not None}

                logits = self.model(**encoded)

                # Loss calculation
                cls_loss = 0.0
                for head_name in ['consistent', 'correct', 'useful']:
                    head_labels = torch.tensor(
                        [l[head_name] for l in labels_dict],
                        dtype=torch.float32
                    ).to(self.device)
                    head_labels_binary = (head_labels >= 0.5).float()

                    head_loss = self.criterion(logits[head_name], head_labels_binary)
                    cls_loss += head_loss

                    preds = torch.sigmoid(logits[head_name]).cpu().numpy()
                    all_preds[head_name].extend(preds)
                    all_labels[head_name].extend(head_labels_binary.cpu().numpy())

                cls_loss = cls_loss / 3
                total_loss += cls_loss.item() * len(questions)

                # Collect answers for quality metrics
                all_answers.extend(answers)
                all_references.extend(
                    [
                        self._reference_lookup.get(
                            (q or "").strip(),
                            (m or {}).get("reference_answer", ""),
                        )
                        for q, m in zip(questions, metadata)
                    ]
                )

                total_samples += len(questions)

        # Calculate classification metrics
        metrics = compute_metrics(all_preds, all_labels)

        # Always compute quality metrics (with answers as both candidate and reference if no reference)
        # This allows tracking code quality even without explicit references
        try:
            # Use empty string when no reference is available — these are filtered
            # out by compute_code_quality_metrics. Self-comparison (r if r else a)
            # produced inflated scores by comparing an answer to itself.
            refs_to_use = [r if r else '' for r in all_references]
            quality_metrics = compute_quality_metrics(all_answers, refs_to_use, all_preds)
            metrics.update(quality_metrics)
        except Exception as e:
            print(f"Warning: Could not compute quality metrics: {e}")
            # Add zero values so they appear in history
            metrics.update({
                'bertscore': 0.0,
                'bleu': 0.0,
                'codebleu': 0.0,
                'rouge': 0.0,
                'ruby': 0.0
            })

        metrics['loss'] = total_loss / total_samples
        return metrics

    def _print_epoch_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Print formatted epoch metrics."""
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

    def save_model(self, filename: str):
        """Save model to file."""
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename: str):
        """Load model from file."""
        self.model.load_state_dict(torch.load(filename))


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    predictions: Dict[str, List[float]],
    labels: Dict[str, List[float]]
) -> Dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    metrics = {}

    for head in ['consistent', 'correct', 'useful']:
        if head in predictions and head in labels:
            preds = predictions[head]
            labs = labels[head]

            pred_binary = [1 if p >= 0.5 else 0 for p in preds]

            metrics[f'{head}_accuracy'] = accuracy_score(labs, pred_binary)
            metrics[f'{head}_balanced_accuracy'] = (accuracy_score(labs, pred_binary) +
                                                   accuracy_score(1 - np.array(labs), 1 - np.array(pred_binary))) / 2
            metrics[f'{head}_precision'] = precision_score(labs, pred_binary, zero_division=0)
            metrics[f'{head}_recall'] = recall_score(labs, pred_binary, zero_division=0)
            metrics[f'{head}_f1'] = f1_score(labs, pred_binary, zero_division=0)

            try:
                metrics[f'{head}_auc'] = roc_auc_score(labs, preds)
            except:
                metrics[f'{head}_auc'] = 0.5

    # Overall metrics
    metrics['consistent_acc'] = metrics.get('consistent_accuracy', 0)
    metrics['correct_acc'] = metrics.get('correct_accuracy', 0)
    metrics['useful_acc'] = metrics.get('useful_accuracy', 0)

    return metrics


def compute_quality_metrics(
    answers: List[str], 
    references: List[str],
    predictions: Optional[Dict[str, List[float]]] = None
) -> Dict[str, float]:
    """Compute code quality metrics using the metrics module."""
    try:
        return compute_code_quality_metrics(answers, references, predictions)
    except Exception as e:
        print(f"Warning: Could not compute quality metrics: {e}")
        # Fallback to zeros
        return {
            'bertscore': 0.0,
            'codebleu': 0.0,
            'bleu': 0.0,
            'rouge': 0.0,
            'ruby': 0.0
        }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

async def train_llm_system_v3(args: argparse.Namespace) -> None:
    """Main training function for LLM classifier v3 with real feedback."""
    import random as _random
    seed = getattr(args, 'seed', 42)
    _random.seed(seed)
    import numpy as _np
    _np.random.seed(seed)
    import torch as _torch
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

    print("=" * 70)
    print(f"  ClassifLLM v3 - Enhanced LLM-based Code Quality Classification | seed={seed}")
    print("  With Real LLM Feedback Integration")
    print("=" * 70)

    # LLM Configuration
    if hasattr(args, 'llm_provider') and args.llm_provider:
        # Get API key from args or environment variables
        api_key = getattr(args, 'api_key', None)
        if not api_key:
            if args.llm_provider == 'huggingface':
                api_key = os.getenv('HF_TOKEN')
            elif args.llm_provider == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')

        llm_config = LLMConfig(
            provider=args.llm_provider,
            model_name=getattr(args, 'llm_model', 'gpt-4'),
            api_key=api_key,
            api_base=getattr(args, 'api_base', None),
            temperature=getattr(args, 'llm_temperature', 0.7),
            max_tokens=getattr(args, 'llm_max_tokens', 1000),
            batch_size=getattr(args, 'llm_batch_size', 3),
            max_retries=getattr(args, 'llm_max_retries', 3),
            timeout=getattr(args, 'llm_timeout', 30)
        )
    else:
        llm_config = None

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
        'gradient_accumulation': args.gradient_accumulation,
        'use_amp': args.use_amp,
        'use_cross_attention': args.use_cross_attention,
        'use_contrastive': args.use_contrastive,
        'label_smoothing': args.label_smoothing,
        'contrastive_weight': args.contrastive_weight,
        'use_ema': args.use_ema,
        'ema_decay': args.ema_decay,
        'max_length': args.max_length,
        'llm_config': llm_config,
        'force_regenerate_feedback': getattr(args, 'force_regenerate_feedback', False),
        'max_feedback_samples': getattr(args, 'max_feedback_samples', 1000),
    }

    print("Configuration:")
    print(f"  Model: {args.model_type}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Frozen layers: {args.freeze_layers}")
    print(f"  Dropout: {args.dropout}")
    print()

    print("LLM Feedback Configuration:")
    if llm_config:
        print(f"  Provider: {llm_config.provider}")
        print(f"  Model: {llm_config.model_name}")
        print(f"  Temperature: {llm_config.temperature}")
        print(f"  Max tokens: {llm_config.max_tokens}")
        print(f"  Batch size: {llm_config.batch_size}")
        print(f"  Force regenerate: {config['force_regenerate_feedback']}")
    else:
        print("  No LLM feedback configured (using synthetic)")
    print()

    print("Enhanced Features:")
    print(f"  Cross-attention: {'Yes' if args.use_cross_attention else 'No'}")
    print(f"  Contrastive learning: {'Yes' if args.use_contrastive else 'No'}")
    print(f"  Mixed precision (AMP): {'Yes' if args.use_amp else 'No'}")
    print(f"  EMA: {'Yes' if args.use_ema else 'No'}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print("=" * 70)

    # Load data with LLM feedback
    feedback_dir = Path(getattr(args, 'feedback_dir', 'evaluation_results_server'))
    dataset_dir = Path(getattr(args, 'dataset_dir', 'stage4/datasets_for_eval'))

    samples = await load_samples_with_llm_feedback(
        feedback_dir=feedback_dir,
        dataset_dir=dataset_dir,
        llm_config=llm_config,
        force_regenerate=config['force_regenerate_feedback'],
        max_samples=config['max_feedback_samples']
    )

    print(f"Total samples: {len(samples)}")

    # Label distribution (for feedback samples)
    feedback_samples = [s for s in samples if 'feedback' in s]
    if feedback_samples:
        label_dist = Counter([
            'positive' if s['labels']['useful'] >= 0.6 else
            'negative' if s['labels']['useful'] <= 0.4 else
            'neutral'
            for s in feedback_samples
        ])
        print(f"LLM Feedback label distribution: {dict(label_dist)}")

        # Confidence stats
        confidences = [s['feedback']['confidence'] for s in feedback_samples]
        print(".3f")
        print(f"  Feedback source: {feedback_samples[0]['feedback']['source']}")
    print()

    # Create dataset and split
    dataset = EnhancedCodeQualityDataset(
        samples,
        augment=getattr(args, 'augment', False),
        use_confidence_weighting=getattr(args, 'use_confidence_weighting', True)
    )
    train_indices, val_indices = balanced_split(samples, val_ratio=0.2)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Train
    pipeline = EnhancedLLMTrainingPipeline(config)
    results = pipeline.train(train_loader, val_loader, num_epochs=args.epochs)

    # Save results — per-seed subfolder when seed != default
    seed = getattr(args, 'seed', 42)
    output_dir = Path(args.output_dir)
    if seed != 42:
        output_dir = output_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "training_history_llm_v3.json"
    with history_path.open('w') as f:
        json.dump(results['history'], f, indent=2)

    print(f"\nTraining completed!")
    print(f"History saved to: {history_path}")
    print(f"Best model saved to: best_model_llm_v3.pt")
    print(f"Best validation F1: {results['best_val_f1']:.4f}")

    if llm_config:
        print(f"LLM feedback cache saved to: llm_feedback_cache.json")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for v3."""
    parser = argparse.ArgumentParser(
        description="ClassifLLM v3 - Enhanced LLM-based Code Quality Classifier with Real LLM Feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with OpenAI GPT-4 feedback
  python clasifLLM/v3/train.py --llm-provider openai --api-key YOUR_KEY --epochs 10

  # Train with HuggingFace API feedback
  export HF_TOKEN="your-hf-token"
  python clasifLLM/v3/train.py --llm-provider huggingface --llm-model gpt2 --epochs 10

  # Train with local model feedback
  python clasifLLM/v3/train.py --llm-provider local --llm-model microsoft/DialoGPT-small --epochs 10

  # Train without LLM feedback (synthetic only)
  python clasifLLM/v3/train.py --epochs 10

  # Regenerate all LLM feedback
  python clasifLLM/v3/train.py --llm-provider openai --api-key YOUR_KEY --force-regenerate-feedback
        """
    )

    # Model settings
    parser.add_argument("--model-type", type=str, default="codebert",
                        choices=['codebert', 'graphcodebert', 'unixcoder', 'codet5', 'roberta-base'],
                        help="Pretrained model to use (default: codebert)")
    parser.add_argument("--freeze-layers", type=int, default=4,
                        help="Number of encoder layers to freeze (default: 4)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum sequence length (default: 128 for 4GB GPU)")

    # LLM Feedback settings
    parser.add_argument("--llm-provider", type=str, default=None,
                        choices=['openai', 'anthropic', 'local', 'huggingface'],
                        help="LLM provider for feedback generation")
    parser.add_argument("--llm-model", type=str, default="gpt-4",
                        help="LLM model name (default: gpt-4)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for LLM provider (or set HF_TOKEN env var for HuggingFace)")
    parser.add_argument("--api-base", type=str, default=None,
                        help="API base URL for custom endpoints")
    parser.add_argument("--llm-temperature", type=float, default=0.7,
                        help="LLM temperature for feedback generation")
    parser.add_argument("--llm-max-tokens", type=int, default=1000,
                        help="Max tokens for LLM response")
    parser.add_argument("--llm-batch-size", type=int, default=3,
                        help="Batch size for LLM API calls")
    parser.add_argument("--llm-max-retries", type=int, default=3,
                        help="Max retries for LLM API calls")
    parser.add_argument("--llm-timeout", type=int, default=30,
                        help="Timeout for LLM API calls")
    parser.add_argument("--force-regenerate-feedback", action="store_true",
                        help="Force regeneration of all LLM feedback")
    parser.add_argument("--max-feedback-samples", type=int, default=1000,
                        help="Maximum samples to generate LLM feedback for")

    # Enhanced features
    parser.add_argument("--use-cross-attention", action="store_true", default=True,
                        help="Use cross-attention between Q and A (default: True)")
    parser.add_argument("--no-cross-attention", action="store_false", dest="use_cross_attention",
                        help="Disable cross-attention")
    parser.add_argument("--use-contrastive", action="store_true", default=True,
                        help="Use contrastive learning (default: True)")
    parser.add_argument("--no-contrastive", action="store_false", dest="use_contrastive",
                        help="Disable contrastive learning")
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="Weight for contrastive loss (default: 0.1)")
    parser.add_argument("--use-amp", action="store_true", default=True,
                        help="Use mixed precision training (default: True)")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp",
                        help="Disable mixed precision")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA for model parameters (default: True)")
    parser.add_argument("--no-ema", action="store_false", dest="use_ema",
                        help="Disable EMA")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate (default: 0.999)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default: 0.1)")
    parser.add_argument("--use-confidence-weighting", action="store_true", default=True,
                        help="Use confidence weighting for LLM feedback")

    # Training settings
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (default: cuda if available)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4 for 4GB GPU)")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs (default: 30)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW (default: 0.01)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Use data augmentation (default: False)")

    # Data settings
    parser.add_argument("--feedback-dir", type=str, default="evaluation_results_server",
                        help="Directory with feedback JSON files")
    parser.add_argument("--dataset-dir", type=str, default="stage4/datasets_for_eval",
                        help="Directory with evaluation CSV files")
    parser.add_argument("--output-dir", type=str, default="stage5/stage5C/artifacts",
                        help="Output directory for model and history")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Run async training
    import asyncio
    asyncio.run(train_llm_system_v3(args))
