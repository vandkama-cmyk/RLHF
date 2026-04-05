"""
Stage 2A - Data Loader
======================

Loads and processes training data from:
- Aggregated SFT dataset (T2C-CoNaLa + T2T-SO) in CSV format
"""

import csv
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class Stage2ADataLoader:
    """
    Data loader for Stage 2A contrastive learning.
    Loads aggregated SFT dataset for contrastive training.
    """
    
    def __init__(
        self,
        sft_path: str = "datasets_for_training/sft_dataset.csv",
        extra_jsonl_paths: Optional[List[str]] = None,
        max_samples: int = 5000,
        negative_ratio: float = 1.0,
        seed: int = 42,
        eval_dir: Optional[str] = None,
        feedback_dir: Optional[str] = None,
    ):
        self.sft_path = Path(sft_path)
        self.extra_jsonl_paths = [Path(p) for p in (extra_jsonl_paths or [])]
        self.max_samples = max_samples
        self.negative_ratio = max(0.0, negative_ratio)
        self.seed = seed
        self.rng = random.Random(seed)
        self.eval_dir = Path(eval_dir) if eval_dir else None
        self.feedback_dir = Path(feedback_dir) if feedback_dir else None
    
    def load_sft_data(self) -> List[Dict[str, Any]]:
        """Load SFT training dataset."""
        samples = []
        
        if not self.sft_path.exists():
            print(f"[DataLoader] SFT path not found: {self.sft_path}")
            return samples
        
        print(f"[DataLoader] Loading SFT data from {self.sft_path}")
        
        with open(self.sft_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = row.get('question', '')
                response = row.get('best_answer', row.get('answer', ''))
                
                if prompt and response:
                    # Clean response
                    response = self._clean_response(response)
                    
                    if len(response) > 10:  # Skip very short responses
                        samples.append({
                            'prompt': prompt,
                            'response': response,
                            'reference': response,
                            'source': row.get('model_tag', 'T2T-SO'),
                            'quality': 1.0,
                            'is_negative': False
                        })
        
        print(f"[DataLoader] Loaded {len(samples)} samples from SFT data")
        return samples
    
    def _load_conala_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load CoNaLa JSONL as additional SFT data."""
        samples = []
        if not path.exists():
            print(f"[DataLoader] Extra JSONL path not found: {path}")
            return samples
        
        print(f"[DataLoader] Loading extra SFT JSONL from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                
                prompt = data.get('rewritten_intent') or data.get('intent', '')
                response = data.get('snippet', '')
                if prompt and response:
                    samples.append({
                        'prompt': prompt,
                        'response': response,
                        'reference': response,
                        'source': 'T2C-CoNaLa',
                        'quality': 1.0,
                        'is_negative': False
                    })
        
        print(f"[DataLoader] Loaded {len(samples)} samples from {path.name}")
        return samples
    
    def _clean_response(self, response: str) -> str:
        """Clean and normalize response text."""
        if not response:
            return ""
        
        # Remove common noise patterns
        response = response.strip()
        
        # Remove repeated "A:" patterns
        lines = response.split('\n')
        cleaned_lines = []
        seen_content = set()
        
        for line in lines:
            line = line.strip()
            if line.startswith('A:'):
                continue
            if line in seen_content:
                continue
            if line:
                seen_content.add(line)
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines[:5])  # Limit to first 5 unique lines
    
    def _build_negative_samples(
        self,
        positive_samples: List[Dict[str, Any]],
        negative_ratio: float
    ) -> List[Dict[str, Any]]:
        """Create negative samples by mismatching prompts and responses."""
        if not positive_samples or negative_ratio <= 0:
            return []
        
        responses = [s['response'] for s in positive_samples]
        num_negatives = int(len(positive_samples) * negative_ratio)
        negatives = []
        
        for idx in range(num_negatives):
            pos = positive_samples[idx % len(positive_samples)]
            prompt = pos['prompt']
            
            # Sample a different response
            chosen = None
            for _ in range(10):
                candidate = self.rng.choice(responses)
                if candidate != pos['response']:
                    chosen = candidate
                    break
            if not chosen:
                continue
            
            negatives.append({
                'prompt': prompt,
                'response': chosen,
                'reference': pos.get('reference', ''),
                'source': f"{pos.get('source', 'T2T-SO')}_neg",
                'quality': 0.0,
                'is_negative': True
            })
        
        print(f"[DataLoader] Created {len(negatives)} negative samples")
        return negatives
    
    def _load_shared_datasets(self) -> List[Dict[str, Any]]:
        """Load real (question, answer, quality) samples from shared stage4 datasets."""
        if not self.eval_dir or not self.feedback_dir:
            return []
        if not self.eval_dir.exists() or not self.feedback_dir.exists():
            print(f"[DataLoader] Shared dataset dirs not found: {self.eval_dir}, {self.feedback_dir}")
            return []

        try:
            import sys
            import os
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from stage4.dataset import load_real_datasets
            raw_samples = load_real_datasets(self.eval_dir, self.feedback_dir)
        except Exception as e:
            print(f"[DataLoader] Could not load shared datasets: {e}")
            return []

        converted = []
        for s in raw_samples:
            question = s.get('question', '')
            answer = s.get('answer', '')
            labels = s.get('labels', {})
            if not question or not answer:
                continue
            quality = float(
                labels.get('consistent', 0.5) +
                labels.get('correct', 0.5) +
                labels.get('useful', 0.5)
            ) / 3.0
            ref = s.get('metadata', {}).get('reference_answer') or answer
            converted.append({
                'prompt': question,
                'response': self._clean_response(answer),
                'reference': ref,
                'source': 'human_feedback',
                'quality': quality,
                'is_negative': quality < 0.4,
            })

        print(f"[DataLoader] Loaded {len(converted)} samples from shared datasets")
        return converted

    def load_training_data(
        self,
        val_ratio: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load train and validation data using SFT positives + synthetic negatives."""
        positive_samples = self.load_sft_data()
        for path in self.extra_jsonl_paths:
            positive_samples.extend(self._load_conala_jsonl(path))
        positive_samples.extend(self._load_shared_datasets())
        
        # Shuffle and cap positives to honor max_samples with negatives included
        self.rng.shuffle(positive_samples)
        if self.max_samples:
            if self.negative_ratio > 0:
                max_pos = max(1, int(self.max_samples / (1 + self.negative_ratio)))
            else:
                max_pos = self.max_samples
            positive_samples = positive_samples[:max_pos]
        
        # Split positives into train/val
        train_pos, val_pos = self.split_train_val(positive_samples, val_ratio=val_ratio)
        
        # Add negatives to training and validation for meaningful evaluation
        train_neg = self._build_negative_samples(train_pos, self.negative_ratio)
        val_neg = self._build_negative_samples(val_pos, self.negative_ratio)
        train_samples = train_pos + train_neg
        val_samples = val_pos + val_neg
        
        self.rng.shuffle(train_samples)
        self.rng.shuffle(val_samples)
        
        print(f"[DataLoader] Train positives: {len(train_pos)}")
        print(f"[DataLoader] Train negatives: {len(train_neg)}")
        print(f"[DataLoader] Train total: {len(train_samples)}")
        print(f"[DataLoader] Val positives: {len(val_pos)}")
        print(f"[DataLoader] Val negatives: {len(val_neg)}")
        print(f"[DataLoader] Val total: {len(val_samples)}")
        
        return train_samples, val_samples
    
    def create_contrastive_pairs(
        self,
        samples: List[Dict[str, Any]],
        num_pairs: int = 1000
    ) -> List[Tuple[Dict, Dict, int]]:
        """
        Create pairs for contrastive learning.
        
        Returns list of (sample1, sample2, label) where:
        - label=0: similar pair (both high or both low quality)
        - label=1: dissimilar pair (one high, one low quality)
        """
        pairs = []
        
        # Separate by quality
        high_quality = [s for s in samples if s.get('quality', 0.5) >= 0.7]
        low_quality = [s for s in samples if s.get('quality', 0.5) <= 0.3]
        mid_quality = [s for s in samples if 0.3 < s.get('quality', 0.5) < 0.7]
        
        print(f"[DataLoader] Quality distribution: high={len(high_quality)}, "
              f"mid={len(mid_quality)}, low={len(low_quality)}")
        
        # Create similar pairs (label=0)
        for _ in range(num_pairs // 3):
            if len(high_quality) >= 2:
                pair = self.rng.sample(high_quality, 2)
                pairs.append((pair[0], pair[1], 0))
            
            if len(low_quality) >= 2:
                pair = self.rng.sample(low_quality, 2)
                pairs.append((pair[0], pair[1], 0))
        
        # Create dissimilar pairs (label=1)
        for _ in range(num_pairs // 3):
            if high_quality and low_quality:
                high = self.rng.choice(high_quality)
                low = self.rng.choice(low_quality)
                pairs.append((high, low, 1))
        
        self.rng.shuffle(pairs)
        print(f"[DataLoader] Created {len(pairs)} contrastive pairs")
        return pairs
    
    def split_train_val(
        self,
        samples: List[Dict[str, Any]],
        val_ratio: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split samples into train and validation sets."""
        self.rng.shuffle(samples)
        split_idx = int(len(samples) * (1 - val_ratio))
        return samples[:split_idx], samples[split_idx:]
