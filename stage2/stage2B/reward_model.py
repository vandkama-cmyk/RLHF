"""
Stage 2B - Feedback-Driven Reward Model
=======================================

Reward model trained with GPT-2 generated feedback to filter out
syntactically incorrect "hallucinations" and improve code quality.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Patch for transformers compatibility
import transformers.utils.hub
_original_list_repo_templates = transformers.utils.hub.list_repo_templates
def _patched_list_repo_templates(*args, **kwargs):
    try:
        return _original_list_repo_templates(*args, **kwargs)
    except Exception:
        return []
transformers.utils.hub.list_repo_templates = _patched_list_repo_templates

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np


class FeedbackRewardModel(nn.Module):
    """
    Reward Model for Stage 2B.
    
    Uses CodeBERT encoder with multi-head reward prediction for:
    - Consistency: Logical coherence with question
    - Agreement: Alignment with coding practices  
    - Usefulness: Practical utility of the answer
    
    Trained with GPT-2 generated synthetic feedback scores.
    """
    
    SUPPORTED_MODELS = {
        'codebert': 'microsoft/codebert-base',
        'graphcodebert': 'microsoft/graphcodebert-base',
        'unixcoder': 'microsoft/unixcoder-base',
    }
    
    def __init__(
        self,
        model_name: str = 'codebert',
        hidden_dim: int = 768,
        num_heads: int = 3,
        dropout: float = 0.3,
        freeze_encoder_layers: int = 4,
        max_length: int = 256,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_length = max_length
        self.device = device
        
        # Resolve model path
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        
        print(f"[FeedbackRewardModel] Loading encoder: {model_path}")
        
        # Load tokenizer and encoder
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.config = AutoConfig.from_pretrained(model_path)
            self.encoder = AutoModel.from_pretrained(model_path)
        except Exception as e1:
            print(f"[FeedbackRewardModel] Standard loading failed: {type(e1).__name__}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
                self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
                print("[FeedbackRewardModel] Loaded from cache")
            except Exception as e2:
                print(f"[FeedbackRewardModel] Falling back to roberta-base")
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", local_files_only=True)
                self.config = AutoConfig.from_pretrained("roberta-base", local_files_only=True)
                self.encoder = AutoModel.from_pretrained("roberta-base", local_files_only=True)
        
        self.hidden_size = getattr(self.config, 'hidden_size', 768)
        
        # Freeze early encoder layers
        if freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)
        
        # Reward prediction heads
        self.shared_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Individual heads for each criterion
        self.consistency_head = nn.Linear(self.hidden_size // 2, 1)
        self.agreement_head = nn.Linear(self.hidden_size // 2, 1)
        self.usefulness_head = nn.Linear(self.hidden_size // 2, 1)
        
        # Quality filter head (for hallucination detection)
        self.quality_filter = nn.Sequential(
            nn.Linear(self.hidden_size // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.to(device)
        print(f"[FeedbackRewardModel] Initialized with hidden_size={self.hidden_size}")
    
    def _freeze_encoder_layers(self, num_layers: int):
        """Freeze the first N encoder layers."""
        if hasattr(self.encoder, 'embeddings'):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            layers = self.encoder.encoder.layer
            for i, layer in enumerate(layers):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        print(f"[FeedbackRewardModel] Frozen {num_layers} encoder layers")
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into embeddings."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        outputs = self.encoder(**encoded)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return cls_embedding
    
    def forward(
        self,
        questions: List[str],
        answers: List[str],
        return_quality_score: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for reward prediction.
        
        Args:
            questions: List of questions
            answers: List of code answers
            return_quality_score: Whether to return quality filter score
        
        Returns:
            Dictionary with scores for consistency, agreement, usefulness
        """
        # Encode separately
        question_embeddings = self.encode(questions)
        answer_embeddings = self.encode(answers)
        
        # Combine embeddings
        combined = torch.cat([question_embeddings, answer_embeddings], dim=-1)
        
        # Shared representation
        shared = self.shared_layer(combined)
        
        # Individual scores
        consistency = torch.sigmoid(self.consistency_head(shared)).squeeze(-1)
        agreement = torch.sigmoid(self.agreement_head(shared)).squeeze(-1)
        usefulness = torch.sigmoid(self.usefulness_head(shared)).squeeze(-1)
        
        outputs = {
            'consistency': consistency,
            'agreement': agreement,
            'usefulness': usefulness,
            'question_embeddings': question_embeddings,
            'answer_embeddings': answer_embeddings,
        }
        
        # Quality filter for hallucination detection
        if return_quality_score:
            quality = self.quality_filter(shared).squeeze(-1)
            outputs['quality'] = quality
        
        return outputs
    
    def predict_reward(
        self,
        questions: List[str],
        answers: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Predict reward scores for Q-A pairs.
        
        Returns numpy arrays for inference.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(questions, answers)
            
            return {
                'consistency': outputs['consistency'].cpu().numpy(),
                'agreement': outputs['agreement'].cpu().numpy(),
                'usefulness': outputs['usefulness'].cpu().numpy(),
                'quality': outputs.get('quality', torch.zeros_like(outputs['consistency'])).cpu().numpy(),
            }
    
    def filter_hallucinations(
        self,
        questions: List[str],
        answers: List[str],
        threshold: float = 0.5
    ) -> Tuple[List[str], List[str], List[int]]:
        """
        Filter out low-quality answers (hallucinations).
        
        Args:
            questions: List of questions
            answers: List of answers
            threshold: Quality threshold for filtering
        
        Returns:
            Filtered questions, answers, and indices of kept samples
        """
        rewards = self.predict_reward(questions, answers)
        quality_scores = rewards['quality']
        
        # Keep samples above threshold
        kept_indices = [i for i, q in enumerate(quality_scores) if q >= threshold]
        
        filtered_questions = [questions[i] for i in kept_indices]
        filtered_answers = [answers[i] for i in kept_indices]
        
        return filtered_questions, filtered_answers, kept_indices
    
    def compute_combined_reward(
        self,
        questions: List[str],
        answers: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute combined reward score.
        
        Args:
            questions: List of questions
            answers: List of answers
            weights: Optional weights for each criterion
        
        Returns:
            Combined reward tensor
        """
        if weights is None:
            weights = {'consistency': 0.4, 'agreement': 0.3, 'usefulness': 0.3}
        
        outputs = self.forward(questions, answers)
        
        combined = (
            weights['consistency'] * outputs['consistency'] +
            weights['agreement'] * outputs['agreement'] +
            weights['usefulness'] * outputs['usefulness']
        )
        
        # Apply quality filter
        if 'quality' in outputs:
            combined = combined * outputs['quality']
        
        return combined
    
    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        os.makedirs(path, exist_ok=True)
        
        self.encoder.save_pretrained(os.path.join(path, 'encoder'))
        self.tokenizer.save_pretrained(os.path.join(path, 'tokenizer'))
        
        torch.save({
            'shared_layer': self.shared_layer.state_dict(),
            'consistency_head': self.consistency_head.state_dict(),
            'agreement_head': self.agreement_head.state_dict(),
            'usefulness_head': self.usefulness_head.state_dict(),
            'quality_filter': self.quality_filter.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'hidden_size': self.hidden_size,
        }, os.path.join(path, 'heads.pt'))
        
        print(f"[FeedbackRewardModel] Saved to {path}")
    
    @classmethod
    def load_pretrained(cls, path: str, device: str = 'cuda') -> 'FeedbackRewardModel':
        """Load model from checkpoint."""
        heads_config = torch.load(os.path.join(path, 'heads.pt'), map_location=device)
        
        model = cls(
            model_name=heads_config['model_name'],
            max_length=heads_config['max_length'],
            device=device
        )
        
        model.encoder = AutoModel.from_pretrained(os.path.join(path, 'encoder'))
        model.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, 'tokenizer'))
        
        model.shared_layer.load_state_dict(heads_config['shared_layer'])
        model.consistency_head.load_state_dict(heads_config['consistency_head'])
        model.agreement_head.load_state_dict(heads_config['agreement_head'])
        model.usefulness_head.load_state_dict(heads_config['usefulness_head'])
        model.quality_filter.load_state_dict(heads_config['quality_filter'])
        
        model.to(device)
        return model


class RewardModelLoss(nn.Module):
    """
    Loss function for reward model training.
    Combines feedback alignment loss with quality filtering loss.
    """
    
    def __init__(self, label_smoothing: float = 0.1, quality_weight: float = 0.2):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.quality_weight = quality_weight
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        quality_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            predictions: Model predictions
            targets: GPT-2 feedback scores
            quality_labels: Optional quality labels for filtering
        
        Returns:
            Dictionary with loss components
        """
        losses = {}
        total_loss = 0.0
        
        for criterion in ['consistency', 'agreement', 'usefulness']:
            if criterion in predictions and criterion in targets:
                pred = predictions[criterion]
                target = targets[criterion]
                
                # Apply label smoothing
                if self.label_smoothing > 0:
                    target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
                
                loss = F.mse_loss(pred, target)
                losses[f'{criterion}_loss'] = loss
                total_loss += loss
        
        # Quality filter loss
        if quality_labels is not None and 'quality' in predictions:
            quality_loss = F.binary_cross_entropy(predictions['quality'], quality_labels)
            losses['quality_loss'] = quality_loss
            total_loss += self.quality_weight * quality_loss
        
        losses['total_loss'] = total_loss
        
        return losses
