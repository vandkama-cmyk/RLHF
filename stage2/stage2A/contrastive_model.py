"""
Stage 2A - Contrastive Reward Model
===================================

Implements contrastive learning for enhanced reward model discriminative capacity.

Key component: Maximum Contrastive Loss Function
L = (1-y) * d² + y * max(0, margin-d)²

Where:
- d: Euclidean distance between two embeddings
- y: Binary label (0 for similar pairs, 1 for dissimilar pairs)
- margin: Hyperparameter defining minimum distance between dissimilar pairs
"""

# Disable chat template lookup for older models (fixes transformers 4.46+ issue)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Patch transformers to skip chat template listing for models that don't have them
import transformers.utils.hub
_original_list_repo_templates = transformers.utils.hub.list_repo_templates
def _patched_list_repo_templates(*args, **kwargs):
    """Patched version that handles 404 errors gracefully."""
    try:
        return _original_list_repo_templates(*args, **kwargs)
    except Exception:
        return []  # Return empty list if templates can't be fetched
transformers.utils.hub.list_repo_templates = _patched_list_repo_templates

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np


class MaxContrastiveLoss(nn.Module):
    """
    Maximum Contrastive Loss Function (Margin-based Contrastive Loss).
    
    L = (1-y) * d² + y * max(0, margin-d)²
    
    For similar pairs (y=0): Loss = d² (minimize distance)
    For dissimilar pairs (y=1): Loss = max(0, margin-d)² (push apart if too close)
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the maximum contrastive loss.
        
        Args:
            embedding1: [batch, dim] - First set of embeddings
            embedding2: [batch, dim] - Second set of embeddings
            labels: [batch] - Binary labels (0=similar, 1=dissimilar)
        
        Returns:
            Contrastive loss value
        """
        # Compute Euclidean distance
        d = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Maximum contrastive loss
        # For similar (y=0): (1-0) * d² = d²
        # For dissimilar (y=1): 0 + 1 * max(0, margin-d)² = max(0, margin-d)²
        similar_loss = (1 - labels) * d.pow(2)
        dissimilar_loss = labels * F.relu(self.margin - d).pow(2)
        
        loss = similar_loss + dissimilar_loss
        return loss.mean()


class InfoNCEContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for in-batch negative sampling.
    Complements the maximum contrastive loss for richer training signal.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        quality_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with in-batch negatives.
        
        Args:
            query: [batch, dim] - Query embeddings (normalized)
            positive: [batch, dim] - Positive pair embeddings (normalized)
            quality_labels: [batch] - Quality scores to weight the loss
        
        Returns:
            InfoNCE loss value
        """
        batch_size = query.size(0)
        
        # Normalize embeddings
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        
        # Compute similarity matrix [batch, batch]
        similarity = torch.mm(query, positive.t()) / self.temperature
        
        # Diagonal elements are positive pairs
        targets = torch.arange(batch_size, device=query.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity, targets)
        
        # Weight by quality if provided
        if quality_labels is not None:
            weight = 0.5 + 0.5 * quality_labels.mean()
            loss = loss * weight
        
        return loss


class ContrastiveProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    Projects embeddings into a lower-dimensional space optimized for contrastive learning.
    """
    
    def __init__(self, input_dim: int, projection_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # Simplified projection - fewer layers, less dropout for better gradient flow
        self.projector = nn.Sequential(
            nn.Linear(input_dim, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Reduced dropout
            nn.Linear(projection_dim * 2, projection_dim)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot for better gradient flow."""
        for module in self.projector.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings and normalize."""
        projected = self.projector(x)
        return F.normalize(projected, dim=-1)


class RewardHead(nn.Module):
    """
    Reward prediction head for quality scoring.
    Predicts consistent, correct, and useful scores.
    Supports both bi-encoder (two separate embeddings) and cross-encoder (single embedding) inputs.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Cross-encoder path (single embedding input)
        self.cross_encoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Individual heads
        self.head_consistent = nn.Linear(hidden_dim // 2, 1)
        self.head_correct = nn.Linear(hidden_dim // 2, 1)
        self.head_useful = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better gradient flow."""
        for module in self.cross_encoder_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize final heads with slightly larger weights for cross-encoder
        for head in [self.head_consistent, self.head_correct, self.head_useful]:
            nn.init.xavier_uniform_(head.weight, gain=0.5)
            nn.init.zeros_(head.bias)
    
    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict reward scores.
        
        Args:
            prompt_embedding: [batch, hidden_dim] - cross-encoded pair embedding
            response_embedding: [batch, hidden_dim] - same as prompt_embedding for cross-encoder
        
        Returns:
            Dictionary with logits for consistent, correct, useful
        """
        # Use cross-encoder path (prompt_embedding is the pair embedding)
        shared_repr = self.cross_encoder_head(prompt_embedding)
        
        return {
            'consistent': self.head_consistent(shared_repr).squeeze(-1),
            'correct': self.head_correct(shared_repr).squeeze(-1),
            'useful': self.head_useful(shared_repr).squeeze(-1)
        }


class ContrastiveRewardModel(nn.Module):
    """
    Stage 2A Contrastive Reward Model
    
    Combines:
    1. Pre-trained CodeBERT encoder
    2. Contrastive projection head
    3. Maximum contrastive loss for pair-wise learning
    4. InfoNCE loss for in-batch negative mining
    5. Reward prediction heads for quality scoring
    """
    
    SUPPORTED_MODELS = {
        'codebert': 'microsoft/codebert-base',
        'graphcodebert': 'microsoft/graphcodebert-base',
        'unixcoder': 'microsoft/unixcoder-base',
    }
    
    def __init__(
        self,
        model_name: str = 'codebert',
        projection_dim: int = 256,
        margin: float = 1.0,
        temperature: float = 0.07,
        dropout: float = 0.3,
        freeze_encoder_layers: int = 4,
        max_length: int = 256,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.margin = margin
        self.temperature = temperature
        
        # Resolve model path
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        
        print(f"[ContrastiveRewardModel] Loading encoder: {model_path}")
        
        # Load tokenizer and encoder
        # Handle different transformers versions and potential network issues
        import warnings
        warnings.filterwarnings("ignore")
        
        # Set environment variables to avoid network issues
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        
        try:
            # Try standard loading first
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.config = AutoConfig.from_pretrained(model_path)
            self.encoder = AutoModel.from_pretrained(model_path)
        except Exception as e1:
            print(f"[ContrastiveRewardModel] Standard loading failed: {type(e1).__name__}")
            try:
                # Try with local_files_only if model is cached
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True
                )
                self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
                self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
                print("[ContrastiveRewardModel] Loaded from cache")
            except Exception as e2:
                print(f"[ContrastiveRewardModel] Cache loading failed: {type(e2).__name__}")
                # Fallback to roberta-base if codebert not available
                fallback_model = "roberta-base"
                print(f"[ContrastiveRewardModel] Falling back to {fallback_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.config = AutoConfig.from_pretrained(fallback_model)
                self.encoder = AutoModel.from_pretrained(fallback_model)
        
        hidden_size = getattr(self.config, 'hidden_size', 768)
        self.hidden_size = hidden_size
        
        # Freeze early encoder layers (if any)
        if freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)
        else:
            print(f"[ContrastiveRewardModel] All encoder layers trainable")
        
        # Don't enable gradient checkpointing - it can hurt gradient flow
        # Only enable for very large models or low-memory situations
        # if hasattr(self.encoder, 'gradient_checkpointing_enable'):
        #     self.encoder.gradient_checkpointing_enable()
        
        # Contrastive projection head
        self.projection_head = ContrastiveProjectionHead(
            hidden_size, projection_dim, dropout
        )
        
        # Reward prediction head
        self.reward_head = RewardHead(hidden_size, dropout)
        
        # Loss functions
        self.max_contrastive_loss = MaxContrastiveLoss(margin=margin)
        self.infonce_loss = InfoNCEContrastiveLoss(temperature=temperature)
        
        self.to(device)
        print(f"[ContrastiveRewardModel] Model initialized with hidden_size={hidden_size}")
    
    def _freeze_encoder_layers(self, num_layers: int):
        """Freeze the first N encoder layers."""
        # Freeze embeddings
        if hasattr(self.encoder, 'embeddings'):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        
        # Freeze encoder layers
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            layers = self.encoder.encoder.layer
            for i, layer in enumerate(layers):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        print(f"[ContrastiveRewardModel] Frozen {num_layers} encoder layers")
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into embeddings using [CLS] token."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        outputs = self.encoder(**encoded)
        
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return cls_embedding
    
    def encode_pair(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Cross-encoder: encode prompt-response pairs together.
        This captures the interaction between prompt and response.
        """
        # Combine prompt and response with separator
        combined_texts = [
            f"{p} [SEP] {r}" for p, r in zip(prompts, responses)
        ]
        
        encoded = self.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        outputs = self.encoder(**encoded)
        
        # Use [CLS] token embedding - this captures prompt-response interaction
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        return cls_embedding
    
    def forward(
        self,
        prompts: List[str],
        responses: List[str],
        return_projections: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive reward model using CROSS-ENCODER.
        
        Args:
            prompts: List of prompt texts
            responses: List of response texts
            return_projections: Whether to return contrastive projections
        
        Returns:
            Dictionary with reward logits and optionally projections
        """
        # CROSS-ENCODER: Encode prompt+response together to capture interaction
        pair_embeddings = self.encode_pair(prompts, responses)
        
        # Also encode separately for contrastive learning (if needed)
        prompt_embeddings = self.encode(prompts) if return_projections else None
        response_embeddings = self.encode(responses) if return_projections else None
        
        # Predict rewards using cross-encoded pair embedding
        # Double the embedding (use same for both inputs to reward head)
        reward_logits = self.reward_head(pair_embeddings, pair_embeddings)
        
        outputs = {
            'consistent': reward_logits['consistent'],
            'correct': reward_logits['correct'],
            'useful': reward_logits['useful'],
            'pair_embeddings': pair_embeddings
        }
        
        if return_projections and prompt_embeddings is not None:
            outputs['prompt_embeddings'] = prompt_embeddings
            outputs['response_embeddings'] = response_embeddings
            prompt_proj = self.projection_head(prompt_embeddings)
            response_proj = self.projection_head(response_embeddings)
            outputs['prompt_projection'] = prompt_proj
            outputs['response_projection'] = response_proj
        
        return outputs
    
    def compute_contrastive_loss(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor,
        use_infonce: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined contrastive losses.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            labels: Binary labels for similarity (0=similar, 1=dissimilar)
            use_infonce: Whether to also compute InfoNCE loss
        
        Returns:
            Dictionary with loss components
        """
        # Maximum contrastive loss
        max_contrast_loss = self.max_contrastive_loss(embeddings1, embeddings2, labels)
        
        losses = {'max_contrastive': max_contrast_loss}
        
        # InfoNCE loss for similar pairs
        if use_infonce:
            # Only use similar pairs for InfoNCE
            similar_mask = (labels == 0)
            if similar_mask.sum() > 1:
                proj1 = F.normalize(embeddings1[similar_mask], dim=-1)
                proj2 = F.normalize(embeddings2[similar_mask], dim=-1)
                infonce = self.infonce_loss(proj1, proj2)
                losses['infonce'] = infonce
            else:
                losses['infonce'] = torch.tensor(0.0, device=embeddings1.device)
        
        # Total contrastive loss
        losses['total_contrastive'] = max_contrast_loss
        if use_infonce and 'infonce' in losses:
            losses['total_contrastive'] = losses['total_contrastive'] + 0.5 * losses['infonce']
        
        return losses
    
    def predict_reward(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Predict reward scores for prompt-response pairs.
        
        Args:
            prompts: List of prompts
            responses: List of responses
        
        Returns:
            Dictionary with reward scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(prompts, responses, return_projections=False)
            
            return {
                'consistent': torch.sigmoid(outputs['consistent']).cpu().numpy(),
                'correct': torch.sigmoid(outputs['correct']).cpu().numpy(),
                'useful': torch.sigmoid(outputs['useful']).cpu().numpy()
            }
    
    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save encoder and tokenizer
        self.encoder.save_pretrained(os.path.join(path, 'encoder'))
        self.tokenizer.save_pretrained(os.path.join(path, 'tokenizer'))
        
        # Save custom heads
        torch.save({
            'projection_head': self.projection_head.state_dict(),
            'reward_head': self.reward_head.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'margin': self.margin,
            'temperature': self.temperature,
            'hidden_size': self.hidden_size
        }, os.path.join(path, 'heads.pt'))
        
        print(f"[ContrastiveRewardModel] Saved to {path}")
    
    @classmethod
    def load_pretrained(cls, path: str, device: str = 'cuda') -> 'ContrastiveRewardModel':
        """Load model from checkpoint."""
        import os
        
        # Load custom heads config
        heads_config = torch.load(os.path.join(path, 'heads.pt'), map_location=device)
        
        # Create model
        model = cls(
            model_name=heads_config['model_name'],
            margin=heads_config.get('margin', 1.0),
            temperature=heads_config.get('temperature', 0.07),
            max_length=heads_config['max_length'],
            device=device
        )
        
        # Load encoder from saved
        model.encoder = AutoModel.from_pretrained(os.path.join(path, 'encoder'))
        model.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, 'tokenizer'))
        
        # Load custom heads
        model.projection_head.load_state_dict(heads_config['projection_head'])
        model.reward_head.load_state_dict(heads_config['reward_head'])
        
        model.to(device)
        return model
