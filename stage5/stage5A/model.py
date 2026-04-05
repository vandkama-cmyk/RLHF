"""
ClassifLLM - LLM-based Classification Models

Supports multiple pretrained models:
- CodeBERT (microsoft/codebert-base)
- CodeT5 (Salesforce/codet5-base)
- GraphCodeBERT (microsoft/graphcodebert-base)
- UniXcoder (microsoft/unixcoder-base)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List, Union
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)


class MultiHeadClassificationHead(nn.Module):
    """
    Multi-head classification head for LLM outputs.
    Predicts: consistent, correct, useful scores.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.3):
        super().__init__()
        
        # Shared projection layer
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Independent heads for each metric
        head_input_size = hidden_size // 2
        self.head_consistent = nn.Linear(head_input_size, 1)
        self.head_correct = nn.Linear(head_input_size, 1)
        self.head_useful = nn.Linear(head_input_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, hidden_size] - pooled output from LLM
            
        Returns:
            Dict with 'consistent', 'correct', 'useful' logits
        """
        projected = self.shared_projection(hidden_states)
        
        return {
            'consistent': self.head_consistent(projected).squeeze(-1),
            'correct': self.head_correct(projected).squeeze(-1),
            'useful': self.head_useful(projected).squeeze(-1)
        }


class LLMClassifier(nn.Module):
    """
    Base LLM Classifier for code quality assessment.
    Uses pretrained encoder model with multi-head classification.
    """
    
    # Supported models with their configurations
    SUPPORTED_MODELS = {
        'codebert': 'microsoft/codebert-base',
        'graphcodebert': 'microsoft/graphcodebert-base',
        'unixcoder': 'microsoft/unixcoder-base',
        'codet5': 'Salesforce/codet5-base',
        'codebert-mlm': 'microsoft/codebert-base-mlm',
        'roberta-base': 'roberta-base',
        'bert-base': 'bert-base-uncased',
    }
    
    def __init__(
        self,
        model_name: str = 'codebert',
        dropout: float = 0.3,
        freeze_encoder_layers: int = 0,
        max_length: int = 512,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # Resolve model name
        if model_name in self.SUPPORTED_MODELS:
            model_path = self.SUPPORTED_MODELS[model_name]
        else:
            model_path = model_name  # Assume it's a HuggingFace path
        
        print(f"[LLMClassifier] Loading model: {model_path}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
            print(f"[LLMClassifier] Loaded from cache")
        except Exception as e:
            print(f"[LLMClassifier] Cache load failed: {e}, trying network")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.config = AutoConfig.from_pretrained(model_path)
            self.encoder = AutoModel.from_pretrained(model_path)
        
        # Get hidden size from config
        hidden_size = getattr(self.config, 'hidden_size', 768)
        
        # Freeze early encoder layers if requested
        if freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)
        
        # Multi-head classification
        self.classifier = MultiHeadClassificationHead(hidden_size, dropout)
        
        # Move to device
        self.to(device)
        
    def _freeze_encoder_layers(self, num_layers: int):
        """Freeze the first N encoder layers."""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
            
        # Freeze specified number of layers
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            layers = self.encoder.encoder.layer
            for i, layer in enumerate(layers):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                        
        print(f"[LLMClassifier] Frozen first {num_layers} encoder layers")
    
    def _pool_output(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool the encoder output to get a single vector per sample.
        Uses mean pooling over non-masked tokens.
        """
        # Expand attention mask for broadcasting
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum of hidden states
        sum_hidden = torch.sum(last_hidden_state * mask, dim=1)
        
        # Sum of mask (for averaging)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        
        return sum_hidden / sum_mask
    
    def encode_text(self, questions: List[str], answers: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize and encode question-answer pairs.
        
        Args:
            questions: List of question texts
            answers: List of answer/code texts
            
        Returns:
            Tokenized inputs ready for the model
        """
        # Combine question and answer with separator
        texts = [f"{q} [SEP] {a}" for q, a in zip(questions, answers)]
        
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoder and classification heads.
        
        Returns:
            Dict with 'consistent', 'correct', 'useful' logits
        """
        # Prepare inputs for encoder
        encoder_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if token_type_ids is not None and hasattr(self.config, 'type_vocab_size'):
            encoder_inputs['token_type_ids'] = token_type_ids
        
        # Get encoder outputs
        outputs = self.encoder(**encoder_inputs)
        
        # Pool the output
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = self._pool_output(outputs.last_hidden_state, attention_mask)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def predict(self, questions: List[str], answers: List[str]) -> Dict[str, torch.Tensor]:
        """
        Predict quality scores for question-answer pairs.
        
        Returns:
            Dict with 'consistent', 'correct', 'useful' probabilities
        """
        self.eval()
        with torch.no_grad():
            encoded = self.encode_text(questions, answers)
            logits = self.forward(**encoded)
            
            return {
                'consistent': torch.sigmoid(logits['consistent']),
                'correct': torch.sigmoid(logits['correct']),
                'useful': torch.sigmoid(logits['useful'])
            }
    
    def save_pretrained(self, path: str):
        """Save model and tokenizer."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save encoder
        self.encoder.save_pretrained(os.path.join(path, 'encoder'))
        self.tokenizer.save_pretrained(os.path.join(path, 'tokenizer'))
        
        # Save classifier head
        torch.save({
            'classifier': self.classifier.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
        }, os.path.join(path, 'classifier_head.pt'))
        
        print(f"[LLMClassifier] Model saved to {path}")
    
    @classmethod
    def load_pretrained(cls, path: str, device: str = 'cuda') -> 'LLMClassifier':
        """Load saved model."""
        import os
        
        # Load classifier config
        head_path = os.path.join(path, 'classifier_head.pt')
        head_data = torch.load(head_path, map_location=device)
        
        # Create instance
        model = cls(
            model_name=os.path.join(path, 'encoder'),
            max_length=head_data['max_length'],
            device=device
        )
        
        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, 'tokenizer'))
        
        # Load classifier head
        model.classifier.load_state_dict(head_data['classifier'])
        
        print(f"[LLMClassifier] Model loaded from {path}")
        return model


class CodeBERTClassifier(LLMClassifier):
    """CodeBERT-based classifier for code quality."""
    
    def __init__(self, dropout: float = 0.3, freeze_layers: int = 6, device: str = 'cuda'):
        super().__init__(
            model_name='codebert',
            dropout=dropout,
            freeze_encoder_layers=freeze_layers,
            device=device
        )


class CodeT5Classifier(LLMClassifier):
    """CodeT5-based classifier for code quality."""
    
    def __init__(self, dropout: float = 0.3, freeze_layers: int = 4, device: str = 'cuda'):
        super().__init__(
            model_name='codet5',
            dropout=dropout,
            freeze_encoder_layers=freeze_layers,
            device=device
        )
    
    def _pool_output(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """CodeT5 uses decoder, so we use the first token (like CLS)."""
        return last_hidden_state[:, 0, :]


class GraphCodeBERTClassifier(LLMClassifier):
    """GraphCodeBERT-based classifier for code quality."""
    
    def __init__(self, dropout: float = 0.3, freeze_layers: int = 6, device: str = 'cuda'):
        super().__init__(
            model_name='graphcodebert',
            dropout=dropout,
            freeze_encoder_layers=freeze_layers,
            device=device
        )


class UniXcoderClassifier(LLMClassifier):
    """UniXcoder-based classifier for code quality."""
    
    def __init__(self, dropout: float = 0.3, freeze_layers: int = 6, device: str = 'cuda'):
        super().__init__(
            model_name='unixcoder',
            dropout=dropout,
            freeze_encoder_layers=freeze_layers,
            device=device
        )

