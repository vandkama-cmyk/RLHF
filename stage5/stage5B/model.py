"""
ClassifLLM v2 - Enhanced LLM-based Classification Models

Improvements over v1:
1. Cross-attention between question and answer
2. Separate encoding for question and answer with attention pooling
3. Contrastive learning auxiliary objective
4. Improved multi-head classification with residual connections
5. Better pooling strategies (attention-based)
"""

from __future__ import annotations

from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig


class AttentionPooling(nn.Module):
    """Attention-based pooling for sequence outputs."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]
        Returns:
            pooled: [batch, hidden_size]
        """
        # Compute attention scores
        attention_scores = self.attention(hidden_states).squeeze(-1)  # [batch, seq_len]
        
        # Mask padding tokens
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, 
            float('-inf')
        )
        
        # Softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, seq_len]
        
        # Weighted sum
        pooled = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return pooled


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for question-answer interaction."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len_q, hidden_size] - question or answer
            key_value: [batch, seq_len_kv, hidden_size] - the other sequence
            key_padding_mask: [batch, seq_len_kv] - padding mask for key/value
        """
        # Cross attention with residual
        attn_out, _ = self.cross_attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask
        )
        query = self.norm1(query + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)
        
        return query


class EnhancedClassificationHead(nn.Module):
    """
    Enhanced multi-head classification with residual connections and gating.
    Predicts: consistent, correct, useful scores.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.3):
        super().__init__()
        
        # Shared projection with residual
        self.shared_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Combined Q+A
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.shared_residual = nn.Linear(hidden_size * 2, hidden_size)
        
        # Second layer
        self.shared_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism for each head
        head_dim = hidden_size // 2
        
        self.gate_consistent = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.Sigmoid()
        )
        self.gate_correct = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.Sigmoid()
        )
        self.gate_useful = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.Sigmoid()
        )
        
        # Final classification heads
        self.head_consistent = nn.Linear(head_dim, 1)
        self.head_correct = nn.Linear(head_dim, 1)
        self.head_useful = nn.Linear(head_dim, 1)
        
    def forward(
        self, 
        question_repr: torch.Tensor, 
        answer_repr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            question_repr: [batch, hidden_size]
            answer_repr: [batch, hidden_size]
        """
        # Combine representations
        combined = torch.cat([question_repr, answer_repr], dim=-1)
        
        # Shared projection with residual
        proj = self.shared_proj(combined)
        residual = self.shared_residual(combined)
        shared = proj + residual
        
        # Second layer
        shared = self.shared_layer2(shared)
        
        # Gated outputs for each head
        consistent_gate = self.gate_consistent(shared)
        correct_gate = self.gate_correct(shared)
        useful_gate = self.gate_useful(shared)
        
        consistent_out = shared * consistent_gate
        correct_out = shared * correct_gate
        useful_out = shared * useful_gate
        
        return {
            'consistent': self.head_consistent(consistent_out).squeeze(-1),
            'correct': self.head_correct(correct_out).squeeze(-1),
            'useful': self.head_useful(useful_out).squeeze(-1)
        }


class ContrastiveLearningHead(nn.Module):
    """Auxiliary contrastive learning head for better representations."""
    
    def __init__(self, hidden_size: int, projection_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(x), dim=-1)


class EnhancedLLMClassifier(nn.Module):
    """
    Enhanced LLM Classifier with cross-attention and contrastive learning.
    """
    
    SUPPORTED_MODELS = {
        'codebert': 'microsoft/codebert-base',
        'graphcodebert': 'microsoft/graphcodebert-base',
        'unixcoder': 'microsoft/unixcoder-base',
        'codet5': 'Salesforce/codet5-base',
        'codebert-mlm': 'microsoft/codebert-base-mlm',
        'roberta-base': 'roberta-base',
    }
    
    def __init__(
        self,
        model_name: str = 'codebert',
        dropout: float = 0.3,
        freeze_encoder_layers: int = 4,
        max_length: int = 256,
        use_cross_attention: bool = True,
        use_contrastive: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.use_cross_attention = use_cross_attention
        self.use_contrastive = use_contrastive
        
        # Resolve model path
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        
        print(f"[EnhancedLLMClassifier] Loading model: {model_path}")
        
        # Load tokenizer and encoder
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
            print(f"[EnhancedLLMClassifier] Loaded from cache")
        except Exception as e:
            print(f"[EnhancedLLMClassifier] Cache load failed: {e}, trying network")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.config = AutoConfig.from_pretrained(model_path)
            self.encoder = AutoModel.from_pretrained(model_path)
        
        hidden_size = getattr(self.config, 'hidden_size', 768)
        self.hidden_size = hidden_size
        
        # Freeze early layers
        if freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
            print("[EnhancedLLMClassifier] Gradient checkpointing enabled")
        
        # Attention pooling
        self.question_pooling = AttentionPooling(hidden_size)
        self.answer_pooling = AttentionPooling(hidden_size)
        
        # Cross-attention layers
        if use_cross_attention:
            self.q_to_a_attention = CrossAttentionLayer(hidden_size, dropout=dropout)
            self.a_to_q_attention = CrossAttentionLayer(hidden_size, dropout=dropout)
        
        # Classification head
        self.classifier = EnhancedClassificationHead(hidden_size, dropout)
        
        # Contrastive head
        if use_contrastive:
            self.contrastive_head = ContrastiveLearningHead(hidden_size)
        
        self.to(device)
    
    def _freeze_encoder_layers(self, num_layers: int):
        """Freeze the first N encoder layers."""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze layers
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            layers = self.encoder.encoder.layer
            for i, layer in enumerate(layers):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        print(f"[EnhancedLLMClassifier] Frozen {num_layers} encoder layers")
    
    def encode_sequence(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Encode a sequence of texts."""
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
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        question_token_type_ids: Optional[torch.Tensor] = None,
        answer_token_type_ids: Optional[torch.Tensor] = None,
        return_contrastive: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with separate question and answer encoding.
        """
        # Encode question
        q_inputs = {'input_ids': question_input_ids, 'attention_mask': question_attention_mask}
        if question_token_type_ids is not None and hasattr(self.config, 'type_vocab_size'):
            q_inputs['token_type_ids'] = question_token_type_ids
        q_outputs = self.encoder(**q_inputs)
        q_hidden = q_outputs.last_hidden_state  # [batch, seq_q, hidden]
        
        # Encode answer
        a_inputs = {'input_ids': answer_input_ids, 'attention_mask': answer_attention_mask}
        if answer_token_type_ids is not None and hasattr(self.config, 'type_vocab_size'):
            a_inputs['token_type_ids'] = answer_token_type_ids
        a_outputs = self.encoder(**a_inputs)
        a_hidden = a_outputs.last_hidden_state  # [batch, seq_a, hidden]
        
        # Cross-attention if enabled
        if self.use_cross_attention:
            # Question attends to answer
            q_enhanced = self.q_to_a_attention(
                q_hidden, a_hidden, 
                key_padding_mask=(answer_attention_mask == 0)
            )
            # Answer attends to question
            a_enhanced = self.a_to_q_attention(
                a_hidden, q_hidden,
                key_padding_mask=(question_attention_mask == 0)
            )
        else:
            q_enhanced = q_hidden
            a_enhanced = a_hidden
        
        # Attention pooling
        q_pooled = self.question_pooling(q_enhanced, question_attention_mask)
        a_pooled = self.answer_pooling(a_enhanced, answer_attention_mask)
        
        # Classification
        logits = self.classifier(q_pooled, a_pooled)
        
        # Contrastive embeddings
        if return_contrastive and self.use_contrastive:
            q_proj = self.contrastive_head(q_pooled)
            a_proj = self.contrastive_head(a_pooled)
            logits['q_contrastive'] = q_proj
            logits['a_contrastive'] = a_proj
        
        return logits
    
    def encode_qa_pair(
        self, 
        questions: List[str], 
        answers: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize question-answer pairs separately."""
        q_encoded = self.encode_sequence(questions)
        a_encoded = self.encode_sequence(answers)
        
        return {
            'question_input_ids': q_encoded['input_ids'],
            'question_attention_mask': q_encoded['attention_mask'],
            'question_token_type_ids': q_encoded.get('token_type_ids'),
            'answer_input_ids': a_encoded['input_ids'],
            'answer_attention_mask': a_encoded['attention_mask'],
            'answer_token_type_ids': a_encoded.get('token_type_ids'),
        }
    
    def predict(
        self, 
        questions: List[str], 
        answers: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Predict quality scores for QA pairs."""
        self.eval()
        with torch.no_grad():
            encoded = self.encode_qa_pair(questions, answers)
            # Filter out None values
            encoded = {k: v for k, v in encoded.items() if v is not None}
            logits = self.forward(**encoded)
            
            return {
                'consistent': torch.sigmoid(logits['consistent']),
                'correct': torch.sigmoid(logits['correct']),
                'useful': torch.sigmoid(logits['useful'])
            }
    
    def save_pretrained(self, path: str):
        """Save model."""
        import os
        os.makedirs(path, exist_ok=True)
        
        self.encoder.save_pretrained(os.path.join(path, 'encoder'))
        self.tokenizer.save_pretrained(os.path.join(path, 'tokenizer'))
        
        torch.save({
            'question_pooling': self.question_pooling.state_dict(),
            'answer_pooling': self.answer_pooling.state_dict(),
            'classifier': self.classifier.state_dict(),
            'q_to_a_attention': self.q_to_a_attention.state_dict() if self.use_cross_attention else None,
            'a_to_q_attention': self.a_to_q_attention.state_dict() if self.use_cross_attention else None,
            'contrastive_head': self.contrastive_head.state_dict() if self.use_contrastive else None,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'use_cross_attention': self.use_cross_attention,
            'use_contrastive': self.use_contrastive,
        }, os.path.join(path, 'classifier_head.pt'))
        
        print(f"[EnhancedLLMClassifier] Model saved to {path}")


class CodeBERTEnhancedClassifier(EnhancedLLMClassifier):
    """CodeBERT-based enhanced classifier."""
    
    def __init__(
        self, 
        dropout: float = 0.3, 
        freeze_layers: int = 4, 
        device: str = 'cuda',
        use_cross_attention: bool = True,
        use_contrastive: bool = True
    ):
        super().__init__(
            model_name='codebert',
            dropout=dropout,
            freeze_encoder_layers=freeze_layers,
            use_cross_attention=use_cross_attention,
            use_contrastive=use_contrastive,
            device=device
        )


class GraphCodeBERTEnhancedClassifier(EnhancedLLMClassifier):
    """GraphCodeBERT-based enhanced classifier."""
    
    def __init__(
        self, 
        dropout: float = 0.3, 
        freeze_layers: int = 4, 
        device: str = 'cuda',
        use_cross_attention: bool = True,
        use_contrastive: bool = True
    ):
        super().__init__(
            model_name='graphcodebert',
            dropout=dropout,
            freeze_encoder_layers=freeze_layers,
            use_cross_attention=use_cross_attention,
            use_contrastive=use_contrastive,
            device=device
        )

