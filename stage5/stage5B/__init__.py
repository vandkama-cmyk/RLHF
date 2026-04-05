"""
ClassifLLM v2 - Enhanced LLM-based Code Quality Classification Pipeline

Key improvements over v1:
1. Cross-attention between question and answer encodings
2. Attention-based pooling instead of simple mean pooling
3. Contrastive learning auxiliary objective
4. Mixed precision training (AMP)
5. Gradient accumulation for larger effective batch sizes
6. Label smoothing for better generalization
7. EMA (Exponential Moving Average) for stable predictions
8. Better metrics with balanced accuracy and AUC

Usage:
    from clasifLLM.v2 import EnhancedLLMClassifier, EnhancedLLMTrainingPipeline
    
    # Create model
    model = EnhancedLLMClassifier(
        model_name='codebert',
        use_cross_attention=True,
        use_contrastive=True
    )
    
    # Predict
    scores = model.predict(
        questions=["How to sort a list?"],
        answers=["sorted_list = sorted(my_list)"]
    )
"""

from .model import (
    EnhancedLLMClassifier,
    CodeBERTEnhancedClassifier,
    GraphCodeBERTEnhancedClassifier,
    AttentionPooling,
    CrossAttentionLayer,
    EnhancedClassificationHead,
    ContrastiveLearningHead,
)

from .integrated_system import (
    EnhancedLLMTrainingPipeline,
    train_llm_system_v2,
    LabelSmoothingBCELoss,
    InfoNCELoss,
    EMA,
)

__all__ = [
    # Models
    'EnhancedLLMClassifier',
    'CodeBERTEnhancedClassifier',
    'GraphCodeBERTEnhancedClassifier',
    
    # Components
    'AttentionPooling',
    'CrossAttentionLayer',
    'EnhancedClassificationHead',
    'ContrastiveLearningHead',
    
    # Training
    'EnhancedLLMTrainingPipeline',
    'train_llm_system_v2',
    
    # Losses
    'LabelSmoothingBCELoss',
    'InfoNCELoss',
    
    # Utils
    'EMA',
]

