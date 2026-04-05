"""
Stage 2A Configuration
======================

Configuration for Stage 2A contrastive learning experiment.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Model configuration."""
    base_model_name: str = "microsoft/codebert-base"  # CodeBERT for reward model
    policy_model_name: str = "microsoft/CodeGPT-small-py"  # Policy for generation
    embedding_dim: int = 768  # CodeBERT hidden size
    projection_dim: int = 256  # Contrastive projection dimension
    dropout: float = 0.1  # Reduced from 0.3 - was causing underfitting
    freeze_encoder_layers: int = 0  # Reduced from 4 - allow full adaptation
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    max_length: int = 256


@dataclass
class ContrastiveConfig:
    """Contrastive learning configuration."""
    # Maximum contrastive loss parameters
    margin: float = 0.5  # Reduced margin for easier learning
    temperature: float = 0.1  # Slightly higher temperature for stability
    
    # Loss weights - reward is primary objective
    contrastive_weight: float = 0.2  # Reduced - auxiliary loss only
    reward_weight: float = 0.8  # Increased - primary learning objective
    
    # Pair mining
    hard_negative_mining: bool = True
    similarity_threshold: float = 0.5  # Threshold for similar/dissimilar


@dataclass
class TrainingConfig:
    """Training configuration."""
    total_epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 5e-5  # Increased from 2e-5 for better learning
    head_learning_rate: float = 1e-3  # Much higher LR for fresh heads
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    
    # Optimization
    use_amp: bool = True  # Mixed precision
    use_ema: bool = False  # Disabled - was slowing learning
    ema_decay: float = 0.99  # Lower decay if enabled
    label_smoothing: float = 0.1
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001


@dataclass
class DataConfig:
    """Data configuration."""
    # Dataset paths
    train_data_path: str = "datasets_for_training/sft_dataset.csv"
    extra_sft_jsonl_paths: List[str] = field(default_factory=lambda: [
        "conala-corpus/conala-train.jsonl",
        "conala-corpus/conala-test.jsonl"
    ])
    eval_data_path: str = "stage4/datasets_for_eval"
    feedback_data_path: str = "stage4/evaluation_results_server"
    
    # Output paths
    output_path: str = "stage2/stage2A/outputs"
    artifacts_path: str = "stage2/stage2A/artifacts"
    
    # Data limits
    max_train_samples: int = 8000  # Increased for better learning
    max_eval_samples: int = 500
    val_ratio: float = 0.1
    negative_ratio: float = 1.0  # 1:1 positive:negative ratio
    
    # Filtering
    min_prompt_length: int = 10
    max_prompt_length: int = 512
    min_response_length: int = 5
    max_response_length: int = 256


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_samples: int = 200
    eval_batch_size: int = 8
    metrics: List[str] = field(default_factory=lambda: [
        'bertscore', 'bleu', 'codebleu', 'rouge', 'ruby'
    ])


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class Stage2AConfig:
    """Complete Stage 2A configuration."""
    experiment_name: str = "stage2a_contrastive_reward_model"
    seed: int = 42
    
    model: ModelConfig = field(default_factory=ModelConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'model': {
                'base_model_name': self.model.base_model_name,
                'policy_model_name': self.model.policy_model_name,
                'embedding_dim': self.model.embedding_dim,
                'projection_dim': self.model.projection_dim,
                'dropout': self.model.dropout,
                'freeze_encoder_layers': self.model.freeze_encoder_layers,
                'max_length': self.model.max_length,
            },
            'contrastive': {
                'margin': self.contrastive.margin,
                'temperature': self.contrastive.temperature,
                'contrastive_weight': self.contrastive.contrastive_weight,
                'reward_weight': self.contrastive.reward_weight,
                'hard_negative_mining': self.contrastive.hard_negative_mining,
                'similarity_threshold': self.contrastive.similarity_threshold,
            },
            'training': {
                'total_epochs': self.training.total_epochs,
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'head_learning_rate': self.training.head_learning_rate,
                'weight_decay': self.training.weight_decay,
                'warmup_ratio': self.training.warmup_ratio,
                'max_grad_norm': self.training.max_grad_norm,
                'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
                'use_amp': self.training.use_amp,
                'use_ema': self.training.use_ema,
                'ema_decay': self.training.ema_decay,
                'label_smoothing': self.training.label_smoothing,
            },
            'data': {
                'train_data_path': self.data.train_data_path,
                'extra_sft_jsonl_paths': self.data.extra_sft_jsonl_paths,
                'output_path': self.data.output_path,
                'max_train_samples': self.data.max_train_samples,
                'max_eval_samples': self.data.max_eval_samples,
                'val_ratio': self.data.val_ratio,
                'negative_ratio': self.data.negative_ratio,
            },
            'evaluation': {
                'eval_samples': self.evaluation.eval_samples,
                'eval_batch_size': self.evaluation.eval_batch_size,
                'metrics': self.evaluation.metrics,
            },
            'hardware': {
                'device': self.hardware.device,
                'num_workers': self.hardware.num_workers,
            }
        }


def get_stage2a_config() -> Stage2AConfig:
    """Get default Stage 2A configuration."""
    return Stage2AConfig()
