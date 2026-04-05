"""
Stage 2B Configuration
======================

Configuration for Stage 2B GPT-2 feedback generator experiment.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class FeedbackGeneratorConfig:
    """GPT-2 Feedback Generator configuration."""
    model_name: str = "gpt2"  # GPT-2 base model (117M params)
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Feedback criteria
    criteria: List[str] = field(default_factory=lambda: [
        'consistency', 'agreement', 'usefulness'
    ])
    
    # Score generation
    score_bins: int = 5  # Scores 1-5
    use_prompt_template: bool = True


@dataclass
class RewardModelConfig:
    """Reward Model configuration."""
    base_model_name: str = "microsoft/codebert-base"
    hidden_dim: int = 768
    num_heads: int = 3  # Consistency, Agreement, Usefulness
    dropout: float = 0.3
    freeze_encoder_layers: int = 4
    max_length: int = 256


@dataclass
class TrainingConfig:
    """Training configuration."""
    total_epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    
    # Optimization
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    label_smoothing: float = 0.1
    
    # Feedback loop
    feedback_update_freq: int = 5  # Update feedback every N batches
    use_synthetic_feedback: bool = True
    feedback_weight: float = 0.3


@dataclass
class DataConfig:
    """Data configuration."""
    # Primary data source
    conala_path: str = "conala-corpus/conala-train.jsonl"
    conala_test_path: str = "conala-corpus/conala-test.jsonl"
    
    # Additional data
    sft_data_path: str = "datasets_for_training/sft_dataset.csv"
    eval_dir: str = "clasifNN/datasets_for_eval"
    
    # Output paths
    output_path: str = "stage_2B/outputs"
    artifacts_path: str = "stage_2B/artifacts"
    
    # Data limits
    max_train_samples: int = 5000
    max_eval_samples: int = 500
    val_ratio: float = 0.1


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_samples: int = 200
    eval_batch_size: int = 8
    metrics: List[str] = field(default_factory=lambda: [
        'bertscore', 'bleu', 'codebleu', 'rouge', 'ruby'
    ])
    
    # Target metrics (from paper)
    target_bertscore: float = 0.90
    target_codebleu: float = 0.80


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class Stage2BConfig:
    """Complete Stage 2B configuration."""
    experiment_name: str = "stage2b_gpt2_feedback_reward_model"
    seed: int = 42
    
    feedback_generator: FeedbackGeneratorConfig = field(default_factory=FeedbackGeneratorConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'feedback_generator': {
                'model_name': self.feedback_generator.model_name,
                'max_length': self.feedback_generator.max_length,
                'temperature': self.feedback_generator.temperature,
                'top_p': self.feedback_generator.top_p,
                'criteria': self.feedback_generator.criteria,
            },
            'reward_model': {
                'base_model_name': self.reward_model.base_model_name,
                'hidden_dim': self.reward_model.hidden_dim,
                'num_heads': self.reward_model.num_heads,
                'dropout': self.reward_model.dropout,
                'freeze_encoder_layers': self.reward_model.freeze_encoder_layers,
            },
            'training': {
                'total_epochs': self.training.total_epochs,
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
                'use_amp': self.training.use_amp,
                'feedback_weight': self.training.feedback_weight,
            },
            'data': {
                'conala_path': self.data.conala_path,
                'output_path': self.data.output_path,
                'max_train_samples': self.data.max_train_samples,
            },
            'evaluation': {
                'metrics': self.evaluation.metrics,
                'target_bertscore': self.evaluation.target_bertscore,
                'target_codebleu': self.evaluation.target_codebleu,
            },
            'hardware': {
                'device': self.hardware.device,
            }
        }


def get_stage2b_config() -> Stage2BConfig:
    """Get default Stage 2B configuration."""
    return Stage2BConfig()
