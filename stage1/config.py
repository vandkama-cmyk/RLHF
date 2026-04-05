"""
Stage 1 RLHF Configuration
==========================

Baseline RLHF setup using:
- CodeGPT-small as the policy model
- CodeBERT-based preference reward model (Bradley–Terry)
- Policy training via Reward-Weighted NLL (not PPO)

This stage verifies the end-to-end pipeline from data preparation,
reward estimation, to policy optimization operates correctly.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Stage1ModelConfig:
    """Model configuration for Stage 1."""
    
    # Base model: CodeGPT-small
    base_model_name: str = "microsoft/CodeGPT-small-py"
    # Reward model: CodeBERT-base
    reward_model_name: str = "microsoft/codebert-base"
    
    policy_model_size: str = "small"
    reward_model_size: str = "base"
    
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    torch_dtype: str = "float16"
    
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12


@dataclass
class Stage1TrainingConfig:
    """Training configuration for Stage 1."""
    
    # Training parameters
    learning_rate: float = 5e-6
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Used by Stage1B (full PPO) only — RewardWeightedNLL ignores these
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    ppo_value_loss_coef: float = 0.1
    ppo_entropy_coef: float = 0.01
    ppo_kl_penalty: float = 0.02
    
    # Total training epochs
    total_epochs: int = 30
    
    # Training schedule
    warmup_steps: int = 100
    total_steps: int = 1000
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    
    # Early stopping disabled for reproducibility
    early_stopping_patience: int = 100
    early_stopping_threshold: float = 0.001


@dataclass
class Stage1GenerationConfig:
    """Generation configuration for Stage 1."""
    
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    do_sample: bool = True
    
    max_prompt_length: int = 256
    max_response_length: int = 512
    min_code_length: int = 20
    
    num_beams: int = 4
    num_return_sequences: int = 1
    early_stopping: bool = True


@dataclass
class Stage1RewardConfig:
    """Reward configuration for Stage 1."""
    
    reward_learning_rate: float = 2e-5
    reward_batch_size: int = 32
    reward_epochs: int = 1
    
    # Preference reward model (Bradley–Terry on encoder + scalar head)
    train_preference_reward: bool = True
    preference_max_pairs: int = 2000
    preference_epochs: int = 2
    preference_lr: float = 2e-5
    reward_checkpoint_name: str = "preference_reward.pt"
    force_retrain_reward: bool = False
    # Optional CSV with columns: prompt, chosen, rejected (or winner/loser)
    preference_csv_path: Optional[str] = None
    
    human_feedback_weight: float = 0.3
    use_human_logits: bool = True
    
    syntax_reward_weight: float = 0.2
    execution_reward_weight: float = 0.3
    semantic_reward_weight: float = 0.3
    human_preference_weight: float = 0.2
    
    reward_normalization: bool = True
    reward_clipping: bool = True
    reward_clip_value: float = 5.0


@dataclass
class Stage1EvaluationConfig:
    """Evaluation configuration for Stage 1."""
    
    # Target metrics (baseline targets)
    target_bertscore: float = 0.75
    target_codebleu: float = 0.20
    target_bleu: float = 0.05
    target_rouge: float = 0.15
    target_ruby: float = 0.15
    
    eval_batch_size: int = 8
    eval_samples: int = 100
    
    # Evaluation datasets (if loading from CSV files)
    eval_datasets: List[str] = field(default_factory=lambda: [])


@dataclass
class Stage1DataConfig:
    """Data configuration for Stage 1."""
    
    train_data_path: str = "./datasets_for_training"
    eval_data_path: str = "./datasets_for_eval"
    human_feedback_path: str = "./stage1/human_feedback"
    output_path: str = "./stage1/output"
    conala_local_path: Optional[str] = "./conala-corpus"  # Local CoNaLa data
    
    max_train_samples: int = 2000
    max_eval_samples: int = 200
    train_test_split: float = 0.9
    
    # Data filtering
    min_prompt_length: int = 10
    max_prompt_length: int = 512
    min_response_length: int = 5
    max_response_length: int = 512
    require_code_like: bool = True
    use_data_augmentation: bool = False
    augmentation_ratio: float = 0.1


@dataclass
class Stage1HardwareConfig:
    """Hardware configuration for Stage 1."""
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True


@dataclass
class Stage1Config:
    """Main configuration for Stage 1 experiment."""
    
    model: Stage1ModelConfig = field(default_factory=Stage1ModelConfig)
    training: Stage1TrainingConfig = field(default_factory=Stage1TrainingConfig)
    generation: Stage1GenerationConfig = field(default_factory=Stage1GenerationConfig)
    reward: Stage1RewardConfig = field(default_factory=Stage1RewardConfig)
    evaluation: Stage1EvaluationConfig = field(default_factory=Stage1EvaluationConfig)
    data: Stage1DataConfig = field(default_factory=Stage1DataConfig)
    hardware: Stage1HardwareConfig = field(default_factory=Stage1HardwareConfig)
    
    # Reproducibility
    seed: int = 42
    debug: bool = False
    verbose: bool = True
    
    experiment_name: str = "stage1_baseline_rlhf"
    
    def __post_init__(self):
        """Post-initialization setup."""
        os.makedirs(self.data.output_path, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "generation": self.generation.__dict__,
            "reward": self.reward.__dict__,
            "evaluation": self.evaluation.__dict__,
            "data": self.data.__dict__,
            "hardware": self.hardware.__dict__,
            "seed": self.seed,
            "debug": self.debug,
            "verbose": self.verbose,
            "experiment_name": self.experiment_name
        }


def get_stage1_config() -> Stage1Config:
    """Get the Stage 1 baseline configuration."""
    return Stage1Config()
