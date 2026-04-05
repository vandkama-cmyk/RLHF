"""
Modern Reward Model with Human Feedback Integration
=================================================

A state-of-the-art reward model that combines multiple signals:
- Syntax correctness
- Execution success
- Semantic similarity
- Human preference feedback
- Code quality metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import warnings
import numpy as np
from dataclasses import dataclass
import json
import os
import ast
import subprocess
import tempfile
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    PreTrainedModel, PreTrainedTokenizer
)
from typing import List, Dict, Any, Optional, Tuple, Union

# Suppress transformers warnings about uninitialized weights
# This is expected when fine-tuning models (we add custom heads)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN.*")

from .metrics import ModernMetricsEvaluator, CodeQualityAnalyzer
from .config import RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Container for different reward components."""
    syntax_reward: float = 0.0
    execution_reward: float = 0.0
    semantic_reward: float = 0.0
    human_preference_reward: float = 0.0
    quality_reward: float = 0.0
    total_reward: float = 0.0


class HumanFeedbackIntegrator:
    """Integrates human feedback into reward computation."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.human_logits_cache = {}
        self.feedback_weights = {}
        
    def load_human_feedback(self, feedback_path_or_items: Union[str, List[Dict[str, Any]]]):
        """Load human feedback data from a file path or directly from a list of items.

        Accepts either a path to a JSON file or a list of feedback dicts. This makes
        it robust to being called with the in-memory list produced by the data loader.
        """
        try:
            # Determine input type
            items: List[Dict[str, Any]] = []
            if isinstance(feedback_path_or_items, list):
                items = feedback_path_or_items
            else:
                feedback_path = str(feedback_path_or_items)
                if os.path.exists(feedback_path):
                    with open(feedback_path, 'r', encoding='utf-8') as f:
                        feedback_data = json.load(f)
                    # Normalize to list of dicts
                    if isinstance(feedback_data, list):
                        items = feedback_data
                    elif isinstance(feedback_data, dict):
                        for key in ['data', 'items', 'examples', 'feedback']:
                            v = feedback_data.get(key)
                            if isinstance(v, list):
                                items = v
                                break
                        if not items:
                            # Try to interpret dict values as entries
                            for v in feedback_data.values():
                                if isinstance(v, dict):
                                    items.append(v)

            # Process feedback data
            for item in items:
                if not isinstance(item, dict):
                    continue
                prompt = item.get('prompt', '')
                response = item.get('response', '')
                rating = item.get('rating', 0.0)
                logits = item.get('logits', None)

                # Store feedback data (with or without logits)
                if prompt and response:  # Only need prompt and response
                    key = f"{prompt[:50]}_{response[:50]}"
                    self.human_logits_cache[key] = {
                        'logits': logits,  # May be None
                        'rating': rating
                    }

            logger.info(f"Loaded {len(self.human_logits_cache)} human feedback entries with ratings")

        except Exception as e:
            logger.warning(f"Failed to load human feedback: {e}")
    
    def get_human_logits(self, prompt: str, response: str) -> Optional[torch.Tensor]:
        """Get human logits for a prompt-response pair."""
        key = f"{prompt[:50]}_{response[:50]}"
        
        if key in self.human_logits_cache:
            logits_data = self.human_logits_cache[key]['logits']
            if isinstance(logits_data, list):
                return torch.tensor(logits_data, dtype=torch.float32)
            elif isinstance(logits_data, dict):
                # Handle different logits formats
                if 'last_layer' in logits_data:
                    return torch.tensor(logits_data['last_layer'], dtype=torch.float32)
                elif 'logits' in logits_data:
                    return torch.tensor(logits_data['logits'], dtype=torch.float32)
        
        return None
    
    def compute_human_preference_reward(self, prompt: str, response: str) -> float:
        """Compute reward based on human preferences."""
        key = f"{prompt[:50]}_{response[:50]}"
        
        if key in self.human_logits_cache:
            rating = self.human_logits_cache[key]['rating']
            # Normalize rating to [0, 1] range
            return max(0.0, min(1.0, rating / 5.0))  # Assuming 5-point scale
        
        return 0.5  # Neutral reward if no human feedback available


class SyntaxChecker:
    """Advanced syntax checking for code."""
    
    def __init__(self):
        self.supported_languages = ['python', 'javascript', 'java', 'cpp']
    
    def check_syntax(self, code: str, language: str = 'python') -> Tuple[bool, float, str]:
        """Check syntax correctness of code."""
        if language == 'python':
            return self._check_python_syntax(code)
        else:
            # For other languages, use basic parsing
            return self._check_generic_syntax(code)
    
    def _check_python_syntax(self, code: str) -> Tuple[bool, float, str]:
        """Check Python syntax."""
        try:
            ast.parse(code)
            return True, 1.0, ""
        except SyntaxError as e:
            return False, 0.0, str(e)
        except Exception as e:
            return False, 0.0, f"Parse error: {str(e)}"
    
    def _check_generic_syntax(self, code: str) -> Tuple[bool, float, str]:
        """Generic syntax checking."""
        # Basic checks
        if not code.strip():
            return False, 0.0, "Empty code"
        
        # Check for balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False, 0.0, "Unbalanced brackets"
                if brackets[stack.pop()] != char:
                    return False, 0.0, "Unbalanced brackets"
        
        if stack:
            return False, 0.0, "Unbalanced brackets"
        
        return True, 0.8, ""  # Partial credit for basic structure


class ExecutionTester:
    """Test code execution in a safe environment."""
    
    def __init__(self):
        self.timeout = 5  # seconds
        self.max_memory = 100 * 1024 * 1024  # 100MB
    
    def test_execution(self, code: str, language: str = 'python') -> Tuple[bool, float, str]:
        """Test if code can be executed successfully."""
        if language == 'python':
            return self._test_python_execution(code)
        else:
            return False, 0.0, f"Execution testing not supported for {language}"
    
    def _test_python_execution(self, code: str) -> Tuple[bool, float, str]:
        """Test Python code execution."""
        try:
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'bool': bool,
                    'type': type,
                    'isinstance': isinstance,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                }
            }
            
            # Try to execute
            exec(code, safe_globals)
            return True, 1.0, ""
            
        except Exception as e:
            return False, 0.0, str(e)


class ModernRewardModel(nn.Module):
    """Modern reward model with multiple signal integration."""
    
    def __init__(self, config: RewardConfig, model_name: str = "microsoft/codebert-base", device: Optional[str] = None):
        super().__init__()
        self.config = config
        self.model_name = model_name
        
        # Determine device - force GPU if available
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"RewardModel: Using GPU {torch.cuda.get_device_name(0)}")
            else:
                raise RuntimeError("GPU is required for training but CUDA is not available!")
        else:
            self.device = torch.device(device)
        
        # Load base model
        # Suppress warnings about uninitialized weights (expected for fine-tuning)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")
            warnings.filterwarnings("ignore", message=".*You should probably TRAIN.*")
            self.base_model = AutoModel.from_pretrained(
                model_name,
                ignore_mismatched_sizes=False,  # Keep False to detect real mismatches
                device_map=None,  # Отключить автоматическое распределение
                low_cpu_mem_usage=False  # Отключить оптимизации памяти
            )
        self.base_model = self.base_model.to(self.device)  # Move to GPU immediately
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.tokenizer.padding_side = 'left'
        except (AttributeError, ValueError) as e:
            logger.debug(f"Could not set padding_side: {e}")
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Reward head
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Component-specific heads
        self.syntax_head = nn.Linear(hidden_size, 1)
        self.execution_head = nn.Linear(hidden_size, 1)
        self.semantic_head = nn.Linear(hidden_size, 1)
        self.quality_head = nn.Linear(hidden_size, 1)
        
        # Human feedback integration
        self.human_feedback_integrator = HumanFeedbackIntegrator(config)
        
        # Utility components
        self.syntax_checker = SyntaxChecker()
        self.execution_tester = ExecutionTester()
        self.metrics_evaluator = ModernMetricsEvaluator()
        self.code_analyzer = CodeQualityAnalyzer()
        
        # Initialize weights
        self._init_weights()
        
        # Ensure model is on GPU
        self.to(self.device)
        
        # Verify GPU usage
        if next(self.parameters()).is_cuda:
            logger.info(f"RewardModel successfully loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            raise RuntimeError(f"RewardModel is not on GPU! Current device: {next(self.parameters()).device}")
    
    def _init_weights(self):
        """Initialize model weights with numerical stability considerations."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use more conservative initialization for reward heads
                if 'reward_head' in name or 'head' in name:
                    # Smaller initialization for reward heads to prevent extreme values
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    # Standard initialization for other layers
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the correct device.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor on the correct device
        """
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor
    
    def forward(self, prompts: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        """Forward pass through the reward model with numerical stability."""
        # Validate inputs
        if not prompts or not responses:
            raise ValueError("prompts and responses cannot be empty")
        if len(prompts) != len(responses):
            raise ValueError(f"prompts ({len(prompts)}) and responses ({len(responses)}) must have the same length")
        
        # Tokenize inputs
        inputs = self._tokenize_pairs(prompts, responses)
        
        # Get base model outputs
        # NOTE: base_model is frozen (no_grad) - only reward heads are trainable
        with torch.no_grad():
            outputs = self.base_model(**inputs)
        
        # Get pooled representation
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Normalize pooled output for numerical stability (prevent extreme values)
        # CRITICAL: Normalization must be outside no_grad to preserve gradients for reward heads
        # Normalize across batch dimension to preserve relative differences between samples
        epsilon = getattr(self.config, 'reward_normalization_epsilon', 1e-6)
        mean = pooled_output.mean(dim=0, keepdim=True)  # Mean across batch
        std = pooled_output.std(dim=0, keepdim=True)    # Std across batch
        std = torch.clamp(std, min=epsilon)  # More robust minimum to prevent extreme values
        pooled_output = (pooled_output - mean) / std
        
        # Compute different reward components
        rewards = {}
        rewards['total'] = self.reward_head(pooled_output)
        rewards['syntax'] = self.syntax_head(pooled_output)
        rewards['execution'] = self.execution_head(pooled_output)
        rewards['semantic'] = self.semantic_head(pooled_output)
        rewards['quality'] = self.quality_head(pooled_output)
        
        # Apply numerical stability: clamp rewards to prevent extreme values
        # This prevents numerical instability during training
        clip_min = getattr(self.config, 'reward_clip_min', -10.0)
        clip_max = getattr(self.config, 'reward_clip_max', 10.0)
        for key in rewards:
            rewards[key] = torch.clamp(rewards[key], clip_min, clip_max)
        
        return rewards
    
    def _tokenize_pairs(self, prompts: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize prompt-response pairs."""
        # Combine prompts and responses
        texts = [f"{prompt} <SEP> {response}" for prompt, response in zip(prompts, responses)]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def compute_reward_components(self, prompts: List[str], responses: List[str]) -> List[RewardComponents]:
        """Compute detailed reward components for each prompt-response pair."""
        components_list = []
        
        for prompt, response in zip(prompts, responses):
            components = RewardComponents()
            
            # Syntax reward
            syntax_correct, syntax_score, syntax_error = self.syntax_checker.check_syntax(response)
            components.syntax_reward = syntax_score
            
            # Execution reward
            exec_success, exec_score, exec_error = self.execution_tester.test_execution(response)
            components.execution_reward = exec_score
            
            # Semantic reward (using BERTScore)
            try:
                semantic_result = self.metrics_evaluator.compute_bertscore([response], [prompt])
                components.semantic_reward = semantic_result.score
            except Exception as e:
                logger.warning(f"Semantic reward computation failed: {e}")
                components.semantic_reward = 0.0
            
            # Human preference reward
            components.human_preference_reward = self.human_feedback_integrator.compute_human_preference_reward(
                prompt, response
            )
            
            # Quality reward
            try:
                quality_analysis = self.code_analyzer.analyze_complexity(response)
                style_analysis = self.code_analyzer.analyze_style(response)
                components.quality_reward = (
                    quality_analysis['complexity_score'] * 0.6 +
                    style_analysis['style_score'] * 0.4
                )
            except Exception as e:
                logger.warning(f"Quality reward computation failed: {e}")
                components.quality_reward = 0.0
            
            # Compute total reward
            components.total_reward = (
                components.syntax_reward * self.config.syntax_reward_weight +
                components.execution_reward * self.config.execution_reward_weight +
                components.semantic_reward * self.config.semantic_reward_weight +
                components.human_preference_reward * self.config.human_preference_weight +
                components.quality_reward * 0.1  # Small weight for quality
            )
            
            components_list.append(components)
        
        return components_list
    
    def compute_reward(self, prompts: List[str], responses: List[str], use_fast_mode: bool = True) -> torch.Tensor:
        """Compute final reward scores.
        
        Args:
            use_fast_mode: If True, use only neural network (10x faster).
                          If False, use full component-based rewards (slow but comprehensive).
        """
        # Get neural network predictions
        neural_rewards = self.forward(prompts, responses)
        
        if use_fast_mode:
            # FAST MODE: Use only neural network rewards (for training)
            # This is 10x faster and sufficient for reward model training
            total_rewards = neural_rewards['total'].squeeze()
            
            # Check for NaN/Inf before processing
            if torch.isnan(total_rewards).any() or torch.isinf(total_rewards).any():
                logger.warning("NaN/Inf detected in rewards, replacing with zeros")
                total_rewards = torch.where(
                    torch.isnan(total_rewards) | torch.isinf(total_rewards),
                    torch.zeros_like(total_rewards),
                    total_rewards
                )
            
            # Apply clipping first, then normalization
            if self.config.reward_clipping:
                total_rewards = torch.clamp(
                    total_rewards,
                    -self.config.reward_clip_value,
                    self.config.reward_clip_value
                )

            # Apply normalization after clipping
            if self.config.reward_normalization:
                total_rewards = torch.sigmoid(total_rewards)
            
            return total_rewards.to(self.device)
        
        else:
            # SLOW MODE: Use full component-based rewards (for evaluation/inference)
            # Get component-based rewards
            component_rewards = self.compute_reward_components(prompts, responses)
            
            # Combine neural and component rewards
            final_reward_tensors: List[torch.Tensor] = []
            total_tensor = neural_rewards['total']
            for i, component_reward in enumerate(component_rewards):
                neural_reward = total_tensor[i]

                # Weighted combination: keep tensor operations to preserve gradients
                combined_reward = neural_reward * 0.7 + float(component_reward.total_reward) * 0.3
                
                # Check for NaN/Inf before processing
                if torch.isnan(combined_reward) or torch.isinf(combined_reward):
                    logger.warning(f"NaN/Inf detected in combined_reward at index {i}, replacing with zero")
                    combined_reward = torch.tensor(0.0, device=self.device, dtype=torch.float32)

                # Apply clipping first, then normalization
                if self.config.reward_clipping:
                    combined_reward = torch.clamp(
                        combined_reward,
                        -self.config.reward_clip_value,
                        self.config.reward_clip_value
                    )

                # Apply normalization after clipping
                if self.config.reward_normalization:
                    combined_reward = torch.sigmoid(combined_reward)

                final_reward_tensors.append(combined_reward)

            if final_reward_tensors:
                result = torch.stack(final_reward_tensors).view(-1).to(self.device)
                # Final check for NaN/Inf in result
                if torch.isnan(result).any() or torch.isinf(result).any():
                    logger.warning("NaN/Inf detected in final rewards, replacing with zeros")
                    result = torch.where(
                        torch.isnan(result) | torch.isinf(result),
                        torch.zeros_like(result),
                        result
                    )
                return result
            else:
                return torch.tensor([], dtype=torch.float32, device=self.device)
    
    def load_human_feedback(self, feedback_path: str):
        """Load human feedback data."""
        self.human_feedback_integrator.load_human_feedback(feedback_path)
    
    def save_model(self, save_path: str):
        """Save the reward model."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_path, "reward_model.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config_dict = {
            "model_name": self.model_name,
            "reward_config": self.config.__dict__
        }
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Reward model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, config: RewardConfig):
        """Load a saved reward model."""
        # Load config
        with open(os.path.join(load_path, "config.json"), 'r') as f:
            config_dict = json.load(f)
        
        # Create model
        model = cls(config, config_dict["model_name"])
        
        # Load state dict
        state_dict = torch.load(os.path.join(load_path, "reward_model.pt"))
        model.load_state_dict(state_dict)
        
        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(load_path)
        
        logger.info(f"Reward model loaded from {load_path}")
        return model


class RewardModelTrainer:
    """Trainer for the reward model."""
    
    def __init__(self, model: ModernRewardModel, config: RewardConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.reward_learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.reward_epochs
        )
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Validate batch
        if not batch or 'prompts' not in batch or 'responses' not in batch:
            logger.warning("Invalid batch format, skipping")
            return {
                'loss': float('inf'),
                'predicted_reward_mean': 0.0,
                'predicted_reward_std': 0.0,
                'skipped': True
            }
        
        prompts = batch['prompts']
        responses = batch['responses']
        
        if not prompts or not responses or len(prompts) != len(responses):
            logger.warning(f"Invalid batch: prompts={len(prompts) if prompts else 0}, responses={len(responses) if responses else 0}")
            return {
                'loss': float('inf'),
                'predicted_reward_mean': 0.0,
                'predicted_reward_std': 0.0,
                'skipped': True
            }
        # Support new structured human feedback field (`human_feedback`) or legacy `human_ratings`
        human_feedback = batch.get('human_feedback', None)
        human_ratings = None
        if human_feedback is not None:
            # extract numeric ratings if present
            try:
                human_ratings = [None if hf is None else (hf.get('rating') if isinstance(hf, dict) else hf) for hf in human_feedback]
            except (AttributeError, TypeError, KeyError) as e:
                logger.warning(f"Error extracting human ratings: {e}")
                human_ratings = None
        else:
            human_ratings = batch.get('human_ratings', None)

        # Compute rewards
        predicted_rewards = self.model.compute_reward(prompts, responses)

        # Compute loss
        # If human_ratings is provided and contains no None values, use them as targets.
        use_human_targets = False
        if human_ratings is not None:
            try:
                # Normalize and check for None
                processed = [None if v is None else float(v) for v in human_ratings]
                if all(v is not None for v in processed) and len(processed) == len(prompts):
                    target_rewards = torch.tensor(processed, dtype=torch.float32, device=self.model.device)
                    use_human_targets = True
                else:
                    use_human_targets = False
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing human ratings: {e}")
                use_human_targets = False

        if use_human_targets:
            loss = F.mse_loss(predicted_rewards, target_rewards)
        else:
            # Use component-based rewards as targets
            component_rewards = self.model.compute_reward_components(prompts, responses)
            target_list = [0.0 if getattr(c, 'total_reward', None) is None else float(c.total_reward) for c in component_rewards]
            target_rewards = torch.tensor(target_list, dtype=torch.float32, device=self.model.device)
            loss = F.mse_loss(predicted_rewards, target_rewards)
        
        # Numerical stability checks
        # Check for NaN or Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Numerical instability detected: loss is NaN or Inf. Skipping this batch.")
            return {
                'loss': float('inf'),
                'predicted_reward_mean': 0.0,
                'predicted_reward_std': 0.0,
                'skipped': True
            }
        
        # Check for NaN or Inf in predicted rewards
        if torch.isnan(predicted_rewards).any() or torch.isinf(predicted_rewards).any():
            logger.warning("Numerical instability detected: predicted_rewards contain NaN or Inf. Skipping this batch.")
            return {
                'loss': float('inf'),
                'predicted_reward_mean': 0.0,
                'predicted_reward_std': 0.0,
                'skipped': True
            }
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Check gradients for numerical issues before clipping
        has_nan_grad = False
        grad_norm_squared = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    logger.warning(f"NaN gradients detected in {name}. Skipping gradient update.")
                    has_nan_grad = True
                    break
                if torch.isinf(param.grad).any():
                    logger.warning(f"Inf gradients detected in {name}. Skipping gradient update.")
                    has_nan_grad = True
                    break
                # Only accumulate grad_norm if no NaN/Inf found
                grad_norm_squared += param.grad.data.norm(2).item() ** 2
        
        if has_nan_grad:
            self.optimizer.zero_grad()  # Clear gradients
            return {
                'loss': loss.item(),
                'predicted_reward_mean': predicted_rewards.mean().item(),
                'predicted_reward_std': predicted_rewards.std().item(),
                'skipped': True
            }
        
        # Apply gradient clipping with numerical stability (only if no NaN/Inf)
        grad_norm = grad_norm_squared ** 0.5
        if grad_norm > 0:
            max_norm = self.config.reward_clip_value if hasattr(self.config, 'reward_clip_value') else 1.0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'predicted_reward_mean': predicted_rewards.mean().item(),
            'predicted_reward_std': predicted_rewards.std().item(),
            'grad_norm': grad_norm
        }
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        from tqdm import tqdm
        import sys
        
        epoch_metrics = []
        
        # Add progress bar for reward model training
        pbar = tqdm(
            dataloader,
            desc="Training Reward Model",
            unit="batch",
            bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
            ascii=True if sys.platform == 'win32' else False
        )
        
        for batch in pbar:
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)
            
            # Update progress bar with current metrics
            if metrics:
                pbar.set_postfix({
                    'loss': f"{metrics.get('loss', 0):.4f}",
                    'reward': f"{metrics.get('predicted_reward_mean', 0):.3f}"
                })
        
        pbar.close()
        
        # Average metrics (skip 'skipped' key and handle non-numeric values)
        if not epoch_metrics:
            return {'loss': 0.0, 'predicted_reward_mean': 0.0, 'predicted_reward_std': 0.0}
        avg_metrics = {}
        skipped_count = sum(1 for m in epoch_metrics if m.get('skipped', False))
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count}/{len(epoch_metrics)} batches due to numerical issues")
        
        for key in epoch_metrics[0].keys():
            if key == 'skipped':
                continue
            # Only average numeric values
            values = [m[key] for m in epoch_metrics if key in m and isinstance(m[key], (int, float)) and not np.isnan(m[key]) and not np.isinf(m[key])]
            if values:
                avg_metrics[key] = float(np.mean(values))
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
