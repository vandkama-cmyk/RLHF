"""
Modern RLHF Trainer with PPO and DPO Support
===========================================

A state-of-the-art trainer that supports both PPO and DPO (Direct Preference Optimization)
for code generation tasks with comprehensive evaluation and monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig,
    PreTrainedModel, PreTrainedTokenizer
)
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import numpy as np
from dataclasses import dataclass
import json
import os
import time
from tqdm import tqdm
import sys
import warnings

# Suppress excessive transformer warnings that interfere with progress bars
warnings.filterwarnings('ignore', message='.*not sharded.*')
warnings.filterwarnings('ignore', message='.*was not found in model.*')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# Set transformers logging to only show errors
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
import transformers
transformers.logging.set_verbosity_error()

# Configure tqdm for better Windows console support
if sys.platform == 'win32':
    # Use ASCII mode for Windows consoles
    tqdm_kwargs = {'ascii': True, 'ncols': 100, 'mininterval': 0.5}
else:
    tqdm_kwargs = {'ncols': 100, 'mininterval': 0.5}

# Disable wandb integration by default for offline or CI runs
wandb = None
_WANDB_AVAILABLE = False
from collections import defaultdict

from .config import ModernRLHFConfig, TrainingConfig
from .reward_model import ModernRewardModel
from .metrics import ModernMetricsEvaluator
from .metrics_tracker import MetricsTracker, EpochMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingStep:
    """Container for training step results."""
    step: int
    loss: float
    reward: float
    kl_divergence: float
    entropy: float
    learning_rate: float
    metrics: Dict[str, float]


class PPOTrainer:
    """Modern PPO trainer for RLHF."""
    
    def __init__(self, config: ModernRLHFConfig, reward_model: ModernRewardModel):
        self.config = config
        self.reward_model = reward_model
        
        dev = config.hardware.device
        if dev == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU (training will be slow).")
            dev = "cpu"
            config.hardware.device = "cpu"
        self.device = torch.device(dev)
        if self.device.type == "cpu":
            logger.warning("PPOTrainer running on CPU — expect long runtimes and higher memory use for large models.")
            print("PPOTrainer initialized on CPU (no GPU).")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"PPOTrainer: Using GPU {gpu_name} ({gpu_memory:.1f} GB)")
            print(f"PPOTrainer initialized on GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Load models
        self.policy_model = self._load_policy_model()
        self.reference_model = self._load_reference_model()
        self.tokenizer = self._load_tokenizer()
        
        # CRITICAL: Synchronize pad_token_id with model vocab size after models are loaded
        self._sync_tokenizer_with_model()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=0.01
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.total_steps
        )
        
        # Metrics evaluator
        self.metrics_evaluator = ModernMetricsEvaluator()
        
        # Metrics tracker for detailed monitoring
        self.metrics_tracker = MetricsTracker(output_dir=config.data.output_path + "/metrics")
        self.metrics_tracker.set_total_epochs(config.training.ppo_epochs)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_reward = -float('inf')
        self.training_history = []
        self.evaluation_history = []  # Track evaluation metrics per epoch
        
        # Initialize wandb if available
        self._wandb_enabled = False
        if config.verbose and not config.debug and _WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=config.experiment_name,
                    name=config.run_name,
                    config=config.to_dict(),
                    tags=config.tags
                )
                self._wandb_enabled = True
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
    
    def _load_policy_model(self) -> PreTrainedModel:
        """Load the policy model (prefer causal LM / seq2seq variants to get logits/generate)."""
        model_name = self.config.model.base_model_name
        torch_dtype = getattr(torch, self.config.model.torch_dtype)

        # Prefer causal LM model to support `.generate()` and `.logits`
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.config.model.trust_remote_code,
                device_map=None,  # Отключить автоматическое распределение
                low_cpu_mem_usage=False  # Отключить оптимизации памяти, которые вызывают предупреждения
            )
        except (OSError, ValueError, RuntimeError) as e:
            logger.debug(f"Failed to load CausalLM model: {e}, trying Seq2SeqLM")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.config.model.trust_remote_code,
                    device_map=None,
                    low_cpu_mem_usage=False
                )
            except (OSError, ValueError, RuntimeError) as e2:
                logger.debug(f"Failed to load Seq2SeqLM model: {e2}, falling back to base AutoModel")
                # Fallback to base AutoModel (may not provide logits/generate)
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.config.model.trust_remote_code,
                    device_map=None,
                    low_cpu_mem_usage=False
                )

        if self.config.hardware.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        model = model.to(self.device)
        if next(model.parameters()).device != self.device:
            raise RuntimeError(f"Policy model device mismatch: {next(model.parameters()).device} vs {self.device}")
        logger.info("Policy model on %s", self.device)
        
        return model
    
    def _load_reference_model(self) -> PreTrainedModel:
        """Load the reference model (frozen). Prefer matching class to policy model."""
        model_name = self.config.model.base_model_name
        torch_dtype = getattr(torch, self.config.model.torch_dtype)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.config.model.trust_remote_code,
                device_map=None,  # Отключить автоматическое распределение
                low_cpu_mem_usage=False
            )
        except Exception:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.config.model.trust_remote_code,
                    device_map=None,
                    low_cpu_mem_usage=False
                )
            except Exception:
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=self.config.model.trust_remote_code,
                    device_map=None,
                    low_cpu_mem_usage=False
                )

        # Freeze reference model
        for param in model.parameters():
            param.requires_grad = False

        model = model.to(self.device)
        if next(model.parameters()).device != self.device:
            raise RuntimeError(f"Reference model device mismatch: {next(model.parameters()).device} vs {self.device}")
        logger.info("Reference model on %s", self.device)
        
        return model
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            use_fast=self.config.model.use_fast_tokenizer,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Ensure pad_token is set - critical for generation
        # Use eos_token as pad_token (standard for GPT models) to avoid resizing model
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif tokenizer.unk_token is not None:
                # Fallback: use unk_token if eos_token is not available
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                logger.warning("Could not set pad_token - generation may fail. Trying to use token 0.")
                # Last resort: try to use token at index 0 (but this is risky)
                try:
                    tokenizer.pad_token_id = 0
                except (AttributeError, ValueError, TypeError) as e:
                    logger.error(f"Failed to set pad_token_id: {e} - generation will likely fail")
        
        # Verify pad_token_id is valid
        vocab_size = getattr(tokenizer, 'vocab_size', None)
        if vocab_size is None:
            try:
                vocab_size = len(tokenizer.get_vocab())
            except (AttributeError, TypeError) as e:
                logger.debug(f"Could not get vocab size from get_vocab(): {e}")
                vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else 50257  # GPT-2 default
        if tokenizer.pad_token_id is not None:
            if tokenizer.pad_token_id < 0 or tokenizer.pad_token_id >= vocab_size:
                logger.warning(f"pad_token_id ({tokenizer.pad_token_id}) is out of vocab range ({vocab_size}), using eos_token_id")
                if tokenizer.eos_token_id is not None and tokenizer.eos_token_id < vocab_size:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Ensure left padding for decoder-only models to avoid generation issues
        try:
            tokenizer.padding_side = 'left'
        except (AttributeError, ValueError) as e:
            logger.debug(f"Could not set padding_side: {e}")
        
        return tokenizer
    
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
    
    def _sync_tokenizer_with_model(self):
        """Synchronize tokenizer pad_token_id with model vocab size to prevent CUDA errors."""
        if not hasattr(self, 'policy_model') or self.policy_model is None:
            return
        
        # Get model vocab size
        model_vocab_size = None
        if hasattr(self.policy_model, 'config'):
            model_vocab_size = getattr(self.policy_model.config, 'vocab_size', None)
        if model_vocab_size is None:
            try:
                emb = self.policy_model.get_input_embeddings()
                if hasattr(emb, 'num_embeddings'):
                    model_vocab_size = emb.num_embeddings
            except (AttributeError, RuntimeError) as e:
                logger.debug(f"Could not get vocab size from embeddings: {e}")
                pass
        
        if model_vocab_size is None:
            logger.warning("Could not determine model vocab size, skipping tokenizer sync")
            return
        
        # Check and fix pad_token_id
        current_pad_token_id = self.tokenizer.pad_token_id
        if current_pad_token_id is not None:
            if current_pad_token_id < 0 or current_pad_token_id >= model_vocab_size:
                # Fix invalid pad_token_id
                if self.tokenizer.eos_token_id is not None:
                    if 0 <= self.tokenizer.eos_token_id < model_vocab_size:
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        logger.info(f"Fixed pad_token_id: {current_pad_token_id} -> {self.tokenizer.eos_token_id} (vocab_size={model_vocab_size})")
                    else:
                        # Use a safe fallback - use token 0 if valid
                        safe_token_id = 0 if model_vocab_size > 0 else None
                        if safe_token_id is not None:
                            self.tokenizer.pad_token_id = safe_token_id
                            logger.warning(f"Fixed pad_token_id: {current_pad_token_id} -> {safe_token_id} (vocab_size={model_vocab_size})")
                        else:
                            logger.error(f"Cannot fix pad_token_id: vocab_size={model_vocab_size}")
        
        # Also update model config
        if hasattr(self.policy_model, 'config'):
            try:
                self.policy_model.config.pad_token_id = self.tokenizer.pad_token_id
                if self.tokenizer.eos_token_id is not None:
                    self.policy_model.config.eos_token_id = self.tokenizer.eos_token_id
            except Exception:
                pass
    
    def generate_responses(self, prompts: List[str]) -> Dict[str, Any]:
        """Generate responses for given prompts."""
        self.policy_model.eval()
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_prompt_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get vocab size from model config (more reliable than tokenizer)
        model_vocab_size = None
        if hasattr(self.policy_model, 'config'):
            model_vocab_size = getattr(self.policy_model.config, 'vocab_size', None)
        
        # Fallback to tokenizer vocab size
        if model_vocab_size is None:
            model_vocab_size = getattr(self.tokenizer, 'vocab_size', None)
        if model_vocab_size is None:
            try:
                model_vocab_size = len(self.tokenizer.get_vocab())
            except Exception:
                # Last resort: try to get from model's embedding layer
                if hasattr(self.policy_model, 'get_input_embeddings'):
                    try:
                        emb = self.policy_model.get_input_embeddings()
                        if hasattr(emb, 'num_embeddings'):
                            model_vocab_size = emb.num_embeddings
                    except Exception:
                        pass
                if model_vocab_size is None:
                    model_vocab_size = 50257  # GPT-2 default fallback
        
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        # Debug: log token info once to diagnose CUDA issues
        if not hasattr(self, '_debug_pad_logged'):
            msg = (
                f"Tokenizer/model token info: pad_token_id={pad_token_id}, "
                f"eos_token_id={eos_token_id}, vocab_size={model_vocab_size}"
            )
            logger.info(msg)
            print(msg)
            self._debug_pad_logged = True
        
        # CRITICAL: Validate pad_token_id against model vocab size
        # This is the source of CUDA errors if pad_token_id >= vocab_size
        if pad_token_id is None:
            pad_token_id = eos_token_id  # Use eos_token as pad_token (standard for GPT models)
        
        if pad_token_id is None or pad_token_id < 0:
            pad_token_id = 0  # Fallback to token 0
            logger.warning(f"pad_token_id was None or negative, using 0")
        
        if pad_token_id >= model_vocab_size:
            # CRITICAL FIX: pad_token_id must be < vocab_size
            if eos_token_id is not None and eos_token_id >= 0 and eos_token_id < model_vocab_size:
                pad_token_id = eos_token_id
                logger.warning(f"pad_token_id ({self.tokenizer.pad_token_id}) >= vocab_size ({model_vocab_size}), using eos_token_id ({eos_token_id})")
            else:
                pad_token_id = model_vocab_size - 1  # Use last valid token
                logger.warning(f"pad_token_id >= vocab_size ({model_vocab_size}), using last valid token ({pad_token_id})")
        
        # Validate eos_token_id
        if eos_token_id is None or eos_token_id < 0:
            # Try to get from model config
            if hasattr(self.policy_model, 'config') and hasattr(self.policy_model.config, 'eos_token_id'):
                eos_token_id = self.policy_model.config.eos_token_id
        
        if eos_token_id is not None and eos_token_id >= model_vocab_size:
            logger.warning(f"eos_token_id ({eos_token_id}) >= vocab_size ({model_vocab_size}), not setting eos_token_id")
            eos_token_id = None  # Don't use invalid eos_token_id
        
        # Also update model config to ensure consistency
        if hasattr(self.policy_model, 'config'):
            try:
                self.policy_model.config.pad_token_id = pad_token_id
                if eos_token_id is not None:
                    self.policy_model.config.eos_token_id = eos_token_id
            except Exception:
                pass  # Some models don't allow config modification
        
        # Prepare generation kwargs
        do_sample = bool(getattr(self.config.generation, 'do_sample', False))
        generation_kwargs = {
            'max_new_tokens': getattr(self.config.generation, 'max_new_tokens', 128),
            'do_sample': do_sample,
            'repetition_penalty': getattr(self.config.generation, 'repetition_penalty', 1.0),
        }

        if do_sample:
            generation_kwargs['temperature'] = getattr(self.config.generation, 'temperature', 1.0)
            generation_kwargs['top_p'] = getattr(self.config.generation, 'top_p', 1.0)
            generation_kwargs['top_k'] = getattr(self.config.generation, 'top_k', 0)
        else:
            generation_kwargs['num_beams'] = max(1, getattr(self.config.generation, 'num_beams', 1))
        
        bos_token_id = getattr(self.tokenizer, 'bos_token_id', None)
        if bos_token_id is None or bos_token_id < 0 or bos_token_id >= model_vocab_size:
            bos_token_id = pad_token_id if pad_token_id is not None else eos_token_id
        if bos_token_id is not None and bos_token_id >= 0 and bos_token_id < model_vocab_size:
            generation_kwargs['bos_token_id'] = bos_token_id

        # Only add pad_token_id if it's valid
        if pad_token_id is not None and pad_token_id >= 0 and pad_token_id < model_vocab_size:
            generation_kwargs['pad_token_id'] = pad_token_id
        else:
            logger.error(f"ERROR: Cannot use pad_token_id={pad_token_id} with vocab_size={model_vocab_size}")
            raise ValueError(f"Invalid pad_token_id={pad_token_id} for vocab_size={model_vocab_size}")
        
        # Only add eos_token_id if it's valid
        if eos_token_id is not None and eos_token_id >= 0 and eos_token_id < model_vocab_size:
            generation_kwargs['eos_token_id'] = eos_token_id
        
        # Generate responses with fallback for sampling issues
        try:
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    **generation_kwargs
                )
        except RuntimeError as err:
            logger.warning(f"Sampling generation failed ({err}). Falling back to greedy decoding.")
            fallback_kwargs = generation_kwargs.copy()
            fallback_kwargs['do_sample'] = False
            fallback_kwargs.pop('temperature', None)
            fallback_kwargs.pop('top_p', None)
            fallback_kwargs.pop('top_k', None)
            fallback_kwargs.setdefault('num_beams', max(1, getattr(self.config.generation, 'num_beams', 1)))

            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    **fallback_kwargs
                )
        
        # Decode responses
        response_texts = []
        for i, output in enumerate(outputs):
            # Remove prompt from output
            prompt_length = inputs['input_ids'][i].shape[0]
            response_tokens = output[prompt_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            if not response_text.strip():
                response_text = self.tokenizer.eos_token or " "
            response_texts.append(response_text)
        
        return {
            "response_texts": response_texts,
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask']
        }
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute rewards for prompt-response pairs."""
        return self.reward_model.compute_reward(prompts, responses)
    
    def compute_kl_divergence(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute KL divergence between policy and reference models."""
        # Tokenize responses
        inputs = self.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_response_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get logits from both models
        with torch.no_grad():
            policy_outputs = self.policy_model(**inputs)
            reference_outputs = self.reference_model(**inputs)

            def _outputs_to_logits(model, outputs):
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    return outputs.logits
                # Try lm_head
                if hasattr(model, 'lm_head') and hasattr(outputs, 'last_hidden_state'):
                    return model.lm_head(outputs.last_hidden_state)
                # Try output embeddings weight multiplication
                try:
                    if hasattr(model, 'get_output_embeddings') and outputs.last_hidden_state is not None:
                        emb = model.get_output_embeddings()
                        if hasattr(emb, 'weight'):
                            return torch.matmul(outputs.last_hidden_state, emb.weight.t())
                except Exception:
                    pass
                raise AttributeError("Model outputs do not contain logits and no lm_head/output_embeddings available")

            policy_logits = _outputs_to_logits(self.policy_model, policy_outputs)
            reference_logits = _outputs_to_logits(self.reference_model, reference_outputs)
        
        # Compute KL divergence
        policy_probs = F.softmax(policy_logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)
        
        kl_div = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            reference_probs,
            reduction='none'
        ).sum(dim=-1)
        
        return kl_div.mean(dim=1)
    
    def compute_entropy(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute entropy of policy model outputs."""
        # Tokenize responses
        inputs = self.tokenizer(
            responses,
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_response_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Get logits
        with torch.no_grad():
            outputs = self.policy_model(**inputs)
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                logits = outputs.logits
            elif hasattr(self.policy_model, 'lm_head') and hasattr(outputs, 'last_hidden_state'):
                logits = self.policy_model.lm_head(outputs.last_hidden_state)
            else:
                emb = self.policy_model.get_output_embeddings() if hasattr(self.policy_model, 'get_output_embeddings') else None
                if emb is not None and hasattr(outputs, 'last_hidden_state'):
                    logits = torch.matmul(outputs.last_hidden_state, emb.weight.t())
                else:
                    raise AttributeError("Unable to obtain logits from policy model outputs")
        
        # Compute entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        
        return entropy.mean(dim=1)
    
    def ppo_step(self, batch: Dict[str, Any]) -> TrainingStep:
        """Single PPO training step with simplified implementation."""
        self.policy_model.train()

        prompts = batch['prompts']

        # Generate new responses for PPO training
        generation_output = self.generate_responses(prompts)
        responses = generation_output['response_texts']

        # Compute rewards
        rewards = self.compute_rewards(prompts, responses)

        # Ensure rewards are on correct device
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        elif rewards.device != self.device:
            rewards = rewards.to(self.device)

        # Compute simplified advantages (can be improved with value function later)
        advantages = rewards - rewards.mean()

        # Simplified policy loss: maximize rewards with basic regularization
        policy_loss = -rewards.mean()

        # Add entropy regularization to encourage exploration
        entropy = self.compute_entropy(prompts, responses)
        entropy_loss = -entropy.mean()

        # Add KL regularization to prevent policy from drifting too far
        kl_div = self.compute_kl_divergence(prompts, responses)
        kl_loss = kl_div.mean()

        # Total loss
        total_loss = (
            policy_loss +
            self.config.training.ppo_entropy_coef * entropy_loss +
            self.config.training.ppo_kl_penalty * kl_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.training.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        # Create training step
        step = TrainingStep(
            step=self.step,
            loss=total_loss.item(),
            reward=rewards.mean().item(),
            kl_divergence=kl_div.mean().item(),
            entropy=entropy.mean().item(),
            learning_rate=self.optimizer.param_groups[0]['lr'],
            metrics={
                'policy_loss': policy_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'kl_loss': kl_loss.item()
            }
        )

        self.step += 1
        return step
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_start_time = time.time()  # Track epoch duration
        epoch_metrics = defaultdict(list)
        
        # Create progress bar with detailed info
        total_batches = len(dataloader)
        pbar = tqdm(
            enumerate(dataloader), 
            total=total_batches,
            desc=f"Epoch {self.epoch+1}/{self.config.training.ppo_epochs}",
            unit="batch",
            bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,  # Keep progress bar after completion
            position=0,  # Top-level progress bar
            **tqdm_kwargs
        )
        
        for batch_idx, batch in pbar:
            step = self.ppo_step(batch)
            
            # Collect metrics
            epoch_metrics['loss'].append(step.loss)
            epoch_metrics['reward'].append(step.reward)
            epoch_metrics['kl_divergence'].append(step.kl_divergence)
            epoch_metrics['entropy'].append(step.entropy)
            epoch_metrics['learning_rate'].append(step.learning_rate)
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'loss': f'{step.loss:.4f}',
                'reward': f'{step.reward:.4f}',
                'lr': f'{step.learning_rate:.2e}',
                'step': f'{step.step}/{self.config.training.total_steps}'
            })
            
            # Log to wandb
            # wandb integration intentionally disabled
            
            # Save checkpoint
            if step.step % self.config.training.save_steps == 0:
                self.save_checkpoint()
                
            # Stop if reached total steps
            if step.step >= self.config.training.total_steps:
                break
        
        # Close progress bar
        pbar.close()
        
        # Average metrics
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        # Record history for plotting/inspection
        try:
            # Ensure training_history exists and append epoch metrics
            if not hasattr(self, 'training_history') or self.training_history is None:
                self.training_history = []
            self.training_history.append(avg_metrics)
        except Exception:
            pass
        
        # Log to metrics tracker
        epoch_time = time.time() - epoch_start_time
        samples_per_second = total_batches * self.config.training.batch_size / epoch_time if epoch_time > 0 else 0
        
        metrics_obj = EpochMetrics(
            epoch=self.epoch + 1,
            timestamp=time.time(),
            loss=avg_metrics['loss'],
            reward=avg_metrics['reward'],
            kl_divergence=avg_metrics['kl_divergence'],
            entropy=avg_metrics['entropy'],
            learning_rate=avg_metrics['learning_rate'],
            epoch_time=epoch_time,
            samples_per_second=samples_per_second
        )
        
        self.metrics_tracker.log_epoch(metrics_obj)

        self.epoch += 1
        return avg_metrics
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate the model."""
        self.policy_model.eval()
        
        all_prompts = []
        all_responses = []
        all_rewards = []
        all_references = []
        
        # If no evaluation dataloader or it's empty, return default zero metrics
        if not eval_dataloader:
            return {'avg_reward': 0.0, 'reward_std': 0.0}

        with torch.no_grad():
            pbar = tqdm(
                eval_dataloader,
                desc="Evaluating",
                unit="batch",
                bar_format='{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                file=sys.stdout,
                dynamic_ncols=True,
                leave=True,
                position=0,
                **tqdm_kwargs
            )
            
            for batch in pbar:
                prompts = batch.get('prompts', [])

                # Generate responses
                generation_output = self.generate_responses(prompts)
                responses = generation_output['response_texts']

                # Compute rewards
                rewards = self.compute_rewards(prompts, responses)

                all_prompts.extend(prompts)
                all_responses.extend(responses)
                all_rewards.extend(rewards.tolist())
                if 'references' in batch:
                    all_references.extend(batch.get('references', []))
                
                # Update progress bar
                pbar.set_postfix({
                    'processed': len(all_responses),
                    'avg_reward': f'{np.mean(all_rewards):.4f}' if all_rewards else '0.0000'
                })
            
            pbar.close()
        
        # If references exist but lengths mismatch, align to the minimum length and warn
        if all_references and len(all_references) != len(all_responses):
            logger.warning(f"Mismatch between number of references ({len(all_references)}) and responses ({len(all_responses)}). Trimming to min length.")
            n = min(len(all_references), len(all_responses))
            all_references = all_references[:n]
            all_responses = all_responses[:n]
        
        # Compute evaluation metrics
        eval_metrics = {}
        if all_rewards:
            eval_metrics['avg_reward'] = float(np.mean(all_rewards))
            eval_metrics['reward_std'] = float(np.std(all_rewards))
        else:
            eval_metrics['avg_reward'] = 0.0
            eval_metrics['reward_std'] = 0.0
        
        # Compute other metrics if references are available (only once)
        if all_references and len(all_references) == len(all_responses):
            metrics_results = self.metrics_evaluator.compute_all_metrics(all_responses, all_references)
            for metric_name, result in metrics_results.items():
                eval_metrics[f'eval_{metric_name}'] = result.score
        
        return eval_metrics
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.data.output_path, f"checkpoint-{self.step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.policy_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state (convert tensors to JSON-serializable types)
        training_state = {
            'step': int(self.step),
            'epoch': int(self.epoch),
            'best_reward': float(self.best_reward),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }

        def _to_json_safe(obj):
            """Recursively convert torch tensors and numpy types to JSON-safe Python types."""
            import numpy as _np
            try:
                import torch as _torch
            except Exception:
                _torch = None

            if _torch is not None and _torch.is_tensor(obj):
                try:
                    return obj.detach().cpu().numpy().tolist()
                except Exception:
                    try:
                        return float(obj.detach().cpu())
                    except Exception:
                        return None
            if isinstance(obj, dict):
                return {k: _to_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json_safe(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_json_safe(v) for v in obj)
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, (_np.bool_,)):
                return bool(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            return obj

        safe_state = _to_json_safe(training_state)
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
            json.dump(safe_state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint."""
        # Load model
        self.policy_model = AutoModel.from_pretrained(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        
        # Load training state
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'r') as f:
            training_state = json.load(f)
        
        self.step = training_state['step']
        self.epoch = training_state['epoch']
        self.best_reward = training_state['best_reward']
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
        self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")


class DPOTrainer:
    """Direct Preference Optimization trainer."""
    
    def __init__(self, config: ModernRLHFConfig, reward_model: ModernRewardModel):
        self.config = config
        self.reward_model = reward_model
        self.device = torch.device(config.hardware.device)
        
        # Load models
        self.policy_model = self._load_policy_model()
        self.reference_model = self._load_reference_model()
        self.tokenizer = self._load_tokenizer()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=0.01
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.training_history = []
    
    def _load_policy_model(self) -> PreTrainedModel:
        """Load the policy model."""
        model = AutoModel.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if self.config.hardware.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model.to(self.device)
    
    def _load_reference_model(self) -> PreTrainedModel:
        """Load the reference model (frozen)."""
        model = AutoModel.from_pretrained(
            self.config.model.base_model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Freeze reference model
        for param in model.parameters():
            param.requires_grad = False
        
        return model.to(self.device)
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            use_fast=self.config.model.use_fast_tokenizer,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def dpo_step(self, batch: Dict[str, Any]) -> TrainingStep:
        """Single DPO training step."""
        self.policy_model.train()
        
        prompts = batch['prompts']
        chosen_responses = batch['chosen_responses']
        rejected_responses = batch['rejected_responses']
        
        # Compute log probabilities for chosen responses
        chosen_log_probs = self._compute_log_probs(prompts, chosen_responses)
        
        # Compute log probabilities for rejected responses
        rejected_log_probs = self._compute_log_probs(prompts, rejected_responses)
        
        # Compute reference log probabilities
        with torch.no_grad():
            chosen_ref_log_probs = self._compute_log_probs(prompts, chosen_responses, use_reference=True)
            rejected_ref_log_probs = self._compute_log_probs(prompts, rejected_responses, use_reference=True)
        
        # Compute DPO loss
        pi_logratios = chosen_log_probs - rejected_log_probs
        ref_logratios = chosen_ref_log_probs - rejected_ref_log_probs
        
        logits = pi_logratios - ref_logratios
        
        if self.config.training.dpo_loss_type == "sigmoid":
            losses = -F.logsigmoid(self.config.training.dpo_beta * logits)
        elif self.config.training.dpo_loss_type == "hinge":
            losses = torch.relu(1 - self.config.training.dpo_beta * logits)
        else:
            raise ValueError(f"Unknown DPO loss type: {self.config.training.dpo_loss_type}")
        
        loss = losses.mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.training.max_grad_norm)
        self.optimizer.step()
        
        # Create training step
        step = TrainingStep(
            step=self.step,
            loss=loss.item(),
            reward=0.0,  # DPO doesn't use explicit rewards
            kl_divergence=0.0,
            entropy=0.0,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            metrics={
                'dpo_loss': loss.item(),
                'chosen_log_prob': chosen_log_probs.mean().item(),
                'rejected_log_prob': rejected_log_probs.mean().item(),
                'log_ratio': logits.mean().item()
            }
        )
        
        self.step += 1
        return step
    
    def _compute_log_probs(self, prompts: List[str], responses: List[str], use_reference: bool = False) -> torch.Tensor:
        """Compute log probabilities for prompt-response pairs."""
        model = self.reference_model if use_reference else self.policy_model
        
        # Tokenize
        inputs = self.tokenizer(
            [f"{prompt} {response}" for prompt, response in zip(prompts, responses)],
            padding=True,
            truncation=True,
            max_length=self.config.generation.max_prompt_length + self.config.generation.max_response_length,
            return_tensors="pt"
        ).to(self.device)
        
        # Compute logits
        with torch.no_grad() if use_reference else torch.enable_grad():
            outputs = model(**inputs)
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                logits = outputs.logits
            elif hasattr(model, 'lm_head') and hasattr(outputs, 'last_hidden_state'):
                logits = model.lm_head(outputs.last_hidden_state)
            else:
                emb = model.get_output_embeddings() if hasattr(model, 'get_output_embeddings') else None
                if emb is not None and hasattr(outputs, 'last_hidden_state'):
                    logits = torch.matmul(outputs.last_hidden_state, emb.weight.t())
                else:
                    raise AttributeError("Unable to obtain logits from model outputs")
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Extract log probabilities for response tokens
        response_log_probs = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            
            # Get log probabilities for response tokens
            response_start = len(prompt_tokens)
            response_end = response_start + len(response_tokens)
            
            if response_end <= logits.shape[1]:
                response_log_prob = log_probs[i, response_start:response_end, response_tokens].sum()
                response_log_probs.append(response_log_prob)
            else:
                response_log_probs.append(torch.tensor(0.0))
        
        return torch.stack(response_log_probs)
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = defaultdict(list)
        
        for batch in tqdm(dataloader, desc=f"DPO Epoch {self.epoch}", file=sys.stdout, dynamic_ncols=True, **tqdm_kwargs):
            step = self.dpo_step(batch)
            
            # Collect metrics
            epoch_metrics['loss'].append(step.loss)
            epoch_metrics['learning_rate'].append(step.learning_rate)
            
            # Log metrics
            for key, value in step.metrics.items():
                epoch_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        self.epoch += 1
        return avg_metrics


class ModernRLHFTrainer:
    """Main trainer that supports both PPO and DPO."""
    
    def __init__(self, config: ModernRLHFConfig, reward_model: ModernRewardModel):
        self.config = config
        self.reward_model = reward_model
        
        # Choose trainer based on config
        if hasattr(config.training, 'use_dpo') and config.training.use_dpo:
            self.trainer = DPOTrainer(config, reward_model)
        else:
            self.trainer = PPOTrainer(config, reward_model)
        
        logger.info(f"Initialized {type(self.trainer).__name__} trainer")
    
    def train(self, train_dataloader, eval_dataloader=None) -> Dict[str, Any]:
        """Main training loop."""
        import time
        start_time = time.time()
        
        logger.info("Starting training...")
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Total steps: {self.config.training.total_steps}")
        print(f"  PPO epochs: {self.config.training.ppo_epochs}")
        print(f"  Batch size: {self.config.training.batch_size}")
        print(f"  Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        print(f"  Device: {self.config.hardware.device}")
        print(f"{'='*60}\n")
        
        best_metrics = {}
        patience_counter = 0
        
        # Main training loop with progress tracking
        for epoch in range(self.config.training.ppo_epochs):
            epoch_start = time.time()
            print(f"\n[Epoch {epoch+1}/{self.config.training.ppo_epochs}]")
            
            # Training
            train_metrics = self.trainer.train_epoch(train_dataloader)
            epoch_time = time.time() - epoch_start
            
            # Evaluation
            if eval_dataloader is not None:
                print(f"\n[Evaluation]")
                eval_start = time.time()
                eval_metrics = self.trainer.evaluate(eval_dataloader)
                eval_time = time.time() - eval_start
                
                # Store evaluation metrics for plotting
                epoch_eval_metrics = {
                    'epoch': epoch + 1,
                    **eval_metrics
                }
                # Store in trainer's evaluation history
                if hasattr(self.trainer, 'evaluation_history'):
                    self.trainer.evaluation_history.append(epoch_eval_metrics)
                
                # Update metrics tracker with evaluation metrics
                if hasattr(self.trainer, 'metrics_tracker') and len(self.trainer.metrics_tracker.history) > 0:
                    # Update the last logged epoch with eval metrics
                    last_epoch = self.trainer.metrics_tracker.history[-1]
                    last_epoch.bertscore = eval_metrics.get('eval_bertscore')
                    last_epoch.codebleu = eval_metrics.get('eval_codebleu')
                    last_epoch.bleu = eval_metrics.get('eval_bleu')
                    last_epoch.rouge = eval_metrics.get('eval_rouge')
                    last_epoch.ruby = eval_metrics.get('eval_ruby')
                    last_epoch.eval_loss = eval_metrics.get('eval_loss')
                    
                    # Re-save updated metrics
                    self.trainer.metrics_tracker._save_to_csv(last_epoch)
                    self.trainer.metrics_tracker._save_to_json()
                    self.trainer.metrics_tracker._generate_plots()
                
                # Check for improvement
                if eval_metrics.get('avg_reward', 0) > best_metrics.get('avg_reward', -float('inf')):
                    best_metrics = eval_metrics
                    patience_counter = 0
                    self.trainer.save_checkpoint()
                    print(f"  New best reward: {best_metrics.get('avg_reward', 0):.4f} - Checkpoint saved!")
                else:
                    patience_counter += 1
                
                # Early stopping - только если reward действительно не улучшается
                # Но не останавливаться слишком рано если метрики низкие
                if patience_counter >= self.config.training.early_stopping_patience:
                    # Проверить что мы действительно не улучшаемся
                    current_reward = eval_metrics.get('avg_reward', 0)
                    best_reward = best_metrics.get('avg_reward', -float('inf'))
                    
                    # Если reward очень низкий, не останавливаться рано (даем больше шансов)
                    if current_reward < 0.1 and epoch < self.config.training.ppo_epochs // 2:
                        logger.info(f"Reward still very low ({current_reward:.4f}), continuing training despite no improvement")
                        patience_counter = patience_counter // 2  # Уменьшаем счетчик, даем больше времени
                    else:
                        logger.info("Early stopping triggered")
                        print(f"\n[Early Stopping] No improvement for {patience_counter} epochs")
                        print(f"  Best reward: {best_reward:.4f}, Current reward: {current_reward:.4f}")
                        break
                
                print(f"  Eval time: {eval_time:.1f}s | Reward: {eval_metrics.get('avg_reward', 0):.4f}")
                # Print evaluation metrics if available
                if any(k.startswith('eval_') for k in eval_metrics.keys()):
                    print(f"  Metrics: ", end="")
                    metric_strs = []
                    for k in ['eval_bertscore', 'eval_codebleu', 'eval_bleu', 'eval_rouge', 'eval_ruby']:
                        if k in eval_metrics:
                            metric_strs.append(f"{k.replace('eval_', '')}={eval_metrics[k]:.4f}")
                    print(", ".join(metric_strs))
            
            # Calculate progress
            total_time = time.time() - start_time
            progress = ((epoch + 1) / self.config.training.ppo_epochs) * 100
            avg_time_per_epoch = total_time / (epoch + 1)
            remaining_epochs = self.config.training.ppo_epochs - (epoch + 1)
            eta_seconds = avg_time_per_epoch * remaining_epochs
            
            # Log metrics
            print(f"\n[Epoch {epoch+1} Summary]")
            print(f"  Time: {epoch_time:.1f}s | Total: {total_time/60:.1f}min")
            print(f"  Progress: {progress:.1f}% | ETA: {eta_seconds/60:.1f}min")
            print(f"  Train metrics: loss={train_metrics.get('loss', 0):.4f}, reward={train_metrics.get('reward', 0):.4f}")
            logger.info(f"Epoch {epoch+1}: {train_metrics}")
            if eval_dataloader is not None:
                logger.info(f"Eval metrics: {eval_metrics}")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.1f} minutes")
        print(f"{'='*60}\n")
        
        return best_metrics
