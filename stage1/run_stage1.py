"""
Stage 1 RLHF Experiment
=======================

Baseline RLHF setup using:
- CodeGPT-small as the policy trained with Reward-Weighted NLL (not PPO)
- CodeBERT-based preference reward model (Bradley–Terry)

Metrics evaluated per epoch:
- BERTScore
- ROUGE
- BLEU
- Ruby
- CodeBLEU
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm

from stage1.config import Stage1Config, get_stage1_config
from stage1.preference_reward_model import (
    PreferenceRewardModel,
    load_preference_triples_from_csv,
    load_reward_checkpoint,
    save_reward_model,
    train_preference_reward_model,
    build_preference_pairs,
)

# Import from modern_rlhf
from modern_rlhf.metrics import ModernMetricsEvaluator, get_metric_float
from modern_rlhf.data_loader import ModernDataLoader
from modern_rlhf.config import ModernRLHFConfig

logger = logging.getLogger(__name__)


class Stage1Experiment:
    """Stage 1 baseline RLHF experiment (policy optimization: Reward-Weighted NLL)."""

    ALGORITHM_NAME = "RewardWeightedNLL"
    
    def __init__(self, config: Optional[Stage1Config] = None):
        self.config = config or get_stage1_config()
        
        # Setup device
        self.device = torch.device(self.config.hardware.device)
        
        # Set random seeds for reproducibility
        self._set_seeds(self.config.seed)
        
        # Initialize metrics evaluator
        self.metrics_evaluator = ModernMetricsEvaluator()
        
        # Results storage
        self.epoch_results = []
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Stage 1 Experiment initialized (%s)", self.ALGORITHM_NAME)
        logger.info(f"Policy model: {self.config.model.base_model_name}")
        logger.info(f"Reward encoder: {self.config.model.reward_model_name}")
        logger.info(f"Device: {self.device}")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.config.data.output_path
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(log_dir, 'stage1_experiment.log'))
            ]
        )
    
    def load_data(self) -> Tuple[List, List]:
        """Load training and evaluation data."""
        logger.info("Loading data...")
        
        # Create a ModernRLHFConfig to use the data loader
        modern_config = ModernRLHFConfig()
        modern_config.data.train_data_path = self.config.data.train_data_path
        modern_config.data.eval_data_path = self.config.data.eval_data_path
        modern_config.data.human_feedback_path = self.config.data.human_feedback_path
        modern_config.data.conala_local_path = self.config.data.conala_local_path
        modern_config.data.max_train_samples = self.config.data.max_train_samples
        modern_config.data.max_eval_samples = self.config.data.max_eval_samples
        modern_config.data.min_prompt_length = self.config.data.min_prompt_length
        modern_config.data.max_prompt_length = self.config.data.max_prompt_length
        modern_config.data.min_response_length = self.config.data.min_response_length
        modern_config.data.max_response_length = self.config.data.max_response_length
        modern_config.data.require_code_like = self.config.data.require_code_like
        modern_config.data.use_data_augmentation = self.config.data.use_data_augmentation
        modern_config.evaluation.eval_datasets = self.config.evaluation.eval_datasets
        
        data_loader = ModernDataLoader(modern_config)
        
        train_data = data_loader.load_training_data()
        eval_data = data_loader.load_evaluation_data()
        
        logger.info(f"Loaded {len(train_data)} training samples, {len(eval_data)} eval samples")
        
        return train_data, eval_data
    
    def initialize_models(self):
        """Initialize policy and reward models."""
        logger.info("Initializing models...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load policy model (CodeGPT-small)
        logger.info(f"Loading policy model: {self.config.model.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            use_fast=False,
            local_files_only=True
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load in float32 so GradScaler+autocast work correctly
        # (autocast handles fp16 compute; model params stay fp32 for gradient accumulation)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.float32,
            local_files_only=True
        ).to(self.device)
        
        # Preference reward model (encoder + scalar head; weights trained in run() or loaded from disk)
        logger.info(f"Initializing preference reward model (encoder): {self.config.model.reward_model_name}")
        self.preference_reward = PreferenceRewardModel(
            self.config.model.reward_model_name,
            self.device,
            trust_remote_code=self.config.model.trust_remote_code,
            local_files_only=True,
        )

        # Create optimizer and AMP scaler once (persistent across epochs)
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.training.learning_rate
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        logger.info("Models initialized successfully")
    
    def compute_reward(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Scalar rewards in (0, 1) from the preference score head (higher = better)."""
        scores = self.preference_reward.score_pairs(prompts, responses)
        return torch.sigmoid(scores)

    def _setup_preference_reward(self, train_data: List) -> None:
        """Train or load Bradley–Terry preference reward model."""
        rc = self.config.reward
        ckpt = os.path.join(self.config.data.output_path, rc.reward_checkpoint_name)

        if rc.train_preference_reward and (rc.force_retrain_reward or not os.path.isfile(ckpt)):
            if getattr(rc, "preference_csv_path", None) and os.path.isfile(rc.preference_csv_path):
                pairs = load_preference_triples_from_csv(rc.preference_csv_path)
                pairs = pairs[: rc.preference_max_pairs]
                logger.info("Loaded %d preference pairs from %s", len(pairs), rc.preference_csv_path)
            else:
                pairs = build_preference_pairs(
                    train_data,
                    max_samples=rc.preference_max_pairs,
                    seed=self.config.seed,
                )
            logger.info("Training preference reward model on %d pairs (%s)", len(pairs), self.ALGORITHM_NAME)
            train_preference_reward_model(
                self.preference_reward,
                pairs,
                epochs=rc.preference_epochs,
                batch_size=rc.reward_batch_size,
                lr=rc.preference_lr,
                max_grad_norm=self.config.training.max_grad_norm,
            )
            save_reward_model(ckpt, self.preference_reward)
        elif os.path.isfile(ckpt):
            logger.info("Loading preference reward checkpoint from %s", ckpt)
            self.preference_reward = load_reward_checkpoint(
                ckpt,
                self.device,
                trust_remote_code=self.config.model.trust_remote_code,
                local_files_only=True,
            )
        else:
            logger.warning(
                "No reward checkpoint at %s and training disabled — using randomly initialized reward head",
                ckpt,
            )
    
    def generate_responses(self, prompts: List[str]) -> List[str]:
        """Generate code responses for prompts."""
        self.policy_model.eval()
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.generation.max_prompt_length,
                    padding=True
                ).to(self.device)
                
                # Generate (fallback to greedy if sampling produces NaN)
                try:
                    outputs = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=self.config.generation.max_new_tokens,
                        temperature=self.config.generation.temperature,
                        top_p=self.config.generation.top_p,
                        top_k=self.config.generation.top_k,
                        do_sample=self.config.generation.do_sample,
                        repetition_penalty=self.config.generation.repetition_penalty,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                except RuntimeError:
                    outputs = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=self.config.generation.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode only the generated part
                input_length = inputs.input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                responses.append(response)
        
        return responses
    
    def evaluate_epoch(self, eval_data: List, epoch: int) -> Dict[str, float]:
        """Evaluate model on all metrics for current epoch."""
        logger.info(f"Evaluating epoch {epoch}...")
        
        # Sample evaluation data
        sample_size = min(self.config.evaluation.eval_samples, len(eval_data))
        eval_samples = eval_data[:sample_size]
        
        # Extract prompts and references
        prompts = []
        references = []
        
        for sample in eval_samples:
            if isinstance(sample, dict):
                prompts.append(sample.get('prompt', ''))
                references.append(sample.get('reference', sample.get('response', '')))
            else:
                prompts.append(getattr(sample, 'prompt', ''))
                references.append(getattr(sample, 'reference', getattr(sample, 'response', '')))
        
        # Generate responses
        logger.info(f"Generating {len(prompts)} responses...")
        predictions = []
        batch_size = self.config.evaluation.eval_batch_size
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = self.generate_responses(batch_prompts)
            predictions.extend(batch_responses)
        
        # Compute all metrics
        logger.info("Computing metrics...")
        metrics_results = self.metrics_evaluator.compute_all_metrics(predictions, references)
        
        # Extract scores
        epoch_metrics = {
            'epoch': epoch,
            'bertscore': get_metric_float(metrics_results, 'bertscore'),
            'rouge': get_metric_float(metrics_results, 'rouge'),
            'bleu': get_metric_float(metrics_results, 'bleu'),
            'ruby': get_metric_float(metrics_results, 'ruby', 'ruby_like_heuristic'),
            'codebleu': get_metric_float(metrics_results, 'codebleu', 'codebleu_proxy'),
            'metric_keys': {
                'codebleu': 'codebleu' if 'codebleu' in metrics_results else 'codebleu_proxy',
                'ruby': 'ruby' if 'ruby' in metrics_results else 'ruby_like_heuristic',
            },
        }
        
        # Add detailed info
        for metric_name, result in metrics_results.items():
            if result.details:
                epoch_metrics[f'{metric_name}_details'] = result.details
            if result.error:
                epoch_metrics[f'{metric_name}_error'] = result.error
        
        logger.info(f"Epoch {epoch} metrics:")
        logger.info(f"  BERTScore: {epoch_metrics['bertscore']:.4f}")
        logger.info(f"  ROUGE:     {epoch_metrics['rouge']:.4f}")
        logger.info(f"  BLEU:      {epoch_metrics['bleu']:.4f}")
        logger.info(f"  Ruby:      {epoch_metrics['ruby']:.4f}")
        logger.info(f"  CodeBLEU:  {epoch_metrics['codebleu']:.4f}")
        
        return epoch_metrics
    
    def train_epoch(self, train_data: List, epoch: int) -> Dict[str, float]:
        """One epoch of Reward-Weighted NLL policy updates."""
        logger.info("Training epoch %s (%s)...", epoch, self.ALGORITHM_NAME)
        
        self.policy_model.train()
        
        # Sample training data
        sample_size = min(500, len(train_data))
        train_samples = train_data[:sample_size]
        
        # Extract prompts
        prompts = []
        for sample in train_samples:
            if isinstance(sample, dict):
                prompts.append(sample.get('prompt', ''))
            else:
                prompts.append(getattr(sample, 'prompt', ''))
        
        # Training metrics
        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0
        
        batch_size = self.config.training.batch_size

        pbar = tqdm(range(0, len(prompts), batch_size), desc=f"Training Epoch {epoch}")

        for i in pbar:
            batch_prompts = prompts[i:i + batch_size]

            # Generate responses (in eval mode, no grad)
            self.policy_model.eval()
            with torch.no_grad():
                responses = self.generate_responses(batch_prompts)

            # Compute rewards
            rewards = self.compute_reward(batch_prompts, responses)
            total_reward += rewards.mean().item()

            # Policy gradient update (one step per batch)
            self.policy_model.train()
            self.optimizer.zero_grad()

            batch_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
            valid_samples = 0

            for prompt, response, reward in zip(batch_prompts, responses, rewards):
                # Tokenize
                full_text = f"{prompt}{response}"
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=False
                ).to(self.device)

                # Build labels: mask padding positions with -100
                labels = inputs.input_ids.clone()
                if self.tokenizer.pad_token_id is not None:
                    labels[labels == self.tokenizer.pad_token_id] = -100

                # Forward pass with AMP autocast
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    outputs = self.policy_model(**inputs, labels=labels)

                if outputs.loss is None or torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                    continue

                # Scale loss by reward; accumulate across samples in batch
                sample_loss = outputs.loss * (1.0 + reward.item()) / max(1, len(batch_prompts))
                self.scaler.scale(sample_loss).backward()
                batch_loss = batch_loss + sample_loss.detach()
                valid_samples += 1

            if valid_samples > 0:
                # Unscale before clipping so clip threshold is in original scale
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(),
                    self.config.training.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += batch_loss.item()

            num_batches += 1
            pbar.set_postfix({
                'loss': f'{total_loss / max(1, num_batches):.4f}',
                'reward': f'{total_reward / max(1, num_batches):.4f}'
            })
        
        avg_loss = total_loss / max(1, num_batches)
        avg_reward = total_reward / max(1, num_batches)
        
        training_metrics = {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'avg_reward': avg_reward,
            'num_batches': num_batches
        }
        
        logger.info(
            "Epoch %s training (%s): loss=%.4f, reward=%.4f",
            epoch,
            self.ALGORITHM_NAME,
            avg_loss,
            avg_reward,
        )
        
        return training_metrics
    
    def run(self) -> Dict[str, Any]:
        """Run the full Stage 1 experiment."""
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("STAGE 1 BASELINE RLHF EXPERIMENT")
        print("=" * 70)
        print(f"Algorithm:    {self.ALGORITHM_NAME}")
        print(f"Policy Model: {self.config.model.base_model_name}")
        print(f"Reward Model: {self.config.model.reward_model_name} (preference / Bradley–Terry)")
        print(f"Training Epochs: {self.config.training.total_epochs}")
        print("=" * 70)
        
        try:
            # Load data
            train_data, eval_data = self.load_data()
            
            # Initialize models
            self.initialize_models()
            self._setup_preference_reward(train_data)
            
            # Evaluate before training (epoch 0)
            epoch_metrics = self.evaluate_epoch(eval_data, epoch=0)
            self.epoch_results.append(epoch_metrics)
            
            # Training loop
            for epoch in range(1, self.config.training.total_epochs + 1):
                # Train
                train_metrics = self.train_epoch(train_data, epoch)
                self.training_history.append(train_metrics)
                
                # Evaluate
                epoch_metrics = self.evaluate_epoch(eval_data, epoch)
                self.epoch_results.append(epoch_metrics)
                
                # Save checkpoint
                self._save_checkpoint(epoch)
            
            # Save final results
            total_time = time.time() - start_time
            results = self._save_results(total_time)
            
            print("\n" + "=" * 70)
            print("EXPERIMENT COMPLETED")
            print("=" * 70)
            self._print_results_table()
            
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.data.output_path, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.policy_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _save_results(self, total_time: float) -> Dict[str, Any]:
        """Save experiment results."""
        results = {
            'experiment_name': self.config.experiment_name,
            'algorithm': self.ALGORITHM_NAME,
            'config': self.config.to_dict(),
            'epoch_results': self.epoch_results,
            'training_history': self.training_history,
            'total_time_seconds': total_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON
        results_path = os.path.join(self.config.data.output_path, 'stage1_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Save metrics table as CSV
        self._save_metrics_csv()
        
        return results
    
    def _save_metrics_csv(self):
        """Save metrics as CSV table."""
        import csv
        
        csv_path = os.path.join(self.config.data.output_path, 'stage1_metrics.csv')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'BERTScore', 'ROUGE', 'BLEU', 'Ruby', 'CodeBLEU'])
            
            for result in self.epoch_results:
                writer.writerow([
                    result['epoch'],
                    f"{result['bertscore']:.4f}",
                    f"{result['rouge']:.4f}",
                    f"{result['bleu']:.4f}",
                    f"{result['ruby']:.4f}",
                    f"{result['codebleu']:.4f}"
                ])
        
        logger.info(f"Metrics CSV saved to {csv_path}")
    
    def _print_results_table(self):
        """Print results table to console."""
        print("\nMetrics by Epoch:")
        print("-" * 70)
        print(f"{'Epoch':>6} {'BERTScore':>10} {'ROUGE':>10} {'BLEU':>10} {'Ruby':>10} {'CodeBLEU':>10}")
        print("-" * 70)
        
        for result in self.epoch_results:
            print(f"{result['epoch']:>6} "
                  f"{result['bertscore']:>10.4f} "
                  f"{result['rouge']:>10.4f} "
                  f"{result['bleu']:>10.4f} "
                  f"{result['ruby']:>10.4f} "
                  f"{result['codebleu']:>10.4f}")
        
        print("-" * 70)


def main():
    """Run Stage 1 experiment.

    Accepts an optional --seed argument (default 42).  When seed != 42
    results are saved to stage1/output/seed_{seed}/ so that multi-seed
    runs don't overwrite each other.
    """
    import argparse as _ap
    parser = _ap.ArgumentParser(description="Stage 1 RLHF Experiment")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    config = get_stage1_config()
    config.seed = args.seed
    if args.seed != 42:
        config.data.output_path = f"./stage1/output/seed_{args.seed}"

    experiment = Stage1Experiment(config)
    results = experiment.run()

    print("\nExperiment completed successfully!")
    print(f"Results saved to: {config.data.output_path}")

    return results


if __name__ == "__main__":
    main()
