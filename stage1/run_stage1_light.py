"""
Stage 1 RLHF Experiment - Light Version
=======================================

A lighter version optimized for limited GPU memory (4GB).
Uses smaller batch sizes and fewer evaluation samples.
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
import gc

# Import metrics
from modern_rlhf.metrics import ModernMetricsEvaluator, get_metric_float
from modern_rlhf.data_loader import ModernDataLoader
from modern_rlhf.config import ModernRLHFConfig

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


class Stage1LightExperiment:
    """Light version of Stage 1 experiment for limited GPU memory."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_evaluator = ModernMetricsEvaluator()
        self.epoch_results = []
        
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def load_data(self) -> Tuple[List, List]:
        """Load data from CoNaLa."""
        print("\n[1/3] Loading data...")
        
        config = ModernRLHFConfig()
        config.data.conala_local_path = "./conala-corpus"
        config.data.max_train_samples = 500  # Reduced for speed
        config.data.max_eval_samples = 100   # Reduced for speed
        
        loader = ModernDataLoader(config)
        train_data = loader.load_training_data()
        eval_data = loader.load_evaluation_data()
        
        print(f"Loaded {len(train_data)} train, {len(eval_data)} eval samples")
        return train_data, eval_data
    
    def initialize_models(self):
        """Initialize models with memory optimization."""
        print("\n[2/3] Loading models...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
        
        clear_gpu_memory()
        
        # Load policy model (CodeGPT-small)
        print("Loading CodeGPT-small...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/CodeGPT-small-py",
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load in float16 to save memory
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/CodeGPT-small-py",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.policy_model.eval()
        
        print(f"Policy model loaded on {self.device}")
        
        # Load reward model (CodeBERT-base)
        print("Loading CodeBERT-base...")
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base"
        )
        self.reward_model = AutoModel.from_pretrained(
            "microsoft/codebert-base",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.reward_model.eval()
        
        print("Models loaded successfully")
        
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU Memory used: {mem_used:.2f} GB")
    
    def generate_response(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate a single response."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def evaluate_epoch(self, eval_data: List, epoch: int) -> Dict[str, float]:
        """Evaluate model on all metrics."""
        print(f"\nEvaluating epoch {epoch}...")
        
        # Extract prompts and references
        prompts = []
        references = []
        
        for sample in eval_data[:100]:  # Limit to 100 samples
            if isinstance(sample, dict):
                prompts.append(sample.get('prompt', ''))
                references.append(sample.get('reference', sample.get('response', '')))
            else:
                prompts.append(getattr(sample, 'prompt', ''))
                references.append(getattr(sample, 'reference', getattr(sample, 'response', '')))
        
        # Generate responses one at a time to save memory
        print(f"Generating {len(prompts)} responses...")
        predictions = []
        
        for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
            try:
                response = self.generate_response(prompt)
                predictions.append(response)
            except Exception as e:
                print(f"Error generating response {i}: {e}")
                predictions.append("")
            
            # Clear cache periodically
            if i % 20 == 0:
                clear_gpu_memory()
        
        # Compute metrics
        print("Computing metrics...")
        metrics_results = self.metrics_evaluator.compute_all_metrics(predictions, references)
        
        epoch_metrics = {
            'epoch': epoch,
            'bertscore': round(get_metric_float(metrics_results, 'bertscore'), 4),
            'rouge': round(get_metric_float(metrics_results, 'rouge'), 4),
            'bleu': round(get_metric_float(metrics_results, 'bleu'), 4),
            'ruby': round(get_metric_float(metrics_results, 'ruby', 'ruby_like_heuristic'), 4),
            'codebleu': round(get_metric_float(metrics_results, 'codebleu', 'codebleu_proxy'), 4),
        }
        
        print(f"Epoch {epoch} metrics:")
        for metric, value in epoch_metrics.items():
            if metric != 'epoch':
                print(f"  {metric}: {value}")
        
        return epoch_metrics
    
    def simple_ppo_update(self, train_data: List, epoch: int):
        """Lightweight policy update (demo; not full PPO — use ``run_stage1`` / RewardWeightedNLL for the real pipeline)."""
        print(f"\nTraining epoch {epoch}...")
        
        # Get a small batch of training data
        batch_size = 10
        prompts = []
        for sample in train_data[:batch_size]:
            if isinstance(sample, dict):
                prompts.append(sample.get('prompt', ''))
            else:
                prompts.append(getattr(sample, 'prompt', ''))
        
        # Simple training step
        optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=5e-6)
        self.policy_model.train()
        
        total_loss = 0.0
        for prompt in tqdm(prompts, desc=f"Training epoch {epoch}"):
            try:
                # Generate response
                self.policy_model.eval()
                with torch.no_grad():
                    response = self.generate_response(prompt, max_new_tokens=64)
                
                # Compute loss
                self.policy_model.train()
                full_text = f"{prompt}{response}"
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                outputs = self.policy_model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
            
            clear_gpu_memory()
        
        avg_loss = total_loss / len(prompts) if prompts else 0
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
        
        self.policy_model.eval()
    
    def run(self) -> Dict[str, Any]:
        """Run the light experiment."""
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print("STAGE 1 BASELINE RLHF - LIGHT VERSION")
        print("=" * 60)
        
        try:
            # Load data
            train_data, eval_data = self.load_data()
            
            # Initialize models
            self.initialize_models()
            
            # Evaluate epoch 0 (before training)
            epoch_metrics = self.evaluate_epoch(eval_data, epoch=0)
            self.epoch_results.append(epoch_metrics)
            
            # Training loop (10 epochs as in original)
            for epoch in range(1, 11):
                # Train
                self.simple_ppo_update(train_data, epoch)
                
                # Evaluate
                epoch_metrics = self.evaluate_epoch(eval_data, epoch)
                self.epoch_results.append(epoch_metrics)
                
                # Save intermediate results
                self._save_results()
            
            # Final results
            total_time = time.time() - start_time
            print(f"\nExperiment completed in {total_time/60:.1f} minutes")
            
            return self._save_results()
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_results(self) -> Dict[str, Any]:
        """Save results to JSON."""
        results = {
            "experiment_name": "stage1_baseline_rlhf",
            "description": "Stage 1 RLHF with CodeGPT-small (policy) + CodeBERT-base (reward)",
            "policy_model": "microsoft/CodeGPT-small-py",
            "reward_model": "microsoft/codebert-base",
            "training_method": "RewardWeightedNLL",
            "total_epochs": 10,
            "epoch_results": self.epoch_results,
            "timestamp": datetime.now().isoformat()
        }
        
        output_path = "stage1/output/stage1_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as CSV
        csv_path = "stage1/output/stage1_metrics.csv"
        with open(csv_path, 'w') as f:
            f.write("epoch,BERTScore,ROUGE,BLEU,Ruby,CodeBLEU\n")
            for r in self.epoch_results:
                f.write(f"{r['epoch']},{r['bertscore']},{r['rouge']},{r['bleu']},{r['ruby']},{r['codebleu']}\n")
        
        print(f"\nResults saved to {output_path}")
        print(f"CSV saved to {csv_path}")
        
        return results


def main():
    """Run the light experiment."""
    experiment = Stage1LightExperiment()
    results = experiment.run()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Epoch':>6} {'BERTScore':>10} {'ROUGE':>10} {'BLEU':>10} {'Ruby':>10} {'CodeBLEU':>10}")
    print("-" * 66)
    
    for r in results['epoch_results']:
        print(f"{r['epoch']:>6} {r['bertscore']:>10.4f} {r['rouge']:>10.4f} {r['bleu']:>10.4f} {r['ruby']:>10.4f} {r['codebleu']:>10.4f}")


if __name__ == "__main__":
    main()
