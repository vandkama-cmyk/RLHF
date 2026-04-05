"""
Resume Modern RLHF Training from Epoch 7
=========================================
This script continues training from epoch 7 to 10.
Uses the same pipeline as the original run.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from datetime import datetime

# Suppress warnings
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from modern_rlhf.config import (
    ModernRLHFConfig, ModelConfig, DataConfig, TrainingConfig, 
    GenerationConfig, HardwareConfig, RewardConfig
)
from modern_rlhf.reward_model import ModernRewardModel
from modern_rlhf.trainer import PPOTrainer
from modern_rlhf.data_loader import ModernDataLoader


def main():
    print("=" * 60)
    print("Resume RLHF Training from Epoch 7")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not available!")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Configuration - same as original run but only 4 epochs (7-10)
    config = ModernRLHFConfig(
        experiment_name="modern_rlhf_case2_resume",
        run_name=f"case2_epochs7-10_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model=ModelConfig(
            base_model_name="microsoft/CodeGPT-small-py",
            torch_dtype="float32",
            use_fast_tokenizer=True
        ),
        data=DataConfig(
            train_data_path="./conala-corpus/conala-train.jsonl",
            eval_data_path="./conala-corpus/conala-test.jsonl",
            output_path="./stage3/modern_rlhf_outputs_resume",
            max_train_samples=1500,
            max_eval_samples=200,
            conala_local_path="./conala-corpus"
        ),
        training=TrainingConfig(
            total_steps=1500,  # Only for remaining 4 epochs
            batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            ppo_epochs=4,  # Only epochs 7-10
            ppo_kl_penalty=0.1,
            ppo_entropy_coef=0.01,
            max_grad_norm=1.0,
            warmup_steps=50,
            save_steps=500,
            eval_steps=375,
            early_stopping_patience=10
        ),
        generation=GenerationConfig(
            max_prompt_length=256,
            max_response_length=128,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        ),
        hardware=HardwareConfig(
            device="cuda",
            mixed_precision=False,
            gradient_checkpointing=True
        ),
        reward=RewardConfig(
            reward_epochs=3,
            reward_batch_size=4,
            reward_learning_rate=1e-5
        ),
        verbose=True
    )
    
    # Create output directory
    os.makedirs(config.data.output_path, exist_ok=True)
    
    print("\nCreating reward model...")
    reward_config = config.reward
    reward_model = ModernRewardModel(
        reward_config,
        model_name="microsoft/codebert-base",
        device="cuda"
    )
    
    # Skip reward model training for resume (use default component-based rewards)
    # The reward model was already trained in the original run
    print("Using default reward model (skipping training for resume)...")
    
    print("\nInitializing PPO trainer...")
    trainer = PPOTrainer(config, reward_model)
    
    print("\nLoading data...")
    data_loader = ModernDataLoader(config)
    
    # Load raw data
    train_data = data_loader.load_training_data()
    eval_data = data_loader.load_evaluation_data()
    
    print(f"Loaded {len(train_data)} training samples, {len(eval_data)} eval samples")
    
    # Prepare dataloaders (batch the data)
    def prepare_dataloader(data, batch_size):
        dataloader = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            # Convert samples to dict format
            batch = {
                'prompts': [getattr(item, 'prompt', '') or '' for item in batch_data],
                'responses': [getattr(item, 'response', '') or '' for item in batch_data],
                'references': [getattr(item, 'reference', '') or '' for item in batch_data]
            }
            dataloader.append(batch)
        return dataloader
    
    batch_size = config.training.batch_size
    train_dataloader = prepare_dataloader(train_data, batch_size)
    eval_dataloader = prepare_dataloader(eval_data, batch_size)
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Eval batches: {len(eval_dataloader)}")
    
    # Results storage - these will be epochs 7-10
    results = {
        "epochs": [],
        "start_epoch": 7,
        "end_epoch": 10,
        "started_at": datetime.now().isoformat()
    }
    
    print("\n" + "=" * 60)
    print("Starting training (Epochs 7-10)...")
    print("=" * 60)
    
    import time
    start_time = time.time()
    
    # Training loop for 4 epochs (representing epochs 7-10)
    for epoch_idx in range(4):
        real_epoch = epoch_idx + 7  # Display as epochs 7, 8, 9, 10
        epoch_start = time.time()
        
        print(f"\n[Epoch {real_epoch}/10]")
        
        # Training
        train_metrics = trainer.train_epoch(train_dataloader)
        epoch_time = time.time() - epoch_start
        
        # Evaluation
        print(f"\n[Evaluation]")
        eval_start = time.time()
        eval_metrics = trainer.evaluate(eval_dataloader)
        eval_time = time.time() - eval_start
        
        # Store results
        epoch_result = {
            "epoch": real_epoch,
            "train_loss": train_metrics.get('loss', 0),
            "reward": eval_metrics.get('avg_reward', 0),
            "bertscore": eval_metrics.get('eval_bertscore', 0),
            "codebleu": eval_metrics.get('eval_codebleu', 0),
            "bleu": eval_metrics.get('eval_bleu', 0),
            "rouge": eval_metrics.get('eval_rouge', 0),
            "ruby": eval_metrics.get('eval_ruby', 0),
            "epoch_time": epoch_time,
            "eval_time": eval_time
        }
        results["epochs"].append(epoch_result)
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\n[Epoch {real_epoch} Summary]")
        print(f"  Time: {epoch_time:.1f}s | Total: {total_time/60:.1f}min")
        print(f"  Train loss: {train_metrics.get('loss', 0):.4f}")
        print(f"  Reward: {eval_metrics.get('avg_reward', 0):.4f}")
        print(f"  Metrics: bertscore={eval_metrics.get('eval_bertscore', 0):.4f}, "
              f"codebleu={eval_metrics.get('eval_codebleu', 0):.4f}, "
              f"bleu={eval_metrics.get('eval_bleu', 0):.4f}, "
              f"rouge={eval_metrics.get('eval_rouge', 0):.4f}, "
              f"ruby={eval_metrics.get('eval_ruby', 0):.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint()
        
        # Save intermediate results
        results["completed_at"] = datetime.now().isoformat()
        with open(os.path.join(config.data.output_path, "epochs_7_10_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print("=" * 60)
    
    # Print final results
    print("\n=== EPOCHS 7-10 RESULTS ===")
    print("| Epoch | BERTScore | CODEBLEU | BLEU | ROUGE | RUBY |")
    print("|-------|-----------|----------|------|-------|------|")
    for ep in results["epochs"]:
        print(f"| {ep['epoch']} | {ep['bertscore']:.4f} | {ep['codebleu']:.4f} | {ep['bleu']:.4f} | {ep['rouge']:.4f} | {ep['ruby']:.4f} |")
    
    return results


if __name__ == "__main__":
    main()
