#!/usr/bin/env python3
"""
Run Stage 2A - Contrastive Learning for Enhanced Reward Model
=============================================================

Stage 2A introduces contrastive learning and an improved reward function design
to enhance the discriminative capacity of the reward model.

Key features:
- Maximum contrastive loss function: L = (1-y) * d² + y * max(0, margin-d)²
- Enhanced reward model with contrastive projections
- Uses aggregated SFT dataset (T2C-CoNaLa + T2T-SO) from sft_dataset.csv
- 20 epochs training with per-epoch metrics

Metrics evaluated per epoch:
- Reward accuracy (mean across heads)
- Reward gap (positive vs negative)
- Validation loss

Usage:
    python run_stage2a.py
    python run_stage2a.py --epochs 20 --margin 1.0
    python run_stage2a.py --batch-size 4 --learning-rate 2e-5
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage2.stage2A.config import get_stage2a_config
from stage2.stage2A.train import Stage2AExperiment


def parse_args():
    """Parse command line arguments."""
    # Get default config to use its values as defaults
    default_config = get_stage2a_config()
    
    parser = argparse.ArgumentParser(
        description="Stage 2A - Contrastive Learning for Enhanced Reward Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (20 epochs)
    python run_stage2a.py

    # Run with custom margin for contrastive loss
    python run_stage2a.py --margin 1.5

    # Run with smaller batch size for low VRAM
    python run_stage2a.py --batch-size 4 --gradient-accumulation 8

    # Quick test run
    python run_stage2a.py --epochs 3 --max-samples 500
        """
    )
    
    # Training settings - use config defaults
    parser.add_argument("--epochs", type=int, default=default_config.training.total_epochs,
                        help=f"Number of training epochs (default: {default_config.training.total_epochs})")
    parser.add_argument("--batch-size", type=int, default=default_config.training.batch_size,
                        help=f"Batch size (default: {default_config.training.batch_size})")
    parser.add_argument("--learning-rate", type=float, default=default_config.training.learning_rate,
                        help=f"Learning rate (default: {default_config.training.learning_rate})")
    parser.add_argument("--gradient-accumulation", type=int, default=default_config.training.gradient_accumulation_steps,
                        help=f"Gradient accumulation steps (default: {default_config.training.gradient_accumulation_steps})")
    
    # Contrastive learning settings - use config defaults
    parser.add_argument("--margin", type=float, default=default_config.contrastive.margin,
                        help=f"Margin for maximum contrastive loss (default: {default_config.contrastive.margin})")
    parser.add_argument("--temperature", type=float, default=default_config.contrastive.temperature,
                        help=f"Temperature for InfoNCE loss (default: {default_config.contrastive.temperature})")
    parser.add_argument("--contrastive-weight", type=float, default=default_config.contrastive.contrastive_weight,
                        help=f"Weight for contrastive loss (default: {default_config.contrastive.contrastive_weight})")
    
    # Model settings - use config defaults
    parser.add_argument("--model", type=str, default=default_config.model.base_model_name,
                        help=f"Base model for reward model (default: {default_config.model.base_model_name})")
    parser.add_argument("--projection-dim", type=int, default=default_config.model.projection_dim,
                        help=f"Projection dimension for contrastive head (default: {default_config.model.projection_dim})")
    parser.add_argument("--dropout", type=float, default=default_config.model.dropout,
                        help=f"Dropout rate (default: {default_config.model.dropout})")
    parser.add_argument("--freeze-layers", type=int, default=default_config.model.freeze_encoder_layers,
                        help=f"Number of encoder layers to freeze (default: {default_config.model.freeze_encoder_layers})")
    
    # Data settings - use config defaults
    parser.add_argument("--max-samples", type=int, default=default_config.data.max_train_samples,
                        help=f"Maximum training samples (default: {default_config.data.max_train_samples})")
    parser.add_argument("--val-ratio", type=float, default=default_config.data.val_ratio,
                        help=f"Validation set ratio (default: {default_config.data.val_ratio})")
    parser.add_argument("--negative-ratio", type=float, default=default_config.data.negative_ratio,
                        help=f"Negative sample ratio per positive (default: {default_config.data.negative_ratio})")
    
    # Hardware settings
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--no-ema", action="store_true",
                        help=f"Disable EMA (EMA is {'enabled' if default_config.training.use_ema else 'disabled'} by default)")
    
    # Other
    parser.add_argument("--seed", type=int, default=default_config.seed,
                        help=f"Random seed (default: {default_config.seed})")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience (default: None, train all epochs)")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get default config
    config = get_stage2a_config()
    
    # Override with command line arguments
    config.training.total_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.gradient_accumulation_steps = args.gradient_accumulation
    config.training.use_amp = not args.no_amp
    # Only disable EMA if --no-ema is explicitly set
    if args.no_ema:
        config.training.use_ema = False
    config.training.patience = args.patience
    
    config.contrastive.margin = args.margin
    config.contrastive.temperature = args.temperature
    config.contrastive.contrastive_weight = args.contrastive_weight
    
    config.model.base_model_name = args.model
    config.model.projection_dim = args.projection_dim
    config.model.dropout = args.dropout
    config.model.freeze_encoder_layers = args.freeze_layers
    
    config.data.max_train_samples = args.max_samples
    config.data.val_ratio = args.val_ratio
    config.data.negative_ratio = args.negative_ratio
    
    config.seed = args.seed
    
    if args.device:
        config.hardware.device = args.device
    
    # Print configuration
    print("\n" + "=" * 70)
    print("STAGE 2A - CONTRASTIVE LEARNING CONFIGURATION")
    print("=" * 70)
    print(f"Epochs:              {config.training.total_epochs}")
    print(f"Batch size:          {config.training.batch_size}")
    print(f"Learning rate:       {config.training.learning_rate}")
    print(f"Gradient accum:      {config.training.gradient_accumulation_steps}")
    print(f"Effective batch:     {config.training.batch_size * config.training.gradient_accumulation_steps}")
    print()
    print("Contrastive Learning:")
    print(f"  Margin:            {config.contrastive.margin}")
    print(f"  Temperature:       {config.contrastive.temperature}")
    print(f"  Contrastive weight: {config.contrastive.contrastive_weight}")
    print()
    print("Model:")
    print(f"  Base model:        {config.model.base_model_name}")
    print(f"  Projection dim:    {config.model.projection_dim}")
    print(f"  Frozen layers:     {config.model.freeze_encoder_layers}")
    print()
    print("Hardware:")
    print(f"  Device:            {config.hardware.device}")
    print(f"  Mixed precision:   {config.training.use_amp}")
    print(f"  EMA:               {config.training.use_ema}")
    print()
    print("Data:")
    print(f"  SFT dataset:       {config.data.train_data_path}")
    if config.data.extra_sft_jsonl_paths:
        print(f"  Extra SFT JSONL:   {', '.join(config.data.extra_sft_jsonl_paths)}")
    print(f"  Negative ratio:    {config.data.negative_ratio}")
    print("=" * 70)
    
    # Run experiment
    experiment = Stage2AExperiment(config)
    results = experiment.run()
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    if results['epoch_results']:
        final = results['epoch_results'][-1]
        print(f"Final epoch ({final['epoch']}) metrics:")
        print(f"  RewardAcc: {final.get('reward_acc_mean', 0):.4f}")
        if 'reward_gap' in final:
            print(f"  RewardGap: {final.get('reward_gap', 0):.4f}")
        print(f"  Val Loss:  {final.get('val_loss', 0):.4f}")
    
    print(f"\nTotal training time: {results['total_time_seconds'] / 60:.2f} minutes")
    print(f"Results saved to: {config.data.output_path}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
