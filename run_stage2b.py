#!/usr/bin/env python
"""
Stage 2B - GPT-2 Feedback Generator for Reward Model
=====================================================

Entry point for Stage 2B experiment.

Usage:
    python run_stage2b.py --epochs 20 --batch-size 8 --max-samples 2000

Model Architecture:
    - GPT-2 Base (117M params) + Score Predictor Head (~0.5M params)
    - CodeBERT-base (125M params) + Reward Heads (~1.5M params)
    - Total: ~244.5M parameters

Expected Results:
    - BERTScore: 0.35 → 0.90 (157% improvement)
    - CodeBLEU: 0.23 → 0.80 (248% improvement)
"""

import os
import sys
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(
        description='Stage 2B: GPT-2 Feedback Generator for Reward Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full training (20 epochs)
    python run_stage2b.py --epochs 20

    # Quick test
    python run_stage2b.py --epochs 2 --max-samples 100 --batch-size 4

Model Architecture:
    Component 1: GPT-2 Feedback Generator
      - GPT-2 Base: 117M parameters
      - Score Predictor Head: ~0.5M parameters
      
    Component 2: CodeBERT Reward Model  
      - CodeBERT-base Encoder: 125M parameters
      - Reward Heads: ~1.5M parameters
      - Quality Filter: ~25K parameters
      
    Total System: ~244.5M parameters
        """
    )
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--max-samples', type=int, default=2000,
                        help='Maximum training samples (default: 2000)')
    
    # Feedback generator arguments
    parser.add_argument('--gpt2-model', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
                        help='GPT-2 model size (default: gpt2)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='GPT-2 sampling temperature (default: 0.7)')
    
    # Reward model arguments
    parser.add_argument('--freeze-layers', type=int, default=4,
                        help='Number of encoder layers to freeze (default: 4)')
    parser.add_argument('--feedback-weight', type=float, default=0.3,
                        help='Weight for feedback loss (default: 0.3)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='stage2/stage2B/outputs',
                        help='Output directory (default: stage2/stage2B/outputs)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 70)
    print("STAGE 2B - GPT-2 FEEDBACK GENERATOR CONFIGURATION")
    print("=" * 70)
    print(f"Epochs:              {args.epochs}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Learning rate:       {args.lr}")
    print(f"Max samples:         {args.max_samples}")
    print(f"")
    print("Feedback Generator (GPT-2):")
    print(f"  Model:             {args.gpt2_model}")
    print(f"  Temperature:       {args.temperature}")
    print(f"")
    print("Reward Model (CodeBERT):")
    print(f"  Frozen layers:     {args.freeze_layers}")
    print(f"  Feedback weight:   {args.feedback_weight}")
    print(f"")
    print("Model Parameters:")
    print(f"  GPT-2 Base:        ~117M parameters")
    print(f"  Score Predictor:   ~0.5M parameters")
    print(f"  CodeBERT:          ~125M parameters")
    print(f"  Reward Heads:      ~1.5M parameters")
    print(f"  Total:             ~244.5M parameters")
    print("=" * 70)
    
    # Import and run
    from stage2.stage2B.config import get_stage2b_config
    from stage2.stage2B.train import Stage2BExperiment
    
    # Create config
    config = get_stage2b_config()
    config.training.total_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.data.max_train_samples = args.max_samples
    config.feedback_generator.model_name = args.gpt2_model
    config.feedback_generator.temperature = args.temperature
    config.reward_model.freeze_encoder_layers = args.freeze_layers
    config.training.feedback_weight = args.feedback_weight
    config.data.output_path = args.output_dir
    
    # Run experiment
    experiment = Stage2BExperiment(config)
    experiment.run()


if __name__ == '__main__':
    main()
