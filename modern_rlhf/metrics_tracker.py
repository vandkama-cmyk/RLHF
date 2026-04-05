"""
Metrics tracking and visualization for RLHF training.
Provides real-time progress monitoring and metrics logging.
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, plotting disabled")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available, advanced metrics analysis disabled")


@dataclass
class EpochMetrics:
    """Container for metrics from a single epoch."""
    epoch: int
    timestamp: float
    
    # Training metrics
    loss: float
    reward: float
    kl_divergence: float
    entropy: float
    learning_rate: float
    
    # Evaluation metrics (optional)
    eval_loss: Optional[float] = None
    bertscore: Optional[float] = None
    codebleu: Optional[float] = None
    bleu: Optional[float] = None
    rouge: Optional[float] = None
    ruby: Optional[float] = None
    
    # Timing
    epoch_time: Optional[float] = None
    samples_per_second: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class MetricsTracker:
    """Track, save, and visualize training metrics."""
    
    def __init__(self, output_dir: str = "./modern_outputs/metrics"):
        """Initialize metrics tracker.
        
        Args:
            output_dir: Directory to save metrics and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history: List[EpochMetrics] = []
        self.start_time = time.time()
        
        # CSV file for continuous logging
        self.csv_file = self.output_dir / "training_metrics.csv"
        self.csv_initialized = False
        
        logger.info(f"MetricsTracker initialized. Output: {self.output_dir}")
    
    def log_epoch(self, metrics: EpochMetrics):
        """Log metrics for a completed epoch.
        
        Args:
            metrics: EpochMetrics object with all relevant data
        """
        # Add to history
        self.history.append(metrics)
        
        # Save to CSV
        self._save_to_csv(metrics)
        
        # Save to JSON (full history)
        self._save_to_json()
        
        # Generate plots
        if MATPLOTLIB_AVAILABLE:
            self._generate_plots()
        
        # Print summary
        self._print_epoch_summary(metrics)
    
    def _save_to_csv(self, metrics: EpochMetrics):
        """Append metrics to CSV file."""
        metrics_dict = metrics.to_dict()
        
        # Initialize CSV with headers if needed
        if not self.csv_initialized:
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
                writer.writeheader()
            self.csv_initialized = True
        
        # Append metrics
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            writer.writerow(metrics_dict)
        
        logger.debug(f"Metrics saved to {self.csv_file}")
    
    def _save_to_json(self):
        """Save full metrics history to JSON."""
        json_file = self.output_dir / "training_metrics.json"
        
        history_dict = [m.to_dict() for m in self.history]
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.debug(f"Full history saved to {json_file}")
    
    def _generate_plots(self):
        """Generate visualization plots."""
        if not self.history:
            return
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Extract data
        epochs = [m.epoch for m in self.history]
        
        # Plot 1: Training Loss and Reward
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        losses = [m.loss for m in self.history]
        rewards = [m.reward for m in self.history]
        
        ax1.plot(epochs, losses, marker='o', label='Loss', color='#e74c3c')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss over Epochs')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(epochs, rewards, marker='o', label='Reward', color='#2ecc71')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Reward')
        ax2.set_title('Average Reward over Epochs')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "training_progress.png", dpi=150)
        plt.close()
        
        # Plot 2: KL Divergence and Entropy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        kl_divs = [m.kl_divergence for m in self.history]
        entropies = [m.entropy for m in self.history]
        
        ax1.plot(epochs, kl_divs, marker='o', label='KL Divergence', color='#3498db')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('KL Divergence')
        ax1.set_title('KL Divergence over Epochs')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(epochs, entropies, marker='o', label='Entropy', color='#9b59b6')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Entropy')
        ax2.set_title('Policy Entropy over Epochs')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "policy_metrics.png", dpi=150)
        plt.close()
        
        # Plot 3: Evaluation Metrics (if available)
        eval_metrics = {
            'BERTScore': [m.bertscore for m in self.history if m.bertscore is not None],
            'CodeBLEU': [m.codebleu for m in self.history if m.codebleu is not None],
            'BLEU': [m.bleu for m in self.history if m.bleu is not None],
            'ROUGE': [m.rouge for m in self.history if m.rouge is not None],
            'RUBY': [m.ruby for m in self.history if m.ruby is not None],
        }
        
        # Only plot if we have eval metrics
        if any(len(v) > 0 for v in eval_metrics.values()):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            eval_epochs = [m.epoch for m in self.history if m.bertscore is not None]
            
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            for (name, values), color in zip(eval_metrics.items(), colors):
                if values:
                    ax.plot(eval_epochs[:len(values)], values, marker='o', label=name, color=color)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title('Evaluation Metrics over Epochs')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / "evaluation_metrics.png", dpi=150)
            plt.close()
        
        logger.info(f"Plots saved to {plots_dir}/")
    
    def _print_epoch_summary(self, metrics: EpochMetrics):
        """Print a formatted summary of epoch metrics."""
        print("\n" + "="*80)
        print(f"EPOCH {metrics.epoch} SUMMARY")
        print("="*80)
        
        # Training metrics
        print("\n[Training Metrics]")
        print(f"  Loss:          {metrics.loss:.6f}")
        print(f"  Reward:        {metrics.reward:.6f}")
        print(f"  KL Divergence: {metrics.kl_divergence:.6f}")
        print(f"  Entropy:       {metrics.entropy:.6f}")
        print(f"  Learning Rate: {metrics.learning_rate:.2e}")
        
        # Evaluation metrics (if available)
        if metrics.bertscore is not None:
            print("\n[Evaluation Metrics]")
            if metrics.bertscore: print(f"  BERTScore:  {metrics.bertscore:.4f}")
            if metrics.codebleu: print(f"  CodeBLEU:   {metrics.codebleu:.4f}")
            if metrics.bleu: print(f"  BLEU:       {metrics.bleu:.4f}")
            if metrics.rouge: print(f"  ROUGE:      {metrics.rouge:.4f}")
            if metrics.ruby: print(f"  RUBY:       {metrics.ruby:.4f}")
        
        # Timing
        if metrics.epoch_time:
            print("\n[Performance]")
            print(f"  Epoch Time:         {metrics.epoch_time:.2f}s")
            if metrics.samples_per_second:
                print(f"  Samples/sec:        {metrics.samples_per_second:.2f}")
            
            # ETA
            if len(self.history) > 1:
                avg_epoch_time = sum(m.epoch_time or 0 for m in self.history) / len(self.history)
                remaining_epochs = self.get_remaining_epochs()
                if remaining_epochs > 0:
                    eta_seconds = avg_epoch_time * remaining_epochs
                    eta_minutes = eta_seconds / 60
                    print(f"  Estimated Time Remaining: {eta_minutes:.1f} min ({eta_seconds:.0f}s)")
        
        print("="*80 + "\n")
    
    def get_remaining_epochs(self) -> int:
        """Calculate remaining epochs (needs to be set externally)."""
        # This should be set by the trainer
        return getattr(self, '_total_epochs', 0) - len(self.history)
    
    def set_total_epochs(self, total: int):
        """Set total number of epochs for ETA calculation."""
        self._total_epochs = total
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics achieved so far."""
        if not self.history:
            return {}
        
        return {
            'best_reward': max(m.reward for m in self.history),
            'lowest_loss': min(m.loss for m in self.history),
            'best_bertscore': max((m.bertscore for m in self.history if m.bertscore), default=0),
            'best_codebleu': max((m.codebleu for m in self.history if m.codebleu), default=0),
            'best_bleu': max((m.bleu for m in self.history if m.bleu), default=0),
        }
    
    def export_summary(self) -> Dict[str, Any]:
        """Export a summary of all metrics."""
        if not self.history:
            return {}
        
        total_time = time.time() - self.start_time
        
        return {
            'total_epochs': len(self.history),
            'total_training_time': total_time,
            'average_epoch_time': sum(m.epoch_time or 0 for m in self.history) / len(self.history),
            'best_metrics': self.get_best_metrics(),
            'final_metrics': self.history[-1].to_dict(),
            'metrics_file': str(self.csv_file),
            'plots_dir': str(self.output_dir / "plots"),
        }

