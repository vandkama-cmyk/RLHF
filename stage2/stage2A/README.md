# Stage 2A - Contrastive Learning for Enhanced Reward Model

> **Model Architecture**: CodeBERT-base (125M) + Contrastive Projection Layer (0.8M) + Reward Head (1.2M) = **~127M parameters**
>
> See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed model and training information.

## Overview

Stage 2A introduces **contrastive learning** and an **improved reward function design** to enhance the discriminative capacity of the reward model. This stage learns an embedded space where consistent, agreeable, and useful solutions are clustered closer together than suboptimal responses.

## Model Summary

| Component | Parameters | Description |
|-----------|------------|-------------|
| CodeBERT-base Encoder | ~125M | Pre-trained code understanding model |
| Contrastive Projection Head | ~0.8M | Projects to 256-dim embedding space |
| Reward Prediction Head | ~1.2M | Predicts Consistent, Correct, Useful scores |
| **Total** | **~127M** | Full Stage 2A model |

## Training Summary

| Parameter | Value |
|-----------|-------|
| Loss Function | Maximum Contrastive Loss: `L = (1-y)*d² + y*max(0,margin-d)²` |
| Margin | 1.0 |
| Temperature (InfoNCE) | 0.07 |
| Epochs | 20 |
| Learning Rate | 2e-5 |

## Key Features

### Maximum Contrastive Loss Function

The core of Stage 2A is the maximum contrastive loss function:

```
L = (1-y) * d² + y * max(0, margin-d)²
```

Where:
- **d**: Euclidean distance between two embeddings
- **y**: Binary label indicating relationship between embeddings
  - `y=0` for similar pairs (both high or both low quality)
  - `y=1` for dissimilar pairs (one high, one low quality)
- **margin**: Hyperparameter defining the minimum distance between dissimilar pairs

**How it works:**
- For **similar pairs** (y=0): Loss = d² → minimize distance (pull together)
- For **dissimilar pairs** (y=1): Loss = max(0, margin-d)² → push apart if too close

### Architecture

1. **CodeBERT Encoder**: Pre-trained encoder for code understanding
2. **Contrastive Projection Head**: Projects embeddings to a lower-dimensional space optimized for contrastive learning
3. **Reward Prediction Heads**: Predicts quality scores (consistent, correct, useful)

### Training Data

Uses the aggregated SFT dataset:
- `datasets_for_training/sft_dataset.csv` (T2C-CoNaLa + T2T-SO)
- Extra SFT JSONL from `conala-corpus/conala-*.jsonl`

Training builds **synthetic negatives** by mismatching prompts with unrelated answers so the contrastive objective has both similar and dissimilar pairs.

## Metrics

Evaluated per epoch:
- **Reward Accuracy**: Mean accuracy across consistent/correct/useful heads
- **Reward Gap**: Mean reward score difference between positive and negative pairs
- **Val Loss**: BCE loss across reward heads

## Usage

### Basic Training (20 epochs)

```bash
python run_stage2a.py
```

### Custom Configuration

```bash
# Adjust contrastive margin
python run_stage2a.py --margin 1.5

# Smaller batch for low VRAM
python run_stage2a.py --batch-size 4 --gradient-accumulation 8

# Quick test
python run_stage2a.py --epochs 3 --max-samples 500
```

### Full Options

```bash
python run_stage2a.py --help
```

## Configuration

Key configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 8 | Batch size |
| `--learning-rate` | 2e-5 | Learning rate |
| `--margin` | 1.0 | Contrastive loss margin |
| `--temperature` | 0.07 | InfoNCE temperature |
| `--contrastive-weight` | 0.5 | Weight for contrastive loss |
| `--model` | microsoft/codebert-base | Base encoder model |

## Output

Training produces:
- `stage_2A/outputs/stage2a_results.json` - Full results with all metrics
- `stage_2A/outputs/stage2a_metrics.csv` - Per-epoch metrics table
- `stage_2A/artifacts/checkpoint_epoch_N/` - Model checkpoints

## Results Format

### Metrics CSV

```csv
epoch,reward_acc_mean,reward_acc_consistent,reward_acc_correct,reward_acc_useful,reward_pos_mean,reward_neg_mean,reward_gap,val_loss
0,0.5020,0.5000,0.5160,0.4900,0.5170,0.4920,0.0250,0.6931
1,0.6120,0.6200,0.5980,0.6180,0.5600,0.4400,0.1200,0.6210
...
20,0.4567,0.2345,0.3567,0.3789,0.4567,0.8901
```

### Results JSON

```json
{
  "experiment_name": "stage2a_contrastive_reward_model",
  "config": {...},
  "epoch_results": [
    {
      "epoch": 0,
      "bertscore": 0.3245,
      "bleu": 0.1234,
      "codebleu": 0.2345,
      "rouge": 0.2567,
      "ruby": 0.3456
    },
    ...
  ],
  "training_history": [...],
  "total_time_seconds": 3600.5
}
```

## Implementation Details

### Contrastive Pair Mining

Pairs are created within each batch:
1. **Similar pairs** (label=0): Both samples have similar quality (both high or both low)
2. **Dissimilar pairs** (label=1): One sample is high quality, the other is low quality

### Loss Components

Total loss = reward_weight × reward_loss + contrastive_weight × contrastive_loss

Where:
- `reward_loss`: BCE loss for quality prediction heads
- `contrastive_loss`: Maximum contrastive loss + InfoNCE loss

### Optimization

- AdamW optimizer with weight decay
- OneCycleLR scheduler
- Gradient accumulation for effective larger batches
- Mixed precision training (AMP) for faster training
- Exponential Moving Average (EMA) for stable predictions

## References

- CoNaLa Dataset: [https://conala-corpus.github.io/](https://conala-corpus.github.io/)
- CodeBERT: [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- Contrastive Learning: [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)
