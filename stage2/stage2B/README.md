# Stage 2B - GPT-2 Feedback Generator for Reward Model

> **Model Architecture**: GPT-2 Base (117M) + Score Predictor (0.5M) + CodeBERT (125M) + Reward Heads (1.5M) = **~244.5M parameters**
>
> See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed model and training information.

## Overview

Stage 2B uses **GPT-2 as a feedback generator** to provide synthetic evaluations along three qualitative criteria (Consistency, Agreement, Usefulness). The feedback loop enables the reward model to filter out syntactically incorrect "hallucinations" and dramatically improve code quality metrics.

## Model Summary

| Component | Parameters | Description |
|-----------|------------|-------------|
| GPT-2 Feedback Generator | ~117.5M | Generates Consistency/Agreement/Usefulness scores |
| CodeBERT Reward Model | ~126.5M | Learns to predict feedback and filter hallucinations |
| **Total System** | **~244.5M** | Combined Stage 2B pipeline |

### GPT-2 Feedback Generator
| Component | Parameters |
|-----------|------------|
| GPT-2 Base | 117M |
| Score Predictor Head | 0.5M |

### CodeBERT Reward Model
| Component | Parameters |
|-----------|------------|
| CodeBERT-base Encoder | 125M |
| Shared Reward Layer | 1.1M |
| Criterion Heads (x3) | 0.4M |
| Quality Filter | 25K |

## Training Summary

| Parameter | Value |
|-----------|-------|
| Loss Function | MSE (Feedback Alignment) + BCE (Quality Filter) |
| Feedback Weight | 0.3 |
| Epochs | 20 |
| Learning Rate | 2e-5 |

## Expected Results

Based on the paper, Stage 2B achieves dramatic improvements:

| Metric | Baseline | Stage 2B Target | Improvement |
|--------|----------|-----------------|-------------|
| BERTScore | 0.35 | **0.90** | +157% |
| CodeBLEU | 0.23 | **0.80** | +248% |

## Key Features

### GPT-2 Synthetic Feedback Loop

The feedback generator evaluates code quality on three criteria:

1. **Consistency**: Does the code logically address the question?
2. **Agreement**: Does the code follow good programming practices?
3. **Usefulness**: Is the code practically useful?

### Hallucination Detection

The quality filter head specifically identifies and filters out:
- Syntactically incorrect code
- Non-functional "hallucinations"
- Code that doesn't match the question

### Training Data

Uses **CoNaLa corpus** as the primary data source:
- Text-to-Code pairs (intent → snippet)
- ~2,379 training examples

## Usage

### Basic Training (20 epochs)

```bash
python run_stage2b.py
```

### Custom Configuration

```bash
# Use larger GPT-2 model
python run_stage2b.py --gpt2-model gpt2-medium

# Adjust feedback weight
python run_stage2b.py --feedback-weight 0.5

# Quick test
python run_stage2b.py --epochs 2 --max-samples 100 --batch-size 4
```

### Full Options

```bash
python run_stage2b.py --help
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 8 | Batch size |
| `--learning-rate` | 2e-5 | Learning rate |
| `--gpt2-model` | gpt2 | GPT-2 model size |
| `--temperature` | 0.7 | GPT-2 sampling temperature |
| `--freeze-layers` | 4 | Encoder layers to freeze |
| `--feedback-weight` | 0.3 | Weight for feedback loss |

## Output

Training produces:
- `stage_2B/outputs/stage2b_results.json` - Full results with all metrics
- `stage_2B/outputs/stage2b_metrics.csv` - Per-epoch metrics table
- `stage_2B/artifacts/checkpoint_epoch_N/` - Model checkpoints

## Comparison with Stage 2A

| Aspect | Stage 2A | Stage 2B |
|--------|----------|----------|
| Primary Technique | Contrastive Learning | Feedback Loop |
| Feedback Source | Heuristic estimation | GPT-2 generated |
| Loss Function | Contrastive + InfoNCE | MSE + BCE |
| Key Innovation | Embedding clustering | Hallucination filtering |
| Total Parameters | ~127M | ~244.5M |
| Target BERTScore | 0.70 | 0.90 |
| Target CodeBLEU | 0.62 | 0.80 |

## Files

| File | Description |
|------|-------------|
| `feedback_generator.py` | GPT-2 feedback generator |
| `reward_model.py` | CodeBERT reward model |
| `train.py` | Training pipeline |
| `config.py` | Configuration classes |
| `ARCHITECTURE.md` | Detailed architecture documentation |

## References

- GPT-2: [OpenAI GPT-2](https://huggingface.co/gpt2)
- CodeBERT: [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)
- CoNaLa: [CoNaLa Corpus](https://conala-corpus.github.io/)
