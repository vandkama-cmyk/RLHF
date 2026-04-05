# ClassifLLM v3 - Real LLM Feedback Integration

## Overview

ClassifLLM v3 introduces **real LLM feedback generation** instead of synthetic labels used in previous versions. The system uses actual LLM API calls to generate authentic human-like feedback for code quality assessment.

## Key Features

### 🚀 Real LLM Feedback
- **OpenAI GPT models** (GPT-3.5, GPT-4)
- **Anthropic Claude** models
- **Local transformers** models
- **Structured prompts** for consistent evaluation
- **Confidence scoring** and weighting

### 🧠 Enhanced Training
- All v2 features: cross-attention, contrastive learning, AMP, EMA
- **Confidence-weighted training** based on LLM certainty
- **Feedback caching** for efficiency
- **Multi-provider support**

### 📊 Quality Assessment
Three evaluation criteria with structured feedback:
- **Consistent**: How well the answer matches the question (-2 to +2)
- **Correct**: Technical correctness of the code (-2 to +2)
- **Useful**: Practical value and helpfulness (-2 to +2)

## Quick Start

### 1. With OpenAI GPT-4

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Train with GPT-4 feedback
python clasifLLM/v3/train.py --llm-provider openai --api-key YOUR_API_KEY --epochs 15
```

### 2. With HuggingFace API

```bash
# Train with HuggingFace Inference API (using HF_TOKEN env var)
export HF_TOKEN="your-huggingface-token"
python clasifLLM/v3/train.py --llm-provider huggingface --llm-model gpt2 --epochs 15

# Note: If HF_TOKEN is invalid/expired, system will automatically use heuristic fallback

# Or specify token directly
python clasifLLM/v3/train.py --llm-provider huggingface --llm-model microsoft/DialoGPT-medium --api-key YOUR_HF_TOKEN
```

### 3. With Local Model

```bash
# Train with local DialoGPT (no API key needed)
python clasifLLM/v3/train.py --llm-provider local --llm-model microsoft/DialoGPT-small --epochs 10
```

### 3. Synthetic Baseline (like v2)

```bash
# Train without LLM feedback (uses synthetic labels)
python clasifLLM/v3/train.py --epochs 15
```

## Installation

### Requirements

```bash
# Core dependencies
pip install torch transformers tqdm scikit-learn

# For OpenAI feedback
pip install openai

# For local models (optional)
pip install accelerate  # For GPU acceleration with local models
```

### Environment Setup

```bash
# Clone and setup
cd clasifLLM/v3

# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Configuration

### LLM Providers

| Provider | Models | Setup |
|----------|--------|-------|
| **OpenAI** | `gpt-4`, `gpt-3.5-turbo` | `export OPENAI_API_KEY="..."` |
| **Anthropic** | `claude-3-sonnet-20240229` | `export ANTHROPIC_API_KEY="..."` |
| **HuggingFace** | `gpt2`, `microsoft/DialoGPT-medium` | `export HF_TOKEN="..."` or `--api-key` |
| **Local** | Generative models (DialoGPT, GPT-2) | No API key needed, fallback to heuristics |

### Example Prompts

The system uses structured prompts like this:

```json
{
  "prompt": "You are an expert code reviewer evaluating the quality of programming answers.\n\nPlease analyze this question and answer pair:\n\nQUESTION: How to sort a list in Python?\n\nANSWER: sorted_list = sorted(my_list)\n\nEvaluate the answer on three criteria using a scale from -2 to +2:\n- consistent: How well does the answer match what was asked?\n- correct: Is the code technically correct and functional?\n- useful: How practical and helpful is this solution?\n\nRespond with valid JSON...",
  "response_format": {
    "consistent": "number",
    "correct": "number",
    "useful": "number",
    "explanation": "text",
    "confidence": "number"
  }
}
```

## Training Examples

### Full Research Configuration

```bash
python clasifLLM/v3/train.py \
    --llm-provider openai \
    --llm-model gpt-4 \
    --model-type graphcodebert \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --epochs 20 \
    --learning-rate 2e-5 \
    --max-feedback-samples 2000 \
    --use-confidence-weighting
```

### Fast Testing Configuration

```bash
python clasifLLM/v3/train.py \
    --llm-provider local \
    --llm-model microsoft/codebert-base \
    --batch-size 8 \
    --epochs 5 \
    --max-feedback-samples 200
```

### Production Configuration

```bash
python clasifLLM/v3/train.py \
    --llm-provider anthropic \
    --llm-model claude-3-sonnet-20240229 \
    --batch-size 2 \
    --gradient-accumulation 8 \
    --epochs 30 \
    --freeze-layers 2 \
    --max-feedback-samples 5000
```

## Output Files

Training generates several files:

```
clasifLLM/v3/artifacts/
├── training_history_llm_v3.json    # Training metrics history
├── best_model_llm_v3.pt            # Best model checkpoint
└── llm_feedback_cache.json         # Cached LLM feedback (if used)
```

## API Reference

### LLMFeedbackGenerator

```python
from clasifLLM.v3.llm_feedback import LLMFeedbackGenerator, LLMConfig

# Configure LLM
config = LLMConfig(
    provider="openai",
    model_name="gpt-4",
    api_key="your-key",
    temperature=0.7,
    batch_size=3
)

# Create generator
generator = LLMFeedbackGenerator(config)

# Generate feedback for samples
results = await generator.generate_feedback_batch(samples)
```

### Training Pipeline

```python
from clasifLLM.v3.integrated_system import train_llm_system_v3

# Configure and train
await train_llm_system_v3(args)
```

## Performance Comparison

| Version | Feedback Type | Training Time | Quality | Cost |
|---------|---------------|---------------|---------|------|
| **v1** | Synthetic | Fast | Baseline | Free |
| **v2** | Synthetic | Medium | Improved | Free |
| **v3** | Real LLM | Slow | Best | API costs |

## Troubleshooting

### Common Issues

1. **OpenAI API Rate Limits**
   ```bash
   # Reduce batch size and increase delays
   --llm-batch-size 1 --llm-max-retries 5
   ```

2. **Transformers Not Available**
   ```bash
   # Install transformers (optional for local models)
   pip install transformers torch accelerate

   # If transformers fails, system will automatically use heuristic fallback
   ```

3. **Wrong Local Model**
   ```bash
   # Use generative models, not understanding models
   --llm-model microsoft/DialoGPT-small  # Good
   --llm-model microsoft/codebert-base    # Bad (cannot generate text)
   ```

2. **Local Model Memory Issues**
   ```bash
   # Use smaller batch sizes
   --batch-size 2 --gradient-accumulation 2
   ```

3. **Feedback Generation Timeout**
   ```bash
   # Increase timeout and reduce complexity
   --llm-timeout 60 --max-feedback-samples 500
   ```

### Cache Management

```bash
# Force regenerate all feedback
--force-regenerate-feedback

# Use existing cache (default)
# Cache is saved as llm_feedback_cache.json
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Question      │    │   LLM Feedback   │    │   Training      │
│   + Answer      │───▶│   Generator      │───▶│   Pipeline      │
│   Samples       │    │   (OpenAI/Local) │    │   (Enhanced)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Base Dataset  │    │  Structured      │    │   Fine-tuned    │
│   Generation    │    │  Quality Scores  │    │   CodeBERT      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Contributing

1. **Add new LLM providers**: Extend `BaseLLMProvider` class
2. **Improve prompts**: Modify prompt templates in provider classes
3. **Add metrics**: Extend evaluation in the training pipeline

## Citation

If you use ClassifLLM v3 in your research:

```bibtex
@misc{classifllm-v3,
  title={ClassifLLM v3: Real LLM Feedback for Code Quality Assessment},
  author={Your Name},
  year={2024},
  note={Real LLM feedback integration for authentic code evaluation}
}
```
