"""
Stage 1 LoRA — policy fine-tuning with Low-Rank Adaptation.

Instead of full fine-tuning of CodeGPT-small-py, only low-rank adapter matrices
(r=8, alpha=16) injected into the attention projection layers are updated.
All original weights remain frozen. This reduces trainable parameters from 124M
to ~0.5M while preserving pre-trained representations.

LoRA config:
  task_type : CAUSAL_LM
  r          : 8
  lora_alpha : 16
  target_modules : ["c_attn"]   (GPT-2 combined QKV projection)
  lora_dropout   : 0.05
  bias           : "none"

Otherwise identical to Stage 1 (Reward-Weighted NLL objective, same data,
same CodeBERT reward model, same eval metrics).

Results saved to stage1/outputs/stage1_lora_results.json.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.config import Stage1Config, get_stage1_config
from stage1.run_stage1 import Stage1Experiment

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)


class Stage1LoRAExperiment(Stage1Experiment):
    """Stage 1 with LoRA adapters on the policy model."""

    ALGORITHM_NAME = "RewardWeightedNLL_LoRA"

    def initialize_models(self) -> None:
        super().initialize_models()

        if not PEFT_AVAILABLE:
            raise ImportError(
                "peft is required for LoRA experiments. "
                "Install it with: pip install peft>=0.3.0"
            )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            # "c_attn" is the combined QKV projection in GPT-2 / CodeGPT
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none",
        )
        self.policy_model = get_peft_model(self.policy_model, lora_config)
        self.policy_model.print_trainable_parameters()

        # Rebuild optimizer over LoRA params only
        trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
        )
        logger.info(
            "Stage1 LoRA: %d trainable parameters",
            sum(p.numel() for p in trainable_params),
        )

    def _get_output_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "outputs",
            "stage1_lora_results.json",
        )

    def run(self) -> None:
        # Delegate to parent run() but override save path
        results = super().run()
        return results


def main():
    config = get_stage1_config()
    experiment = Stage1LoRAExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
