"""
Stage 1B — policy optimization with clipped PPO-style updates.

Includes: value head for advantage estimation, KL to a frozen reference policy,
and a multi-epoch PPO inner loop with importance sampling (policy / old-policy ratio).

Stage 1 (`run_stage1.py`) uses Reward-Weighted NLL only; this module is the true PPO experiment.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.config import Stage1Config, get_stage1_config
from stage1.run_stage1 import Stage1Experiment

logger = logging.getLogger(__name__)


class Stage1BExperiment(Stage1Experiment):
    """Same data and metrics as Stage 1, but policy training uses PPO."""

    ALGORITHM_NAME = "PPO"

    def initialize_models(self) -> None:
        super().initialize_models()
        # Frozen snapshot for KL and ratio baseline
        self.ref_policy = copy.deepcopy(self.policy_model)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        hs = getattr(
            self.policy_model.config,
            "hidden_size",
            getattr(self.policy_model.config, "n_embd", 768),
        )
        self.value_head = nn.Linear(hs, 1).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.policy_model.parameters()) + list(self.value_head.parameters()),
            lr=self.config.training.learning_rate,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        logger.info("Stage1B: reference policy frozen; value head dim=%s", hs)

    def _prompt_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def _mean_response_logprob(
        self,
        model: nn.Module,
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """Mean log p(response tokens | prompt) under `model` (single sequence)."""
        full_text = f"{prompt}{response}"
        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        plen = self._prompt_token_len(prompt)

        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = out.logits
            logp = F.log_softmax(logits[:, :-1, :], dim=-1)
            labels = input_ids[:, 1:]
            tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            seq_len = input_ids.size(1)
            mask = (torch.arange(seq_len - 1, device=self.device) >= (plen - 1)).float()
            denom = mask.sum().clamp(min=1.0)
            return (tok_lp * mask).sum() / denom

    def _value_at_last_token(self, prompt: str, response: str) -> torch.Tensor:
        full_text = f"{prompt}{response}"
        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            out = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hid = out.hidden_states[-1]
            last_i = attention_mask.sum(dim=1) - 1
            v = self.value_head(hid[0, last_i[0]])
        return v.squeeze()

    def _entropy_mean_response(self, prompt: str, response: str) -> torch.Tensor:
        full_text = f"{prompt}{response}"
        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        plen = self._prompt_token_len(prompt)
        with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            out = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            ent = -(p * logp).sum(dim=-1)
            seq_len = input_ids.size(1)
            mask = (torch.arange(seq_len - 1, device=self.device) >= (plen - 1)).float()
            return (ent * mask).sum() / mask.sum().clamp(min=1.0)

    def train_epoch(self, train_data: List, epoch: int) -> Dict[str, float]:
        """PPO-style epoch with inner multi-pass optimization on rollouts."""
        logger.info("Stage1B PPO training epoch %s...", epoch)
        self.policy_model.train()
        self.value_head.train()

        sample_size = min(500, len(train_data))
        train_samples = train_data[:sample_size]
        prompts: List[str] = []
        for sample in train_samples:
            if isinstance(sample, dict):
                prompts.append(sample.get("prompt", ""))
            else:
                prompts.append(getattr(sample, "prompt", ""))

        batch_size = self.config.training.batch_size
        ppo_epochs = self.config.training.ppo_epochs
        clip = self.config.training.ppo_clip_ratio
        vf_coef = self.config.training.ppo_value_loss_coef
        ent_coef = self.config.training.ppo_entropy_coef
        kl_coef = self.config.training.ppo_kl_penalty

        total_loss = 0.0
        total_reward = 0.0
        num_batches = 0

        pbar = tqdm(range(0, len(prompts), batch_size), desc=f"Stage1B PPO Epoch {epoch}")

        for i in pbar:
            batch_prompts = prompts[i : i + batch_size]

            self.policy_model.eval()
            self.value_head.eval()
            with torch.no_grad():
                responses = self.generate_responses(batch_prompts)
                rewards = self.compute_reward(batch_prompts, responses)

                rollouts: List[Tuple[str, str, float, torch.Tensor, torch.Tensor, torch.Tensor]] = []
                for p, r, rew in zip(batch_prompts, responses, rewards):
                    old_lp = self._mean_response_logprob(self.policy_model, p, r)
                    ref_lp = self._mean_response_logprob(self.ref_policy, p, r)
                    v_old = self._value_at_last_token(p, r)
                    rollouts.append(
                        (
                            p,
                            r,
                            float(rew.item()),
                            old_lp.detach(),
                            ref_lp.detach(),
                            v_old.detach(),
                        )
                    )

            self.policy_model.train()
            self.value_head.train()

            batch_loss_acc = 0.0
            valid = 0

            for _ in range(ppo_epochs):
                for p, r, rew, old_lp, ref_lp, v_old in rollouts:
                    new_lp = self._mean_response_logprob(self.policy_model, p, r)
                    v = self._value_at_last_token(p, r)
                    ent = self._entropy_mean_response(p, r)

                    rew_t = torch.tensor(rew, device=self.device, dtype=v.dtype)
                    advantage = rew_t - v_old
                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
                    pol_loss = -torch.min(surr1, surr2)
                    vf_loss = F.mse_loss(v, rew_t)
                    kl = ref_lp - new_lp
                    loss = pol_loss + vf_coef * vf_loss - ent_coef * ent + kl_coef * kl

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy_model.parameters()) + list(self.value_head.parameters()),
                        self.config.training.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    batch_loss_acc += float(loss.detach().item())
                    valid += 1

            if valid > 0:
                total_loss += batch_loss_acc / valid
            total_reward += float(rewards.mean().item())
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / max(1, num_batches):.4f}",
                    "reward": f"{total_reward / max(1, num_batches):.4f}",
                }
            )

        avg_loss = total_loss / max(1, num_batches)
        avg_reward = total_reward / max(1, num_batches)
        logger.info(
            "Epoch %s Stage1B (%s): loss=%.4f, reward=%.4f",
            epoch,
            self.ALGORITHM_NAME,
            avg_loss,
            avg_reward,
        )
        return {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_reward": avg_reward,
            "num_batches": num_batches,
            "algorithm": self.ALGORITHM_NAME,
        }

    def run(self) -> Dict[str, Any]:
        start_time = time.time()
        print("\n" + "=" * 70)
        print("STAGE 1B — PPO RLHF EXPERIMENT")
        print("=" * 70)
        print(f"Algorithm:    {self.ALGORITHM_NAME}")
        print(f"Policy Model: {self.config.model.base_model_name}")
        print(f"Reward Model: {self.config.model.reward_model_name} (preference / Bradley–Terry)")
        print(f"Training Epochs: {self.config.training.total_epochs}")
        print("=" * 70)

        train_data, eval_data = self.load_data()
        self.epoch_results = []
        self.training_history = []
        self.initialize_models()
        self._setup_preference_reward(train_data)

        self.epoch_results.append(self.evaluate_epoch(eval_data, epoch=0))

        for epoch in range(1, self.config.training.total_epochs + 1):
            self.training_history.append(self.train_epoch(train_data, epoch))
            self.epoch_results.append(self.evaluate_epoch(eval_data, epoch))
            self._save_checkpoint(epoch)

        results = self._save_results_stage1b(time.time() - start_time)
        print("\n" + "=" * 70)
        print("STAGE 1B EXPERIMENT COMPLETED")
        print("=" * 70)
        self._print_results_table()
        return results

    def _save_results_stage1b(self, total_time: float) -> Dict[str, Any]:
        import json

        out_dir = self.config.data.output_path
        os.makedirs(out_dir, exist_ok=True)
        results = {
            "experiment_name": self.config.experiment_name + "_stage1b_ppo",
            "algorithm": self.ALGORITHM_NAME,
            "config": self.config.to_dict(),
            "epoch_results": self.epoch_results,
            "training_history": self.training_history,
            "total_time_seconds": total_time,
            "timestamp": datetime.now().isoformat(),
        }
        path = os.path.join(out_dir, "stage1b_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Stage1B results saved to %s", path)
        self._save_metrics_csv_stage1b(out_dir)
        return results

    def _save_metrics_csv_stage1b(self, out_dir: str) -> None:
        import csv

        csv_path = os.path.join(out_dir, "stage1b_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "BERTScore", "ROUGE", "BLEU", "Ruby", "CodeBLEU"])
            for result in self.epoch_results:
                w.writerow(
                    [
                        result["epoch"],
                        f"{result['bertscore']:.4f}",
                        f"{result['rouge']:.4f}",
                        f"{result['bleu']:.4f}",
                        f"{result['ruby']:.4f}",
                        f"{result['codebleu']:.4f}",
                    ]
                )
        logger.info("Stage1B metrics CSV saved to %s", csv_path)


def main() -> Dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1B PPO RLHF Experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = get_stage1_config()
    config.seed = args.seed
    base_out = config.data.output_path
    if args.seed != 42:
        base_out = f"./stage1/output/seed_{args.seed}"
    config.data.output_path = os.path.join(base_out, "stage1b_ppo")
    os.makedirs(config.data.output_path, exist_ok=True)

    experiment = Stage1BExperiment(config)
    return experiment.run()


if __name__ == "__main__":
    main()
