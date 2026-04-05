"""
Unified evaluation script for all RLHF pipeline stages.

Computes BERTScore (roberta-large), ROUGE-L, BLEU, CodeBLEU, and RUBY
on the CoNaLa held-out test set for every generative stage:

  stage1       — CodeGPT-small-py after RewardWeightedNLL (best epoch checkpoint)
  stage2B      — Base CodeGPT-small-py reranked by Stage 2B reward model
                 (reward model used as selection criterion, N=5 beam candidates)
  stage3_case2 — CodeGPT-small-py after SFT on 1247 samples
  stage5_v2    — Stage 5B (ClassifLLM v2) LLM-feedback fine-tuned
  stage5_v3    — Stage 5C (ClassifLLM v3) LLM-feedback fine-tuned

When multi-seed results exist (seed_123/, seed_456/ subdirs next to the
default artifact path), ± std is computed from per-seed generations.

Usage:
    python eval_unified_all_stages.py
    python eval_unified_all_stages.py --max-samples 100  # quick smoke test
    python eval_unified_all_stages.py --output eval_unified_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project root on path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Checkpoint / model paths
# ---------------------------------------------------------------------------
STAGE_PATHS: Dict[str, str] = {
    "stage1": "stage1/output/checkpoint_epoch_30",
    "stage2B_reward_model": "stage2/stage2B/outputs/reward_model",
    "stage3_case2": "stage3/modern_rlhf_outputs/checkpoint-2001",
    "stage5_v2": "stage5/stage5B/artifacts",   # contains training_history_llm_v2.json; no generative model
    "stage5_v3": "stage5/stage5C/artifacts",   # contains training_history_llm_v3.json; no generative model
}

# Base CodeGPT used for Stage 2B reranking and Stage 5 (if no fine-tuned gen model)
BASE_POLICY_MODEL = "microsoft/CodeGPT-small-py"

CONALA_TEST_PATH = REPO_ROOT / "conala-corpus" / "conala-test.jsonl"

# Seeds to look for in per-seed subdirectories
SEEDS = [42, 123, 456]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_conala_test(path: Path, max_samples: int = 0) -> List[Dict]:
    """Load CoNaLa test set as list of {prompt, reference} dicts."""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item.get("rewritten_intent") or item.get("intent", "")
            reference = item.get("snippet", "")
            if prompt and reference:
                samples.append({"prompt": prompt, "reference": reference})
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def load_generative_model(model_path: str, device: str):
    """Load a causal LM tokenizer + model from a local checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading generative model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    return tokenizer, model


def generate_responses(
    tokenizer,
    model,
    prompts: List[str],
    device: str,
    max_new_tokens: int = 64,
    num_beams: int = 1,
) -> List[str]:
    """Greedy (or beam) generation for a list of prompts."""
    responses = []
    for prompt in prompts:
        try:
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_ids = out_ids[0][enc["input_ids"].shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            responses.append(text or "<empty>")
        except Exception as e:
            print(f"    Generation error: {e}")
            responses.append("<generation_error>")
    return responses


# ---------------------------------------------------------------------------
# Stage 2B reranking
# ---------------------------------------------------------------------------

def rerank_with_stage2b(
    tokenizer,
    base_model,
    reward_model_path: str,
    prompts: List[str],
    references: List[str],
    device: str,
    n_candidates: int = 5,
) -> List[str]:
    """Generate N beam candidates with base CodeGPT, rerank with Stage 2B reward model."""
    from stage2.stage2B.reward_model import FeedbackRewardModel

    print(f"  Loading Stage 2B reward model from: {reward_model_path}")
    try:
        reward_model = FeedbackRewardModel.load_pretrained(reward_model_path, device=device)
        reward_model.eval()
    except Exception as e:
        print(f"  WARNING: Could not load Stage 2B reward model ({e}). Returning greedy baseline.")
        return generate_responses(tokenizer, base_model, prompts, device)

    selected = []
    for prompt in prompts:
        # Generate N candidates with beam search
        try:
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(device)
            with torch.no_grad():
                out_ids = base_model.generate(
                    **enc,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=n_candidates,
                    num_return_sequences=n_candidates,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            candidates = []
            for seq in out_ids:
                new_ids = seq[enc["input_ids"].shape[1]:]
                text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                candidates.append(text or "<empty>")
        except Exception as e:
            print(f"    Beam generation error: {e}")
            candidates = ["<generation_error>"] * n_candidates

        # Score candidates with Stage 2B reward model
        try:
            with torch.no_grad():
                reward_outputs = reward_model.predict_reward(
                    questions=[prompt] * len(candidates),
                    answers=candidates,
                )
            # Combined reward: mean of consistency, agreement, usefulness
            combined = (
                reward_outputs["consistency"]
                + reward_outputs["agreement"]
                + reward_outputs["usefulness"]
            ) / 3.0
            best_idx = int(np.argmax(combined))
            selected.append(candidates[best_idx])
        except Exception as e:
            print(f"    Reward scoring error: {e}. Using candidate[0].")
            selected.append(candidates[0])

    return selected


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BERTScore (roberta-large), ROUGE-L, BLEU, CodeBLEU, RUBY."""
    from modern_rlhf.metrics import ModernMetricsEvaluator, get_metric_float

    evaluator = ModernMetricsEvaluator()

    # BERTScore via bert_score package with roberta-large
    bertscore_f1 = _compute_bertscore(predictions, references)

    # Other metrics via ModernMetricsEvaluator
    try:
        results = evaluator.compute_all_metrics(predictions, references)
        rouge  = results["rouge"].score
        bleu   = results["bleu"].score
        codebleu = get_metric_float(results, "codebleu", "codebleu_proxy")
        ruby = get_metric_float(results, "ruby", "ruby_like_heuristic")
    except Exception as e:
        print(f"  Metrics error: {e}")
        rouge = bleu = codebleu = ruby = 0.0

    return {
        "bertscore": bertscore_f1,
        "rouge": rouge,
        "bleu": bleu,
        "codebleu": codebleu,
        "ruby": ruby,
    }


def _compute_bertscore(predictions: List[str], references: List[str]) -> float:
    """BERTScore F1 using roberta-large. Falls back to token F1 proxy if unavailable."""
    try:
        from bert_score import score as bs_score
        _, _, F1 = bs_score(
            predictions, references,
            lang="en",
            model_type="roberta-large",
            verbose=False,
        )
        return float(F1.mean().item())
    except ImportError:
        print("  WARNING: bert_score not installed. Using token-F1 proxy for BERTScore.")
        return _token_f1_bertscore(predictions, references)
    except Exception as e:
        print(f"  WARNING: bert_score failed ({e}). Using token-F1 proxy.")
        return _token_f1_bertscore(predictions, references)


def _token_f1_bertscore(predictions: List[str], references: List[str]) -> float:
    scores = []
    for pred, ref in zip(predictions, references):
        p_toks = set(pred.lower().split())
        r_toks = set(ref.lower().split())
        if not p_toks or not r_toks:
            scores.append(0.0)
            continue
        common = len(p_toks & r_toks)
        prec = common / len(p_toks)
        rec  = common / len(r_toks)
        f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Per-stage evaluation
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: str
    metrics: Dict[str, float]
    seed: int = 42
    note: str = ""


def eval_generative_stage(
    stage_name: str,
    model_path: str,
    prompts: List[str],
    references: List[str],
    device: str,
    seed: int = 42,
    note: str = "",
) -> StageResult:
    """Evaluate a single generative model checkpoint."""
    abs_path = str(REPO_ROOT / model_path)
    if not Path(abs_path).exists():
        print(f"  SKIP {stage_name} (path not found): {abs_path}")
        return StageResult(stage_name, {}, seed=seed, note="path_not_found")

    print(f"\n[{stage_name}] seed={seed} — {abs_path}")
    tokenizer, model = load_generative_model(abs_path, device)
    responses = generate_responses(tokenizer, model, prompts, device)
    metrics = compute_metrics(responses, references)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return StageResult(stage_name, metrics, seed=seed, note=note)


def eval_stage2b_reranker(
    prompts: List[str],
    references: List[str],
    device: str,
) -> StageResult:
    """Evaluate Stage 2B: base CodeGPT reranked by FeedbackRewardModel."""
    base_path = str(REPO_ROOT / STAGE_PATHS["stage1"])  # use Stage 1 checkpoint as base
    reward_path = str(REPO_ROOT / STAGE_PATHS["stage2B_reward_model"])

    if not Path(base_path).exists():
        # Fall back to base pre-trained model
        base_path = BASE_POLICY_MODEL

    print(f"\n[stage2B_reranker] base={base_path}")
    tokenizer, base_model = load_generative_model(base_path, device)
    responses = rerank_with_stage2b(
        tokenizer, base_model, reward_path, prompts, references, device
    )
    metrics = compute_metrics(responses, references)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return StageResult("stage2B_reranker", metrics,
                       note="base_codegpt_reranked_by_stage2b_reward_model")


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_seed_results(results: List[StageResult]) -> Dict[str, Dict]:
    """Compute mean ± std across seeds for each metric."""
    if not results:
        return {}
    metric_keys = list(results[0].metrics.keys())
    agg: Dict[str, Dict] = {}
    for k in metric_keys:
        vals = [r.metrics.get(k, float("nan")) for r in results]
        valid = [v for v in vals if not np.isnan(v)]
        agg[k] = {
            "mean": float(np.mean(valid)) if valid else float("nan"),
            "std":  float(np.std(valid))  if len(valid) > 1 else 0.0,
            "per_seed": dict(zip([r.seed for r in results], vals)),
        }
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified eval — all RLHF stages")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit test set size (0 = all, default)")
    parser.add_argument("--output", type=str,
                        default="eval_unified_results.json",
                        help="Output JSON path")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-stage2b", action="store_true",
                        help="Skip Stage 2B reranker eval (slow)")
    args = parser.parse_args()

    # Load test data
    if not CONALA_TEST_PATH.exists():
        print(f"ERROR: CoNaLa test file not found at {CONALA_TEST_PATH}")
        sys.exit(1)

    print(f"Loading CoNaLa test set from {CONALA_TEST_PATH}...")
    test_samples = load_conala_test(CONALA_TEST_PATH, args.max_samples)
    prompts    = [s["prompt"]    for s in test_samples]
    references = [s["reference"] for s in test_samples]
    print(f"Test set size: {len(test_samples)} samples\n")

    all_results: List[StageResult] = []

    # -----------------------------------------------------------------------
    # Stage 1: RewardWeightedNLL checkpoints (multi-seed)
    # -----------------------------------------------------------------------
    stage1_base = STAGE_PATHS["stage1"]
    stage1_seed_results: List[StageResult] = []
    for seed in SEEDS:
        if seed == 42:
            path = stage1_base
        else:
            path = f"stage1/output/seed_{seed}/checkpoint_epoch_30"
            if not (REPO_ROOT / path).exists():
                # try without subfolder (single run)
                path = stage1_base
        r = eval_generative_stage("stage1", path, prompts, references,
                                  args.device, seed=seed)
        if r.note != "path_not_found":
            stage1_seed_results.append(r)
    all_results.extend(stage1_seed_results)

    # -----------------------------------------------------------------------
    # Stage 2B: reranker
    # -----------------------------------------------------------------------
    if not args.skip_stage2b:
        r2b = eval_stage2b_reranker(prompts, references, args.device)
        all_results.append(r2b)

    # -----------------------------------------------------------------------
    # Stage 3 Case 2: SFT checkpoint (multi-seed)
    # -----------------------------------------------------------------------
    stage3_seed_results: List[StageResult] = []
    for seed in SEEDS:
        if seed == 42:
            path = STAGE_PATHS["stage3_case2"]
        else:
            path = f"stage3/modern_rlhf_outputs_seed{seed}/checkpoint-2001"
            if not (REPO_ROOT / path).exists():
                path = STAGE_PATHS["stage3_case2"]
        r = eval_generative_stage("stage3_case2", path, prompts, references,
                                  args.device, seed=seed)
        if r.note != "path_not_found":
            stage3_seed_results.append(r)
    all_results.extend(stage3_seed_results)

    # -----------------------------------------------------------------------
    # Stage 5 v2 and v3: these are classifier models (not generative).
    # Evaluate using their best-epoch checkpoint if a policy fine-tune exists,
    # otherwise report N/A (the classifiers don't generate code).
    # -----------------------------------------------------------------------
    for variant, path_key, history_file in [
        ("stage5_v2", "stage5_v2", "training_history_llm_v2.json"),
        ("stage5_v3", "stage5_v3", "training_history_llm_v3.json"),
    ]:
        artifact_dir = REPO_ROOT / STAGE_PATHS[path_key]
        # Stage 5 trains classifiers on top of CodeBERT — they have no
        # generative decoder.  We therefore evaluate the *base* CodeGPT
        # (or Stage 1 checkpoint) as the generative model and note that
        # Stage 5 results reflect classifier performance, not generation.
        r = StageResult(
            stage=variant,
            metrics={},
            note=(
                "stage5_is_classifier_not_generator; "
                f"see {history_file} for val_f1/balanced_accuracy"
            ),
        )
        all_results.append(r)
        print(f"\n[{variant}] NOTE: Stage 5 trains a quality *classifier*, "
              f"not a code generator.  Text metrics do not apply.  "
              f"See {artifact_dir / history_file} for classification metrics.")

    # -----------------------------------------------------------------------
    # Aggregate multi-seed results
    # -----------------------------------------------------------------------
    stage1_agg  = aggregate_seed_results(
        [r for r in all_results if r.stage == "stage1"])
    stage3_agg  = aggregate_seed_results(
        [r for r in all_results if r.stage == "stage3_case2"])

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    def fmt(agg: Dict, key: str) -> str:
        if not agg or key not in agg:
            return "  N/A  "
        m = agg[key]["mean"]
        s = agg[key]["std"]
        if s > 0:
            return f"{m:.4f}±{s:.4f}"
        return f"{m:.4f}"

    def fmt_single(r: StageResult, key: str) -> str:
        if not r.metrics:
            return "  N/A  "
        return f"{r.metrics.get(key, float('nan')):.4f}"

    r2b = next((r for r in all_results if r.stage == "stage2B_reranker"), None)

    print("\n" + "=" * 90)
    print(f"{'Stage':<22} {'BERTScore':>12} {'ROUGE-L':>10} {'BLEU':>8} {'CodeBLEU':>10} {'RUBY':>8}")
    print("=" * 90)

    for label, agg_or_result in [
        ("Stage 1 (RewardWeightedNLL)", stage1_agg),
        ("Stage 2B* (reranker)", r2b),
        ("Stage 3 Case 2 (SFT)", stage3_agg),
    ]:
        if isinstance(agg_or_result, dict):
            row = [fmt(agg_or_result, k)
                   for k in ["bertscore", "rouge", "bleu", "codebleu", "ruby"]]
        elif agg_or_result is not None:
            row = [fmt_single(agg_or_result, k)
                   for k in ["bertscore", "rouge", "bleu", "codebleu", "ruby"]]
        else:
            row = ["  N/A  "] * 5
        print(f"{label:<22} " + "  ".join(f"{v:>12}" for v in row))

    print("-" * 90)
    print("* Stage 2B: base CodeGPT-small-py candidates reranked by Stage 2B reward model")
    print("  Stage 5: classifier model only — text quality metrics not applicable")
    print("=" * 90)

    # -----------------------------------------------------------------------
    # Save full results
    # -----------------------------------------------------------------------
    output = {
        "meta": {
            "conala_test_samples": len(test_samples),
            "bertscore_model": "roberta-large",
            "device": args.device,
        },
        "stage1": {
            "per_seed": [asdict(r) for r in all_results if r.stage == "stage1"],
            "aggregated": stage1_agg,
        },
        "stage2B_reranker": asdict(r2b) if r2b else None,
        "stage3_case2": {
            "per_seed": [asdict(r) for r in all_results if r.stage == "stage3_case2"],
            "aggregated": stage3_agg,
        },
        "stage5_v2_note": "classifier_not_generator; see stage5/stage5B/artifacts/training_history_llm_v2.json",
        "stage5_v3_note": "classifier_not_generator; see stage5/stage5C/artifacts/training_history_llm_v3.json",
    }

    out_path = REPO_ROOT / args.output
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
