"""
Export per-epoch results from all stages into one JSON file.

This script is intentionally conservative: it only consumes artifacts that
already exist in the repo (stage*.json / stage*.txt logs) and writes a single
combined JSON for easier downstream analysis / Excel validation.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
OUT_PATH = REPO_ROOT / "general_all_stages_results_by_epoch.json"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_stage2b_log(log_path: Path) -> List[Dict[str, Any]]:
    """
    Parse stage2B `run_log_30epochs.txt` into a list of per-epoch metric dicts.

    Expected lines (as used in `fill_excel.py`):
        Epoch 11 Metrics: { ... }
    """
    if not log_path.exists():
        return []

    content = log_path.read_text(encoding="utf-8", errors="ignore")

    # The log uses a python-dict-like representation without nested braces.
    pattern = r"Epoch (\d+) Metrics: (\{[^}]+\})"
    matches = re.findall(pattern, content)
    results: List[Dict[str, Any]] = []

    TEXT_METRIC_KEYS = ["bertscore", "rouge", "bleu", "ruby", "codebleu"]

    for epoch_str, metrics_str in matches:
        epoch = int(epoch_str)
        metrics: Dict[str, Any]
        try:
            metrics = ast.literal_eval(metrics_str)
        except Exception:
            # Fallback for slightly non-literal cases.
            metrics = eval(metrics_str, {"__builtins__": {}}, {})  # noqa: S307

        if isinstance(metrics, dict):
            metrics_out: Dict[str, Any] = {"epoch": epoch}
            metrics_out.update(metrics)
            # Stage 2B: text metrics are known-invalid due to self-comparison.
            # Keep val_accuracy_* but blank out text metrics so downstream
            # doesn't accidentally treat constant "1.0" as real embeddings.
            for k in TEXT_METRIC_KEYS:
                if k in metrics_out:
                    metrics_out[k] = None
            metrics_out["text_metrics_status"] = "invalid_self_comparison"
            results.append(metrics_out)

    # De-duplicate by epoch while keeping order.
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for r in results:
        ep = r.get("epoch")
        if ep is None or ep in seen:
            continue
        seen.add(ep)
        deduped.append(r)
    return deduped


def main() -> None:
    stages: Dict[str, Any] = {}

    # Stage 1
    s1 = REPO_ROOT / "stage1" / "output" / "stage1_results.json"
    if s1.exists():
        stages["stage1"] = {
            "source_file": str(s1),
            "data": _read_json(s1),
        }

    # Stage 2A
    s2a = REPO_ROOT / "stage2" / "stage2A" / "outputs" / "stage2a_results.json"
    if s2a.exists():
        stages["stage2A"] = {
            "source_file": str(s2a),
            "data": _read_json(s2a),
            # Stage 2A trains only the reward model; the policy (CodeGPT-small-py)
            # is frozen and generates identical outputs every epoch.
            # Text metrics (BERTScore/ROUGE/BLEU/RUBY/CodeBLEU) are therefore
            # constant across all epochs and reflect the pre-trained baseline,
            # NOT Stage 2A learning progress.  The relevant learning signal is
            # reward_acc (0.486 → 0.958) and reward_gap (0.0003 → 0.905).
            "text_metrics_status": "baseline_policy_frozen",
        }

    # Stage 2B (log only)
    s2b_log = REPO_ROOT / "stage2" / "stage2B" / "outputs" / "run_log_30epochs.txt"
    if s2b_log.exists():
        stages["stage2B"] = {
            "source_file": str(s2b_log),
            "epoch_results": _parse_stage2b_log(s2b_log),
        }

    # Stage 3
    s3_case1 = REPO_ROOT / "stage3" / "outputs" / "case1_11samples_history.json"
    if s3_case1.exists():
        stages["stage3_case1"] = {
            "source_file": str(s3_case1),
            "epoch_results": _read_json(s3_case1),
        }

    s3_case2 = REPO_ROOT / "stage3" / "outputs" / "case2_1247samples_history.json"
    if s3_case2.exists():
        stages["stage3_case2"] = {
            "source_file": str(s3_case2),
            "epoch_results": _read_json(s3_case2),
        }

    # Stage 4A (three classifier sections)
    s4a_cons = REPO_ROOT / "stage4" / "stage4A_consist" / "artifacts" / "training_history_consistent.json"
    if s4a_cons.exists():
        stages["stage4A_consistency"] = {
            "source_file": str(s4a_cons),
            "epoch_results": _read_json(s4a_cons),
        }

    s4a_corr = REPO_ROOT / "stage4" / "stage4B_corct" / "artifacts" / "training_history_correct.json"
    if s4a_corr.exists():
        stages["stage4B_agreement"] = {
            "source_file": str(s4a_corr),
            "epoch_results": _read_json(s4a_corr),
        }

    s4a_use = REPO_ROOT / "stage4" / "stage4C_useful" / "artifacts" / "training_history_useful.json"
    if s4a_use.exists():
        stages["stage4A_usefulness"] = {
            "source_file": str(s4a_use),
            "epoch_results": _read_json(s4a_use),
        }

    # Stage 5
    # There are 3 JSON versions (v1/v2/v3), but Excel has 2 sheets (5A/5B).
    # Real mapping used for the paper:
    #   Excel "Stage 5A" -> JSON stage5B (v2)
    #   Excel "Stage 5B" -> JSON stage5C (v3)
    # JSON v1 is not documented in Excel (4 epochs; val_loss diverges).
    s5_v1 = REPO_ROOT / "stage5" / "stage5A" / "artifacts" / "training_history_llm.json"
    if s5_v1.exists():
        stages["stage5_v1"] = {
            "source_file": str(s5_v1),
            "epoch_results": _read_json(s5_v1),
        }

    s5_v2 = REPO_ROOT / "stage5" / "stage5B" / "artifacts" / "training_history_llm_v2.json"
    if s5_v2.exists():
        stages["stage5_v2"] = {
            "source_file": str(s5_v2),
            "epoch_results": _read_json(s5_v2),
        }

    s5_v3 = REPO_ROOT / "stage5" / "stage5C" / "artifacts" / "training_history_llm_v3.json"
    if s5_v3.exists():
        stages["stage5_v3"] = {
            "source_file": str(s5_v3),
            "epoch_results": _read_json(s5_v3),
        }

    excel_stage5_mapping = {
        "Stage 5A (Excel)": "stage5_v2 (JSON v2)",
        "Stage 5B (Excel)": "stage5_v3 (JSON v3)",
        "Excel omitted": "stage5_v1 (JSON v1, 4 epochs; val_loss diverges)",
    }

    # If nothing was loaded, fail loudly.
    if not stages:
        raise SystemExit("No stage artifacts found; nothing exported.")

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(REPO_ROOT),
            "output_schema": "general_all_stages_results_by_epoch_v2",
            "excel_stage5_mapping": excel_stage5_mapping,
        },
        "stages": stages,
    }

    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported: {OUT_PATH}")

    # Small sanity check for the user.
    for stage_name, stage_data in stages.items():
        if "epoch_results" in stage_data and isinstance(stage_data["epoch_results"], list):
            epochs = sorted(
                {int(r.get("epoch")) for r in stage_data["epoch_results"] if isinstance(r, dict) and r.get("epoch") is not None}
            )
            print(f"  {stage_name}: epochs={epochs[0]}..{epochs[-1]} (count={len(epochs)})")
        elif "data" in stage_data and isinstance(stage_data["data"], dict) and "epoch_results" in stage_data["data"]:
            epoch_results = stage_data["data"].get("epoch_results", [])
            if isinstance(epoch_results, list):
                epochs = sorted(
                    {int(r.get("epoch")) for r in epoch_results if isinstance(r, dict) and r.get("epoch") is not None}
                )
                print(f"  {stage_name}: epochs={epochs[0]}..{epochs[-1]} (count={len(epochs)})")


if __name__ == "__main__":
    main()

