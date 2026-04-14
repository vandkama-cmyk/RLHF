"""
Version 2: Rater Bias Investigation — Full Pipeline Runner
===========================================================
Runs all 7 stages in sequence.

Stages 1-5: core bias detection + reward model training
Stage 6:    Experiment — L and R sides treated as separate virtual raters
            (doubles dataset to 614 evaluations; tests positional asymmetry)
Stage 7:    Experiment — Pair comparison (slider) vs individual ratings
            (tests hypothesis: humans compare pairs better than they evaluate)

Usage:
    python Version_2/run_all.py               # run all stages
    python Version_2/run_all.py --stages 1 2  # run only stages 1 and 2
    python Version_2/run_all.py --skip 4      # skip stage 4 (model training)
    python Version_2/run_all.py --stages 6 7  # run only experiments
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STAGES = {
    1: ("stage1_eda/parse_and_explore.py",                    "Data Parsing & EDA"),
    2: ("stage2_bias_detection/detect_bias.py",               "Rater Bias Detection"),
    3: ("stage3_expertise/estimate_expertise.py",             "Expertise Proxy & Weighting"),
    4: ("stage4_reward_model/train_weighted.py",              "Bias-Corrected Reward Model"),
    5: ("stage5_report/generate_report.py",                   "Report & Visualization"),
    6: ("stage6_lr_as_raters/lr_raters_experiment.py",        "L/R as Separate Raters Experiment"),
    7: ("stage7_pairs_vs_individual/pairs_vs_individual.py",  "Pair vs Individual Evaluation"),
}


def run_stage(stage_num: int, extra_args: list = None):
    script_rel, name = STAGES[stage_num]
    script = BASE_DIR / script_rel
    if not script.exists():
        print(f"  [ERROR] Script not found: {script}")
        return False

    cmd = [sys.executable, str(script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  Stage {stage_num}: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  [OK] Stage {stage_num} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  [FAILED] Stage {stage_num} exited with code {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Version 2 RLHF pipeline")
    parser.add_argument("--stages", nargs="+", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Run only specific stages (default: all)")
    parser.add_argument("--skip", nargs="+", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                        help="Skip specific stages")
    parser.add_argument("--mode", choices=["weighted", "baseline", "both"],
                        default="both", help="Stage 4 training mode")
    args = parser.parse_args()

    stages_to_run = args.stages or list(STAGES.keys())
    if args.skip:
        stages_to_run = [s for s in stages_to_run if s not in args.skip]

    print("=" * 60)
    print("  Version 2: Rater Bias Investigation Pipeline")
    print("=" * 60)
    print(f"  Stages to run: {stages_to_run}")

    total_start = time.time()
    results = {}
    for stage in sorted(stages_to_run):
        extra = ["--mode", args.mode] if stage == 4 else None
        ok = run_stage(stage, extra)
        results[stage] = ok
        if not ok and stage < 4 and stage not in (6, 7):
            print(f"\n[ABORT] Stage {stage} failed. Later stages depend on this output.")
            break

    print(f"\n{'='*60}")
    print(f"  Pipeline Summary (total: {time.time()-total_start:.1f}s)")
    print(f"{'='*60}")
    for s, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  Stage {s}: {STAGES[s][1]:<40} [{status}]")


if __name__ == "__main__":
    main()
