"""
Stage 1 PPO — entry point.

Delegates to stage1/run_stage1b.py which implements the full clipped-PPO loop:
  - value head for advantage estimation
  - frozen reference policy for KL penalty
  - multi-epoch inner PPO loop with importance-sampling ratio
  - entropy bonus

Results are saved to stage1/outputs/stage1b_results.json.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage1.run_stage1b import Stage1BExperiment
from stage1.config import get_stage1_config


def main():
    config = get_stage1_config()
    experiment = Stage1BExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
