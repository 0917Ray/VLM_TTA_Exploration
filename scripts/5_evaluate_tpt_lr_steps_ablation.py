"""Run experiment 5 with the shared TPT evaluator."""

import sys
from pathlib import Path

from evaluate_tpt import main


if __name__ == "__main__":
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        config = Path(__file__).resolve().parents[1] / "configs/evaluation/5_tpt_lr_steps_ablation.yaml"
        sys.argv[1:1] = ["--config", str(config)]
    main()
