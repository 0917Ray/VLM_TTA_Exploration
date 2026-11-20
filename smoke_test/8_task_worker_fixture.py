"""Small CPU-only subprocess fixture for experiment 8 scheduling tests."""

import json
import os
import sys
import time
from pathlib import Path


def main():
    config_path = Path(sys.argv[1])
    spec = json.loads(config_path.read_text())
    task_root = config_path.parent
    stage = sys.argv[2]
    started = time.monotonic()
    if stage == "postprocess":
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        compute = json.loads((task_root / "compute_event.json").read_text())
        try:
            os.kill(compute["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("Compute process must have exited before postprocessing")
        if spec["dataset_name"] == "plot-fail":
            raise RuntimeError("Injected plotting failure")
        time.sleep(0.03)
        (task_root / "COMPLETED.json").write_text('{"status": "complete"}')
    else:
        if spec["dataset_name"] == "compute-fail":
            raise RuntimeError("Injected compute failure")
        if spec["dataset_name"] == "slow" and spec["seed"] == spec["args"]["seeds"][0]:
            last_seed = spec["args"]["seeds"][-1]
            gate = Path(spec["args"]["output_root"]) / f"seed_{last_seed}" / "m__queued__lr_0p003" / "postprocess_event.json"
            deadline = time.monotonic() + 15
            while not gate.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Fast/queued tasks must finish while the slow task is running")
                time.sleep(0.02)
        record = {
            "model": spec["model_name"], "dataset": spec["dataset_name"],
            "learning_rate": spec["learning_rate"], "seed": spec["seed"],
            "sample_index": 0, "sample_id": "sample.jpg", "initial_correct": True,
            "raw_norm_T": 1.0, "raw_norm_C": 2.0, "raw_cosine": 0.5, "raw_dot": 1.0,
            "update_norm_T": 0.1, "update_norm_C": 0.2, "update_cosine": 0.5, "update_dot": 0.01,
        }
        result = {key: record[key] for key in ("model", "dataset", "learning_rate", "seed")}
        result.update(samples=1, device=spec["device_name"], sample_results="/old/server/path.json")
        (task_root / "sample_gradient_metrics.json").write_text(json.dumps([record]))
        (task_root / "task_result.json").write_text(json.dumps(result))
    (task_root / f"{stage}_event.json").write_text(json.dumps({
        "pid": os.getpid(), "start": started, "end": time.monotonic(),
        "device": spec["device_name"], "seed": spec["seed"], "dataset": spec["dataset_name"],
    }))


if __name__ == "__main__":
    main()
