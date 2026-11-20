"""Exercise the real CPU postprocess child and portable plot bundle without CLIP."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
alignment = importlib.import_module("8_evaluate_tpt_gradient_alignment")


def main():
    root = ROOT / "smoke_test/outputs/8_postprocess" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task = root / "seed_123" / alignment.task_slug("ViT-B/16", "imagenet-a", 0.003)
    task.mkdir(parents=True)
    records = []
    for index in range(12):
        cosine = (index - 5) / 7
        records.append({
            "model": "ViT-B/16", "dataset": "imagenet-a", "learning_rate": 0.003, "seed": 123,
            "sample_index": index, "sample_id": f"sample_{index}.jpg", "initial_correct": index % 2 == 0,
            "raw_norm_T": 1 + index / 10, "raw_norm_C": 2 + index / 10,
            "raw_cosine": cosine, "raw_dot": cosine * 2,
            "update_norm_T": 0.1 + index / 100, "update_norm_C": 0.2 + index / 100,
            "update_cosine": cosine / 2, "update_dot": cosine / 100,
        })
    result = {key: records[0][key] for key in ("model", "dataset", "learning_rate", "seed")}
    result.update(samples=len(records), sample_results="/old/server/path.json")
    alignment.atomic_write_json(task / "sample_gradient_metrics.json", records)
    alignment.atomic_write_json(task / "task_result.json", result)
    config = alignment.load_yaml(ROOT / "configs/visualization/8_tpt_gradient_alignment.yaml")
    config["figure"]["dpi"] = 70
    plot_config = root / "plot_config.yaml"
    plot_config.write_text(yaml.safe_dump(config))
    spec_path = task / "task_config.json"
    alignment.atomic_write_json(spec_path, {"args": {"plot": {"enabled": True, "config": str(plot_config)}}})
    before = (task / "task_result.json").read_bytes()
    alignment.run_task_process(spec_path, None, postprocess=True)
    completion = json.loads((task / "COMPLETED.json").read_text())
    assert completion["figure_files"] == 16
    assert len(list((task / "plots").rglob("*.png"))) == 8
    assert len(list((task / "plots").rglob("*.pdf"))) == 8
    assert (task / "task_result.json").read_bytes() == before
    bundle = task / "plots"
    redraw = root / "redraw"
    # A different cwd confirms that paths resolve from the saved config.
    subprocess.run([
        sys.executable, str(bundle / "8_plot_tpt_gradient_alignment.py"),
        "--config", str(bundle / "plot_config.yaml"), "--output-dir", str(redraw),
    ], cwd=ROOT.parent, env=alignment.task_environment(None), check=True)
    assert len(list(redraw.rglob("*.png"))) == 8
    print(f"[SUCCESS] CPU postprocess and independent redraw verified: {root}")


if __name__ == "__main__":
    main()
