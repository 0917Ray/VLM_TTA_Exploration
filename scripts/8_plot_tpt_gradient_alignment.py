#!/usr/bin/env python3
"""Plot experiment 8: saved T/C gradient-alignment distributions."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "data": {"evaluation_config": "../evaluation/8_tpt_gradient_alignment.yaml", "distribution": None},
    "figure": {"figsize": [15.0, 10.0], "dpi": 300, "transparent": False},
    "histogram": {"bins": 40, "density": True, "alpha": 0.45, "linewidth": 1.7, "grid": True, "grid_color": "#D8DCE2", "grid_alpha": 0.75, "x_quantile_limits": [0.005, 0.995]},
    "colors": {"T": "#5B7CA7", "C": "#BA6580"},
    "spines": {"linewidth": 2.0, "color": "#3A3D42"},
    "text": {"title_prefix": "TPT Gradient Alignment", "xlabel_norm": "L2 norm", "xlabel_cosine": "Cosine similarity", "xlabel_dot": "Inner product", "ylabel": "Density"},
    "output": {"pdf": True},
}
GROUPS = ("all", "initial_correct", "initial_wrong")
METRICS = {"raw": ("raw_norm_T", "raw_norm_C", "raw_cosine", "raw_dot"), "update": ("update_norm_T", "update_norm_C", "update_cosine", "update_dot")}


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        override = yaml.safe_load(stream) or {}
    return merge(DEFAULT_CONFIG, override)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"group", "metric", "value"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Distribution CSV missing columns: {required}")
    for row in rows:
        row["value"] = float(row["value"])
        if not math.isfinite(row["value"]):
            raise ValueError("Distribution contains non-finite value")
    return rows


def safe_limits(values: np.ndarray, quantiles: list[float]) -> tuple[float, float]:
    low, high = np.quantile(values, quantiles)
    if low == high:
        pad = max(abs(float(low)) * 0.05, 1e-6)
        return float(low - pad), float(high + pad)
    pad = (high - low) * 0.08
    return float(low - pad), float(high + pad)


def plot_kind(rows: list[dict[str, Any]], kind: str, output_dir: Path, config: dict[str, Any]) -> list[Path]:
    fig, axes = plt.subplots(3, 3, figsize=config["figure"]["figsize"], squeeze=False)
    prefix = config["text"]["title_prefix"]
    metric_specs = [("norm", [f"{kind}_norm_T", f"{kind}_norm_C"], config["text"]["xlabel_norm"]), ("cosine", [f"{kind}_cosine"], config["text"]["xlabel_cosine"]), ("dot", [f"{kind}_dot"], config["text"]["xlabel_dot"])]
    for row_index, group in enumerate(GROUPS):
        for column, (_, metrics, xlabel) in enumerate(metric_specs):
            axis = axes[row_index, column]
            if len(metrics) == 2:
                values = []
                for metric in metrics:
                    values.append(np.asarray([r["value"] for r in rows if r["group"] == group and r["metric"] == metric], dtype=float))
                combined = np.concatenate([value for value in values if len(value)])
                limits = safe_limits(combined, config["histogram"]["x_quantile_limits"])
                for metric, value in zip(metrics, values):
                    if len(value):
                        axis.hist(value, bins=config["histogram"]["bins"], range=limits, density=config["histogram"]["density"], alpha=config["histogram"]["alpha"], linewidth=config["histogram"]["linewidth"], color=config["colors"]["T" if metric.endswith("_T") else "C"], label=metric[-1])
                axis.legend(frameon=False)
            else:
                value = np.asarray([r["value"] for r in rows if r["group"] == group and r["metric"] == metrics[0]], dtype=float)
                if len(value):
                    limits = safe_limits(value, config["histogram"]["x_quantile_limits"])
                    limits = (min(limits[0], 0.0), max(limits[1], 0.0))
                    axis.hist(value, bins=config["histogram"]["bins"], range=limits, density=config["histogram"]["density"], alpha=0.72, linewidth=config["histogram"]["linewidth"], color=config["colors"]["T"] if metrics[0].endswith("_T") else config["colors"]["C"])
            axis.set_title(f"{group.replace('_', ' ').title()} / {['norm', 'cosine', 'dot'][column]}")
            axis.set_xlabel(xlabel)
            axis.set_ylabel(config["text"]["ylabel"])
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.8g"))
            axis.tick_params(axis="x", labelsize=8)
            if column > 0:
                axis.axvline(0.0, color="#3A3D42", linewidth=1.0, linestyle=":", zorder=0)
            if config["histogram"]["grid"]:
                axis.grid(axis="y", color=config["histogram"]["grid_color"], alpha=config["histogram"]["grid_alpha"], linestyle="--", linewidth=0.7)
            for spine in axis.spines.values():
                spine.set_linewidth(config["spines"]["linewidth"]); spine.set_color(config["spines"]["color"])
    description = "Raw Gradient Distributions" if kind == "raw" else "Optimizer-applied Update Distributions"
    fig.suptitle(f"{prefix}: {description}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{kind}_gradient_alignment.png"]
    fig.savefig(paths[0], dpi=config["figure"]["dpi"], bbox_inches="tight", transparent=config["figure"]["transparent"])
    if config["output"]["pdf"]:
        paths.append(output_dir / f"{kind}_gradient_alignment.pdf")
        fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)
    return paths


def generate_plots(config_path: Path, distribution_path: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if not config["enabled"]:
        return []
    if distribution_path is None:
        distribution_path = Path(config["data"]["distribution"])
        if not distribution_path.is_absolute():
            distribution_path = (config_path.parent / distribution_path).resolve()
    distribution_path = distribution_path.resolve()
    destination = (output_dir or config_path.parent).resolve()
    rows = read_rows(distribution_path)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(distribution_path, destination / "distribution_data.csv")
    shutil.copy2(Path(__file__).resolve(), destination / Path(__file__).name)
    effective = merge(config, {"data": {"evaluation_config": None, "distribution": "distribution_data.csv"}, "output": {"directory": "."}})
    with (destination / "plot_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(effective, stream, sort_keys=False, allow_unicode=True)
    groups = list(dict.fromkeys(
        (row.get("model", "model"), row.get("dataset", "dataset"), float(row.get("learning_rate", 0.0)))
        for row in rows
    ))
    paths: list[Path] = []
    for model, dataset, rate in groups:
        group_rows = [
            row for row in rows
            if row.get("model", "model") == model and row.get("dataset", "dataset") == dataset
            and float(row.get("learning_rate", 0.0)) == rate
        ]
        slug = f"{model.lower().replace('/', '-') }__{dataset}__lr_{rate:.12g}".replace(".", "p")
        group_dir = destination / slug
        paths.extend(plot_kind(group_rows, "raw", group_dir, config))
        paths.extend(plot_kind(group_rows, "update", group_dir, config))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/visualization/8_tpt_gradient_alignment.yaml")
    parser.add_argument("--distribution", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = generate_plots(args.config, args.distribution, args.output_dir)
    print(f"[SUCCESS] generated {len(paths)} gradient alignment figures")


if __name__ == "__main__":
    main()
