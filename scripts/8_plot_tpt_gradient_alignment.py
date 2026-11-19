#!/usr/bin/env python3
"""Plot experiment 8 from saved T/C gradient-alignment distributions."""

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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "data": {"evaluation_config": "../evaluation/8_tpt_gradient_alignment.yaml", "distribution": None},
    "figure": {"figsize": [15.0, 10.0], "metric_figsize": [15.0, 4.2], "dpi": 300, "transparent": False},
    "histogram": {"bins": 40, "density": True, "alpha": 0.45, "linewidth": 1.7, "grid": True, "grid_color": "#D8DCE2", "grid_alpha": 0.75, "x_quantile_limits": [0.005, 0.995]},
    "colors": {"T": "#4F7C65", "C": "#A75B73"},
    "spines": {"linewidth": 2.0, "color": "#3A3D42"},
    "text": {"title_prefix": "TPT Gradient Alignment", "xlabel_norm": "L2 norm", "xlabel_cosine": "Cosine similarity", "xlabel_dot": "Inner product", "ylabel": "Density"},
    "output": {"pdf": True},
}
GROUPS = ("all", "initial_correct", "initial_wrong")
MEASURES = ("norm", "cosine", "dot")


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


def index_values(rows: list[dict[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    values: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        values.setdefault((row["group"], row["metric"]), []).append(row["value"])
    return {key: np.asarray(value, dtype=float) for key, value in values.items()}


def plot_panel(axis: Any, values: dict[tuple[str, str], np.ndarray], kind: str, group: str, measure: str, config: dict[str, Any], compact: bool = False) -> None:
    histogram = config["histogram"]
    color = config["colors"]
    label_size = 8.5 if compact else 10
    if measure == "norm":
        t_values = values.get((group, f"{kind}_norm_T"), np.array([]))
        c_values = values.get((group, f"{kind}_norm_C"), np.array([]))
        present = [v for v in (t_values, c_values) if len(v)]
        if present:
            limits = safe_limits(np.concatenate(present), histogram["x_quantile_limits"])
            for label, data in (("T", t_values), ("C", c_values)):
                if len(data):
                    axis.hist(data, bins=histogram["bins"], range=limits, density=histogram["density"], alpha=histogram["alpha"], linewidth=histogram["linewidth"], color=color[label], edgecolor=color[label])
        axis.text(0.02, 1.015, "T: entropy loss", transform=axis.transAxes, ha="left", va="bottom", color=color["T"], fontsize=label_size, fontweight="bold", clip_on=False)
        axis.text(0.98, 1.015, "C: supervised CE", transform=axis.transAxes, ha="right", va="bottom", color=color["C"], fontsize=label_size, fontweight="bold", clip_on=False)
    else:
        data = values.get((group, f"{kind}_{measure}"), np.array([]))
        if len(data):
            limits = safe_limits(data, histogram["x_quantile_limits"])
            limits = (min(limits[0], 0.0), max(limits[1], 0.0))
            edges = np.linspace(*limits, histogram["bins"] + 1)
            if limits[0] < 0 < limits[1]:
                edges = np.sort(np.unique(np.append(edges, 0.0)))
            heights, edges = np.histogram(data, bins=edges, density=histogram["density"])
            widths = np.diff(edges)
            bar_colors = [color["C"] if right <= 0 else color["T"] for right in edges[1:]]
            axis.bar(edges[:-1], heights, width=widths, align="edge", alpha=0.72, color=bar_colors, edgecolor=bar_colors, linewidth=histogram["linewidth"] / 2)
            axis.set_xlim(limits)
            negative_ratio = np.count_nonzero(data < 0) / len(data)
            positive_ratio = np.count_nonzero(data > 0) / len(data)
            for x, label, shade in (
                ((limits[0] + 0.0) / 2, f"Negative {negative_ratio:.2%}", color["C"]),
                ((limits[1] + 0.0) / 2, f"Positive {positive_ratio:.2%}", color["T"]),
            ):
                axis.text(x, 1.015, label, transform=axis.get_xaxis_transform(), ha="center", va="bottom", color=shade, fontsize=label_size, fontweight="bold", clip_on=False)
        axis.axvline(0.0, color=config["spines"]["color"], linewidth=1.0, linestyle=":", zorder=3)
    axis.set_title(f"{group.replace('_', ' ').title()} / {measure}", pad=32)
    axis.set_xlabel(config["text"][f"xlabel_{measure}"])
    axis.set_ylabel(config["text"]["ylabel"])
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.8g"))
    axis.tick_params(axis="x", labelsize=8)
    if histogram["grid"]:
        axis.grid(axis="y", color=histogram["grid_color"], alpha=histogram["grid_alpha"], linestyle="--", linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_linewidth(config["spines"]["linewidth"])
        spine.set_color(config["spines"]["color"])


def save_figure(fig: Any, output_dir: Path, stem: str, config: dict[str, Any]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png"]
    fig.savefig(paths[0], dpi=config["figure"]["dpi"], bbox_inches="tight", transparent=config["figure"]["transparent"])
    if config["output"]["pdf"]:
        paths.append(output_dir / f"{stem}.pdf")
        fig.savefig(paths[-1], bbox_inches="tight", transparent=config["figure"]["transparent"])
    plt.close(fig)
    return paths


def plot_kind(rows: list[dict[str, Any]], kind: str, output_dir: Path, config: dict[str, Any]) -> list[Path]:
    values = index_values(rows)
    prefix = config["text"]["title_prefix"]
    description = "Raw Gradient Distributions" if kind == "raw" else "Optimizer-applied Update Distributions"
    fig, axes = plt.subplots(3, 3, figsize=config["figure"]["figsize"], squeeze=False)
    for row_index, group in enumerate(GROUPS):
        for column, measure in enumerate(MEASURES):
            plot_panel(axes[row_index, column], values, kind, group, measure, config, compact=True)
    fig.suptitle(f"{prefix}: {description}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    paths = save_figure(fig, output_dir, f"{kind}_gradient_alignment", config)
    for measure in MEASURES:
        fig, axes = plt.subplots(1, 3, figsize=config["figure"]["metric_figsize"], squeeze=False)
        for column, group in enumerate(GROUPS):
            plot_panel(axes[0, column], values, kind, group, measure, config)
        fig.suptitle(f"{prefix}: {description} / {measure.title()}", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        paths.extend(save_figure(fig, output_dir, f"{kind}_gradient_alignment_{measure}", config))
    return paths


def generate_plots(config_path: Path, distribution_path: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if not config["enabled"]:
        return []
    if distribution_path is None:
        configured = config["data"]["distribution"]
        if configured is None:
            raise ValueError("Provide --distribution or data.distribution in the plot config")
        distribution_path = Path(configured)
        if not distribution_path.is_absolute():
            distribution_path = (config_path.parent / distribution_path).resolve()
    distribution_path = distribution_path.resolve()
    destination = (output_dir or config_path.parent).resolve()
    rows = read_rows(distribution_path)
    destination.mkdir(parents=True, exist_ok=True)
    if distribution_path != destination / "distribution_data.csv":
        shutil.copy2(distribution_path, destination / "distribution_data.csv")
    script_path = Path(__file__).resolve()
    if script_path != destination / script_path.name:
        shutil.copy2(script_path, destination / script_path.name)
    effective = merge(config, {"data": {"evaluation_config": None, "distribution": "distribution_data.csv"}, "output": {"directory": "."}})
    with (destination / "plot_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(effective, stream, sort_keys=False, allow_unicode=True)
    groups = list(dict.fromkeys(
        (row.get("model", "model"), row.get("dataset", "dataset"), float(row.get("learning_rate", 0.0)))
        for row in rows
    ))
    paths: list[Path] = []
    for model, dataset, rate in groups:
        group_rows = [row for row in rows if row.get("model", "model") == model and row.get("dataset", "dataset") == dataset and float(row.get("learning_rate", 0.0)) == rate]
        slug = f"{model.lower().replace('/', '-')}__{dataset}__lr_{rate:.12g}".replace(".", "p")
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
    print(f"[SUCCESS] generated {len(paths)} gradient alignment files")


if __name__ == "__main__":
    main()
