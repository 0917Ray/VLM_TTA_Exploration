#!/usr/bin/env python3
"""Plot experiment 7: TPT prediction and loss transition proportions."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tpt-matplotlib-cache")
)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "data": {
        "evaluation_config": "../evaluation/7_tpt_transition_stats.yaml",
        "accuracy": None,
        "loss": None,
    },
    "figure": {
        "figsize": [8.8, 5.2],
        "dpi": 300,
        "transparent": False,
    },
    "bars": {
        "width": 0.68,
        "face_alpha": 0.5,
        "edge_alpha": 0.95,
        "edge_width": 2.0,
        "value_labels": True,
        "value_format": "{:.1f}%",
        "value_fontsize": 9.0,
    },
    "errorbar": {
        "enabled": True,
        "color": "#30343B",
        "alpha": 0.82,
        "linewidth": 1.1,
        "capsize": 3.5,
    },
    "axis": {
        "ylabel": "Samples (%)",
        "ylim": [0, 105],
        "grid": True,
        "grid_color": "#D8DCE2",
        "grid_alpha": 0.8,
        "tick_rotation": 0,
    },
    "spines": {"linewidth": 2.5, "color": "#3A3D42"},
    "colors": {
        "accuracy": ["#5E887E", "#BA6580", "#5B7CA7", "#75668A"],
        "loss": ["#5E887E", "#BA6580", "#75668A", "#5B7CA7"],
    },
    "annotations": {
        "accuracy": {
            "enabled": True,
            "offset_points": 20.0,
            "marker_size": 18.0,
            "symbol_linewidth": 2.4,
            "symbol_color": "#FFFFFF",
            "cross_color": "#C74E52",
            "check_color": "#4F806B",
            "transitions": {
                "correct_to_wrong": "cross",
                "wrong_to_correct": "check",
            },
        },
        "loss": {
            "enabled": True,
            "offset_points": 20.0,
            "marker_size": 18.0,
            "symbol_linewidth": 2.4,
            "symbol_color": "#FFFFFF",
            "cross_color": "#C74E52",
            "check_color": "#4F806B",
            "transitions": {
                "entropy_down_ce_down": "check",
                "entropy_down_ce_up_or_equal": "cross",
            },
        },
    },
    "titles": {
        "accuracy": "Prediction Correctness Transitions",
        "loss": "Entropy and Supervised Cross-Entropy Transitions",
    },
    "output": {"directory": None, "pdf": True},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_relative(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Plot config does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        override = yaml.safe_load(stream) or {}
    if not isinstance(override, dict):
        raise ValueError("Plot config must contain a YAML mapping")
    unknown = set(override) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown plot config keys: {', '.join(sorted(unknown))}")
    config = deep_merge(DEFAULT_CONFIG, override)
    if not isinstance(config["enabled"], bool):
        raise ValueError("enabled must be true or false")
    if (not isinstance(config["figure"]["figsize"], list) or
            len(config["figure"]["figsize"]) != 2 or
            any(float(value) <= 0 for value in config["figure"]["figsize"])):
        raise ValueError("figure.figsize must contain two positive values")
    ylim = config["axis"]["ylim"]
    if (not isinstance(ylim, list) or len(ylim) != 2 or
            float(ylim[0]) >= float(ylim[1])):
        raise ValueError("axis.ylim must contain increasing values")
    for palette in config["colors"].values():
        if not isinstance(palette, list) or len(palette) < 4:
            raise ValueError("Each colors palette must contain at least four colors")
        for color in palette:
            mcolors.to_rgba(color)
    for key in ("face_alpha", "edge_alpha"):
        value = float(config["bars"][key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"bars.{key} must be between 0 and 1")
    error_alpha = float(config["errorbar"]["alpha"])
    if not 0.0 <= error_alpha <= 1.0:
        raise ValueError("errorbar.alpha must be between 0 and 1")
    for kind, annotation in config["annotations"].items():
        if float(annotation["marker_size"]) <= 0:
            raise ValueError(f"annotations.{kind}.marker_size must be positive")
        if float(annotation["symbol_linewidth"]) <= 0:
            raise ValueError(f"annotations.{kind}.symbol_linewidth must be positive")
        for key in ("symbol_color", "cross_color", "check_color"):
            mcolors.to_rgba(annotation[key])
        invalid_markers = set(annotation["transitions"].values()) - {"cross", "check"}
        if invalid_markers:
            raise ValueError(
                f"annotations.{kind}.transitions supports only 'cross' and 'check'"
            )
    return config


def infer_latest_run(config: dict[str, Any], config_path: Path) -> Path:
    evaluation_value = config["data"]["evaluation_config"]
    if evaluation_value is None:
        raise ValueError("data.evaluation_config is required when data paths are omitted")
    evaluation_path = resolve_relative(evaluation_value, config_path.parent)
    with evaluation_path.open(encoding="utf-8") as stream:
        evaluation = yaml.safe_load(stream) or {}
    output_root = Path(evaluation["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    idea = evaluation.get("idea")
    if idea:
        output_root = output_root / f"idea_{'_'.join(str(idea).split())}"
    completed = sorted(
        path for path in output_root.glob("run_*")
        if (path / "COMPLETED.json").is_file()
    )
    if not completed:
        raise FileNotFoundError(f"No completed run found under {output_root}")
    return completed[-1]


def infer_paths(
    config: dict[str, Any], config_path: Path
) -> tuple[Path, Path, Path]:
    accuracy_value = config["data"]["accuracy"]
    loss_value = config["data"]["loss"]
    if (accuracy_value is None) != (loss_value is None):
        raise ValueError("data.accuracy and data.loss must be set together")
    if accuracy_value is not None:
        output_value = config["output"]["directory"]
        output_dir = (
            resolve_relative(output_value, config_path.parent)
            if output_value is not None else config_path.parent
        )
        return (
            resolve_relative(accuracy_value, config_path.parent),
            resolve_relative(loss_value, config_path.parent),
            output_dir,
        )
    summary = infer_latest_run(config, config_path) / "summary"
    return (
        summary / "accuracy_transition_plot_data.csv",
        summary / "loss_transition_plot_data.csv",
        summary / "plots",
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Plot data does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No plot rows found in {path}")
    required = {
        "model", "dataset", "learning_rate", "transition", "label", "order",
        "proportion_percent", "proportion_percent_std", "num_seeds",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    for row_number, row in enumerate(rows, start=2):
        row["learning_rate"] = float(row["learning_rate"])
        row["order"] = int(row["order"])
        row["proportion_percent"] = float(row["proportion_percent"])
        std = row["proportion_percent_std"]
        row["proportion_percent_std"] = None if std in {"", "None", "null"} else float(std)
        row["num_seeds"] = int(row["num_seeds"])
        for field in ("learning_rate", "proportion_percent"):
            if not math.isfinite(row[field]):
                raise ValueError(f"Non-finite {field} at {path}:{row_number}")
    return rows


def configure_style(config: dict[str, Any]) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def group_slug(model: str, dataset: str, learning_rate: float) -> str:
    rate = f"{learning_rate:.12g}".replace("-", "m").replace(".", "p")
    model_slug = "".join(char.lower() if char.isalnum() else "-" for char in model).strip("-")
    return f"{model_slug}__{dataset}__lr_{rate}"


def save_figure(figure, base: Path, config: dict[str, Any]) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(".png")]
    figure.savefig(
        paths[0], dpi=config["figure"]["dpi"], bbox_inches="tight",
        transparent=config["figure"]["transparent"],
    )
    if config["output"]["pdf"]:
        paths.append(base.with_suffix(".pdf"))
        figure.savefig(paths[-1], bbox_inches="tight")
    plt.close(figure)
    return paths


def transition_marker(marker_type: str, config: dict[str, Any]) -> DrawingArea:
    size = float(config["marker_size"])
    linewidth = float(config["symbol_linewidth"])
    symbol_color = config["symbol_color"]
    drawing = DrawingArea(size, size, 0, 0)
    if marker_type == "cross":
        drawing.add_artist(Circle(
            (size / 2, size / 2), size * 0.47,
            facecolor=config["cross_color"], edgecolor="none",
        ))
        start, end = size * 0.30, size * 0.70
        for x_values, y_values in (
            ([start, end], [start, end]),
            ([start, end], [end, start]),
        ):
            drawing.add_artist(Line2D(
                x_values, y_values, color=symbol_color, linewidth=linewidth,
                solid_capstyle="round",
            ))
    else:
        inset = size * 0.04
        drawing.add_artist(FancyBboxPatch(
            (inset, inset), size - 2 * inset, size - 2 * inset,
            boxstyle=f"round,pad=0,rounding_size={size * 0.16}",
            facecolor=config["check_color"], edgecolor="none",
        ))
        drawing.add_artist(Line2D(
            [size * 0.24, size * 0.43, size * 0.77],
            [size * 0.51, size * 0.31, size * 0.70],
            color=symbol_color, linewidth=linewidth,
            solid_capstyle="round", solid_joinstyle="round",
        ))
    return drawing


def add_transition_annotations(
    axis, bars, rows: list[dict[str, Any]], annotation: dict[str, Any]
) -> None:
    if not annotation["enabled"]:
        return
    transitions = annotation["transitions"]
    for bar, row in zip(bars, rows):
        marker_type = transitions.get(row["transition"])
        if marker_type is None:
            continue
        marker = transition_marker(marker_type, annotation)
        artist = AnnotationBbox(
            marker,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xybox=(0, annotation["offset_points"]),
            xycoords="data",
            boxcoords="offset points",
            box_alignment=(0.5, 0.0),
            frameon=False,
            pad=0,
            zorder=6,
        )
        axis.add_artist(artist)


def plot_group(
    rows: list[dict[str, Any]],
    kind: str,
    output_dir: Path,
    config: dict[str, Any],
) -> list[Path]:
    rows = sorted(rows, key=lambda row: row["order"])
    model = rows[0]["model"]
    dataset = rows[0]["dataset"]
    learning_rate = rows[0]["learning_rate"]
    values = np.asarray([row["proportion_percent"] for row in rows], dtype=float)
    std_values = [row["proportion_percent_std"] for row in rows]
    errors = None
    if config["errorbar"]["enabled"] and any(value is not None for value in std_values):
        errors = np.asarray([0.0 if value is None else value for value in std_values], dtype=float)

    figure, axis = plt.subplots(figsize=config["figure"]["figsize"])
    x = np.arange(len(rows), dtype=float)
    palette = config["colors"][kind][:len(rows)]
    bars = axis.bar(
        x,
        values,
        width=config["bars"]["width"],
        color=[
            mcolors.to_rgba(color, config["bars"]["face_alpha"])
            for color in palette
        ],
        edgecolor=[
            mcolors.to_rgba(color, config["bars"]["edge_alpha"])
            for color in palette
        ],
        linewidth=config["bars"]["edge_width"],
        yerr=errors,
        error_kw={
            "ecolor": mcolors.to_rgba(
                config["errorbar"]["color"], config["errorbar"]["alpha"]
            ),
            "elinewidth": config["errorbar"]["linewidth"],
            "capsize": config["errorbar"]["capsize"],
        },
        zorder=3,
    )
    labels = [row["label"] for row in rows]
    if kind == "loss":
        labels = [label.replace(" / ", "\n") for label in labels]
    axis.set_xticks(x, labels, rotation=config["axis"]["tick_rotation"])
    axis.set_ylabel(config["axis"]["ylabel"])
    axis.set_ylim(*config["axis"]["ylim"])
    axis.set_title(
        f"{config['titles'][kind]}\n{model} / {dataset} / LR={learning_rate:g}"
    )
    if config["axis"]["grid"]:
        axis.grid(
            axis="y", color=config["axis"]["grid_color"],
            alpha=config["axis"]["grid_alpha"], linestyle="--", linewidth=0.8, zorder=0,
        )
    for spine in axis.spines.values():
        spine.set_linewidth(config["spines"]["linewidth"])
        spine.set_color(config["spines"]["color"])
    if config["bars"]["value_labels"]:
        for bar, value in zip(bars, values):
            axis.annotate(
                config["bars"]["value_format"].format(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=config["bars"]["value_fontsize"],
            )
    annotation = config["annotations"].get(kind)
    if annotation is not None:
        add_transition_annotations(axis, bars, rows, annotation)
    figure.tight_layout()
    return save_figure(figure, output_dir / f"{kind}_transitions", config)


def write_bundle(
    output_dir: Path,
    config: dict[str, Any],
    accuracy_path: Path,
    loss_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    accuracy_copy = output_dir / "accuracy_transition_plot_data.csv"
    loss_copy = output_dir / "loss_transition_plot_data.csv"
    if accuracy_path.resolve() != accuracy_copy.resolve():
        shutil.copy2(accuracy_path, accuracy_copy)
    if loss_path.resolve() != loss_copy.resolve():
        shutil.copy2(loss_path, loss_copy)
    script_copy = output_dir / Path(__file__).name
    source_script = Path(__file__).resolve()
    if source_script != script_copy.resolve():
        shutil.copy2(source_script, script_copy)
    effective = deep_merge(config, {
        "data": {
            "evaluation_config": None,
            "accuracy": accuracy_copy.name,
            "loss": loss_copy.name,
        },
        "output": {"directory": "."},
    })
    with (output_dir / "plot_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(effective, stream, sort_keys=False, allow_unicode=True)


def generate_plots(
    config_path: Path,
    accuracy_data: Path | None = None,
    loss_data: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if not config["enabled"]:
        return []
    explicit_data = accuracy_data is not None or loss_data is not None
    if explicit_data and (accuracy_data is None or loss_data is None):
        raise ValueError("accuracy_data and loss_data must be provided together")
    if accuracy_data is not None and loss_data is not None:
        accuracy_path = accuracy_data.resolve()
        loss_path = loss_data.resolve()
        destination = (
            output_dir.resolve() if output_dir is not None else config_path.parent
        )
    else:
        accuracy_path, loss_path, inferred_output = infer_paths(config, config_path)
        destination = output_dir.resolve() if output_dir is not None else inferred_output
    accuracy_rows = read_rows(accuracy_path)
    loss_rows = read_rows(loss_path)
    configure_style(config)
    write_bundle(destination, config, accuracy_path, loss_path)

    accuracy_groups = {
        (row["model"], row["dataset"], row["learning_rate"])
        for row in accuracy_rows
    }
    loss_groups = {
        (row["model"], row["dataset"], row["learning_rate"])
        for row in loss_rows
    }
    if accuracy_groups != loss_groups:
        raise ValueError("Accuracy and loss plot data cover different experiment groups")
    paths: list[Path] = []
    for model, dataset, learning_rate in sorted(accuracy_groups):
        group_dir = destination / group_slug(model, dataset, learning_rate)
        paths.extend(plot_group([
            row for row in accuracy_rows
            if (row["model"], row["dataset"], row["learning_rate"])
            == (model, dataset, learning_rate)
        ], "accuracy", group_dir, config))
        paths.extend(plot_group([
            row for row in loss_rows
            if (row["model"], row["dataset"], row["learning_rate"])
            == (model, dataset, learning_rate)
        ], "loss", group_dir, config))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/visualization/7_tpt_transition_stats.yaml",
    )
    parser.add_argument("--accuracy-data", type=Path)
    parser.add_argument("--loss-data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_plots(args.config, args.accuracy_data, args.loss_data, args.output_dir)
    print(f"[SUCCESS] generated {len(paths)} TPT transition figures")


if __name__ == "__main__":
    main()
