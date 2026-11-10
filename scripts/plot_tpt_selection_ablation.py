#!/usr/bin/env python3
"""Plot TPT ablation accuracy from chart-ready evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tpt-matplotlib-cache")
)

try:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import yaml
except ModuleNotFoundError as error:
    raise SystemExit(
        "Plotting requires matplotlib and numpy. Install the visualization dependencies."
    ) from error


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "data": {
        "input": None,
        "evaluation_config": None,
        "source": None,
        "model": None,
        "dataset": None,
        "learning_rate": None,
        "x_column": "selection_percent",
        "series": [
            {
                "column": "top1_accuracy_percent",
                "error_column": "top1_std_percent",
                "label": "Top-1",
                "color": "#4F7C65",
                "marker": "o",
            },
            {
                "column": "top2_accuracy_percent",
                "error_column": "top2_std_percent",
                "label": "Top-2",
                "color": "#587FA3",
                "marker": "^",
            },
            {
                "column": "top3_accuracy_percent",
                "error_column": "top3_std_percent",
                "label": "Top-3",
                "color": "#B5823A",
                "marker": "D",
            },
            {
                "column": "top4_accuracy_percent",
                "error_column": "top4_std_percent",
                "label": "Top-4",
                "color": "#7A6B9A",
                "marker": "v",
            },
            {
                "column": "top5_accuracy_percent",
                "error_column": "top5_std_percent",
                "label": "Top-5",
                "color": "#A75B73",
                "marker": "s",
            },
        ],
    },
    "figure": {
        "figsize": [8.2, 4.8],
        "dpi": 300,
        "transparent": True,
        "spine_width": 2.5,
    },
    "font": {
        "family": "sans-serif",
        "sans_serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "size": 12,
        "title_size": 13,
        "label_size": 12,
        "tick_size": 11,
        "legend_size": 10.5,
    },
    "axis": {
        "xlabel": "Lowest-entropy views used for TTA (%)",
        "ylabel": "Accuracy (%)",
        "ylim": [0, 105],
        "target_y_ticks": 6,
        "grid": True,
        "grid_color": "#DADDE2",
        "grid_alpha": 0.8,
        "grid_linewidth": 0.75,
        "grid_linestyle": "--",
        "axis_color": "#3A3D42",
    },
    "errorbar": {
        "enabled": False,
        "meaning": "population standard deviation across seeds",
        "color": "#3A3D42",
        "alpha": 0.82,
        "linewidth": 1.05,
        "capsize": 3.2,
    },
    "zero_shot": {
        "enabled": True,
        "label": "CLIP zero-shot",
    },
    "legend": {
        "enabled": True,
        "loc": "upper center",
        "bbox_to_anchor": [0.5, 0.99],
        "frameon": False,
        "ncol": 5,
        "columnspacing": 1.4,
        "handlelength": 1.8,
    },
    "bar": {
        "enabled": True,
        "filename": "selection_fraction_bar.png",
        "figsize": [16, 5],
        "title": "TPT Accuracy by Entropy Selection Fraction",
        "group_width": 0.74,
        "face_alpha": 0.5,
        "edge_alpha": 0.95,
        "edge_width": 2.0,
        "value_labels": True,
        "value_format": "{:.1f}",
        "value_fontsize": 8.5,
        "value_rotation": 90,
    },
    "line": {
        "enabled": True,
        "filename": "selection_fraction_line.png",
        "figsize": [16, 5],
        "y_min": 20,
        "title": "TPT Accuracy vs. Entropy Selection Fraction",
        "linewidth": 2.0,
        "line_alpha": 0.92,
        "marker_size": 5.5,
        "marker_face_alpha": 0.6,
        "marker_edge_alpha": 0.95,
        "marker_edge_width": 1.35,
        "value_labels": True,
        "value_format": "{:.1f}",
        "value_fontsize": 8.5,
        "value_offset_points": 7.0,
        "highlight_best": {
            "enabled": False,
            "columns": ["top1_accuracy_percent"],
            "marker": "*",
            "marker_size": 15.0,
            "edge_color": "#1F2328",
            "edge_width": 1.2,
        },
    },
    "output": {"directory": None, "pdf": True, "bbox_inches": "tight", "pad_inches": 0.04},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Plot config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Plot config must contain a JSON object")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown top-level plot config keys: {', '.join(sorted(unknown))}")
    return deep_merge(DEFAULT_CONFIG, config)


def resolve_from_config(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def infer_run_paths(config: dict[str, Any], config_path: Path) -> tuple[Path, Path] | None:
    evaluation_config = config["data"].get("evaluation_config")
    if evaluation_config is None:
        return None
    evaluation_path = resolve_from_config(evaluation_config, config_path)
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {evaluation_path}")
    with evaluation_path.open(encoding="utf-8") as stream:
        evaluation = yaml.safe_load(stream) or {}
    if not isinstance(evaluation, dict):
        raise ValueError(f"Evaluation config must contain a YAML mapping: {evaluation_path}")
    if "output_root" not in evaluation:
        raise ValueError(f"Evaluation config is missing output_root: {evaluation_path}")

    project_root = Path(__file__).resolve().parents[1]
    run_root = Path(evaluation["output_root"])
    if not run_root.is_absolute():
        run_root = project_root / run_root
    idea = evaluation.get("idea")
    if idea is not None:
        if not isinstance(idea, str):
            raise ValueError(f"idea must be a string or null: {evaluation_path}")
        idea = "_".join(idea.split())
        if idea:
            run_root = run_root / f"idea_{idea}"
    summary_root = run_root / "summary"
    return summary_root / "selection_fraction_chart_data.csv", summary_root / "plots"


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config["enabled"], bool):
        raise ValueError("enabled must be true or false")
    if not isinstance(config["data"]["series"], list) or not config["data"]["series"]:
        raise ValueError("data.series must be a non-empty list")
    required_series_keys = {"column", "label", "color", "marker"}
    for series in config["data"]["series"]:
        if not isinstance(series, dict) or not required_series_keys <= set(series):
            raise ValueError(
                "Every data.series entry needs column, label, color, and marker"
            )
        mcolors.to_rgba(series["color"])
    for chart_name in ("bar", "line"):
        if not isinstance(config[chart_name]["enabled"], bool):
            raise ValueError(f"{chart_name}.enabled must be true or false")
        if Path(config[chart_name]["filename"]).suffix.lower() != ".png":
            raise ValueError(f"{chart_name}.filename must end in .png")
    if not isinstance(config["line"]["value_labels"], bool):
        raise ValueError("line.value_labels must be true or false")
    try:
        config["line"]["value_format"].format(0.0)
    except (AttributeError, KeyError, ValueError) as error:
        raise ValueError("line.value_format must be a valid numeric format string") from error
    if float(config["line"]["value_fontsize"]) <= 0:
        raise ValueError("line.value_fontsize must be positive")
    if not math.isfinite(float(config["line"]["value_offset_points"])):
        raise ValueError("line.value_offset_points must be finite")
    highlight = config["line"]["highlight_best"]
    if not isinstance(highlight, dict) or not isinstance(highlight.get("enabled"), bool):
        raise ValueError("line.highlight_best.enabled must be true or false")
    if highlight["enabled"]:
        columns = highlight.get("columns")
        if (not isinstance(columns, list) or not columns or
                any(not isinstance(column, str) or not column for column in columns)):
            raise ValueError(
                "line.highlight_best.columns must be a non-empty list of strings"
            )
        if len(set(columns)) != len(columns):
            raise ValueError("line.highlight_best.columns must not contain duplicates")
        if float(highlight["marker_size"]) <= 0 or float(highlight["edge_width"]) < 0:
            raise ValueError("line.highlight_best marker size/edge width is invalid")
        mcolors.to_rgba(highlight["edge_color"])
    if not isinstance(config["zero_shot"]["enabled"], bool):
        raise ValueError("zero_shot.enabled must be true or false")
    if not isinstance(config["zero_shot"]["label"], str) or not config["zero_shot"]["label"].strip():
        raise ValueError("zero_shot.label must be a non-empty string")
    if len(config["figure"]["figsize"]) != 2 or any(
            float(value) <= 0 for value in config["figure"]["figsize"]):
        raise ValueError("figure.figsize must contain two positive values")
    ylim = config["axis"]["ylim"]
    if ylim is not None and (len(ylim) != 2 or float(ylim[0]) >= float(ylim[1])):
        raise ValueError("axis.ylim must be null or [minimum, maximum]")
    for chart_name in ("bar", "line"):
        figsize = config[chart_name]["figsize"]
        if len(figsize) != 2 or any(float(value) <= 0 for value in figsize):
            raise ValueError(f"{chart_name}.figsize must contain two positive values")
    line_y_min = config["line"]["y_min"]
    if line_y_min is not None:
        line_y_min = float(line_y_min)
        if not math.isfinite(line_y_min):
            raise ValueError("line.y_min must be finite or null")
        if config["axis"]["ylim"] is not None and line_y_min >= float(config["axis"]["ylim"][1]):
            raise ValueError("line.y_min must be less than axis.ylim upper bound")


def read_chart_data(path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Chart data does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    model = config["data"]["model"]
    dataset = config["data"]["dataset"]
    learning_rate = config["data"]["learning_rate"]
    if model is not None:
        rows = [row for row in rows if row.get("model") == model]
    if dataset is not None:
        rows = [row for row in rows if row.get("dataset") == dataset]
    if learning_rate is not None:
        try:
            target_rate = float(learning_rate)
            rows = [
                row for row in rows
                if math.isclose(float(row.get("learning_rate", "nan")), target_rate,
                                rel_tol=1e-12, abs_tol=0.0)
            ]
        except (TypeError, ValueError) as error:
            raise ValueError("data.learning_rate must be numeric or null") from error
    if not rows:
        raise ValueError(f"No chart rows remain after filtering {path}")

    numeric_columns = {config["data"]["x_column"]}
    for series in config["data"]["series"]:
        numeric_columns.add(series["column"])
        if series.get("error_column"):
            numeric_columns.add(series["error_column"])
    zero_shot_value_columns = [
        f"zero_shot_{series['column']}" for series in config["data"]["series"]
    ]
    zero_shot_error_columns = [
        f"zero_shot_{series['error_column']}"
        for series in config["data"]["series"] if series.get("error_column")
    ]
    zero_shot_present = [
        all(column in row for row in rows) for column in zero_shot_value_columns
    ]
    if any(zero_shot_present) and not all(zero_shot_present):
        raise ValueError("Chart data contains only part of the configured zero-shot columns")
    if config["zero_shot"]["enabled"] and all(zero_shot_present):
        numeric_columns.update(zero_shot_value_columns)
        error_present = [
            all(column in row for row in rows) for column in zero_shot_error_columns
        ]
        if any(error_present) and not all(error_present):
            raise ValueError(
                "Chart data contains only part of the configured zero-shot error columns"
            )
        if all(error_present):
            numeric_columns.update(zero_shot_error_columns)
    for row_number, row in enumerate(rows, start=2):
        missing = numeric_columns - set(row)
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        for column in numeric_columns:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid numeric value at row {row_number}, column {column}: {row[column]!r}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite value at row {row_number}, column {column}: {row[column]!r}"
                )
            row[column] = value

    x_column = config["data"]["x_column"]
    rows.sort(key=lambda row: row[x_column])
    x_values = [row[x_column] for row in rows]
    if len(set(x_values)) != len(x_values):
        raise ValueError(
            "Duplicate x values remain after filtering; set data.model, data.dataset, "
            "and data.learning_rate as needed"
        )
    if config["zero_shot"]["enabled"] and all(zero_shot_present):
        zero_shot_row = dict(rows[0])
        zero_shot_row[x_column] = min(x_values) - 1
        zero_shot_row["_tick_label"] = config["zero_shot"]["label"].strip().replace(
            " ", "\n", 1
        )
        zero_shot_row["only_origin"] = "false"
        for series in config["data"]["series"]:
            zero_shot_row[series["column"]] = float(np.mean([
                row[f"zero_shot_{series['column']}"] for row in rows
            ]))
            error_column = series.get("error_column")
            zero_shot_error_column = f"zero_shot_{error_column}"
            if error_column and zero_shot_error_column in rows[0]:
                zero_shot_row[error_column] = float(np.mean([
                    row[zero_shot_error_column] for row in rows
                ]))
            elif error_column:
                zero_shot_row[error_column] = 0.0
        rows.insert(0, zero_shot_row)
    return rows


def configure_matplotlib(config: dict[str, Any]) -> None:
    font = config["font"]
    transparent = config["figure"]["transparent"]
    mpl.rcParams.update({
        "font.family": font["family"],
        "font.sans-serif": font["sans_serif"],
        "font.size": font["size"],
        "axes.titlesize": font["title_size"],
        "axes.labelsize": font["label_size"],
        "xtick.labelsize": font["tick_size"],
        "ytick.labelsize": font["tick_size"],
        "legend.fontsize": font["legend_size"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "none" if transparent else "white",
        "axes.facecolor": "none" if transparent else "white",
        "savefig.facecolor": "none" if transparent else "white",
    })


def style_axis(axis, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    axis_config = config["axis"]
    axis.set_xlabel(axis_config["xlabel"])
    axis.set_ylabel(axis_config["ylabel"])
    if axis_config["ylim"] is not None:
        axis.set_ylim(*axis_config["ylim"])
    axis.yaxis.set_major_locator(mticker.MaxNLocator(axis_config["target_y_ticks"]))
    x_values = [row[config["data"]["x_column"]] for row in rows]
    tick_labels = [
        row["_tick_label"] if "_tick_label" in row else
        "Only\norigin" if str(row.get("only_origin", "false")).lower() == "true"
        else f"{value:g}"
        for row, value in zip(rows, x_values)
    ]
    axis.set_xticks(range(len(x_values)), tick_labels)
    if axis_config["grid"]:
        axis.grid(
            axis="y", color=axis_config["grid_color"], alpha=axis_config["grid_alpha"],
            linewidth=axis_config["grid_linewidth"],
            linestyle=axis_config["grid_linestyle"], zorder=0,
        )
    axis.grid(False, axis="x")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(config["figure"]["spine_width"])
        spine.set_color(axis_config["axis_color"])
    axis.tick_params(colors=axis_config["axis_color"])
    if config["legend"]["enabled"]:
        legend = config["legend"]
        axis.legend(
            loc=legend["loc"], bbox_to_anchor=legend["bbox_to_anchor"],
            frameon=legend["frameon"], ncol=legend["ncol"],
            columnspacing=legend["columnspacing"], handlelength=legend["handlelength"],
        )


def save_figure(figure, png_path: Path, config: dict[str, Any]) -> list[Path]:
    output = config["output"]
    save_args = {
        "dpi": config["figure"]["dpi"],
        "transparent": config["figure"]["transparent"],
        "bbox_inches": output["bbox_inches"],
        "pad_inches": output["pad_inches"],
    }
    figure.savefig(png_path, **save_args)
    paths = [png_path]
    if output["pdf"]:
        pdf_path = png_path.with_suffix(".pdf")
        figure.savefig(pdf_path, **save_args)
        paths.append(pdf_path)
    plt.close(figure)
    return paths


def chart_title(title: str, rows: list[dict[str, Any]]) -> str:
    if rows and all(str(row.get("only_origin", "false")).lower() == "true" for row in rows):
        return f"{title} (Only original image)" if title else "Only original image"
    return title or ""


def plot_bar(rows: list[dict[str, Any]], output_dir: Path,
             config: dict[str, Any]) -> list[Path]:
    x_column = config["data"]["x_column"]
    positions = np.arange(len(rows), dtype=float)
    series_items = config["data"]["series"]
    group_width = float(config["bar"]["group_width"])
    bar_width = group_width / len(series_items)
    figure, axis = plt.subplots(figsize=config["bar"]["figsize"])
    for index, series in enumerate(series_items):
        values = np.array([row[series["column"]] for row in rows])
        offset = (index - (len(series_items) - 1) / 2) * bar_width
        error_values = None
        if config["errorbar"]["enabled"] and series.get("error_column"):
            error_values = np.array([row[series["error_column"]] for row in rows])
        bars = axis.bar(
            positions + offset, values, width=bar_width * 0.88, label=series["label"],
            color=mcolors.to_rgba(series["color"], config["bar"]["face_alpha"]),
            edgecolor=mcolors.to_rgba(series["color"], config["bar"]["edge_alpha"]),
            linewidth=config["bar"]["edge_width"], zorder=3, yerr=error_values,
            error_kw={
                "ecolor": mcolors.to_rgba(
                    config["errorbar"]["color"], config["errorbar"]["alpha"]
                ),
                "elinewidth": config["errorbar"]["linewidth"],
                "capsize": config["errorbar"]["capsize"],
            },
        )
        if config["bar"]["value_labels"]:
            axis.bar_label(
                bars, fmt=config["bar"]["value_format"], padding=3,
                fontsize=config["bar"]["value_fontsize"],
                rotation=config["bar"]["value_rotation"],
            )
    axis.set_title(chart_title(config["bar"]["title"], rows))
    style_axis(axis, config, rows)
    figure.tight_layout()
    return save_figure(figure, output_dir / config["bar"]["filename"], config)


def plot_line(rows: list[dict[str, Any]], output_dir: Path,
              config: dict[str, Any]) -> list[Path]:
    x_column = config["data"]["x_column"]
    positions = np.arange(len(rows), dtype=float)
    figure, axis = plt.subplots(figsize=config["line"]["figsize"])
    series_by_column = {
        series["column"]: series for series in config["data"]["series"]
    }
    for series in config["data"]["series"]:
        values = np.array([row[series["column"]] for row in rows])
        error_values = None
        if config["errorbar"]["enabled"] and series.get("error_column"):
            error_values = np.array([row[series["error_column"]] for row in rows])
        axis.errorbar(
            positions, values, yerr=error_values, label=series["label"],
            color=mcolors.to_rgba(series["color"], config["line"]["line_alpha"]),
            linewidth=config["line"]["linewidth"], marker=series["marker"],
            markersize=config["line"]["marker_size"],
            markerfacecolor=mcolors.to_rgba(
                series["color"], config["line"]["marker_face_alpha"]
            ),
            markeredgecolor=mcolors.to_rgba(
                series["color"], config["line"]["marker_edge_alpha"]
            ),
            markeredgewidth=config["line"]["marker_edge_width"],
            ecolor=mcolors.to_rgba(
                config["errorbar"]["color"], config["errorbar"]["alpha"]
            ),
            elinewidth=config["errorbar"]["linewidth"],
            capsize=config["errorbar"]["capsize"], zorder=3,
        )
        if config["line"]["value_labels"]:
            configured_ylim = config["axis"]["ylim"]
            for position, value in zip(positions, values):
                offset = float(config["line"]["value_offset_points"])
                vertical_alignment = "bottom"
                if configured_ylim is not None:
                    lower, upper = map(float, configured_ylim)
                    if value >= upper - 0.08 * (upper - lower):
                        offset = -abs(offset)
                        vertical_alignment = "top"
                axis.annotate(
                    config["line"]["value_format"].format(value),
                    xy=(position, value), xytext=(0, offset),
                    textcoords="offset points", ha="center", va=vertical_alignment,
                    fontsize=config["line"]["value_fontsize"],
                    color=series["color"], zorder=5,
                )
    highlight = config["line"]["highlight_best"]
    if highlight["enabled"]:
        missing_columns = set(highlight["columns"]) - set(series_by_column)
        if missing_columns:
            raise ValueError(
                "Every line.highlight_best.columns entry must match a configured "
                f"data.series column; missing: {', '.join(sorted(missing_columns))}"
            )
        for column in highlight["columns"]:
            values = np.array([row[column] for row in rows], dtype=float)
            best_value = float(np.max(values))
            best_positions = positions[
                np.isclose(values, best_value, rtol=1e-12, atol=1e-12)
            ]
            series = series_by_column[column]
            axis.scatter(
                best_positions, np.full(len(best_positions), best_value),
                marker=highlight["marker"], s=float(highlight["marker_size"]) ** 2,
                facecolor=series["color"], edgecolor=highlight["edge_color"],
                linewidth=float(highlight["edge_width"]), zorder=6, clip_on=False,
            )
    axis.set_title(chart_title(config["line"]["title"], rows))
    style_axis(axis, config, rows)
    if config["line"]["y_min"] is not None:
        axis.set_ylim(bottom=config["line"]["y_min"])
    figure.tight_layout()
    return save_figure(figure, output_dir / config["line"]["filename"], config)


def persist_replot_bundle(config: dict[str, Any], data_path: Path,
                          output_dir: Path) -> tuple[dict[str, Any], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundled_data = output_dir / "selection_fraction_plot_data.csv"
    if data_path.resolve() != bundled_data.resolve():
        shutil.copyfile(data_path, bundled_data)
    bundled_script = output_dir / Path(__file__).name
    if Path(__file__).resolve() != bundled_script.resolve():
        shutil.copyfile(__file__, bundled_script)

    bundled_config = deep_merge({}, config)
    bundled_config["data"]["source"] = os.path.relpath(data_path, output_dir)
    bundled_config["data"]["input"] = bundled_data.name
    bundled_config["output"]["directory"] = "."
    config_path = output_dir / "selection_fraction_plots_config.json"
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(bundled_config, stream, indent=2, ensure_ascii=False)
    return bundled_config, bundled_data


def generate_plots(config_path: Path, data_path: Path | None = None,
                   output_dir: Path | None = None) -> list[Path]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    validate_config(config)
    if not config["enabled"]:
        return []
    inferred_paths = infer_run_paths(config, config_path)
    if data_path is None:
        configured_input = config["data"]["input"]
        if configured_input is not None:
            data_path = resolve_from_config(configured_input, config_path)
        elif inferred_paths is not None:
            data_path = inferred_paths[0]
        else:
            raise ValueError(
                "Set data.input or data.evaluation_config when --data is not provided"
            )
    else:
        data_path = Path(data_path).resolve()
    if output_dir is None:
        configured_output = config["output"]["directory"]
        if configured_output is not None:
            output_dir = resolve_from_config(configured_output, config_path)
        elif inferred_paths is not None:
            output_dir = inferred_paths[1]
        else:
            output_dir = config_path.parent
    else:
        output_dir = Path(output_dir).resolve()

    config, bundled_data = persist_replot_bundle(config, data_path, output_dir)
    rows = read_chart_data(bundled_data, config)
    configure_matplotlib(config)
    paths: list[Path] = []
    if config["bar"]["enabled"]:
        paths.extend(plot_bar(rows, output_dir, config))
    if config["line"]["enabled"]:
        paths.extend(plot_line(rows, output_dir, config))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="JSON plot config; paths inside it are relative to the config file.",
    )
    parser.add_argument("--data", type=Path, help="Optional chart-ready CSV override.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_plots(args.config, args.data, args.output_dir)
    if paths:
        print("[INFO] generated " + ", ".join(str(path) for path in paths))
    else:
        print("[INFO] plotting is disabled in the config")


if __name__ == "__main__":
    main()
