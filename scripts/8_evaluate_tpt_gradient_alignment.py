#!/usr/bin/env python3
"""Measure experiment 8: raw-gradient and AdamW-update alignment."""

from __future__ import annotations

import argparse
import csv
import json
import importlib
import logging
import math
import multiprocessing
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import yaml

_datawise = importlib.import_module("6_evaluate_datawise_lr_tpt")
IdentifiedVariantDataset = _datawise.IdentifiedVariantDataset
atomic_write_json = _datawise.atomic_write_json
atomic_write_text = _datawise.atomic_write_text
autocast_context = _datawise.autocast_context
json_ready = _datawise.json_ready
safe_component = _datawise.safe_component
slugify = _datawise.slugify
from evaluate_tpt import (
    DATASET_NAMES,
    MODEL_FILES,
    TPTClassifier,
    TPTTransform,
    average_entropy,
    candidate_class_indices,
    initialize_progress_lock,
    load_class_index,
    load_class_names,
    set_seed,
)
_transitions = importlib.import_module("7_evaluate_tpt_transition_stats")
marginal_cross_entropy = _transitions.marginal_cross_entropy
select_low_entropy_count = _transitions.select_low_entropy_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "raw_norm_T", "raw_norm_C", "raw_cosine", "raw_dot",
    "update_norm_T", "update_norm_C", "update_cosine", "update_dot",
)
NEAR_ZERO_THRESHOLD = 1e-4
CONFIG_KEYS = {
    "data_root", "class_index", "class_names", "model_root", "output_root", "idea",
    "run_name", "datasets", "models", "devices", "seeds", "learning_rates",
    "num_workers", "views", "selection_fraction", "weight_decay", "prompt_init",
    "resolution", "print_frequency", "limit", "gradient_epsilon", "save_vectors",
    "log_file", "plot",
}
DEFAULTS: dict[str, Any] = {
    "data_root": "data/raw",
    "class_index": "configs/visualization/imagenet_class_index.json",
    "class_names": "configs/evaluation/imagenet_classes_tpt.json",
    "model_root": "models",
    "output_root": "runs/8_tpt_gradient_alignment",
    "idea": None,
    "run_name": None,
    "datasets": ["imagenet-a"],
    "models": ["ViT-B/16"],
    "devices": [f"cuda:{index}" for index in range(4)],
    "seeds": [123],
    "learning_rates": [0.003],
    "num_workers": 2,
    "views": 64,
    "selection_fraction": 0.1,
    "weight_decay": 0.01,
    "prompt_init": "a photo of a",
    "resolution": 224,
    "print_frequency": 100,
    "limit": None,
    "gradient_epsilon": 1e-12,
    "save_vectors": False,
    "log_file": "run.log",
    "plot": {"enabled": True, "config": "configs/visualization/8_tpt_gradient_alignment.yaml"},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Config file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise SystemExit("Config must contain a YAML mapping")
    return values


def validate_device(name: str) -> None:
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Invalid device {name}: {error}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"CUDA is unavailable, cannot use {name}")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise SystemExit(f"CUDA device index {index} is unavailable")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/evaluation/8_tpt_gradient_alignment.yaml")
    for name in ("data-root", "class-index", "class-names", "model-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--idea")
    parser.add_argument("--run-name")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--learning-rates", type=float, nargs="+")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--views", type=int)
    parser.add_argument("--selection-fraction", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--prompt-init")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--print-frequency", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gradient-epsilon", type=float)
    parser.add_argument("--save-vectors", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None)
    cli = parser.parse_args(argv)
    config_path = resolve_project_path(cli.config)
    config = load_yaml(config_path)
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise SystemExit(f"Unknown config keys: {', '.join(sorted(unknown))}")
    values = deep_merge(DEFAULTS, config)
    cli_map = {
        "data_root": cli.data_root, "class_index": cli.class_index, "class_names": cli.class_names,
        "model_root": cli.model_root, "output_root": cli.output_root, "idea": cli.idea,
        "run_name": cli.run_name, "datasets": cli.datasets, "models": cli.models,
        "devices": cli.devices, "seeds": cli.seeds, "learning_rates": cli.learning_rates,
        "num_workers": cli.num_workers, "views": cli.views, "selection_fraction": cli.selection_fraction,
        "weight_decay": cli.weight_decay, "prompt_init": cli.prompt_init, "resolution": cli.resolution,
        "print_frequency": cli.print_frequency, "limit": cli.limit, "gradient_epsilon": cli.gradient_epsilon,
        "save_vectors": cli.save_vectors,
    }
    for key, value in cli_map.items():
        if value is not None:
            values[key] = value
    for key in ("data_root", "class_index", "class_names", "model_root", "output_root"):
        values[key] = resolve_project_path(values[key])
    values["idea"] = safe_component(values["idea"], "idea")
    values["run_name"] = safe_component(values["run_name"], "run_name")
    if cli.plot is not None:
        values["plot"] = deep_merge(values.get("plot", {}), {"enabled": cli.plot})
    plot = deep_merge(DEFAULTS["plot"], values.get("plot", {}))
    if set(plot) - {"enabled", "config"}:
        raise SystemExit("plot accepts only enabled and config")
    plot["config"] = resolve_project_path(plot["config"]) if plot["config"] else None
    if plot["enabled"] and (plot["config"] is None or not plot["config"].is_file()):
        raise SystemExit("plot.config must exist when plotting is enabled")
    values["plot"] = plot
    if not isinstance(values["datasets"], list) or not values["datasets"] or len(set(values["datasets"])) != len(values["datasets"]):
        raise SystemExit("datasets must be a non-empty list without duplicates")
    if any(name not in DATASET_NAMES for name in values["datasets"]):
        raise SystemExit(f"datasets must contain only: {', '.join(DATASET_NAMES)}")
    if not isinstance(values["models"], list) or not values["models"] or len(set(values["models"])) != len(values["models"]):
        raise SystemExit("models must be a non-empty list without duplicates")
    if any(name not in MODEL_FILES for name in values["models"]):
        raise SystemExit(f"models must contain only: {', '.join(MODEL_FILES)}")
    if not isinstance(values["devices"], list) or not values["devices"] or len(set(values["devices"])) != len(values["devices"]):
        raise SystemExit("devices must be a non-empty list without duplicates")
    for name in values["devices"]:
        validate_device(name)
    if not isinstance(values["seeds"], list) or not values["seeds"] or len(set(values["seeds"])) != len(values["seeds"]):
        raise SystemExit("seeds must be a non-empty list without duplicates")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in values["seeds"]):
        raise SystemExit("seeds must contain integers")
    rates = values["learning_rates"]
    if not isinstance(rates, list) or not rates or len(set(rates)) != len(rates):
        raise SystemExit("learning_rates must be a non-empty list without duplicates")
    if any(isinstance(rate, bool) or not math.isfinite(float(rate)) or float(rate) < 0 for rate in rates):
        raise SystemExit("learning_rates must be finite and non-negative")
    values["learning_rates"] = [float(rate) for rate in rates]
    for field in ("views", "resolution", "print_frequency"):
        if not isinstance(values[field], int) or isinstance(values[field], bool) or values[field] < 1:
            raise SystemExit(f"{field} must be a positive integer")
    if values["views"] < 2:
        raise SystemExit("views must be at least 2")
    if not isinstance(values["num_workers"], int) or values["num_workers"] < 0:
        raise SystemExit("num_workers must be a non-negative integer")
    fraction = float(values["selection_fraction"])
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise SystemExit("selection_fraction must be in (0, 1]")
    values["selection_fraction"] = fraction
    values["selection_count"] = max(1, int(values["views"] * fraction))
    if values["selection_count"] > values["views"]:
        raise SystemExit("selection_fraction selects more views than available")
    for field in ("weight_decay", "gradient_epsilon"):
        value = float(values[field])
        if not math.isfinite(value) or value < 0 or (field == "gradient_epsilon" and value == 0):
            raise SystemExit(f"{field} must be finite and positive/non-negative as appropriate")
        values[field] = value
    if not isinstance(values["save_vectors"], bool):
        raise SystemExit("save_vectors must be true or false")
    if values["limit"] is not None and (not isinstance(values["limit"], int) or isinstance(values["limit"], bool) or values["limit"] < 1):
        raise SystemExit("limit must be a positive integer or null")
    run_name = values["run_name"] or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_parent = values["output_root"] / (f"idea_{values['idea']}" if values["idea"] else "")
    values["output_parent"] = output_parent
    values["output_root"] = output_parent / f"run_{run_name}"
    return argparse.Namespace(**values), config_path


def setup_logging(output_root: Path, log_file: str) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=False)
    path = Path(log_file)
    log_path = output_root / f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{path.suffix or '.log'}"
    logger = logging.getLogger("tpt_gradient_alignment")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(processName)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def rate_slug(rate: float) -> str:
    return f"{rate:.12g}".replace("-", "m").replace(".", "p")


def task_slug(model: str, dataset: str, rate: float) -> str:
    return f"{slugify(model)}__{slugify(dataset)}__lr_{rate_slug(rate)}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    temp.replace(path)


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def metric_stats(values: list[float], metric: str) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    result = {
        "mean": None, "median": None, "standard_deviation": None, "iqr": None,
        "positive_count": 0, "positive_ratio": None,
        "negative_count": 0, "negative_ratio": None,
        "exact_zero_count": 0, "exact_zero_ratio": None,
        "near_zero_count": 0, "near_zero_ratio": None,
        "valid_count": len(finite), "nonfinite_count": len(values) - len(finite),
    }
    if not finite:
        return result
    array = np.asarray(finite, dtype=float)
    result.update({
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": sample_std(finite),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
    })
    for name, condition in (
        ("positive", array > 0),
        ("negative", array < 0),
        ("exact_zero", array == 0),
        ("near_zero", np.abs(array) < NEAR_ZERO_THRESHOLD),
    ):
        count = int(np.count_nonzero(condition))
        result[f"{name}_count"] = count
        result[f"{name}_ratio"] = count / len(finite)
    return result


def summarize_records(records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    experiment_keys = list(dict.fromkeys(
        (record["model"], record["dataset"], float(record["learning_rate"]))
        for record in records
    ))
    for model, dataset, rate in experiment_keys:
        task_records = [
            record for record in records
            if record["model"] == model and record["dataset"] == dataset
            and float(record["learning_rate"]) == rate
        ]
        groups = {
            "all": task_records,
            "initial_correct": [row for row in task_records if row["initial_correct"]],
            "initial_wrong": [row for row in task_records if not row["initial_correct"]],
        }
        for group, group_records in groups.items():
            for metric in METRICS:
                row = {
                    "model": model, "dataset": dataset, "learning_rate": rate,
                    "scope": scope, "group": group, "metric": metric,
                }
                row.update(metric_stats([float(record[metric]) for record in group_records], metric))
                rows.append(row)
    return rows


def distribution_rows(records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        groups = ["all", "initial_correct" if record["initial_correct"] else "initial_wrong"]
        for group in groups:
            for metric in METRICS:
                rows.append({
                    "scope": scope, "sample_index": record["sample_index"],
                    "sample_id": record["sample_id"], "group": group,
                    "metric": metric, "value": record[metric],
                    "model": record["model"], "dataset": record["dataset"],
                    "learning_rate": record["learning_rate"], "seed": record["seed"],
                })
    return rows


def flatten_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().reshape(-1).clone()


def vector_metrics(t_gradient: torch.Tensor, c_gradient: torch.Tensor, t_update: torch.Tensor, c_update: torch.Tensor, epsilon: float) -> dict[str, float]:
    t_gradient, c_gradient = flatten_tensor(t_gradient), flatten_tensor(c_gradient)
    t_update, c_update = flatten_tensor(t_update), flatten_tensor(c_update)
    def pair(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float, float, float]:
        left_norm = torch.linalg.vector_norm(left)
        right_norm = torch.linalg.vector_norm(right)
        dot = torch.dot(left, right)
        cosine = dot / (left_norm * right_norm + epsilon)
        return float(left_norm), float(right_norm), float(cosine), float(dot)
    raw_t, raw_c, raw_cos, raw_dot = pair(t_gradient, c_gradient)
    update_t, update_c, update_cos, update_dot = pair(t_update, c_update)
    values = {
        "raw_norm_T": raw_t, "raw_norm_C": raw_c, "raw_cosine": raw_cos, "raw_dot": raw_dot,
        "update_norm_T": update_t, "update_norm_C": update_c,
        "update_cosine": update_cos, "update_dot": update_dot,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("Non-finite gradient alignment metric")
    return values


def branch_gradient_update(
    model: TPTClassifier,
    selected_views: torch.Tensor,
    theta0: torch.Tensor,
    objective: str,
    local_target: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    with torch.no_grad():
        model.prompt_learner.ctx.copy_(theta0)
    optimizer = torch.optim.AdamW([model.prompt_learner.ctx], lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=False):
        logits = model(selected_views.float())
        loss = average_entropy(logits) if objective == "T" else marginal_cross_entropy(logits, local_target)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite {objective} loss")
    loss.backward()
    gradient = model.prompt_learner.ctx.grad
    if gradient is None:
        raise RuntimeError(f"{objective} gradient is missing")
    gradient = gradient.detach().float().clone()
    if not torch.isfinite(gradient).all():
        raise RuntimeError(f"Non-finite {objective} gradient")
    optimizer.step()
    update = model.prompt_learner.ctx.detach().float().clone() - theta0.detach().float()
    if not torch.isfinite(update).all():
        raise RuntimeError(f"Non-finite {objective} optimizer update")
    return gradient, update, float(loss.detach().float().item())


def evaluate_task(serialized_args: dict[str, Any], model_name: str, dataset_name: str, learning_rate: float, seed: int, device_name: str, position: int, seed_root: str) -> dict[str, Any]:
    args = argparse.Namespace(**serialized_args)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(seed)
    task_root = Path(seed_root) / task_slug(model_name, dataset_name, learning_rate)
    task_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"tpt_gradient_alignment.worker.{os.getpid()}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(task_root / "task.log", mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(processName)s: %(message)s"))
    logger.addHandler(handler)
    import clip
    class_index = load_class_index(Path(args.class_index))
    all_names = load_class_names(Path(args.class_names))
    transform = TPTTransform(args.resolution, args.views, True)
    full_dataset = IdentifiedVariantDataset(Path(args.data_root) / dataset_name, dataset_name, class_index, transform)
    candidate_indices = candidate_class_indices(full_dataset, dataset_name)
    candidate_names = [all_names[index] for index in candidate_indices]
    global_to_local = {global_index: local_index for local_index, global_index in enumerate(candidate_indices)}
    dataset: Dataset = full_dataset if args.limit is None else Subset(full_dataset, range(min(args.limit, len(full_dataset))))
    checkpoint = Path(args.model_root) / MODEL_FILES[model_name]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")
    model = TPTClassifier(clip, checkpoint, candidate_names, args.prompt_init, device).to(device)
    model.eval()
    loader_kwargs: dict[str, Any] = {"batch_size": 1, "shuffle": False, "num_workers": args.num_workers, "pin_memory": device.type == "cuda", "persistent_workers": args.num_workers > 0, "generator": torch.Generator().manual_seed(seed)}
    if args.num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(dataset, **loader_kwargs)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    incremental = task_root / "sample_gradient_metrics.jsonl"
    progress = tqdm(loader, desc=f"{device} / gradient alignment / lr={learning_rate:g} / seed={seed} / {dataset_name}", unit="image", position=position, leave=True, disable=not os.isatty(1))
    with incremental.open("w", encoding="utf-8", buffering=1) as stream:
        for view_list, labels, sample_ids in progress:
            views = torch.cat(view_list, dim=0).to(device, non_blocking=True)
            true_label = int(labels.item())
            if true_label not in global_to_local:
                raise RuntimeError(f"Ground-truth label {true_label} is outside candidate classes")
            local_target = global_to_local[true_label]
            with torch.no_grad():
                model.reset()
            with torch.no_grad(), autocast_context(device):
                initial_logits = model(views)
            _, selected_indices = select_low_entropy_count(initial_logits, args.selection_count)
            theta0 = model.prompt_learner.ctx.detach().clone()
            selected_views = views[selected_indices]
            t_gradient, t_update, t_loss = branch_gradient_update(model, selected_views, theta0, "T", local_target, learning_rate, args.weight_decay, device)
            c_gradient, c_update, c_loss = branch_gradient_update(model, selected_views, theta0, "C", local_target, learning_rate, args.weight_decay, device)
            metrics = vector_metrics(t_gradient, c_gradient, t_update, c_update, args.gradient_epsilon)
            canonical_prediction = candidate_indices[int(initial_logits[:1].argmax(dim=1).item())]
            initial_correct = canonical_prediction == true_label
            record = {
                "sample_index": len(records), "sample_id": sample_ids[0], "model": model_name,
                "dataset": dataset_name, "seed": seed, "learning_rate": learning_rate,
                "views": args.views, "selection_fraction": args.selection_fraction,
                "selection_count": args.selection_count, "selected_view_indices": selected_indices.cpu().tolist(),
                "true_label": true_label, "local_target": local_target,
                "initial_prediction": canonical_prediction, "initial_correct": initial_correct,
                "T_loss": t_loss, "C_loss": c_loss, **metrics,
            }
            if args.save_vectors:
                record.update({
                    "raw_gradient_T": t_gradient.cpu().tolist(), "raw_gradient_C": c_gradient.cpu().tolist(),
                    "update_T": t_update.cpu().tolist(), "update_C": c_update.cpu().tolist(),
                })
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.set_postfix(correct=f"{sum(row['initial_correct'] for row in records) / len(records):.3f}")
            if len(records) % args.print_frequency == 0 or len(records) == len(dataset):
                logger.info("progress %d/%d", len(records), len(dataset))
    if not records:
        raise RuntimeError("No samples were processed")
    atomic_write_json(task_root / "sample_gradient_metrics.json", records)
    write_csv(task_root / "sample_gradient_metrics.csv", records)
    elapsed = (time.perf_counter() - started) / 60
    summaries = summarize_records(records, "per_task")
    result = {
        "method": "TPT Gradient Alignment", "model": model_name, "dataset": dataset_name,
        "seed": seed, "learning_rate": learning_rate, "samples": len(records),
        "initial_correct_count": sum(row["initial_correct"] for row in records),
        "initial_correct_ratio": sum(row["initial_correct"] for row in records) / len(records),
        "candidate_classes": len(candidate_indices), "views": args.views,
        "selection_fraction": args.selection_fraction, "selection_count": args.selection_count,
        "weight_decay": args.weight_decay, "prompt_init": args.prompt_init,
        "device": str(device), "elapsed_minutes": elapsed,
        "sample_results": str(task_root / "sample_gradient_metrics.json"),
        "summary_rows": summaries,
    }
    atomic_write_json(task_root / "task_result.json", result)
    logger.info("completed samples=%d in %.2f min", len(records), elapsed)
    if device.type == "cuda":
        del model
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return result


def build_jobs(args: argparse.Namespace) -> list[tuple[str, str, float, int]]:
    return list(product(args.models, args.datasets, args.learning_rates, args.seeds))


def assign_jobs(jobs: list[tuple[str, str, float, int]], devices: list[str]) -> list[list[tuple[str, str, float, int]]]:
    assignments = [[] for _ in devices]
    for index, job in enumerate(jobs):
        assignments[index % len(devices)].append(job)
    return assignments


def load_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    with Path(result["sample_results"]).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_summary_files(root: Path, results: list[dict[str, Any]], records: list[dict[str, Any]], title: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    scope = "pooled" if title == "All tasks" else "per_seed"
    summary_rows = summarize_records(records, scope)
    distribution = distribution_rows(records, scope)
    atomic_write_json(root / "results.json", results)
    write_csv(root / "results.csv", results)
    atomic_write_json(root / "gradient_summary.json", summary_rows)
    write_csv(root / "gradient_summary.csv", summary_rows)
    write_csv(root / "distribution_data.csv", distribution)
    lines = [f"# {title}", "", f"Samples: {len(records)}", "", "| Model | Dataset | LR | Group | Metric | Mean | Median | Std | IQR | Negative ratio | Valid |", "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary_rows:
        def fmt(key: str) -> str:
            value = row[key]
            return "n/a" if value is None else f"{value:.8g}"
        lines.append(f"| {row['model']} | {row['dataset']} | {row['learning_rate']:g} | {row['group']} | {row['metric']} | {fmt('mean')} | {fmt('median')} | {fmt('standard_deviation')} | {fmt('iqr')} | {fmt('negative_ratio')} | {row['valid_count']} |")
    lines.extend([
        "", "## Sign and near-zero counts", "",
        f"Counts use finite samples. Positive means x > 0; negative means x < 0; exact zero means x = 0; near zero means |x| < {NEAR_ZERO_THRESHOLD:g}. Near zero overlaps the sign counts. Percentages use the valid count for that row.",
        "", "| Model | Dataset | LR | Group | Metric | Positive | Negative | Exact zero | Near zero | Valid |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary_rows:
        def count_percent(name: str) -> str:
            ratio = row[f"{name}_ratio"]
            return "n/a" if ratio is None else f"{row[f'{name}_count']} ({ratio:.2%})"
        lines.append(f"| {row['model']} | {row['dataset']} | {row['learning_rate']:g} | {row['group']} | {row['metric']} | {count_percent('positive')} | {count_percent('negative')} | {count_percent('exact_zero')} | {count_percent('near_zero')} | {row['valid_count']} |")
    atomic_write_text(root / "summary.md", "\n".join(lines) + "\n")


def run_jobs(args: argparse.Namespace, devices: list[str]) -> list[dict[str, Any]]:
    jobs = build_jobs(args)
    assignments = assign_jobs(jobs, devices)
    for seed in args.seeds:
        (args.output_root / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
    spawn = multiprocessing.get_context("spawn")
    lock = spawn.RLock()
    results: list[dict[str, Any]] = []
    with ExitStack() as stack:
        executors = [stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=spawn, initializer=initialize_progress_lock, initargs=(lock,))) for _ in devices]
        futures = {}
        serialized = json_ready(vars(args))
        for position, (executor, device, device_jobs) in enumerate(zip(executors, devices, assignments)):
            for model_name, dataset_name, rate, seed in device_jobs:
                future = executor.submit(evaluate_task, serialized, model_name, dataset_name, rate, seed, device, position, str(args.output_root / f"seed_{seed}"))
                futures[future] = (model_name, dataset_name, rate, seed)
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[SUCCESS] seed={result['seed']} {result['model']} {result['dataset']} lr={result['learning_rate']:g} samples={result['samples']}", flush=True)
    return results


def main(argv: list[str] | None = None) -> None:
    args, config_path = parse_args(argv)
    logger = setup_logging(args.output_root, args.log_file)
    payload = json_ready(vars(args)); payload["source_config"] = str(config_path); payload["generated_at"] = datetime.now().astimezone().isoformat()
    atomic_write_text(args.output_root / "resolved_config.yaml", yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    jobs = build_jobs(args)
    devices = args.devices[:min(len(args.devices), len(jobs))]
    logger.info("resolved args=%s", payload)
    print(f"[INFO] TPT gradient alignment: {len(jobs)} tasks, devices={', '.join(devices)}", flush=True)
    results = run_jobs(args, devices)
    results.sort(key=lambda row: (args.models.index(row["model"]), args.datasets.index(row["dataset"]), args.learning_rates.index(float(row["learning_rate"])), args.seeds.index(int(row["seed"]))))
    all_records: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_results = [row for row in results if int(row["seed"]) == seed]
        seed_records: list[dict[str, Any]] = []
        for result in seed_results:
            seed_records.extend(load_records(result))
        write_summary_files(args.output_root / f"seed_{seed}" / "summary", seed_results, seed_records, f"Seed {seed}")
        all_records.extend(seed_records)
    summary_root = args.output_root / "summary"
    write_summary_files(summary_root, results, all_records, "All tasks")
    if args.plot["enabled"]:
        generate_plots = importlib.import_module("8_plot_tpt_gradient_alignment").generate_plots
        for root in [args.output_root / f"seed_{seed}" / "summary" for seed in args.seeds] + [summary_root]:
            generate_plots(Path(args.plot["config"]), root / "distribution_data.csv", root / "plots")
        print("[SUCCESS] generated gradient alignment plots", flush=True)
    atomic_write_json(args.output_root / "COMPLETED.json", {"status": "complete", "completed_at": datetime.now().astimezone().isoformat(), "total_tasks": len(jobs), "expected_seeds": args.seeds, "completed_seeds": args.seeds, "summary": str(summary_root)})
    logger.info("completed workflow; summary=%s", summary_root)
    print(f"[SUCCESS] results written to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
