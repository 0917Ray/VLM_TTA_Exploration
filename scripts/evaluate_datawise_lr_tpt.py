#!/usr/bin/env python3
"""Evaluate TPT with entropy-selected per-sample learning rates."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import yaml

from evaluate_tpt import (
    DATASET_NAMES,
    MODEL_FILES,
    TPTClassifier,
    TPTTransform,
    VariantDataset,
    accumulate_topk_correct,
    average_entropy,
    candidate_class_indices,
    initialize_progress_lock,
    load_class_index,
    load_class_names,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = {
    "data_root", "class_index", "class_names", "model_root", "output_root",
    "idea", "run_name", "datasets", "models", "devices", "seeds", "num_workers",
    "views", "selection_fractions", "fraction_options", "selection_counts",
    "count_options", "top_k", "tta_steps", "tta_step_options",
    "candidate_lrs", "weight_decay", "prompt_init", "resolution", "print_frequency",
    "limit", "log_file", "plot",
}
DEFAULTS: dict[str, Any] = {
    "data_root": "data/raw",
    "class_index": "configs/visualization/imagenet_class_index.json",
    "class_names": "configs/evaluation/imagenet_classes_tpt.json",
    "model_root": "models",
    "output_root": "runs/datawise_lr_tpt",
    "idea": "entropy_min",
    "run_name": None,
    "datasets": list(DATASET_NAMES),
    "models": ["RN50", "ViT-B/16"],
    "devices": ["cuda:0"] if torch.cuda.is_available() else ["cpu"],
    "seeds": [11, 42, 123],
    "num_workers": min(4, os.cpu_count() or 1),
    "views": 64,
    "selection_fractions": False,
    "fraction_options": [0.1],
    "selection_counts": True,
    "count_options": [1],
    "top_k": 5,
    "tta_steps": 1,
    "tta_step_options": None,
    "candidate_lrs": [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02],
    "weight_decay": 0.01,
    "prompt_init": "a photo of a",
    "resolution": 224,
    "print_frequency": 100,
    "limit": None,
    "log_file": "run.log",
    "plot": {
        "enabled": True,
        "config": "configs/visualization/datawise_lr_tpt_plots.json",
    },
}


@dataclass
class CandidateSelection:
    initial_logits: torch.Tensor
    selected_indices: torch.Tensor
    initial_entropy: float
    candidate_entropies: list[float]
    selected_lr: float
    selected_lr_index: int
    amp_scale: float
    amp_retries: int


class IdentifiedVariantDataset(VariantDataset):
    """Return the stable dataset-relative path with each existing TPT sample."""

    def __getitem__(self, index: int):
        views, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return views, label, path.relative_to(self.root).as_posix()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Config file does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError) as error:
        raise SystemExit(f"Cannot read config {path}: {error}") from error
    if not isinstance(config, dict):
        raise SystemExit("Config must contain a YAML mapping")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise SystemExit(f"Unknown config keys: {', '.join(sorted(unknown))}")
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def safe_component(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SystemExit(f"{field} must be a string or null")
    cleaned = "_".join(value.split())
    if not cleaned:
        return None
    if cleaned in {".", ".."} or any(char in cleaned for char in ("/", "\\", "\0")):
        raise SystemExit(f"{field} must be a directory-safe name without path separators")
    return cleaned


def validate_device(device_name: str) -> None:
    try:
        device = torch.device(device_name)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Invalid device {device_name}: {error}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"CUDA is unavailable, cannot use {device_name}")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise SystemExit(
                f"CUDA device index {index} is unavailable; found {torch.cuda.device_count()} GPUs"
            )


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/evaluation/datawise_lr_tpt.yaml",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--class-index", type=Path)
    parser.add_argument("--class-names", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--idea")
    parser.add_argument("--run-name")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--views", type=int)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--selection-fractions", type=float, nargs="+")
    selection_group.add_argument("--selection-counts", type=int, nargs="+")
    parser.add_argument("--top-k", type=int)
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument("--tta-steps", type=int)
    step_group.add_argument("--tta-step-options", type=int, nargs="+")
    parser.add_argument("--candidate-lrs", type=float, nargs="+")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--prompt-init")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--print-frequency", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None)
    cli = parser.parse_args(argv)

    config_path = resolve_project_path(cli.config)
    config = load_config(config_path)
    if "tta_steps" in config and "tta_step_options" in config:
        raise SystemExit("Config cannot contain both tta_steps and tta_step_options")

    values = {**DEFAULTS, **config}
    for key in (
        "data_root", "class_index", "class_names", "model_root", "output_root", "idea",
        "run_name", "datasets", "models", "devices", "seeds", "num_workers", "views",
        "top_k", "weight_decay", "prompt_init", "resolution",
        "print_frequency", "limit", "candidate_lrs",
    ):
        value = getattr(cli, key, None)
        if value is not None:
            values[key] = value
    if cli.selection_fractions is not None:
        values["selection_fractions"] = True
        values["selection_counts"] = False
        values["fraction_options"] = cli.selection_fractions
    elif cli.selection_counts is not None:
        values["selection_counts"] = True
        values["selection_fractions"] = False
        values["count_options"] = cli.selection_counts
    if cli.tta_steps is not None:
        values["tta_steps"] = cli.tta_steps
        values["tta_step_options"] = [cli.tta_steps]
    elif cli.tta_step_options is not None:
        values["tta_step_options"] = cli.tta_step_options

    for key in ("data_root", "class_index", "class_names", "model_root", "output_root"):
        values[key] = resolve_project_path(values[key])
    values["idea"] = safe_component(values["idea"], "idea")
    values["run_name"] = safe_component(values["run_name"], "run_name")

    if not isinstance(values["datasets"], list) or not values["datasets"]:
        raise SystemExit("datasets must be a non-empty list")
    if len(set(values["datasets"])) != len(values["datasets"]):
        raise SystemExit("datasets must not contain duplicates")
    if any(name not in DATASET_NAMES for name in values["datasets"]):
        raise SystemExit(f"datasets must contain only: {', '.join(DATASET_NAMES)}")
    if not isinstance(values["models"], list) or not values["models"]:
        raise SystemExit("models must be a non-empty list")
    if len(set(values["models"])) != len(values["models"]):
        raise SystemExit("models must not contain duplicates")
    if any(name not in MODEL_FILES for name in values["models"]):
        raise SystemExit(f"models must contain only: {', '.join(MODEL_FILES)}")
    if (not isinstance(values["devices"], list) or not values["devices"] or
            any(not isinstance(device, str) or not device for device in values["devices"])):
        raise SystemExit("devices must be a non-empty list of device names")
    if len(set(values["devices"])) != len(values["devices"]):
        raise SystemExit("devices must not contain duplicates")
    for device_name in values["devices"]:
        validate_device(device_name)
    if (not isinstance(values["seeds"], list) or not values["seeds"] or
            any(not isinstance(seed, int) or isinstance(seed, bool) for seed in values["seeds"])):
        raise SystemExit("seeds must be a non-empty list of integers")
    if len(set(values["seeds"])) != len(values["seeds"]):
        raise SystemExit("seeds must not contain duplicates")

    for field in ("views", "top_k", "resolution", "print_frequency"):
        value = values[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SystemExit(f"{field} must be a positive integer")
    if values["views"] < 2:
        raise SystemExit("views must be at least 2 for augmented TPT")
    fraction_mode = values["selection_fractions"]
    count_mode = values["selection_counts"]
    if not isinstance(fraction_mode, bool) or not isinstance(count_mode, bool):
        raise SystemExit("selection_fractions and selection_counts must be true or false")
    if fraction_mode == count_mode:
        raise SystemExit("Set exactly one of selection_fractions or selection_counts to true")
    selection_specs: list[dict[str, Any]] = []
    if count_mode:
        counts = values["count_options"]
        if (not isinstance(counts, list) or not counts or
                any(not isinstance(count, int) or isinstance(count, bool) for count in counts)):
            raise SystemExit("count_options must be a non-empty list of integers")
        if len(set(counts)) != len(counts) or any(count < 1 or count > values["views"] for count in counts):
            raise SystemExit(f"count_options must be unique and between 1 and {values['views']}")
        selection_specs = [
            {
                "selection_mode": "count",
                "selection_value": count,
                "selection_count": count,
                "selection_fraction": count / values["views"],
            }
            for count in counts
        ]
    else:
        fractions = values["fraction_options"]
        if (not isinstance(fractions, list) or not fractions or
                any(isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                    for fraction in fractions)):
            raise SystemExit("fraction_options must be a non-empty list of numbers")
        fractions = [float(fraction) for fraction in fractions]
        if len(set(fractions)) != len(fractions):
            raise SystemExit("fraction_options must not contain duplicates")
        if any(not math.isfinite(fraction) or not 0 < fraction <= 1 for fraction in fractions):
            raise SystemExit("every fraction_options value must be finite and in (0, 1]")
        if any(int(values["views"] * fraction) < 1 for fraction in fractions):
            raise SystemExit("views * every fraction_options value must select at least one view")
        values["fraction_options"] = fractions
        selection_specs = [
            {
                "selection_mode": "fraction",
                "selection_value": fraction,
                "selection_count": int(values["views"] * fraction),
                "selection_fraction": fraction,
            }
            for fraction in fractions
        ]
    values["selection_specs"] = selection_specs
    if not isinstance(values["num_workers"], int) or values["num_workers"] < 0:
        raise SystemExit("num_workers must be a non-negative integer")
    if values["limit"] is not None and (
        not isinstance(values["limit"], int) or isinstance(values["limit"], bool) or
        values["limit"] < 1
    ):
        raise SystemExit("limit must be a positive integer or null")

    steps = values["tta_step_options"]
    if steps is None:
        steps = [values["tta_steps"]]
    if (not isinstance(steps, list) or not steps or
            any(not isinstance(step, int) or isinstance(step, bool) or step < 1 for step in steps)):
        raise SystemExit("tta_step_options must be a non-empty list of positive integers")
    if len(set(steps)) != len(steps):
        raise SystemExit("tta_step_options must not contain duplicates")
    values["tta_step_values"] = steps
    values["tta_steps"] = steps[0]

    rates = values["candidate_lrs"]
    if (not isinstance(rates, list) or not rates or
            any(isinstance(rate, bool) or not isinstance(rate, (int, float)) for rate in rates)):
        raise SystemExit("candidate_lrs must be a non-empty list of numbers")
    rates = [float(rate) for rate in rates]
    if any(not math.isfinite(rate) or rate < 0 for rate in rates):
        raise SystemExit("candidate_lrs must be finite and non-negative")
    if len(set(rates)) != len(rates):
        raise SystemExit("candidate_lrs must not contain duplicates")
    if rates != sorted(rates):
        raise SystemExit("candidate_lrs must be sorted in ascending order")
    values["candidate_lrs"] = rates
    if (isinstance(values["weight_decay"], bool) or
            not isinstance(values["weight_decay"], (int, float)) or
            not math.isfinite(float(values["weight_decay"])) or
            values["weight_decay"] < 0):
        raise SystemExit("weight_decay must be finite and non-negative")
    values["weight_decay"] = float(values["weight_decay"])
    if not isinstance(values["prompt_init"], str) or not values["prompt_init"].strip():
        raise SystemExit("prompt_init must be a non-empty string")

    plot = {**DEFAULTS["plot"], **values.get("plot", {})}
    if set(plot) - {"enabled", "config"}:
        raise SystemExit("plot must contain only enabled and config")
    if cli.plot is not None:
        plot["enabled"] = cli.plot
    if not isinstance(plot["enabled"], bool):
        raise SystemExit("plot.enabled must be true or false")
    if plot["config"] is not None:
        plot["config"] = resolve_project_path(plot["config"])
    if plot["enabled"] and (plot["config"] is None or not plot["config"].is_file()):
        raise SystemExit("plot.config must exist when automatic plotting is enabled")
    values["plot"] = plot

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = values["run_name"] or timestamp
    run_root = values["output_root"]
    if values["idea"]:
        run_root = run_root / f"idea_{values['idea']}"
    run_root = run_root / f"run_{run_name}"
    values["output_parent"] = values["output_root"]
    values["output_root"] = run_root
    return argparse.Namespace(**values), config_path


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_resolved_config(args: argparse.Namespace, config_path: Path) -> None:
    payload = json_ready(vars(args))
    payload["source_config"] = str(config_path)
    payload["generated_at"] = datetime.now().astimezone().isoformat()
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    atomic_write_text(args.output_root / "resolved_config.yaml", text)


def setup_main_logging(output_root: Path, log_file: str) -> tuple[logging.Logger, Path]:
    output_root.mkdir(parents=True, exist_ok=False)
    configured = Path(log_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = output_root / configured.with_name(
        f"{configured.stem}_{timestamp}{configured.suffix or '.log'}"
    ).name
    logger = logging.getLogger("datawise_lr_tpt")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(processName)s: %(message)s"
    ))
    logger.addHandler(handler)
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.addHandler(handler)
    warnings_logger.propagate = False
    worker_filter = "ignore:pkg_resources is deprecated as an API:UserWarning"
    existing_filters = os.environ.get("PYTHONWARNINGS")
    os.environ["PYTHONWARNINGS"] = (
        f"{existing_filters},{worker_filter}" if existing_filters else worker_filter
    )
    return logger, log_path


def setup_worker_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"datawise_lr_tpt.worker.{os.getpid()}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(processName)s: %(message)s"
    ))
    logger.addHandler(handler)
    return logger


def slugify(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")


def selection_slug(selection_spec: dict[str, Any]) -> str:
    value = f"{selection_spec['selection_value']:g}".replace(".", "p")
    return f"{selection_spec['selection_mode']}_{value}"


def selection_spec_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_mode": result["selection_mode"],
        "selection_value": result["selection_value"],
        "selection_count": result["selection_count"],
        "selection_fraction": result["selection_fraction"],
    }


def task_slug(
    model: str, dataset: str, steps: int, selection_spec: dict[str, Any]
) -> str:
    return (
        f"{slugify(model)}__{slugify(dataset)}__{selection_slug(selection_spec)}"
        f"__steps_{steps}"
    )


def autocast_context(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    )


def make_scaler(device: torch.device):
    return torch.amp.GradScaler(device.type, init_scale=1000, enabled=device.type == "cuda")


def first_step_adamw_context(
    theta0: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
    weight_decay: float,
    epsilon: float,
) -> torch.Tensor:
    """Functional form of the first bias-corrected AdamW update."""
    direction = gradient / (gradient.abs() + epsilon)
    return theta0 * (1.0 - learning_rate * weight_decay) - learning_rate * direction


def select_low_entropy_count(
    logits: torch.Tensor, selection_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    entropy = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)
    indices = torch.argsort(entropy)[:selection_count]
    return logits[indices], indices


def select_one_step_lr(
    model: TPTClassifier,
    views: torch.Tensor,
    candidate_lrs: list[float],
    selection_count: int,
    weight_decay: float,
    device: torch.device,
    scaler,
) -> CandidateSelection:
    theta0 = model.prompt_learner.ctx.detach().clone()
    optimizer = torch.optim.AdamW(
        [model.prompt_learner.ctx], lr=candidate_lrs[0], weight_decay=weight_decay
    )
    amp_retries = 0
    while True:
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(theta0)
        with autocast_context(device):
            initial_logits = model(views)
            selected_logits, selected_indices = select_low_entropy_count(
                initial_logits, selection_count
            )
            initial_loss = average_entropy(selected_logits)
        if not torch.isfinite(initial_loss):
            raise RuntimeError("Non-finite entropy during one-step LR selection")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(initial_loss).backward()
        scaler.unscale_(optimizer)
        gradient = model.prompt_learner.ctx.grad
        if gradient is None:
            raise RuntimeError("Missing prompt gradient during one-step LR selection")
        if torch.isfinite(gradient).all():
            break
        previous_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        amp_retries += 1
        if float(scaler.get_scale()) >= previous_scale or amp_retries >= 16:
            raise RuntimeError(
                "Prompt gradient remained non-finite after AMP scale recovery; "
                f"last scale={scaler.get_scale():g}, retries={amp_retries}"
            )
    gradient = gradient.detach().clone()
    epsilon = float(optimizer.defaults["eps"])
    selected_views = views[selected_indices]

    entropies: list[float] = []
    contexts: list[torch.Tensor] = []
    for learning_rate in candidate_lrs:
        candidate = first_step_adamw_context(
            theta0, gradient, learning_rate, weight_decay, epsilon
        )
        contexts.append(candidate)
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(candidate)
            with autocast_context(device):
                entropy = average_entropy(model(selected_views))
        if not torch.isfinite(entropy):
            raise RuntimeError(f"Non-finite entropy for learning rate {learning_rate:g}")
        entropies.append(float(entropy.item()))

    best_index = min(range(len(candidate_lrs)), key=lambda index: (entropies[index], index))
    best_lr = candidate_lrs[best_index]
    with torch.no_grad():
        model.prompt_learner.ctx.copy_(theta0)
    optimizer.param_groups[0]["lr"] = best_lr
    scaler.step(optimizer)
    scaler.update()
    actual = model.prompt_learner.ctx.detach()
    expected = contexts[best_index]
    if not torch.allclose(actual, expected, rtol=2e-5, atol=2e-6):
        maximum_error = float((actual - expected).abs().max().item())
        raise RuntimeError(
            "Functional first-step AdamW candidate does not match optimizer.step(); "
            f"max absolute error={maximum_error:.3e}"
        )
    return CandidateSelection(
        initial_logits=initial_logits.detach(),
        selected_indices=selected_indices.detach(),
        initial_entropy=float(initial_loss.detach().item()),
        candidate_entropies=entropies,
        selected_lr=best_lr,
        selected_lr_index=best_index,
        amp_scale=float(scaler.get_scale()),
        amp_retries=amp_retries,
    )


def adamw_step_with_amp_recovery(
    model: TPTClassifier,
    selected_views: torch.Tensor,
    optimizer,
    scaler,
    device: torch.device,
) -> int:
    """Execute one real AdamW step, lowering the AMP scale after overflow."""
    retries = 0
    while True:
        with autocast_context(device):
            loss = average_entropy(model(selected_views))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite entropy during multi-step LR selection")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        previous_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) >= previous_scale:
            return retries
        retries += 1
        if retries >= 16:
            raise RuntimeError(
                "AdamW step kept overflowing after AMP scale recovery; "
                f"last scale={scaler.get_scale():g}, retries={retries}"
            )


def select_multi_step_lr(
    model: TPTClassifier,
    views: torch.Tensor,
    candidate_lrs: list[float],
    selection_count: int,
    tta_steps: int,
    weight_decay: float,
    device: torch.device,
) -> CandidateSelection:
    if tta_steps < 2:
        raise ValueError("select_multi_step_lr requires tta_steps >= 2")
    theta0 = model.prompt_learner.ctx.detach().clone()
    with torch.no_grad(), autocast_context(device):
        initial_logits = model(views)
        selected_logits, selected_indices = select_low_entropy_count(
            initial_logits, selection_count
        )
        initial_entropy = float(average_entropy(selected_logits).item())
    selected_views = views[selected_indices]

    entropies: list[float] = []
    contexts: list[torch.Tensor] = []
    total_amp_retries = 0
    candidate_amp_scales: list[float] = []
    for learning_rate in candidate_lrs:
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(theta0)
        if learning_rate > 0:
            optimizer = torch.optim.AdamW(
                [model.prompt_learner.ctx], lr=learning_rate, weight_decay=weight_decay
            )
            scaler = make_scaler(device)
            for _ in range(tta_steps):
                total_amp_retries += adamw_step_with_amp_recovery(
                    model, selected_views, optimizer, scaler, device
                )
            candidate_amp_scales.append(float(scaler.get_scale()))
        with torch.no_grad(), autocast_context(device):
            final_entropy = average_entropy(model(selected_views))
        if not torch.isfinite(final_entropy):
            raise RuntimeError(f"Non-finite entropy for learning rate {learning_rate:g}")
        entropies.append(float(final_entropy.item()))
        contexts.append(model.prompt_learner.ctx.detach().clone())

    best_index = min(range(len(candidate_lrs)), key=lambda index: (entropies[index], index))
    with torch.no_grad():
        model.prompt_learner.ctx.copy_(contexts[best_index])
    return CandidateSelection(
        initial_logits=initial_logits,
        selected_indices=selected_indices,
        initial_entropy=initial_entropy,
        candidate_entropies=entropies,
        selected_lr=candidate_lrs[best_index],
        selected_lr_index=best_index,
        amp_scale=min(candidate_amp_scales, default=1.0),
        amp_retries=total_amp_retries,
    )


def write_records(
    path: Path,
    records: list[dict[str, Any]],
    scope_label: str | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path / "results.json", records)
    if not records:
        return
    csv_path = path / "results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    fieldnames = list(records[0])
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in record.items()
            })
    temporary.replace(csv_path)
    lines = [
        "# Data-wise LR TPT Task Results", "",
    ]
    if scope_label is not None:
        lines.extend([f"Scope: **{scope_label}**.", ""])
    lines.extend([
        "| Model | Dataset | Selection | Seed | Steps | Samples | Selected Top-1 | Zero-shot Top-1 | Runtime (min) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for record in records:
        selection = (
            f"count={record['selection_count']}" if record["selection_mode"] == "count"
            else f"fraction={record['selection_fraction']:.4g}"
        )
        lines.append(
            f"| {record['model']} | {record['dataset']} | {selection} | {record['seed']} | "
            f"{record['tta_steps']} | {record['samples']} | "
            f"{100 * record['top1_accuracy']:.4f}% | "
            f"{100 * record['zero_shot_top1_accuracy']:.4f}% | "
            f"{record['elapsed_minutes']:.2f} |"
        )
    lines.append("")
    atomic_write_text(path / "results.md", "\n".join(lines))


def write_sample_records(task_root: Path, records: list[dict[str, Any]]) -> None:
    task_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(task_root / "sample_lr_selection.json", records)
    csv_path = task_root / "sample_lr_selection.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    fieldnames = list(records[0]) if records else []
    if fieldnames:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow({
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                    for key, value in record.items()
                })
        temporary.replace(csv_path)


def evaluate_task(
    serialized_args: dict[str, Any],
    model_name: str,
    dataset_name: str,
    seed: int,
    tta_steps: int,
    selection_spec: dict[str, Any],
    device_name: str,
    progress_position: int,
    seed_root: str,
) -> dict[str, Any]:
    args = argparse.Namespace(**serialized_args)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    selection_count = int(selection_spec["selection_count"])
    task_name = task_slug(model_name, dataset_name, tta_steps, selection_spec)
    task_root = Path(seed_root) / task_name
    log_path = task_root / "task.log"
    logger = setup_worker_logging(log_path)
    logger.info(
        "starting model=%s dataset=%s seed=%d steps=%d selection=%s device=%s candidate_lrs=%s",
        model_name, dataset_name, seed, tta_steps, selection_spec, device,
        args.candidate_lrs,
    )

    import clip

    class_index = load_class_index(Path(args.class_index))
    all_names = load_class_names(Path(args.class_names))
    transform = TPTTransform(args.resolution, args.views, True)
    full_dataset = IdentifiedVariantDataset(
        Path(args.data_root) / dataset_name, dataset_name, class_index, transform
    )
    candidate_indices = candidate_class_indices(full_dataset, dataset_name)
    candidate_names = [all_names[index] for index in candidate_indices]
    dataset: Dataset = full_dataset
    if args.limit is not None:
        dataset = Subset(full_dataset, range(min(args.limit, len(full_dataset))))

    checkpoint = Path(args.model_root) / MODEL_FILES[model_name]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")
    model = TPTClassifier(clip, checkpoint, candidate_names, args.prompt_init, device).to(device)
    model.eval()
    model_resolution = int(model.image_encoder.input_resolution)
    if model_resolution != args.resolution:
        raise ValueError(
            f"{model_name} expects resolution {model_resolution}, got {args.resolution}"
        )

    generator = torch.Generator().manual_seed(seed)
    loader_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "generator": generator,
    }
    if args.num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(dataset, **loader_kwargs)
    global_indices = torch.tensor(candidate_indices, device=device)
    reported_k = max(args.top_k, 5)
    correct_topk = [0] * reported_k
    zero_shot_correct_topk = [0] * reported_k
    sample_records: list[dict[str, Any]] = []
    lr_counts = {str(rate): 0 for rate in args.candidate_lrs}
    one_step_scaler = make_scaler(device) if tta_steps == 1 else None
    total_amp_retries = 0
    started = time.perf_counter()
    task_root.mkdir(parents=True, exist_ok=True)
    incremental_path = task_root / "sample_lr_selection.jsonl"

    progress = tqdm(
        loader,
        desc=(
            f"{device} / data-wise LR / {selection_slug(selection_spec)} / "
            f"steps={tta_steps} / seed={seed} / {model_name} / {dataset_name}"
        ),
        unit="image",
        position=progress_position,
        leave=True,
        disable=not os.isatty(1),
    )
    with incremental_path.open("w", encoding="utf-8", buffering=1) as incremental_stream:
        for view_list, labels, sample_ids in progress:
            views = torch.cat(view_list, dim=0).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            sample_id = sample_ids[0]
            with torch.no_grad():
                model.reset()

            if tta_steps == 1:
                selection = select_one_step_lr(
                    model, views, args.candidate_lrs, selection_count,
                    args.weight_decay, device, one_step_scaler,
                )
            else:
                selection = select_multi_step_lr(
                    model, views, args.candidate_lrs, selection_count,
                    tta_steps, args.weight_decay, device,
                )
            total_amp_retries += selection.amp_retries
            if selection.amp_retries:
                logger.warning(
                    "AMP overflow recovered sample=%s retries=%d resulting_scale=%g",
                    sample_id, selection.amp_retries, selection.amp_scale,
                )

            with torch.no_grad(), autocast_context(device):
                final_logits = model(views[:1])
            selected_topk = accumulate_topk_correct(
                final_logits, labels, global_indices, correct_topk
            )
            zero_shot_topk = accumulate_topk_correct(
                selection.initial_logits[:1], labels, global_indices, zero_shot_correct_topk
            )
            selected_label = int(selected_topk[0, 0].item())
            zero_shot_label = int(zero_shot_topk[0, 0].item())
            true_label = int(labels.item())
            selected_lr_key = str(selection.selected_lr)
            lr_counts[selected_lr_key] += 1
            record = {
                "sample_index": len(sample_records),
                "sample_id": sample_id,
                "model": model_name,
                "dataset": dataset_name,
                "seed": seed,
                "tta_steps": tta_steps,
                "views": args.views,
                "selection_mode": selection_spec["selection_mode"],
                "selection_value": selection_spec["selection_value"],
                "selection_fraction": selection_spec["selection_fraction"],
                "selection_count": selection_count,
                "candidate_lrs": list(args.candidate_lrs),
                "candidate_entropies": selection.candidate_entropies,
                "initial_entropy": selection.initial_entropy,
                "selected_lr": selection.selected_lr,
                "selected_lr_index": selection.selected_lr_index,
                "amp_scale": selection.amp_scale,
                "amp_retries": selection.amp_retries,
                "selected_entropy": selection.candidate_entropies[selection.selected_lr_index],
                "entropy_reduction": (
                    selection.initial_entropy -
                    selection.candidate_entropies[selection.selected_lr_index]
                ),
                "true_label": true_label,
                "zero_shot_prediction": zero_shot_label,
                "selected_prediction": selected_label,
                "zero_shot_correct": zero_shot_label == true_label,
                "selected_correct": selected_label == true_label,
                "prediction_changed": selected_label != zero_shot_label,
            }
            sample_records.append(record)
            incremental_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            samples = len(sample_records)
            progress.set_postfix(
                top1=f"{100 * correct_topk[0] / samples:.2f}%",
                lr=f"{selection.selected_lr:g}",
            )
            if samples % args.print_frequency == 0 or samples == len(dataset):
                logger.info(
                    "progress %d/%d top1=%.4f%% selected_lr=%g",
                    samples, len(dataset), 100 * correct_topk[0] / samples,
                    selection.selected_lr,
                )

    elapsed = time.perf_counter() - started
    samples = len(sample_records)
    write_sample_records(task_root, sample_records)
    result = {
        "method": "Data-wise LR TPT",
        "idea": args.idea,
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "tta_steps": tta_steps,
        "samples": samples,
        "correct_top1": correct_topk[0],
        "correct_top5": correct_topk[4],
        "top1_accuracy": correct_topk[0] / samples,
        "top5_accuracy": correct_topk[4] / samples,
        "topk_correct": correct_topk[:args.top_k],
        "topk_accuracy": [count / samples for count in correct_topk[:args.top_k]],
        "zero_shot_correct_top1": zero_shot_correct_topk[0],
        "zero_shot_correct_top5": zero_shot_correct_topk[4],
        "zero_shot_top1_accuracy": zero_shot_correct_topk[0] / samples,
        "zero_shot_top5_accuracy": zero_shot_correct_topk[4] / samples,
        "zero_shot_topk_correct": zero_shot_correct_topk[:args.top_k],
        "zero_shot_topk_accuracy": [
            count / samples for count in zero_shot_correct_topk[:args.top_k]
        ],
        "candidate_classes": len(candidate_indices),
        "class_names_source": Path(args.class_names).name,
        "prompt_init": args.prompt_init,
        "views": args.views,
        "selection_mode": selection_spec["selection_mode"],
        "selection_value": selection_spec["selection_value"],
        "selection_fraction": selection_spec["selection_fraction"],
        "selection_count": selection_count,
        "candidate_lrs": list(args.candidate_lrs),
        "weight_decay": args.weight_decay,
        "lr_selection_counts": lr_counts,
        "amp_retries": total_amp_retries,
        "average_initial_entropy": statistics.fmean(
            record["initial_entropy"] for record in sample_records
        ),
        "average_selected_entropy": statistics.fmean(
            record["selected_entropy"] for record in sample_records
        ),
        "average_entropy_reduction": statistics.fmean(
            record["entropy_reduction"] for record in sample_records
        ),
        "device": str(device),
        "elapsed_minutes": elapsed / 60,
        "sample_results": str(task_root / "sample_lr_selection.json"),
    }
    logger.info(
        "completed model=%s dataset=%s seed=%d steps=%d selection=%s top1=%.4f%% in %.2f min",
        model_name, dataset_name, seed, tta_steps, selection_spec,
        100 * result["top1_accuracy"], result["elapsed_minutes"],
    )
    atomic_write_json(task_root / "task_result.json", result)
    if device.type == "cuda":
        del model
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return result


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def build_summary(
    results: list[dict[str, Any]],
    models: list[str],
    datasets: list[str],
    steps: list[int],
    selection_specs: list[dict[str, Any]],
    candidate_lrs: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    averages: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    result_seeds = list(dict.fromkeys(int(result["seed"]) for result in results))
    summary_scope = (
        f"Seed {result_seeds[0]}" if len(result_seeds) == 1
        else f"Mean over seeds {', '.join(map(str, result_seeds))}"
    )
    for model_name in models:
        for dataset_name in datasets:
            for selection_spec, step_count in product(selection_specs, steps):
                group = [
                    result for result in results
                    if result["model"] == model_name and
                    result["dataset"] == dataset_name and
                    result["selection_mode"] == selection_spec["selection_mode"] and
                    result["selection_value"] == selection_spec["selection_value"] and
                    result["tta_steps"] == step_count
                ]
                if not group:
                    raise ValueError(
                        f"Missing result group for {model_name}, {dataset_name}, "
                        f"selection={selection_spec}, steps={step_count}"
                    )
                top1 = [float(result["top1_accuracy"]) for result in group]
                top5 = [float(result["top5_accuracy"]) for result in group]
                zero_top1 = [float(result["zero_shot_top1_accuracy"]) for result in group]
                zero_top5 = [float(result["zero_shot_top5_accuracy"]) for result in group]
                record = {
                    "method": "Data-wise LR TPT",
                    "idea": group[0]["idea"],
                    "summary_scope": summary_scope,
                    "model": model_name,
                    "dataset": dataset_name,
                    "tta_steps": step_count,
                    "selection_mode": selection_spec["selection_mode"],
                    "selection_value": selection_spec["selection_value"],
                    "selection_fraction": selection_spec["selection_fraction"],
                    "selection_count": selection_spec["selection_count"],
                    "num_seeds": len(group),
                    "seeds": [int(result["seed"]) for result in group],
                    "samples_per_seed": [int(result["samples"]) for result in group],
                    "candidate_lrs": candidate_lrs,
                    "views": group[0]["views"],
                    "weight_decay": group[0]["weight_decay"],
                    "average_top1_accuracy": statistics.fmean(top1),
                    "std_top1_accuracy": sample_std(top1),
                    "average_top5_accuracy": statistics.fmean(top5),
                    "std_top5_accuracy": sample_std(top5),
                    "average_zero_shot_top1_accuracy": statistics.fmean(zero_top1),
                    "std_zero_shot_top1_accuracy": sample_std(zero_top1),
                    "average_zero_shot_top5_accuracy": statistics.fmean(zero_top5),
                    "std_zero_shot_top5_accuracy": sample_std(zero_top5),
                    "average_initial_entropy": statistics.fmean(
                        float(result["average_initial_entropy"]) for result in group
                    ),
                    "average_selected_entropy": statistics.fmean(
                        float(result["average_selected_entropy"]) for result in group
                    ),
                    "average_entropy_reduction": statistics.fmean(
                        float(result["average_entropy_reduction"]) for result in group
                    ),
                    "average_elapsed_minutes": statistics.fmean(
                        float(result["elapsed_minutes"]) for result in group
                    ),
                }
                averages.append(record)
                accuracy_rows.append({
                    "summary_scope": summary_scope,
                    "model": model_name,
                    "dataset": dataset_name,
                    "tta_steps": step_count,
                    "selection_mode": selection_spec["selection_mode"],
                    "selection_value": selection_spec["selection_value"],
                    "selection_fraction": selection_spec["selection_fraction"],
                    "selection_count": selection_spec["selection_count"],
                    "num_seeds": len(group),
                    "top1_accuracy_percent": 100 * record["average_top1_accuracy"],
                    "top1_std_percent": (
                        None if record["std_top1_accuracy"] is None
                        else 100 * record["std_top1_accuracy"]
                    ),
                    "top5_accuracy_percent": 100 * record["average_top5_accuracy"],
                    "top5_std_percent": (
                        None if record["std_top5_accuracy"] is None
                        else 100 * record["std_top5_accuracy"]
                    ),
                    "zero_shot_top1_accuracy_percent": 100 * record[
                        "average_zero_shot_top1_accuracy"
                    ],
                    "zero_shot_top1_std_percent": (
                        None if record["std_zero_shot_top1_accuracy"] is None
                        else 100 * record["std_zero_shot_top1_accuracy"]
                    ),
                    "zero_shot_top5_accuracy_percent": 100 * record[
                        "average_zero_shot_top5_accuracy"
                    ],
                    "zero_shot_top5_std_percent": (
                        None if record["std_zero_shot_top5_accuracy"] is None
                        else 100 * record["std_zero_shot_top5_accuracy"]
                    ),
                })
                for learning_rate in candidate_lrs:
                    per_seed_counts = [
                        int(result["lr_selection_counts"][str(learning_rate)])
                        for result in group
                    ]
                    per_seed_fractions = [
                        count / int(result["samples"])
                        for count, result in zip(per_seed_counts, group)
                    ]
                    distribution_rows.append({
                        "summary_scope": summary_scope,
                        "model": model_name,
                        "dataset": dataset_name,
                        "tta_steps": step_count,
                        "selection_mode": selection_spec["selection_mode"],
                        "selection_value": selection_spec["selection_value"],
                        "selection_fraction": selection_spec["selection_fraction"],
                        "selection_count": selection_spec["selection_count"],
                        "candidate_lr": learning_rate,
                        "num_seeds": len(group),
                        "total_samples": sum(int(result["samples"]) for result in group),
                        "total_selected": sum(per_seed_counts),
                        "lr_selection_fraction": statistics.fmean(per_seed_fractions),
                        "selection_percent": 100 * statistics.fmean(per_seed_fractions),
                        "lr_selection_fraction_std": sample_std(per_seed_fractions),
                        "selection_percent_std": (
                            None if len(per_seed_fractions) == 1
                            else 100 * statistics.stdev(per_seed_fractions)
                        ),
                        "per_seed_counts": per_seed_counts,
                        "per_seed_fractions": per_seed_fractions,
                    })
    return averages, accuracy_rows, distribution_rows


def write_table_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in record.items()
            })


def write_summary(
    summary_root: Path,
    results: list[dict[str, Any]],
    averages: list[dict[str, Any]],
    accuracy_rows: list[dict[str, Any]],
    distribution_rows: list[dict[str, Any]],
    seed: int | None = None,
) -> None:
    scope_label = f"Seed {seed}" if seed is not None else "All seeds and cross-seed mean"
    statistics_basename = "summary_results" if seed is not None else "average_results"
    write_records(summary_root, results, scope_label)
    atomic_write_json(summary_root / f"{statistics_basename}.json", averages)
    write_table_csv(summary_root / f"{statistics_basename}.csv", averages)
    write_table_csv(summary_root / "accuracy_plot_data.csv", accuracy_rows)
    write_table_csv(summary_root / "lr_distribution_plot_data.csv", distribution_rows)
    atomic_write_json(summary_root / "lr_distribution.json", distribution_rows)

    if seed is None:
        statistics_title = "# Data-wise LR TPT Cross-seed Average"
        statistics_description = (
            "Accuracy values are means across seeds. Uncertainty is sample standard "
            "deviation (`ddof=1`)."
        )
    else:
        statistics_title = f"# Data-wise LR TPT Seed {seed} Summary"
        statistics_description = (
            f"These statistics contain only seed {seed}. Cross-seed uncertainty is not "
            "applicable and is recorded as blank/null."
        )
    statistics_lines = [
        statistics_title, "", statistics_description, "",
        "| Model | Dataset | View selection | Steps | Seeds | Top-1 | Top-5 | Entropy reduction | Runtime (min) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in averages:
        top1 = f"{100 * row['average_top1_accuracy']:.4f}%"
        top5 = f"{100 * row['average_top5_accuracy']:.4f}%"
        if row["std_top1_accuracy"] is not None:
            top1 += f" +/- {100 * row['std_top1_accuracy']:.4f}%"
        if row["std_top5_accuracy"] is not None:
            top5 += f" +/- {100 * row['std_top5_accuracy']:.4f}%"
        selection = (
            f"count={row['selection_count']}" if row["selection_mode"] == "count"
            else f"fraction={row['selection_fraction']:.4g}"
        )
        statistics_lines.append(
            f"| {row['model']} | {row['dataset']} | {selection} | {row['tta_steps']} | "
            f"{row['num_seeds']} | {top1} | {top5} | "
            f"{row['average_entropy_reduction']:.6f} | "
            f"{row['average_elapsed_minutes']:.2f} |"
        )
    statistics_lines.append("")
    atomic_write_text(
        summary_root / f"{statistics_basename}.md", "\n".join(statistics_lines)
    )

    distribution_description = (
        "Percentages are first computed within each seed, then averaged across seeds. "
        "Uncertainty is sample standard deviation across seed-level percentages."
        if seed is None else
        f"Percentages contain only seed {seed}; cross-seed uncertainty is not applicable."
    )
    distribution_lines = [
        "# Per-sample Learning-rate Selection Distribution", "",
        f"Scope: **{scope_label}**.", "", distribution_description, "",
        "| Model | Dataset | View selection | Steps | LR | Selected | Mean proportion | Std |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in distribution_rows:
        std = "n/a" if row["selection_percent_std"] is None else f"{row['selection_percent_std']:.4f}%"
        selection = (
            f"count={row['selection_count']}" if row["selection_mode"] == "count"
            else f"fraction={row['selection_fraction']:.4g}"
        )
        distribution_lines.append(
            f"| {row['model']} | {row['dataset']} | {selection} | {row['tta_steps']} | "
            f"{row['candidate_lr']:g} | {row['total_selected']}/{row['total_samples']} | "
            f"{row['selection_percent']:.4f}% | {std} |"
        )
    distribution_lines.append("")
    atomic_write_text(
        summary_root / "lr_distribution.md", "\n".join(distribution_lines)
    )

    combined_lines = [
        f"# Data-wise LR TPT {'Seed ' + str(seed) if seed is not None else 'Overall'} Summary",
        "",
        f"Scope: **{scope_label}**.", "",
        "## Individual Task Results", "",
        *write_task_table_lines(results),
        "## Summary Statistics", "",
        *statistics_lines[4:],
        "## Learning-rate Selection", "",
        *distribution_lines[6:],
    ]
    atomic_write_text(summary_root / "summary.md", "\n".join(combined_lines))


def write_task_table_lines(records: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Dataset | Selection | Seed | Steps | Samples | Selected Top-1 | Zero-shot Top-1 | Runtime (min) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        selection = (
            f"count={record['selection_count']}" if record["selection_mode"] == "count"
            else f"fraction={record['selection_fraction']:.4g}"
        )
        lines.append(
            f"| {record['model']} | {record['dataset']} | {selection} | {record['seed']} | "
            f"{record['tta_steps']} | {record['samples']} | "
            f"{100 * record['top1_accuracy']:.4f}% | "
            f"{100 * record['zero_shot_top1_accuracy']:.4f}% | "
            f"{record['elapsed_minutes']:.2f} |"
        )
    lines.append("")
    return lines


def build_jobs(
    args: argparse.Namespace,
) -> list[tuple[int, str, str, dict[str, Any], int]]:
    return [
        (seed, model_name, dataset_name, selection_spec, steps)
        for model_name, dataset_name, selection_spec, steps, seed in product(
            args.models, args.datasets, args.selection_specs,
            args.tta_step_values, args.seeds,
        )
    ]


def assign_jobs(
    jobs: list[tuple[int, str, str, dict[str, Any], int]],
    devices: list[str],
) -> list[list[tuple[int, str, str, dict[str, Any], int]]]:
    assignments: list[list[tuple[int, str, str, dict[str, Any], int]]] = [
        [] for _ in devices
    ]
    for index, job in enumerate(jobs):
        assignments[index % len(devices)].append(job)
    return assignments


def run_jobs(
    args: argparse.Namespace,
    serialized_args: dict[str, Any],
    devices: list[str],
) -> list[dict[str, Any]]:
    jobs = build_jobs(args)
    assignments = assign_jobs(jobs, devices)
    for seed in args.seeds:
        (args.output_root / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
    spawn_context = multiprocessing.get_context("spawn")
    progress_lock = spawn_context.RLock()
    results: list[dict[str, Any]] = []
    results_by_seed: dict[int, list[dict[str, Any]]] = {
        seed: [] for seed in args.seeds
    }
    with ExitStack() as stack:
        executors = [
            stack.enter_context(ProcessPoolExecutor(
                max_workers=1,
                mp_context=spawn_context,
                initializer=initialize_progress_lock,
                initargs=(progress_lock,),
            ))
            for _ in devices
        ]
        futures = {}
        for position, (executor, device, device_jobs) in enumerate(
            zip(executors, devices, assignments)
        ):
            for seed, model_name, dataset_name, selection_spec, steps in device_jobs:
                seed_root = args.output_root / f"seed_{seed}"
                future = executor.submit(
                    evaluate_task, serialized_args, model_name, dataset_name, seed,
                    steps, selection_spec, device, position, str(seed_root),
                )
                futures[future] = (model_name, dataset_name, selection_spec, steps, device)
        for future in as_completed(futures):
            result = future.result()
            result_selection_spec = selection_spec_from_result(result)
            results.append(result)
            seed_results = results_by_seed[int(result["seed"])]
            seed_results.append(result)
            seed_results.sort(key=lambda row: (
                args.models.index(row["model"]),
                args.datasets.index(row["dataset"]),
                args.selection_specs.index(selection_spec_from_result(row)),
                args.tta_step_values.index(row["tta_steps"]),
            ))
            write_records(
                args.output_root / f"seed_{result['seed']}", seed_results,
                f"Seed {result['seed']}",
            )
            print(
                f"[SUCCESS] seed={result['seed']} {result['model']} {result['dataset']} "
                f"{selection_slug(result_selection_spec)} "
                f"steps={result['tta_steps']} top1={100 * result['top1_accuracy']:.4f}%",
                flush=True,
            )
    return results


def main(argv: list[str] | None = None) -> None:
    args, config_path = parse_args(argv)
    logger, _ = setup_main_logging(args.output_root, args.log_file)
    write_resolved_config(args, config_path)
    logger.info("source config: %s", config_path)
    logger.info("resolved arguments: %s", json_ready(vars(args)))
    job_count_per_seed = (
        len(args.models) * len(args.datasets) * len(args.selection_specs)
        * len(args.tta_step_values)
    )
    total_job_count = job_count_per_seed * len(args.seeds)
    devices = args.devices[:min(len(args.devices), total_job_count)]
    print(
        f"[INFO] data-wise LR TPT: {total_job_count} tasks "
        f"({len(args.seeds)} seeds, {job_count_per_seed} tasks/seed), "
        f"devices={', '.join(devices)}",
        flush=True,
    )

    serialized_args = json_ready(vars(args))
    assignments = assign_jobs(build_jobs(args), devices)
    for device, device_jobs in zip(devices, assignments):
        logger.info("device assignment %s: %s", device, device_jobs)
    all_results = run_jobs(args, serialized_args, devices)

    all_results.sort(key=lambda row: (
        args.models.index(row["model"]),
        args.datasets.index(row["dataset"]),
        args.selection_specs.index(selection_spec_from_result(row)),
        args.tta_step_values.index(row["tta_steps"]),
        args.seeds.index(row["seed"]),
    ))

    if args.plot["enabled"]:
        from plot_datawise_lr_tpt import generate_plots

    plot_paths: list[Path] = []
    seed_summary_paths: dict[str, str] = {}
    for seed in args.seeds:
        seed_results = [result for result in all_results if int(result["seed"]) == seed]
        seed_averages, seed_accuracy_rows, seed_distribution_rows = build_summary(
            seed_results, args.models, args.datasets, args.tta_step_values,
            args.selection_specs, args.candidate_lrs,
        )
        seed_summary_root = args.output_root / f"seed_{seed}" / "summary"
        write_summary(
            seed_summary_root, seed_results, seed_averages,
            seed_accuracy_rows, seed_distribution_rows, seed=seed,
        )
        seed_summary_paths[str(seed)] = str(seed_summary_root)
        if args.plot["enabled"]:
            seed_plot_paths = generate_plots(
                Path(args.plot["config"]),
                accuracy_data=seed_summary_root / "accuracy_plot_data.csv",
                distribution_data=seed_summary_root / "lr_distribution_plot_data.csv",
                output_dir=seed_summary_root / "plots",
            )
            plot_paths.extend(seed_plot_paths)
            logger.info(
                "generated seed %d plots: %s", seed,
                ", ".join(map(str, seed_plot_paths)),
            )

    averages, accuracy_rows, distribution_rows = build_summary(
        all_results, args.models, args.datasets, args.tta_step_values,
        args.selection_specs, args.candidate_lrs,
    )
    summary_root = args.output_root / "summary"
    write_summary(summary_root, all_results, averages, accuracy_rows, distribution_rows)

    if args.plot["enabled"]:
        overall_plot_paths = generate_plots(
            Path(args.plot["config"]),
            accuracy_data=summary_root / "accuracy_plot_data.csv",
            distribution_data=summary_root / "lr_distribution_plot_data.csv",
            output_dir=summary_root / "plots",
        )
        plot_paths.extend(overall_plot_paths)
        logger.info("generated overall plots: %s", ", ".join(map(str, overall_plot_paths)))
        print(f"[SUCCESS] generated {len(plot_paths)} plot artifacts", flush=True)

    atomic_write_json(args.output_root / "COMPLETED.json", {
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "expected_seeds": args.seeds,
        "completed_seeds": args.seeds,
        "tasks_per_seed": job_count_per_seed,
        "total_tasks": total_job_count,
        "seed_summaries": seed_summary_paths,
        "summary": str(summary_root),
    })
    logger.info("completed workflow; summary=%s", summary_root)
    print(f"[SUCCESS] results written to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
