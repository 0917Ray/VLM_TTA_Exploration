#!/usr/bin/env python3
"""Evaluate one-step TPT prediction and loss transitions."""

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
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import yaml

from evaluate_datawise_lr_tpt import (
    IdentifiedVariantDataset,
    atomic_write_json,
    atomic_write_text,
    autocast_context,
    json_ready,
    make_scaler,
    safe_component,
    select_low_entropy_count,
    slugify,
)
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_TRANSITIONS = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
)
LOSS_TRANSITIONS = (
    "entropy_down_ce_down",
    "entropy_down_ce_up_or_equal",
    "entropy_up_or_equal_ce_up_or_equal",
    "entropy_up_or_equal_ce_down",
)
ACCURACY_LABELS = {
    "correct_to_correct": "Correct -> Correct",
    "correct_to_wrong": "Correct -> Wrong",
    "wrong_to_correct": "Wrong -> Correct",
    "wrong_to_wrong": "Wrong -> Wrong",
}
LOSS_LABELS = {
    "entropy_down_ce_down": "Entropy down / CE down",
    "entropy_down_ce_up_or_equal": "Entropy down / CE up or equal",
    "entropy_up_or_equal_ce_up_or_equal": "Entropy up or equal / CE up or equal",
    "entropy_up_or_equal_ce_down": "Entropy up or equal / CE down",
}
CONFIG_KEYS = {
    "data_root", "class_index", "class_names", "model_root", "output_root",
    "idea", "run_name", "datasets", "models", "devices", "seeds",
    "learning_rates", "num_workers", "views", "selection_fraction",
    "weight_decay", "prompt_init", "resolution", "print_frequency", "limit",
    "loss_tolerance", "log_file", "plot",
}
DEFAULTS: dict[str, Any] = {
    "data_root": "data/raw",
    "class_index": "configs/visualization/imagenet_class_index.json",
    "class_names": "configs/evaluation/imagenet_classes_tpt.json",
    "model_root": "models",
    "output_root": "runs/tpt_transition_stats",
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
    "loss_tolerance": 1e-8,
    "log_file": "run.log",
    "plot": {
        "enabled": True,
        "config": "configs/visualization/tpt_transition_stats.yaml",
    },
}


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
        default=PROJECT_ROOT / "configs/evaluation/tpt_transition_stats.yaml",
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
    parser.add_argument("--learning-rates", type=float, nargs="+")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--views", type=int)
    parser.add_argument("--selection-fraction", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--prompt-init")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--print-frequency", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--loss-tolerance", type=float)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None)
    cli = parser.parse_args(argv)

    config_path = resolve_project_path(cli.config)
    values = {**DEFAULTS, **load_config(config_path)}
    for key in CONFIG_KEYS - {"log_file", "plot"}:
        value = getattr(cli, key, None)
        if value is not None:
            values[key] = value

    for key in ("data_root", "class_index", "class_names", "model_root", "output_root"):
        values[key] = resolve_project_path(values[key])
    values["idea"] = safe_component(values["idea"], "idea")
    values["run_name"] = safe_component(values["run_name"], "run_name")

    for field, allowed in (("datasets", DATASET_NAMES), ("models", MODEL_FILES)):
        entries = values[field]
        if not isinstance(entries, list) or not entries:
            raise SystemExit(f"{field} must be a non-empty list")
        if len(set(entries)) != len(entries):
            raise SystemExit(f"{field} must not contain duplicates")
        if any(entry not in allowed for entry in entries):
            raise SystemExit(f"{field} contains unsupported values; allowed: {', '.join(allowed)}")

    devices = values["devices"]
    if (not isinstance(devices, list) or not devices or
            any(not isinstance(device, str) or not device for device in devices)):
        raise SystemExit("devices must be a non-empty list of device names")
    if len(set(devices)) != len(devices):
        raise SystemExit("devices must not contain duplicates")
    for device in devices:
        validate_device(device)

    seeds = values["seeds"]
    if (not isinstance(seeds, list) or not seeds or
            any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)):
        raise SystemExit("seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("seeds must not contain duplicates")

    rates = values["learning_rates"]
    if (not isinstance(rates, list) or not rates or
            any(isinstance(rate, bool) or not isinstance(rate, (int, float)) for rate in rates)):
        raise SystemExit("learning_rates must be a non-empty list of numbers")
    rates = [float(rate) for rate in rates]
    if len(set(rates)) != len(rates) or any(not math.isfinite(rate) or rate < 0 for rate in rates):
        raise SystemExit("learning_rates must contain unique, finite, non-negative values")
    values["learning_rates"] = rates

    for field in ("views", "resolution", "print_frequency"):
        value = values[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SystemExit(f"{field} must be a positive integer")
    if values["views"] < 2:
        raise SystemExit("views must be at least 2")
    if not isinstance(values["num_workers"], int) or values["num_workers"] < 0:
        raise SystemExit("num_workers must be a non-negative integer")
    fraction = values["selection_fraction"]
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or
            not math.isfinite(float(fraction)) or not 0 < float(fraction) <= 1):
        raise SystemExit("selection_fraction must be in (0, 1]")
    values["selection_fraction"] = float(fraction)
    values["selection_count"] = max(1, int(values["views"] * values["selection_fraction"]))
    if values["selection_count"] > values["views"]:
        raise SystemExit("selection_fraction selects more views than available")

    weight_decay = values["weight_decay"]
    if (isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) or
            not math.isfinite(float(weight_decay)) or weight_decay < 0):
        raise SystemExit("weight_decay must be finite and non-negative")
    values["weight_decay"] = float(weight_decay)
    tolerance = values["loss_tolerance"]
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or
            not math.isfinite(float(tolerance)) or tolerance < 0):
        raise SystemExit("loss_tolerance must be finite and non-negative")
    values["loss_tolerance"] = float(tolerance)
    if not isinstance(values["prompt_init"], str) or not values["prompt_init"].strip():
        raise SystemExit("prompt_init must be a non-empty string")
    if values["limit"] is not None and (
            not isinstance(values["limit"], int) or isinstance(values["limit"], bool) or
            values["limit"] < 1):
        raise SystemExit("limit must be a positive integer or null")

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

    run_name = values["run_name"] or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_parent = values["output_root"]
    if values["idea"]:
        output_parent = output_parent / f"idea_{values['idea']}"
    values["output_parent"] = output_parent
    values["output_root"] = output_parent / f"run_{run_name}"
    return argparse.Namespace(**values), config_path


def setup_main_logging(output_root: Path, log_file: str) -> tuple[logging.Logger, Path]:
    output_root.mkdir(parents=True, exist_ok=False)
    configured = Path(log_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = output_root / configured.with_name(
        f"{configured.stem}_{timestamp}{configured.suffix or '.log'}"
    ).name
    logger = logging.getLogger("tpt_transition_stats")
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


def setup_worker_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"tpt_transition_stats.worker.{os.getpid()}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(processName)s: %(message)s"
    ))
    logger.addHandler(handler)
    return logger


def write_resolved_config(args: argparse.Namespace, source: Path) -> None:
    payload = json_ready(vars(args))
    payload["source_config"] = str(source)
    payload["generated_at"] = datetime.now().astimezone().isoformat()
    atomic_write_text(
        args.output_root / "resolved_config.yaml",
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )


def marginal_cross_entropy(logits: torch.Tensor, local_target: int) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] < 1:
        raise ValueError("logits must have shape [views, classes]")
    if not 0 <= local_target < logits.shape[1]:
        raise ValueError("local_target is outside the logits class dimension")
    log_probabilities = logits.log_softmax(dim=-1)
    average_log_probability = (
        log_probabilities.logsumexp(dim=0) - math.log(logits.shape[0])
    )
    return -average_log_probability[local_target]


def tune_one_step(
    model: TPTClassifier,
    views: torch.Tensor,
    learning_rate: float,
    selection_count: int,
    weight_decay: float,
    device: torch.device,
    scaler,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    """Apply one real AdamW entropy step and return the pre-update state."""
    theta0 = model.prompt_learner.ctx.detach().clone()
    optimizer = torch.optim.AdamW(
        [model.prompt_learner.ctx], lr=learning_rate, weight_decay=weight_decay
    )
    retries = 0
    while True:
        with torch.no_grad():
            model.prompt_learner.ctx.copy_(theta0)
        with autocast_context(device):
            initial_logits = model(views)
            selected_logits, selected_indices = select_low_entropy_count(
                initial_logits, selection_count
            )
            loss = average_entropy(selected_logits)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite entropy during one-step TPT")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        previous_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) >= previous_scale:
            return (
                initial_logits.detach(),
                selected_indices.detach(),
                float(loss.detach().item()),
                float(scaler.get_scale()),
                retries,
            )
        retries += 1
        if retries >= 16:
            raise RuntimeError(
                "One-step TPT kept overflowing after AMP scale recovery; "
                f"last scale={scaler.get_scale():g}, retries={retries}"
            )


def accuracy_transition(before_correct: bool, after_correct: bool) -> str:
    if before_correct:
        return "correct_to_correct" if after_correct else "correct_to_wrong"
    return "wrong_to_correct" if after_correct else "wrong_to_wrong"


def loss_transition(
    entropy_before: float,
    entropy_after: float,
    cross_entropy_before: float,
    cross_entropy_after: float,
    tolerance: float,
) -> str:
    entropy_down = entropy_after < entropy_before - tolerance
    cross_entropy_down = cross_entropy_after < cross_entropy_before - tolerance
    if entropy_down:
        return (
            "entropy_down_ce_down" if cross_entropy_down
            else "entropy_down_ce_up_or_equal"
        )
    return (
        "entropy_up_or_equal_ce_down" if cross_entropy_down
        else "entropy_up_or_equal_ce_up_or_equal"
    )


def rate_slug(rate: float) -> str:
    return f"{rate:.12g}".replace("-", "m").replace(".", "p")


def task_slug(model: str, dataset: str, learning_rate: float) -> str:
    return f"{slugify(model)}__{slugify(dataset)}__lr_{rate_slug(learning_rate)}"


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        atomic_write_text(path, "")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in record.items()
            })
    temporary.replace(path)


def write_task_records(path: Path, records: list[dict[str, Any]], title: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path / "results.json", records)
    write_csv(path / "results.csv", records)
    lines = [f"# {title}", ""]
    lines.extend([
        "| Model | Dataset | LR | Seed | Samples | Zero-shot Top-1 | TPT Top-1 | Runtime (min) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for record in records:
        lines.append(
            f"| {record['model']} | {record['dataset']} | {record['learning_rate']:g} | "
            f"{record['seed']} | {record['samples']} | "
            f"{100 * record['zero_shot_top1_accuracy']:.4f}% | "
            f"{100 * record['tpt_top1_accuracy']:.4f}% | "
            f"{record['elapsed_minutes']:.2f} |"
        )
    lines.append("")
    atomic_write_text(path / "results.md", "\n".join(lines))


def write_sample_records(task_root: Path, records: list[dict[str, Any]]) -> None:
    atomic_write_json(task_root / "sample_transitions.json", records)
    write_csv(task_root / "sample_transitions.csv", records)


def evaluate_task(
    serialized_args: dict[str, Any],
    model_name: str,
    dataset_name: str,
    learning_rate: float,
    seed: int,
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
    task_root = Path(seed_root) / task_slug(model_name, dataset_name, learning_rate)
    logger = setup_worker_logging(task_root / "task.log")
    logger.info(
        "starting model=%s dataset=%s lr=%g seed=%d device=%s",
        model_name, dataset_name, learning_rate, seed, device,
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
    global_to_local = {global_index: local_index for local_index, global_index in enumerate(candidate_indices)}
    dataset: Dataset = full_dataset
    if args.limit is not None:
        dataset = Subset(full_dataset, range(min(args.limit, len(full_dataset))))

    checkpoint = Path(args.model_root) / MODEL_FILES[model_name]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")
    model = TPTClassifier(clip, checkpoint, candidate_names, args.prompt_init, device).to(device)
    model.eval()
    if int(model.image_encoder.input_resolution) != args.resolution:
        raise ValueError(
            f"{model_name} expects resolution {model.image_encoder.input_resolution}, "
            f"got {args.resolution}"
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

    accuracy_counts = {key: 0 for key in ACCURACY_TRANSITIONS}
    loss_counts = {key: 0 for key in LOSS_TRANSITIONS}
    sample_records: list[dict[str, Any]] = []
    scaler = make_scaler(device)
    total_amp_retries = 0
    entropy_equal_count = 0
    cross_entropy_equal_count = 0
    started = time.perf_counter()
    task_root.mkdir(parents=True, exist_ok=True)
    incremental_path = task_root / "sample_transitions.jsonl"

    progress = tqdm(
        loader,
        desc=f"{device} / transition stats / lr={learning_rate:g} / seed={seed} / {dataset_name}",
        unit="image",
        position=progress_position,
        leave=True,
        disable=not os.isatty(1),
    )
    with incremental_path.open("w", encoding="utf-8", buffering=1) as incremental_stream:
        for view_list, labels, sample_ids in progress:
            views = torch.cat(view_list, dim=0).to(device, non_blocking=True)
            true_label = int(labels.item())
            if true_label not in global_to_local:
                raise RuntimeError(
                    f"Ground-truth label {true_label} is outside the candidate class set"
                )
            local_target = global_to_local[true_label]
            with torch.no_grad():
                model.reset()

            (
                initial_logits,
                selected_indices,
                initial_entropy,
                amp_scale,
                amp_retries,
            ) = tune_one_step(
                model, views, learning_rate, args.selection_count,
                args.weight_decay, device, scaler,
            )
            total_amp_retries += amp_retries
            pre_logits = initial_logits.float()
            pre_selected_logits = pre_logits[selected_indices]
            with torch.no_grad(), autocast_context(device):
                post_canonical_logits = model(views[:1]).float()
                post_selected_logits = model(views[selected_indices]).float()

            entropy_before = float(average_entropy(pre_selected_logits).item())
            entropy_after = float(average_entropy(post_selected_logits).item())
            cross_entropy_before = float(
                marginal_cross_entropy(pre_selected_logits, local_target).item()
            )
            cross_entropy_after = float(
                marginal_cross_entropy(post_selected_logits, local_target).item()
            )
            if not all(math.isfinite(value) for value in (
                entropy_before, entropy_after, cross_entropy_before, cross_entropy_after
            )):
                raise RuntimeError(f"Non-finite loss statistics for sample {sample_ids[0]}")

            zero_local = int(pre_logits[:1].argmax(dim=1).item())
            tpt_local = int(post_canonical_logits.argmax(dim=1).item())
            zero_prediction = candidate_indices[zero_local]
            tpt_prediction = candidate_indices[tpt_local]
            zero_correct = zero_prediction == true_label
            tpt_correct = tpt_prediction == true_label
            accuracy_key = accuracy_transition(zero_correct, tpt_correct)
            loss_key = loss_transition(
                entropy_before, entropy_after,
                cross_entropy_before, cross_entropy_after,
                args.loss_tolerance,
            )
            accuracy_counts[accuracy_key] += 1
            loss_counts[loss_key] += 1

            entropy_delta = entropy_after - entropy_before
            cross_entropy_delta = cross_entropy_after - cross_entropy_before
            entropy_equal = abs(entropy_delta) <= args.loss_tolerance
            cross_entropy_equal = abs(cross_entropy_delta) <= args.loss_tolerance
            entropy_equal_count += int(entropy_equal)
            cross_entropy_equal_count += int(cross_entropy_equal)
            record = {
                "sample_index": len(sample_records),
                "sample_id": sample_ids[0],
                "model": model_name,
                "dataset": dataset_name,
                "seed": seed,
                "learning_rate": learning_rate,
                "views": args.views,
                "selection_fraction": args.selection_fraction,
                "selection_count": args.selection_count,
                "selected_view_indices": selected_indices.detach().cpu().tolist(),
                "true_label": true_label,
                "local_target": local_target,
                "zero_shot_prediction": zero_prediction,
                "tpt_prediction": tpt_prediction,
                "zero_shot_correct": zero_correct,
                "tpt_correct": tpt_correct,
                "accuracy_transition": accuracy_key,
                "entropy_before": entropy_before,
                "entropy_after": entropy_after,
                "entropy_delta": entropy_delta,
                "entropy_decreased": entropy_after < entropy_before - args.loss_tolerance,
                "entropy_equal_within_tolerance": entropy_equal,
                "cross_entropy_before": cross_entropy_before,
                "cross_entropy_after": cross_entropy_after,
                "cross_entropy_delta": cross_entropy_delta,
                "cross_entropy_decreased": (
                    cross_entropy_after < cross_entropy_before - args.loss_tolerance
                ),
                "cross_entropy_equal_within_tolerance": cross_entropy_equal,
                "loss_transition": loss_key,
                "optimizer_pre_entropy": initial_entropy,
                "amp_scale": amp_scale,
                "amp_retries": amp_retries,
            }
            sample_records.append(record)
            incremental_stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            samples = len(sample_records)
            tpt_correct_count = (
                accuracy_counts["correct_to_correct"]
                + accuracy_counts["wrong_to_correct"]
            )
            entropy_down_count = (
                loss_counts["entropy_down_ce_down"]
                + loss_counts["entropy_down_ce_up_or_equal"]
            )
            progress.set_postfix(
                tpt_acc=f"{100 * tpt_correct_count / samples:.2f}%",
                ent_down=f"{100 * entropy_down_count / samples:.1f}%",
            )
            if samples % args.print_frequency == 0 or samples == len(dataset):
                logger.info(
                    "progress %d/%d tpt_top1=%.4f%% entropy_down=%.4f%%",
                    samples, len(dataset),
                    100 * tpt_correct_count / samples,
                    100 * entropy_down_count / samples,
                )

    samples = len(sample_records)
    if sum(accuracy_counts.values()) != samples or sum(loss_counts.values()) != samples:
        raise RuntimeError("Transition counts do not cover every processed sample")
    write_sample_records(task_root, sample_records)
    elapsed_minutes = (time.perf_counter() - started) / 60
    result = {
        "method": "One-step TPT Transition Statistics",
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "learning_rate": learning_rate,
        "samples": samples,
        "zero_shot_correct": sum(record["zero_shot_correct"] for record in sample_records),
        "tpt_correct": sum(record["tpt_correct"] for record in sample_records),
        "zero_shot_top1_accuracy": (
            sum(record["zero_shot_correct"] for record in sample_records) / samples
        ),
        "tpt_top1_accuracy": sum(record["tpt_correct"] for record in sample_records) / samples,
        "accuracy_transition_counts": accuracy_counts,
        "accuracy_transition_proportions": {
            key: accuracy_counts[key] / samples for key in ACCURACY_TRANSITIONS
        },
        "loss_transition_counts": loss_counts,
        "loss_transition_proportions": {
            key: loss_counts[key] / samples for key in LOSS_TRANSITIONS
        },
        "entropy_equal_count": entropy_equal_count,
        "cross_entropy_equal_count": cross_entropy_equal_count,
        "mean_entropy_before": statistics.fmean(
            record["entropy_before"] for record in sample_records
        ),
        "mean_entropy_after": statistics.fmean(
            record["entropy_after"] for record in sample_records
        ),
        "mean_entropy_delta": statistics.fmean(
            record["entropy_delta"] for record in sample_records
        ),
        "mean_cross_entropy_before": statistics.fmean(
            record["cross_entropy_before"] for record in sample_records
        ),
        "mean_cross_entropy_after": statistics.fmean(
            record["cross_entropy_after"] for record in sample_records
        ),
        "mean_cross_entropy_delta": statistics.fmean(
            record["cross_entropy_delta"] for record in sample_records
        ),
        "candidate_classes": len(candidate_indices),
        "class_names_source": Path(args.class_names).name,
        "prompt_init": args.prompt_init,
        "views": args.views,
        "selection_fraction": args.selection_fraction,
        "selection_count": args.selection_count,
        "weight_decay": args.weight_decay,
        "loss_tolerance": args.loss_tolerance,
        "amp_retries": total_amp_retries,
        "device": str(device),
        "elapsed_minutes": elapsed_minutes,
        "sample_results": str(task_root / "sample_transitions.json"),
    }
    atomic_write_json(task_root / "task_result.json", result)
    logger.info(
        "completed model=%s dataset=%s lr=%g seed=%d samples=%d tpt_top1=%.4f%% in %.2f min",
        model_name, dataset_name, learning_rate, seed, samples,
        100 * result["tpt_top1_accuracy"], elapsed_minutes,
    )
    if device.type == "cuda":
        del model
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return result


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def build_summary(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    group_keys: list[tuple[str, str, float]] = []
    for result in results:
        key = (result["model"], result["dataset"], float(result["learning_rate"]))
        if key not in group_keys:
            group_keys.append(key)

    averages: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    for model_name, dataset_name, learning_rate in group_keys:
        group = [
            result for result in results
            if result["model"] == model_name and result["dataset"] == dataset_name
            and float(result["learning_rate"]) == learning_rate
        ]
        seeds = [int(result["seed"]) for result in group]
        average: dict[str, Any] = {
            "model": model_name,
            "dataset": dataset_name,
            "learning_rate": learning_rate,
            "num_seeds": len(group),
            "seeds": seeds,
            "samples_per_seed": [int(result["samples"]) for result in group],
            "average_zero_shot_top1_accuracy": statistics.fmean(
                result["zero_shot_top1_accuracy"] for result in group
            ),
            "std_zero_shot_top1_accuracy": sample_std([
                result["zero_shot_top1_accuracy"] for result in group
            ]),
            "average_tpt_top1_accuracy": statistics.fmean(
                result["tpt_top1_accuracy"] for result in group
            ),
            "std_tpt_top1_accuracy": sample_std([
                result["tpt_top1_accuracy"] for result in group
            ]),
            "average_entropy_delta": statistics.fmean(
                result["mean_entropy_delta"] for result in group
            ),
            "average_cross_entropy_delta": statistics.fmean(
                result["mean_cross_entropy_delta"] for result in group
            ),
        }
        for prefix, transitions, result_key, target_rows, labels in (
            ("accuracy", ACCURACY_TRANSITIONS, "accuracy_transition_proportions", accuracy_rows, ACCURACY_LABELS),
            ("loss", LOSS_TRANSITIONS, "loss_transition_proportions", loss_rows, LOSS_LABELS),
        ):
            for order, transition in enumerate(transitions):
                proportions = [result[result_key][transition] for result in group]
                mean = statistics.fmean(proportions)
                std = sample_std(proportions)
                average[f"{prefix}_{transition}_proportion"] = mean
                average[f"{prefix}_{transition}_std"] = std
                counts_key = f"{prefix}_transition_counts"
                total_count = sum(result[counts_key][transition] for result in group)
                total_samples = sum(int(result["samples"]) for result in group)
                target_rows.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "learning_rate": learning_rate,
                    "transition": transition,
                    "label": labels[transition],
                    "order": order,
                    "total_count": total_count,
                    "total_samples": total_samples,
                    "proportion_percent": 100 * mean,
                    "proportion_percent_std": None if std is None else 100 * std,
                    "num_seeds": len(group),
                    "seeds": seeds,
                })
        averages.append(average)
    return averages, accuracy_rows, loss_rows


def write_summary(
    summary_root: Path,
    results: list[dict[str, Any]],
    averages: list[dict[str, Any]],
    accuracy_rows: list[dict[str, Any]],
    loss_rows: list[dict[str, Any]],
    seed: int | None = None,
) -> None:
    write_task_records(
        summary_root, results,
        f"TPT Transition Statistics {'Seed ' + str(seed) if seed is not None else 'All Tasks'}",
    )
    atomic_write_json(summary_root / "average_results.json", averages)
    write_csv(summary_root / "average_results.csv", averages)
    write_csv(summary_root / "accuracy_transition_plot_data.csv", accuracy_rows)
    write_csv(summary_root / "loss_transition_plot_data.csv", loss_rows)

    scope = f"Seed {seed}" if seed is not None else "All seeds; mean and sample standard deviation"
    lines = [
        "# TPT Transition Statistics Summary", "", f"Scope: {scope}.", "",
        "## Accuracy Transitions", "",
        "| Model | Dataset | LR | Transition | Count | Proportion | Std across seeds |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in accuracy_rows:
        std = "n/a" if row["proportion_percent_std"] is None else f"{row['proportion_percent_std']:.4f}%"
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['learning_rate']:g} | "
            f"{row['label']} | {row['total_count']}/{row['total_samples']} | "
            f"{row['proportion_percent']:.4f}% | {std} |"
        )
    lines.extend([
        "", "## Loss Transitions", "",
        "The `up or equal` buckets include changes within the configured tolerance.", "",
        "| Model | Dataset | LR | Transition | Count | Proportion | Std across seeds |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ])
    for row in loss_rows:
        std = "n/a" if row["proportion_percent_std"] is None else f"{row['proportion_percent_std']:.4f}%"
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['learning_rate']:g} | "
            f"{row['label']} | {row['total_count']}/{row['total_samples']} | "
            f"{row['proportion_percent']:.4f}% | {std} |"
        )
    lines.append("")
    atomic_write_text(summary_root / "summary.md", "\n".join(lines))


def build_jobs(args: argparse.Namespace) -> list[tuple[str, str, float, int]]:
    return list(product(args.models, args.datasets, args.learning_rates, args.seeds))


def assign_jobs(
    jobs: list[tuple[str, str, float, int]], devices: list[str]
) -> list[list[tuple[str, str, float, int]]]:
    assignments: list[list[tuple[str, str, float, int]]] = [[] for _ in devices]
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
    results_by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in args.seeds}
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
            for model_name, dataset_name, learning_rate, seed in device_jobs:
                future = executor.submit(
                    evaluate_task,
                    serialized_args,
                    model_name,
                    dataset_name,
                    learning_rate,
                    seed,
                    device,
                    position,
                    str(args.output_root / f"seed_{seed}"),
                )
                futures[future] = (model_name, dataset_name, learning_rate, seed, device)
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            seed_results = results_by_seed[int(result["seed"])]
            seed_results.append(result)
            seed_results.sort(key=lambda row: (
                args.models.index(row["model"]),
                args.datasets.index(row["dataset"]),
                args.learning_rates.index(float(row["learning_rate"])),
            ))
            write_task_records(
                args.output_root / f"seed_{result['seed']}",
                seed_results,
                f"TPT Transition Statistics Seed {result['seed']}",
            )
            print(
                f"[SUCCESS] seed={result['seed']} {result['model']} {result['dataset']} "
                f"lr={result['learning_rate']:g} tpt_top1={100 * result['tpt_top1_accuracy']:.4f}%",
                flush=True,
            )
    return results


def main(argv: list[str] | None = None) -> None:
    args, config_path = parse_args(argv)
    logger, _ = setup_main_logging(args.output_root, args.log_file)
    write_resolved_config(args, config_path)
    logger.info("source config: %s", config_path)
    logger.info("resolved arguments: %s", json_ready(vars(args)))

    jobs = build_jobs(args)
    devices = args.devices[:min(len(args.devices), len(jobs))]
    assignments = assign_jobs(jobs, devices)
    for device, device_jobs in zip(devices, assignments):
        logger.info("device assignment %s: %s", device, device_jobs)
    print(
        f"[INFO] TPT transition statistics: {len(jobs)} tasks, "
        f"devices={', '.join(devices)}",
        flush=True,
    )

    all_results = run_jobs(args, json_ready(vars(args)), devices)
    all_results.sort(key=lambda row: (
        args.models.index(row["model"]),
        args.datasets.index(row["dataset"]),
        args.learning_rates.index(float(row["learning_rate"])),
        args.seeds.index(int(row["seed"])),
    ))

    if args.plot["enabled"]:
        from plot_tpt_transition_stats import generate_plots

    plot_paths: list[Path] = []
    seed_summaries: dict[str, str] = {}
    for seed in args.seeds:
        seed_results = [result for result in all_results if int(result["seed"]) == seed]
        seed_averages, seed_accuracy_rows, seed_loss_rows = build_summary(seed_results)
        seed_summary = args.output_root / f"seed_{seed}" / "summary"
        write_summary(
            seed_summary, seed_results, seed_averages,
            seed_accuracy_rows, seed_loss_rows, seed=seed,
        )
        seed_summaries[str(seed)] = str(seed_summary)
        if args.plot["enabled"]:
            plot_paths.extend(generate_plots(
                Path(args.plot["config"]),
                accuracy_data=seed_summary / "accuracy_transition_plot_data.csv",
                loss_data=seed_summary / "loss_transition_plot_data.csv",
                output_dir=seed_summary / "plots",
            ))

    averages, accuracy_rows, loss_rows = build_summary(all_results)
    summary_root = args.output_root / "summary"
    write_summary(summary_root, all_results, averages, accuracy_rows, loss_rows)
    if args.plot["enabled"]:
        plot_paths.extend(generate_plots(
            Path(args.plot["config"]),
            accuracy_data=summary_root / "accuracy_transition_plot_data.csv",
            loss_data=summary_root / "loss_transition_plot_data.csv",
            output_dir=summary_root / "plots",
        ))
        print(f"[SUCCESS] generated {len(plot_paths)} plot artifacts", flush=True)

    atomic_write_json(args.output_root / "COMPLETED.json", {
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "expected_seeds": args.seeds,
        "completed_seeds": args.seeds,
        "total_tasks": len(jobs),
        "seed_summaries": seed_summaries,
        "summary": str(summary_root),
    })
    logger.info("completed workflow; summary=%s", summary_root)
    print(f"[SUCCESS] results written to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
