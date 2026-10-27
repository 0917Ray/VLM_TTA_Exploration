"""Evaluate OpenAI CLIP zero-shot top-1 accuracy on ImageNet variants."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from threading import Lock
from typing import Iterable

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("imagenetv2", "imagenet-a", "imagenet-r", "imagenet-sketch")
MODEL_PATHS = {
    "RN50": PROJECT_ROOT / "models/RN50.pt",
    "ViT-B/16": PROJECT_ROOT / "models/ViT-B-16.pt",
}
MODEL_FILES = {name: path.name for name, path in MODEL_PATHS.items()}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jpeg"}
CONFIG_KEYS = {
    "data_root", "class_index", "class_names", "model_root", "output_root", "datasets", "models",
    "batch_size", "num_workers", "devices", "device", "prompt_template", "seeds", "seed",
    "limit", "log_file",
}
DEFAULTS = {
    "data_root": "data/raw",
    "class_index": "configs/visualization/imagenet_class_index.json",
    "class_names": "configs/evaluation/imagenet_classes_tpt.json",
    "model_root": "models",
    "output_root": "runs/clip_zero_shot",
    "datasets": list(DATASET_NAMES),
    "models": list(MODEL_PATHS),
    "batch_size": 256,
    "num_workers": min(8, os.cpu_count() or 1),
    "devices": ["cuda:0"] if torch.cuda.is_available() else ["cpu"],
    "prompt_template": "a photo of a {}.",
    "seeds": [42],
    "limit": None,
    "log_file": "run.log",
}


@dataclass
class Result:
    model: str
    dataset: str
    samples: int
    correct: int
    top1_accuracy: float
    candidate_classes: int
    class_names_source: str
    prompt_template: str
    seed: int
    device: str


class VariantDataset(Dataset):
    def __init__(self, root: Path, dataset_name: str, class_index: dict[str, list[str]], transform):
        self.root = root
        self.dataset_name = dataset_name
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        by_wnid = {entry[0]: int(index) for index, entry in class_index.items()}

        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        for category in sorted(path for path in root.iterdir() if path.is_dir()):
            if dataset_name == "imagenetv2":
                try:
                    label = int(category.name)
                except ValueError as error:
                    raise ValueError(f"ImageNet-V2 category is not numeric: {category}") from error
                if label not in range(len(class_index)):
                    raise ValueError(f"ImageNet-V2 category is outside [0, 999]: {category.name}")
            else:
                if category.name not in by_wnid:
                    raise ValueError(f"Unknown ImageNet WNID in {root}: {category.name}")
                label = by_wnid[category.name]
            for path in sorted(category.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No supported images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                return self.transform(image), label
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Cannot read image {path}: {error}") from error


def load_config(path: Path) -> dict:
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


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> tuple[argparse.Namespace, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs/evaluation/clip_zero_shot.yaml")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--class-index", type=Path)
    parser.add_argument("--class-names", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--devices", nargs="+")
    device_group.add_argument("--device", help=argparse.SUPPRESS)
    parser.add_argument("--prompt-template")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", type=int, nargs="+")
    seed_group.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int,
                        help="Optionally evaluate only the first N samples of each dataset.")
    cli = parser.parse_args()
    config_path = resolve_path(cli.config)
    config = load_config(config_path)
    if "device" in config:
        if "devices" in config:
            raise SystemExit("Config cannot contain both device and devices")
        config["devices"] = [config.pop("device")]
    if "seed" in config:
        if "seeds" in config:
            raise SystemExit("Config cannot contain both seed and seeds")
        config["seeds"] = [config.pop("seed")]
    values = dict(DEFAULTS)
    values.update(config)
    for key in CONFIG_KEYS - {"log_file", "seed", "device"}:
        value = getattr(cli, key, None)
        if value is not None:
            values[key] = value
    if cli.device is not None:
        values["devices"] = [cli.device]
    if cli.seed is not None:
        values["seeds"] = [cli.seed]
    for key in ("data_root", "class_index", "class_names", "model_root", "output_root"):
        values[key] = resolve_path(values[key])
    if not values["datasets"] or any(name not in DATASET_NAMES for name in values["datasets"]):
        raise SystemExit(f"datasets must contain only: {', '.join(DATASET_NAMES)}")
    if not values["models"] or any(name not in MODEL_PATHS for name in values["models"]):
        raise SystemExit(f"models must contain only: {', '.join(MODEL_PATHS)}")
    if values["batch_size"] < 1 or values["num_workers"] < 0:
        raise SystemExit("batch_size must be positive and num_workers must be non-negative")
    if (not isinstance(values["devices"], list) or not values["devices"] or
            any(not isinstance(device, str) or not device for device in values["devices"])):
        raise SystemExit("devices must be a non-empty list of device names")
    if len(set(values["devices"])) != len(values["devices"]):
        raise SystemExit("devices must not contain duplicate values")
    for device_name in values["devices"]:
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
    if (not isinstance(values["seeds"], list) or not values["seeds"] or
            any(not isinstance(seed, int) or isinstance(seed, bool) for seed in values["seeds"])):
        raise SystemExit("seeds must be a non-empty list of integers")
    if len(set(values["seeds"])) != len(values["seeds"]):
        raise SystemExit("seeds must not contain duplicate values")
    if values["limit"] is not None and values["limit"] < 1:
        raise SystemExit("limit must be positive")
    return argparse.Namespace(**values), config_path


def setup_logging(output_root: Path, log_file: str) -> logging.Logger:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file)
    if not log_path.is_absolute():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = output_root / log_path.with_name(
            f"{log_path.stem}_{timestamp}{log_path.suffix or '.log'}"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("clip_zero_shot")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.propagate = False
    warnings_logger.addHandler(handler)
    logging.captureWarnings(True)
    worker_filter = "ignore:pkg_resources is deprecated as an API:UserWarning"
    existing_filters = os.environ.get("PYTHONWARNINGS")
    os.environ["PYTHONWARNINGS"] = (
        f"{existing_filters},{worker_filter}" if existing_filters else worker_filter
    )
    logger.info("run log: %s", log_path)
    return logger


def load_class_index(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8") as stream:
        class_index = json.load(stream)
    if not isinstance(class_index, dict) or len(class_index) != 1000:
        raise ValueError(f"Expected a 1000-class ImageNet mapping: {path}")
    return {str(index): value for index, value in class_index.items()}


def load_class_names(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    names = payload.get("classes") if isinstance(payload, dict) else payload
    if (not isinstance(names, list) or len(names) != 1000 or
            any(not isinstance(name, str) or not name.strip() for name in names)):
        raise ValueError(f"Expected 1000 non-empty ImageNet class names: {path}")
    return names


def candidate_class_indices(dataset: VariantDataset, dataset_name: str) -> list[int]:
    if dataset_name not in {"imagenet-a", "imagenet-r"}:
        return list(range(1000))
    indices = sorted({label for _, label in dataset.samples})
    if len(indices) != 200:
        raise ValueError(
            f"{dataset_name} must contain exactly 200 classes for the standard protocol; "
            f"found {len(indices)}"
        )
    return indices


def encode_text(clip_module, model, names: Iterable[str], prompt_template: str, device: torch.device):
    prompts = [prompt_template.format(name) for name in names]
    tokens = clip_module.tokenize(prompts).to(device)
    with torch.inference_mode():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features


def evaluate_dataset(model, text_features, dataset: Dataset, batch_size: int, num_workers: int,
                     device: torch.device, model_name: str, dataset_name: str, seed: int,
                     limit: int | None, progress_position: int,
                     candidate_indices: list[int]):
    if limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(limit, len(dataset))))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(dataset, **loader_kwargs)
    correct = 0
    samples = 0
    global_class_indices = torch.tensor(candidate_indices, device=device)
    progress = tqdm(
        loader, desc=f"{device} / seed={seed} / {model_name} / {dataset_name}",
        unit="batch", position=progress_position, leave=True,
    )
    with torch.inference_mode():
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            local_predictions = (100.0 * image_features @ text_features.T).argmax(dim=-1)
            predictions = global_class_indices[local_predictions]
            correct += int((predictions == labels).sum().item())
            samples += labels.numel()
            progress.set_postfix(accuracy=f"{100 * correct / samples:.2f}%")
    return samples, correct


def write_results(results: list[Result], output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "results.json").open("w", encoding="utf-8") as stream:
        json.dump([asdict(result) for result in results], stream, indent=2, ensure_ascii=False)
    with (output_root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def average_results(results: list[Result], models: list[str], datasets: list[str]) -> list[dict]:
    averages = []
    for model_name in models:
        for dataset_name in datasets:
            group = [result for result in results
                     if result.model == model_name and result.dataset == dataset_name]
            if not group:
                continue
            accuracies = [result.top1_accuracy for result in group]
            averages.append({
                "model": model_name,
                "dataset": dataset_name,
                "num_seeds": len(group),
                "seeds": [result.seed for result in group],
                "samples_per_seed": group[0].samples,
                "device": group[0].device,
                "average_top1_accuracy": statistics.fmean(accuracies),
                "std_top1_accuracy": statistics.pstdev(accuracies),
                "min_top1_accuracy": min(accuracies),
                "max_top1_accuracy": max(accuracies),
                "prompt_template": group[0].prompt_template,
            })
    return averages


def write_average_results(averages: list[dict], output_root: Path,
                          models: list[str], datasets: list[str]):
    with (output_root / "average_results.json").open("w", encoding="utf-8") as stream:
        json.dump(averages, stream, indent=2, ensure_ascii=False)
    csv_rows = [{**row, "seeds": " ".join(map(str, row["seeds"]))} for row in averages]
    with (output_root / "average_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    by_pair = {(row["model"], row["dataset"]): row for row in averages}
    lines = [
        "# Average Top-1 Accuracy",
        "",
        "Values are mean +/- population standard deviation across seeds.",
        "",
        "| Model | " + " | ".join(datasets) + " |",
        "| --- | " + " | ".join("---:" for _ in datasets) + " |",
    ]
    for model_name in models:
        cells = []
        for dataset_name in datasets:
            row = by_pair[(model_name, dataset_name)]
            mean = 100 * row["average_top1_accuracy"]
            std = 100 * row["std_top1_accuracy"]
            cells.append(f"{mean:.4f}% +/- {std:.4f}%")
        lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
    lines.append("")
    (output_root / "average_results.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_job(clip_module, args, class_index, names, model_name: str, dataset_name: str,
                 device_name: str, progress_position: int, logger: logging.Logger) -> list[Result]:
    device = torch.device(device_name)
    model_path = args.model_root / MODEL_FILES[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")

    logger.info("starting %s / %s on %s", model_name, dataset_name, device)
    print(f"[INFO] {device}: loading {model_name} for {dataset_name}", flush=True)
    model, preprocess = clip_module.load(str(model_path), device=device, jit=False)
    model.eval()
    dataset = VariantDataset(args.data_root / dataset_name, dataset_name, class_index, preprocess)
    candidate_indices = candidate_class_indices(dataset, dataset_name)
    candidate_names = [names[index] for index in candidate_indices]
    text_features = encode_text(
        clip_module, model, candidate_names, args.prompt_template, device
    )
    logger.info("%s uses %d candidate classes", dataset_name, len(candidate_indices))
    job_results = []
    try:
        for seed in args.seeds:
            samples, correct = evaluate_dataset(
                model, text_features, dataset, args.batch_size, args.num_workers,
                device, model_name, dataset_name, seed, args.limit, progress_position,
                candidate_indices,
            )
            result = Result(
                model=model_name,
                dataset=dataset_name,
                samples=samples,
                correct=correct,
                top1_accuracy=correct / samples,
                candidate_classes=len(candidate_indices),
                class_names_source=args.class_names.name,
                prompt_template=args.prompt_template,
                seed=seed,
                device=str(device),
            )
            job_results.append(result)
            print(f"[RESULT] {device} seed={seed} {model_name} {dataset_name}: "
                  f"{correct}/{samples} ({100 * result.top1_accuracy:.4f}%)", flush=True)
            logger.info("result %s seed=%d %s %s: %d/%d (%.4f%%)", device, seed,
                        model_name, dataset_name, correct, samples,
                        100 * result.top1_accuracy)
    finally:
        del text_features, model
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    return job_results


def run_device_queue(clip_module, args, class_index, names, device_name: str,
                     jobs: list[tuple[str, str]], progress_position: int,
                     logger: logging.Logger, record_results):
    logger.info("assigned to %s: %s", device_name, jobs)
    for model_name, dataset_name in jobs:
        job_results = evaluate_job(
            clip_module, args, class_index, names, model_name, dataset_name,
            device_name, progress_position, logger,
        )
        record_results(job_results)


def main():
    args, config_path = parse_args()
    logger = setup_logging(args.output_root, args.log_file)
    logger.info("config: %s", config_path)
    logger.info("resolved arguments: %s", vars(args))
    import clip

    class_index = load_class_index(args.class_index)
    names = load_class_names(args.class_names)
    results: list[Result] = []
    results_lock = Lock()
    model_order = {name: index for index, name in enumerate(args.models)}
    dataset_order = {name: index for index, name in enumerate(args.datasets)}
    seed_order = {seed: index for index, seed in enumerate(args.seeds)}

    def record_results(job_results: list[Result]):
        with results_lock:
            results.extend(job_results)
            results.sort(key=lambda result: (
                model_order[result.model], dataset_order[result.dataset], seed_order[result.seed]
            ))
            write_results(results, args.output_root)

    jobs = list(product(args.models, args.datasets))
    worker_count = min(len(args.devices), len(jobs))
    devices = args.devices[:worker_count]
    queues = [[] for _ in devices]
    for index, job in enumerate(jobs):
        queues[index % worker_count].append(job)
    print(f"[INFO] scheduling {len(jobs)} jobs across {len(devices)} devices: "
          f"{', '.join(devices)}", flush=True)
    logger.info("scheduling %d jobs across devices: %s", len(jobs), devices)

    if worker_count == 1:
        run_device_queue(
            clip, args, class_index, names, devices[0], queues[0], 0, logger, record_results
        )
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="clip-gpu") as executor:
            futures = [
                executor.submit(
                    run_device_queue, clip, args, class_index, names, device_name,
                    device_jobs, position, logger, record_results,
                )
                for position, (device_name, device_jobs) in enumerate(zip(devices, queues))
            ]
            for future in as_completed(futures):
                future.result()

    averages = average_results(results, args.models, args.datasets)
    write_average_results(averages, args.output_root, args.models, args.datasets)
    for row in averages:
        print(f"[AVERAGE] {row['model']} {row['dataset']}: "
              f"{100 * row['average_top1_accuracy']:.4f}% +/- "
              f"{100 * row['std_top1_accuracy']:.4f}%", flush=True)
        logger.info("average %s %s: %.4f%% +/- %.4f%%", row["model"], row["dataset"],
                    100 * row["average_top1_accuracy"], 100 * row["std_top1_accuracy"])
    logger.info("wrote individual and average results to %s", args.output_root)
    print(f"[INFO] wrote individual and average results to {args.output_root}")


if __name__ == "__main__":
    main()
