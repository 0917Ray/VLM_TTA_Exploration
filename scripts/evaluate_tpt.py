"""Evaluate Test-Time Prompt Tuning on ImageNet variants."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("imagenetv2", "imagenet-a", "imagenet-r", "imagenet-sketch")
MODEL_FILES = {"RN50": "RN50.pt", "ViT-B/16": "ViT-B-16.pt"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CONFIG_KEYS = {
    "data_root", "class_index", "class_names", "model_root", "output_root", "idea",
    "datasets", "models", "devices", "device", "seeds", "seed", "num_workers", "views",
    "selection_fraction", "tta_steps", "learning_rate", "prompt_init", "resolution",
    "print_frequency", "limit", "log_file",
}
DEFAULTS = {
    "data_root": "data/raw",
    "class_index": "configs/visualization/imagenet_class_index.json",
    "class_names": "configs/evaluation/imagenet_classes_tpt.json",
    "model_root": "models",
    "output_root": "runs/tpt_classification",
    "idea": None,
    "datasets": list(DATASET_NAMES),
    "models": ["RN50"],
    "devices": ["cuda:0"] if torch.cuda.is_available() else ["cpu"],
    "seeds": [0],
    "num_workers": min(4, os.cpu_count() or 1),
    "views": 64,
    "selection_fraction": 0.1,
    "tta_steps": 1,
    "learning_rate": 5e-3,
    "prompt_init": "a photo of a",
    "resolution": 224,
    "print_frequency": 100,
    "limit": None,
    "log_file": "run.log",
}


@dataclass
class Result:
    method: str
    idea: str | None
    model: str
    dataset: str
    samples: int
    correct_top1: int
    correct_top5: int
    top1_accuracy: float
    top5_accuracy: float
    candidate_classes: int
    class_names_source: str
    prompt_init: str
    views: int
    selection_fraction: float
    tta_steps: int
    learning_rate: float
    seed: int
    device: str
    elapsed_minutes: float


class TPTTransform:
    """Return the canonical CLIP view followed by stochastic TPT views."""

    def __init__(self, resolution: int, views: int):
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        self.canonical = transforms.Compose([
            transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ])
        self.preaugment = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
        ])
        self.preprocess = transforms.Compose([transforms.ToTensor(), normalize])
        self.views = views

    def __call__(self, image: Image.Image) -> list[torch.Tensor]:
        output = [self.canonical(image)]
        output.extend(self.preprocess(self.preaugment(image)) for _ in range(self.views - 1))
        return output


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
                if label not in range(1000):
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
                return self.transform(image.convert("RGB")), label
        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(f"Cannot read image {path}: {error}") from error


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.to(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(self.dtype)
        indices = tokenized_prompts.argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), indices] @ self.text_projection


class PromptLearner(nn.Module):
    def __init__(self, clip_module, clip_model, class_names: list[str], prompt_init: str):
        super().__init__()
        prompt_init = prompt_init.replace("_", " ").strip()
        if not prompt_init:
            raise ValueError("prompt_init must not be empty")
        self.clip_dtype = clip_model.dtype
        self.n_ctx = len(prompt_init.split())
        init_tokens = clip_module.tokenize([prompt_init]).to(clip_model.positional_embedding.device)
        with torch.no_grad():
            init_embedding = clip_model.token_embedding(init_tokens).to(self.clip_dtype)
        initial_context = init_embedding[0, 1:1 + self.n_ctx].float()
        if initial_context.shape[0] != self.n_ctx:
            raise ValueError(f"Cannot initialize {self.n_ctx} context tokens from {prompt_init!r}")
        self.ctx = nn.Parameter(initial_context)
        self.register_buffer("initial_context", initial_context.detach().clone())

        prompts = [f"{prompt_init} {name}." for name in class_names]
        tokenized = clip_module.tokenize(prompts).to(clip_model.positional_embedding.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).to(self.clip_dtype)
        self.register_buffer("tokenized_prompts", tokenized)
        self.register_buffer("token_prefix", embedding[:, :1])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:])

    def reset(self):
        self.ctx.copy_(self.initial_context)

    def forward(self) -> torch.Tensor:
        context = self.ctx.to(self.clip_dtype).unsqueeze(0).expand(
            self.token_prefix.shape[0], -1, -1
        )
        return torch.cat((self.token_prefix, context, self.token_suffix), dim=1)


class TPTClassifier(nn.Module):
    def __init__(self, clip_module, checkpoint: Path, class_names: list[str],
                 prompt_init: str, device: torch.device):
        super().__init__()
        clip_model, _ = clip_module.load(str(checkpoint), device=device, jit=False)
        # The official TPT CLIP fork keeps master weights in FP32 and uses AMP at runtime.
        clip_model.float()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_learner = PromptLearner(clip_module, clip_model, class_names, prompt_init)
        self.register_buffer("logit_scale", clip_model.logit_scale.detach().clone())
        self.clip_dtype = clip_model.dtype

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.prompt_learner.ctx.requires_grad_(True)

    def reset(self):
        self.prompt_learner.reset()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.image_encoder(images.to(self.clip_dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * image_features @ text_features.t()


def select_confident_samples(logits: torch.Tensor, fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    entropy = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)
    count = int(logits.shape[0] * fraction)
    indices = torch.argsort(entropy)[:count]
    return logits[indices], indices


def average_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probabilities = logits - logits.logsumexp(dim=-1, keepdim=True)
    average_log_probability = log_probabilities.logsumexp(dim=0) - math.log(logits.shape[0])
    average_log_probability = torch.clamp(
        average_log_probability, min=torch.finfo(average_log_probability.dtype).min
    )
    return -(average_log_probability * average_log_probability.exp()).sum()


def tune_prompt(model: TPTClassifier, views: torch.Tensor, optimizer, scaler,
                selection_fraction: float, tta_steps: int, device: torch.device):
    selected_indices = None
    for _ in range(tta_steps):
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            logits = model(views)
            if selected_indices is None:
                logits, selected_indices = select_confident_samples(logits, selection_fraction)
            else:
                logits = logits[selected_indices]
            loss = average_entropy(logits)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


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
                        default=PROJECT_ROOT / "configs/evaluation/tpt_classification.yaml")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--class-index", type=Path)
    parser.add_argument("--class-names", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--idea")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--devices", nargs="+")
    device_group.add_argument("--device", help=argparse.SUPPRESS)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", type=int, nargs="+")
    seed_group.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--views", type=int)
    parser.add_argument("--selection-fraction", type=float)
    parser.add_argument("--tta-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--prompt-init")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--print-frequency", type=int)
    parser.add_argument("--limit", type=int)
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
    for key in CONFIG_KEYS - {"log_file", "device", "seed"}:
        value = getattr(cli, key, None)
        if value is not None:
            values[key] = value
    if cli.device is not None:
        values["devices"] = [cli.device]
    if cli.seed is not None:
        values["seeds"] = [cli.seed]
    for key in ("data_root", "class_index", "class_names", "model_root", "output_root"):
        values[key] = resolve_path(values[key])

    idea = values["idea"]
    if idea is not None and not isinstance(idea, str):
        raise SystemExit("idea must be a string or null")
    if isinstance(idea, str):
        idea = "_".join(idea.split())
        if idea in {".", ".."} or any(character in idea for character in ("/", "\\", "\0")):
            raise SystemExit("idea must be a single directory-safe name without path separators")
    values["idea"] = idea or None
    if values["idea"] is not None:
        values["output_root"] = values["output_root"] / f"idea_{values['idea']}"

    if not values["datasets"] or any(name not in DATASET_NAMES for name in values["datasets"]):
        raise SystemExit(f"datasets must contain only: {', '.join(DATASET_NAMES)}")
    if not values["models"] or any(name not in MODEL_FILES for name in values["models"]):
        raise SystemExit(f"models must contain only: {', '.join(MODEL_FILES)}")
    if (not isinstance(values["devices"], list) or not values["devices"] or
            len(set(values["devices"])) != len(values["devices"])):
        raise SystemExit("devices must be a non-empty list without duplicates")
    for device_name in values["devices"]:
        device = torch.device(device_name)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise SystemExit(f"CUDA is unavailable, cannot use {device_name}")
            index = device.index if device.index is not None else 0
            if index >= torch.cuda.device_count():
                raise SystemExit(f"CUDA device {device_name} is unavailable")
    if (not isinstance(values["seeds"], list) or not values["seeds"] or
            any(not isinstance(seed, int) or isinstance(seed, bool) for seed in values["seeds"])):
        raise SystemExit("seeds must be a non-empty list of integers")
    if len(set(values["seeds"])) != len(values["seeds"]):
        raise SystemExit("seeds must not contain duplicates")
    if values["num_workers"] < 0:
        raise SystemExit("num_workers must be non-negative")
    if values["views"] < 2:
        raise SystemExit("views must be at least 2")
    if not 0 < values["selection_fraction"] <= 1:
        raise SystemExit("selection_fraction must be in (0, 1]")
    if int(values["views"] * values["selection_fraction"]) < 1:
        raise SystemExit("views * selection_fraction must select at least one view")
    if values["tta_steps"] < 1 or values["learning_rate"] <= 0:
        raise SystemExit("tta_steps and learning_rate must be positive")
    if values["resolution"] < 1 or values["print_frequency"] < 1:
        raise SystemExit("resolution and print_frequency must be positive")
    if not isinstance(values["prompt_init"], str) or not values["prompt_init"].strip():
        raise SystemExit("prompt_init must be a non-empty string")
    if values["limit"] is not None and values["limit"] < 1:
        raise SystemExit("limit must be positive")
    return argparse.Namespace(**values), config_path


def setup_main_logging(output_root: Path, log_file: str) -> tuple[logging.Logger, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file)
    if not log_path.is_absolute():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = output_root / log_path.with_name(
            f"{log_path.stem}_{timestamp}{log_path.suffix or '.log'}"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tpt")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(processName)s: %(message)s"))
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


def setup_worker_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"tpt.worker.{os.getpid()}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(processName)s: %(message)s"))
    logger.addHandler(handler)
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.addHandler(handler)
    warnings_logger.propagate = False
    return logger


def initialize_progress_lock(lock):
    tqdm.set_lock(lock)


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
        raise ValueError(f"{dataset_name} must contain exactly 200 classes; found {len(indices)}")
    return indices


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_job(serialized_args: dict, model_name: str, dataset_name: str, seed: int,
                 device_name: str, progress_position: int, log_path: str) -> dict:
    logger = setup_worker_logging(log_path)
    args = argparse.Namespace(**serialized_args)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    logger.info("starting %s / %s / seed=%d on %s", model_name, dataset_name, seed, device)

    import clip

    class_index = load_class_index(args.class_index)
    all_names = load_class_names(args.class_names)
    transform = TPTTransform(args.resolution, args.views)
    full_dataset = VariantDataset(
        args.data_root / dataset_name, dataset_name, class_index, transform
    )
    candidate_indices = candidate_class_indices(full_dataset, dataset_name)
    candidate_names = [all_names[index] for index in candidate_indices]
    dataset: Dataset = full_dataset
    if args.limit is not None:
        dataset = Subset(full_dataset, range(min(args.limit, len(full_dataset))))

    checkpoint = args.model_root / MODEL_FILES[model_name]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")
    model = TPTClassifier(clip, checkpoint, candidate_names, args.prompt_init, device).to(device)
    model.eval()
    model_resolution = int(model.image_encoder.input_resolution)
    if model_resolution != args.resolution:
        raise ValueError(
            f"{model_name} expects resolution {model_resolution}, got {args.resolution}"
        )
    optimizer = torch.optim.AdamW(
        [model.prompt_learner.ctx], lr=args.learning_rate
    )
    scaler = torch.amp.GradScaler(device.type, init_scale=1000, enabled=device.type == "cuda")

    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "generator": generator,
    }
    if args.num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(dataset, **loader_kwargs)
    global_indices = torch.tensor(candidate_indices, device=device)
    correct_top1 = 0
    correct_top5 = 0
    samples = 0
    started = time.perf_counter()

    progress = tqdm(
        loader,
        desc=f"{device} / seed={seed} / {model_name} / {dataset_name}",
        unit="image",
        position=progress_position,
        leave=True,
    )
    for view_list, labels in progress:
        views = torch.cat(view_list, dim=0).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            model.reset()
        optimizer.state.clear()
        tune_prompt(
            model, views, optimizer, scaler, args.selection_fraction, args.tta_steps, device
        )
        with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(views[:1])
        local_top5 = logits.topk(k=min(5, len(candidate_indices)), dim=1).indices
        predictions = global_indices[local_top5]
        correct_top1 += int((predictions[:, 0] == labels).sum().item())
        correct_top5 += int((predictions == labels.unsqueeze(1)).any(dim=1).sum().item())
        samples += labels.numel()
        accuracy = 100 * correct_top1 / samples
        progress.set_postfix(top1=f"{accuracy:.2f}%")
        if samples % args.print_frequency == 0 or samples == len(dataset):
            logger.info("progress %d/%d top1=%.4f%%", samples, len(dataset), accuracy)

    elapsed = time.perf_counter() - started
    result = Result(
        method="TPT",
        idea=args.idea,
        model=model_name,
        dataset=dataset_name,
        samples=samples,
        correct_top1=correct_top1,
        correct_top5=correct_top5,
        top1_accuracy=correct_top1 / samples,
        top5_accuracy=correct_top5 / samples,
        candidate_classes=len(candidate_indices),
        class_names_source=args.class_names.name,
        prompt_init=args.prompt_init,
        views=args.views,
        selection_fraction=args.selection_fraction,
        tta_steps=args.tta_steps,
        learning_rate=args.learning_rate,
        seed=seed,
        device=str(device),
        elapsed_minutes=elapsed / 60,
    )
    logger.info("completed %s / %s / seed=%d: top1=%.4f%% top5=%.4f%% in %.2f min",
                model_name, dataset_name, seed, 100 * result.top1_accuracy,
                100 * result.top5_accuracy, result.elapsed_minutes)
    return asdict(result)


def write_results(results: list[Result], output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "results.json").open("w", encoding="utf-8") as stream:
        json.dump([asdict(result) for result in results], stream, indent=2, ensure_ascii=False)
    with (output_root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def write_average_results(results: list[Result], output_root: Path,
                          models: list[str], datasets: list[str]):
    output_root.mkdir(parents=True, exist_ok=True)
    averages = []
    for model_name in models:
        for dataset_name in datasets:
            group = [result for result in results
                     if result.model == model_name and result.dataset == dataset_name]
            top1 = [result.top1_accuracy for result in group]
            top5 = [result.top5_accuracy for result in group]
            elapsed_minutes = [result.elapsed_minutes for result in group]
            averages.append({
                "idea": results[0].idea,
                "model": model_name,
                "dataset": dataset_name,
                "num_seeds": len(group),
                "seeds": [result.seed for result in group],
                "samples_per_seed": group[0].samples,
                "average_top1_accuracy": statistics.fmean(top1),
                "std_top1_accuracy": statistics.pstdev(top1),
                "average_top5_accuracy": statistics.fmean(top5),
                "std_top5_accuracy": statistics.pstdev(top5),
                "average_elapsed_minutes": statistics.fmean(elapsed_minutes),
            })
    with (output_root / "average_results.json").open("w", encoding="utf-8") as stream:
        json.dump(averages, stream, indent=2, ensure_ascii=False)
    csv_rows = [{**row, "seeds": " ".join(map(str, row["seeds"]))} for row in averages]
    with (output_root / "average_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    by_pair = {(row["model"], row["dataset"]): row for row in averages}
    lines = ["# TPT Average Top-1 Accuracy", ""]
    if results[0].idea is not None:
        lines.extend([f"Idea: `{results[0].idea}`", ""])
    lines.extend([
        "Values are mean +/- population standard deviation across seeds.", "",
        "| Model | " + " | ".join(datasets) + " |",
        "| --- | " + " | ".join("---:" for _ in datasets) + " |",
    ])
    for model_name in models:
        cells = []
        for dataset_name in datasets:
            row = by_pair[(model_name, dataset_name)]
            cells.append(
                f"{100 * row['average_top1_accuracy']:.4f}% +/- "
                f"{100 * row['std_top1_accuracy']:.4f}%"
            )
        lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.extend([
        "## Average Runtime (minutes)", "",
        "| Model | " + " | ".join(datasets) + " |",
        "| --- | " + " | ".join("---:" for _ in datasets) + " |",
    ])
    for model_name in models:
        cells = [
            f"{by_pair[(model_name, dataset_name)]['average_elapsed_minutes']:.2f}"
            for dataset_name in datasets
        ]
        lines.append(f"| {model_name} | " + " | ".join(cells) + " |")
    lines.append("")
    (output_root / "average_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args, config_path = parse_args()
    logger, _ = setup_main_logging(args.output_root, args.log_file)
    logger.info("config: %s", config_path)
    logger.info("resolved arguments: %s", vars(args))

    jobs = list(product(args.models, args.datasets, args.seeds))
    worker_count = min(len(args.devices), len(jobs))
    devices = args.devices[:worker_count]
    assignments = [[] for _ in devices]
    for index, job in enumerate(jobs):
        assignments[index % worker_count].append(job)
    print(
        f"[INFO] scheduling {len(jobs)} TPT jobs across {worker_count} devices: "
        f"{', '.join(devices)}",
        flush=True,
    )
    logger.info("assignments: %s", dict(zip(devices, assignments)))

    seed_log_paths = {}
    for seed in args.seeds:
        seed_root = args.output_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        configured_log = Path(args.log_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        seed_log_name = configured_log.with_name(
            f"{configured_log.stem}_{timestamp}{configured_log.suffix or '.log'}"
        )
        seed_log_paths[seed] = str(seed_root / seed_log_name)

    serialized_args = vars(args)
    results: list[Result] = []
    model_order = {name: index for index, name in enumerate(args.models)}
    dataset_order = {name: index for index, name in enumerate(args.datasets)}
    seed_order = {seed: index for index, seed in enumerate(args.seeds)}
    spawn_context = multiprocessing.get_context("spawn")
    progress_lock = spawn_context.RLock()

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
        for executor, device_name, device_jobs in zip(executors, devices, assignments):
            for model_name, dataset_name, seed in device_jobs:
                future = executor.submit(
                    evaluate_job, serialized_args, model_name, dataset_name, seed,
                    device_name, devices.index(device_name), seed_log_paths[seed],
                )
                futures[future] = (model_name, dataset_name, seed, device_name)

        for future in as_completed(futures):
            result = Result(**future.result())
            results.append(result)
            results.sort(key=lambda item: (
                model_order[item.model], dataset_order[item.dataset], seed_order[item.seed]
            ))
            seed_results = [item for item in results if item.seed == result.seed]
            write_results(seed_results, args.output_root / f"seed_{result.seed}")
            print(
                f"[RESULT] {result.device} seed={result.seed} {result.model} {result.dataset}: "
                f"{result.correct_top1}/{result.samples} "
                f"({100 * result.top1_accuracy:.4f}%)",
                flush=True,
            )

    summary_root = args.output_root / "summary"
    write_results(results, summary_root)
    write_average_results(results, summary_root, args.models, args.datasets)
    logger.info("wrote seed-specific results under %s and summary to %s",
                args.output_root, summary_root)
    print(f"[INFO] wrote seed-specific TPT results under {args.output_root}")


if __name__ == "__main__":
    main()
