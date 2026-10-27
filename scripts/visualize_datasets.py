"""Create class-balanced sample galleries for ImageNet and its variants."""

import argparse
import json
import math
import random
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, UnidentifiedImageError
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/visualization/datasets.yaml"
DATASETS = ("imagenet", "imagenetv2", "imagenet-a", "imagenet-r", "imagenet-sketch")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CONFIG_KEYS = {
    "data_root", "output_root", "datasets", "num_classes", "rows", "seed",
    "dpi", "figsize", "font_size", "title_font_size", "border_width",
    "border_color", "pad_inches", "class_index", "show_title", "title_pad",
    "show_figure_title", "figure_title_pad",
}


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        super().error(f"[FAIL] {message}")


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--class-index", type=Path)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--figsize", type=float, nargs=2)
    parser.add_argument("--font-size", type=float)
    parser.add_argument("--title-font-size", type=float)
    parser.add_argument("--show-title", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--title-pad", type=float)
    parser.add_argument("--show-figure-title", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--figure-title-pad", type=float)
    parser.add_argument("--border-width", type=float)
    parser.add_argument("--border-color")
    parser.add_argument("--pad-inches", type=float)
    cli = parser.parse_args(argv)

    config_path = cli.config if cli.config.is_absolute() else Path.cwd() / cli.config
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    except OSError as error:
        parser.error(f"Cannot read config {config_path}: {error}")
    if not isinstance(config, dict):
        parser.error("Config must contain a YAML mapping")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        parser.error(f"Unknown config keys: {', '.join(sorted(unknown))}")

    values = {
        "data_root": "data/raw",
        "output_root": "runs/dataset_visualizations",
        "class_index": "configs/visualization/imagenet_class_index.json",
        "datasets": list(DATASETS),
        "num_classes": 10,
        "rows": 2,
        "seed": 42,
        "dpi": 180,
        "figsize": [15, 7],
        "font_size": 10,
        "title_font_size": 14,
        "show_title": True,
        "title_pad": 3,
        "show_figure_title": True,
        "figure_title_pad": 8,
        "border_width": 2.2,
        "border_color": "#3A3D42",
        "pad_inches": 0.12,
    }
    values.update(config)
    for key in CONFIG_KEYS:
        override = getattr(cli, key, None)
        if override is not None:
            values[key] = override
    for key in ("data_root", "output_root", "class_index"):
        path = Path(values[key])
        values[key] = path if path.is_absolute() else PROJECT_ROOT / path

    if not values["datasets"] or any(name not in DATASETS for name in values["datasets"]):
        parser.error("datasets must contain one or more supported dataset names")
    if values["num_classes"] < 1 or values["rows"] < 1:
        parser.error("num_classes and rows must be positive")
    if values["dpi"] <= 0 or values["font_size"] <= 0 or values["title_font_size"] <= 0:
        parser.error("dpi and font sizes must be positive")
    if values["border_width"] < 0 or values["pad_inches"] < 0:
        parser.error("border width and padding must be non-negative")
    if values["title_pad"] < 0 or values["figure_title_pad"] < 0:
        parser.error("title spacing must be non-negative")
    if not isinstance(values["show_title"], bool) or not isinstance(values["show_figure_title"], bool):
        parser.error("show_title and show_figure_title must be true or false")
    if len(values["figsize"]) != 2 or any(size <= 0 for size in values["figsize"]):
        parser.error("figsize must contain two positive values")
    return values


def load_class_names(path: Path):
    with path.open(encoding="utf-8") as stream:
        class_index = json.load(stream)
    if len(class_index) != 1000:
        raise ValueError(f"Expected 1000 ImageNet classes in {path}")
    by_index = {index: names[1].replace("_", " ") for index, names in class_index.items()}
    by_wnid = {names[0]: names[1].replace("_", " ") for names in class_index.values()}
    return by_index, by_wnid


def find_categories(root: Path):
    categories = []
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        images = [path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
        if images:
            categories.append((directory.name, images))
    return categories


def sample_categories(root: Path, count: int, seed: int):
    categories = find_categories(root)
    if len(categories) < count:
        raise ValueError(f"Found {len(categories)} image categories under {root}; requested {count}")
    rng = random.Random(seed)
    selected = rng.sample(categories, count)
    return [(name, rng.choice(images)) for name, images in selected]


def style_axis(axis, image, args):
    tile = Image.new("RGB", (256, 256), "white")
    thumbnail = ImageOps.contain(image.convert("RGB"), tile.size)
    tile.paste(thumbnail, ((256 - thumbnail.width) // 2, (256 - thumbnail.height) // 2))
    axis.imshow(tile, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(args["border_width"] > 0)
        spine.set_color(args["border_color"])
        spine.set_linewidth(args["border_width"])


def make_contact_sheet(name: str, root: Path, output: Path, seed: int, args, class_names):
    selected = sample_categories(root, args["num_classes"], seed)
    columns = math.ceil(len(selected) / args["rows"])
    fig, axes = plt.subplots(args["rows"], columns, figsize=args["figsize"], squeeze=False)
    for axis, (category, path) in zip(axes.flat, selected):
        label = class_names[0].get(category) if name == "imagenetv2" else class_names[1].get(category)
        if label is None:
            raise ValueError(f"No ImageNet class name for {name} category {category}")
        try:
            with Image.open(path) as image:
                style_axis(axis, image, args)
        except (OSError, UnidentifiedImageError) as error:
            print(f"[WARNING] Cannot read {path}: {error}", file=sys.stderr)
            axis.text(0.5, 0.5, f"Unreadable image\n{error}", ha="center", va="center")
            axis.set_xticks([])
            axis.set_yticks([])
        if args["show_title"]:
            axis.set_title(textwrap.fill(label, width=22), fontsize=args["font_size"],
                           pad=args["title_pad"])
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    top = 0.97
    if args["show_figure_title"]:
        fig.suptitle(f"{name}: {len(selected)} randomly selected classes",
                     fontsize=args["title_font_size"], y=top, va="top")
        top -= (args["title_font_size"] + args["figure_title_pad"]) / (72 * args["figsize"][1])
    if args["show_title"]:
        top -= (args["font_size"] + args["title_pad"]) / (72 * args["figsize"][1])
    row_spacing = 0.18 if args["show_title"] else 0.06
    cell_width = args["figsize"][0] * 0.94 / columns
    grid_height = cell_width * (args["rows"] + row_spacing * (args["rows"] - 1))
    bottom = max(0.04, top - grid_height / args["figsize"][1])
    fig.subplots_adjust(left=0.03, right=0.97, top=top, bottom=bottom,
                        hspace=row_spacing, wspace=0.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=args["dpi"],
                bbox_inches="tight", pad_inches=args["pad_inches"])
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight",
                pad_inches=args["pad_inches"])
    plt.close(fig)
    print(f"[SUCCESS] {name}: saved {len(selected)} classes to {output.with_suffix('.png')}")
    print(f"[SUCCESS] {name}: saved {output.with_suffix('.pdf')}")


def main():
    args = parse_args()
    try:
        class_names = load_class_names(args["class_index"])
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"[FAIL] Cannot load class index: {error}") from error
    for index, name in enumerate(args["datasets"]):
        root = args["data_root"] / name
        if name == "imagenet" and (root / "train").is_dir():
            root = root / "train"
        if not root.is_dir():
            raise SystemExit(f"[FAIL] Dataset directory does not exist: {root}")
        print(f"[INFO] {name}: selecting {args['num_classes']} classes from {root}", flush=True)
        try:
            make_contact_sheet(
                name, root, args["output_root"] / name / "contact_sheet.png",
                args["seed"] + index, args, class_names,
            )
        except (OSError, ValueError) as error:
            raise SystemExit(f"[FAIL] {name}: {error}") from error


if __name__ == "__main__":
    main()
