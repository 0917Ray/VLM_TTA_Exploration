# 准备工作

本指南完成环境、ImageNet Variants 数据和 CLIP 权重准备。所有命令均在仓库根目录执行。

## 1. 创建环境

```bash
conda create -n clip python=3.10 -y
conda activate clip
python -m pip install --upgrade pip
```

先安装与本机匹配的 PyTorch。以下是 CUDA 13.0 示例：

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

其他 CUDA 版本或 CPU 环境使用 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/)给出的命令。随后安装项目、OpenAI CLIP 和绘图依赖：

```bash
python -m pip install 'setuptools<81'
python -m pip install --no-build-isolation -r requirements.txt
```

验证解释器、CLIP、绘图库和 CUDA：

```bash
which python
python - <<'PY'
import clip
import matplotlib
import numpy
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("NumPy:", numpy.__version__)
print("Matplotlib:", matplotlib.__version__)
PY

clip-evaluate --help
```

GPU 复现需要 `CUDA available: True`。`clip-evaluate --help` 应显示 zero-shot 评测参数；该命令与 `python scripts/evaluate_clip_zeroshot.py` 使用同一个实现。已有环境只缺绘图依赖时可运行 `python -m pip install -e '.[visualization]'`。

## 2. 数据集准备

创建目录：

```bash
mkdir -p data/downloads data/raw
```

评测需要四个 ImageNet Variants。脚本要求类别目录直接位于各数据集根目录，不能再多套一层同名目录：

```text
data/raw/imagenetv2/0/...                 # 1000 个数字类别目录
data/raw/imagenet-a/n01498041/...         # 200 个 WNID 目录
data/raw/imagenet-r/n01443537/...         # 200 个 WNID 目录
data/raw/imagenet-sketch/n01440764/...    # 1000 个 WNID 目录
```

### ImageNet-V2

本项目使用 MatchedFrequency 版本：

```bash
wget -c -O data/downloads/imagenetv2-matched-frequency.tar \
  'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz'
mkdir -p data/raw/imagenetv2
tar -xf data/downloads/imagenetv2-matched-frequency.tar \
  -C data/raw/imagenetv2
```

### ImageNet-A

```bash
wget -c -O data/downloads/imagenet-a.tar \
  'https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar'
mkdir -p data/raw/imagenet-a
tar -xf data/downloads/imagenet-a.tar \
  -C data/raw/imagenet-a
```

### ImageNet-R

```bash
wget -c -O data/downloads/imagenet-r.tar \
  'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
mkdir -p data/raw/imagenet-r
tar -xf data/downloads/imagenet-r.tar \
  -C data/raw/imagenet-r
```

### ImageNet-Sketch

作者通过 [Google Drive](https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA) 提供压缩包。安装可选下载工具并解压：

```bash
python -m pip install gdown
gdown 'https://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA' \
  -O data/downloads/imagenet-sketch.zip
mkdir -p data/raw/imagenet-sketch
unzip -q data/downloads/imagenet-sketch.zip -d data/raw/imagenet-sketch
```

若解压后得到 `data/raw/imagenet-sketch/sketch/<wnid>/`，将 `sketch/` 下的 WNID 目录上移一层，使其符合上面的最终结构。

tar/zip 发布包可能包含顶层目录。解压后如看到 `data/raw/<dataset>/<archive-root>/<class>/`，应将 `<archive-root>/` 内类别目录移到 `data/raw/<dataset>/`；不要把图像文件直接移到数据集根目录。可先查看压缩包前几条路径确认层级：

```bash
tar -tf data/downloads/imagenetv2-matched-frequency.tar | head
tar -tf data/downloads/imagenet-a.tar | head
tar -tf data/downloads/imagenet-r.tar | head
unzip -l data/downloads/imagenet-sketch.zip | head
```

若确认该顶层目录只包含类别目录，按实际目录名移动。例如 ImageNet-A 解压为 `data/raw/imagenet-a/imagenet-a/<wnid>/` 时：

```bash
mv data/raw/imagenet-a/imagenet-a/* data/raw/imagenet-a/
rmdir data/raw/imagenet-a/imagenet-a
```

将示例中的两处目录名替换为归档实际生成的路径；执行前先用 `find data/raw/imagenet-a -mindepth 1 -maxdepth 2 -type d | head` 核对（其他数据集替换目录名），避免移动错误目录。

### 校验目录

```bash
for dataset in imagenetv2 imagenet-a imagenet-r imagenet-sketch; do
  printf '%-18s ' "$dataset"
  find "data/raw/$dataset" -mindepth 1 -maxdepth 1 -type d | wc -l
done
```

期望类别数依次为 `1000`、`200`、`200`、`1000`。数量不对时先检查是否多了一层顶级目录。使用或再分发数据前请遵守各数据集许可。

完整 ImageNet 仅用于可选的数据画廊，不是上述评测的必需项。如需使用，应整理为 `data/raw/imagenet/train/<wnid>/` 和 `data/raw/imagenet/val/<wnid>/`。

## 3. 下载 CLIP 权重

```bash
mkdir -p models
python - <<'PY'
import clip

for name in ("RN50", "ViT-B/16"):
    clip.load(name, device="cpu", download_root="models")
PY
```

校验文件：

```bash
ls -lh models/RN50.pt models/ViT-B-16.pt
```

脚本中的模型名与文件对应如下：

| 模型参数 | 权重文件 |
| --- | --- |
| `RN50` | `models/RN50.pt` |
| `ViT-B/16` | `models/ViT-B-16.pt` |

## 4. 可视化检查

生成每个数据集的类别样例画廊：

```bash
python scripts/visualize_datasets.py \
  --config configs/visualization/datasets.yaml
```

输出位于 `runs/dataset_visualizations/<dataset>/contact_sheet.png` 和 `.pdf`。只检查 ImageNet-A：

```bash
python scripts/visualize_datasets.py \
  --config configs/visualization/datasets.yaml \
  --datasets imagenet-a --num-classes 10 --rows 2
```

可视化配置中的 `data_root`、`output_root`、`datasets`、`num_classes`、`rows`、`seed`、画布和字体选项均可由同名 CLI 参数临时覆盖。

## 常见错误

- `Dataset directory does not exist`：数据集名称或 `data_root` 不匹配。
- `No supported images found`：类别目录层级错误，或目录中没有受支持的图像格式。
- `Unknown ImageNet WNID`：目录名不是 `configs/visualization/imagenet_class_index.json` 中的 WNID。
- `CUDA device ... is unavailable`：把配置中的 `devices` 改为本机存在的 GPU，例如只保留 `cuda:0`。
- `Plotting requires matplotlib and numpy`：确认已激活 `clip` 环境并安装 `.[visualization]`。
