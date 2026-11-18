# 准备工作

本部分完成
1. 环境创建
2. ImageNet Variants 数据集准备和可视化审阅
3. CLIP 权重准备

## 1. 创建环境

```bash
conda create -n clip python=3.10 -y # clip 可以换成任意其他环境名
conda activate clip
python -m pip install --upgrade pip
```

安装与本机匹配的 PyTorch，以 CUDA 13.0 示例：

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

其他 CUDA 版本或 CPU 环境使用 [PyTorch 官方选择器](https://pytorch.org/get-started/locally/) 给出的命令.

安装项目、OpenAI CLIP 和绘图依赖：

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

预期结果：
- GPU： `CUDA available: True`
- `clip-evaluate --help` 应显示 zero-shot 评测参数


## 2. 数据集准备

创建目录：

```bash
mkdir -p data/downloads data/raw
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

通过 [Google Drive](https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA) 压缩包下载并解压：

```bash
python -m pip install gdown
gdown 'https://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA' \
  -O data/downloads/imagenet-sketch.zip
mkdir -p data/raw/imagenet-sketch
unzip -q data/downloads/imagenet-sketch.zip -d data/raw/imagenet-sketch
```

解压后确保 (手动移动、增删文件即可) 每个数据集目录下面直接是类别目录（数字或 WNID），图像放在类别目录内. 例如：

```text
data/raw/imagenetv2/0/...                 # 1000 个数字类别目录
data/raw/imagenet-a/n01498041/...         # 200 个 WNID 目录
data/raw/imagenet-r/n01443537/...         # 200 个 WNID 目录
data/raw/imagenet-sketch/n01440764/...    # 1000 个 WNID 目录
```

### 校验目录

```bash
for dataset in imagenetv2 imagenet-a imagenet-r imagenet-sketch; do
  printf '%-18s ' "$dataset"
  find "data/raw/$dataset" -mindepth 1 -maxdepth 1 -type d | wc -l
done
```

预期结果：
- 上面四行应依次显示 `1000`、`200`、`200`、`1000`


## 3. 下载 CLIP 权重

创建目录并下载模型权重
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
python scripts/0_visualize_datasets.py \
  --config configs/visualization/0_datasets.yaml
```

输出位于 `runs/0_dataset_visualizations/<dataset>/contact_sheet.png` 和 `.pdf`.

只检查 ImageNet-A：

```bash
python scripts/0_visualize_datasets.py \
  --config configs/visualization/0_datasets.yaml \
  --datasets imagenet-a --num-classes 10 --rows 2
```
