# 准备工作

本文说明 ImageNet 及四个分布偏移数据集的下载、目录整理和可视化方法。

- 原始图像放在 `data/raw/`。
- 下载文件放在 `data/downloads/`。
- 可视化结果放在 `runs/dataset_visualizations/`。

## 环境准备

在运行 zero-shot 评测前创建一次 `clip` 环境。以下命令适用于带 NVIDIA GPU、CUDA 13.0 的环境：

```bash
conda create -n clip python=3.10 -y
conda activate clip
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install 'setuptools<81'
python -m pip install --no-build-isolation \
  'git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33'
python -m pip install Pillow PyYAML tqdm ftfy regex matplotlib
```

验证安装：

```bash
python -c "import torch, clip; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

输出应包含 `True` 和可用 GPU 数量。没有 NVIDIA GPU 时，请按 PyTorch 官网选择 CPU 版安装命令。`setuptools<81` 和 `--no-build-isolation` 用于兼容 OpenAI CLIP 的旧版打包配置。

## 数据集准备

| 数据集 | 说明 | 来源 |
| --- | --- | --- |
| ImageNet (ILSVRC 2012) | 1000 类图像分类基准。下载需要注册 ImageNet 账号并接受使用条款。 | [ImageNet 下载页](https://image-net.org/download.php) |
| ImageNet-V2 | 重新采样的测试集。本项目使用 MatchedFrequency 版本。 | [项目说明](https://github.com/modestyachts/ImageNetV2)、[数据文件](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main) |
| ImageNet-A | 覆盖 200 个类别的自然对抗样本。 | [论文](https://arxiv.org/abs/1907.07174)、[下载页](https://github.com/hendrycks/natural-adv-examples) |
| ImageNet-R | 包含绘画、卡通等不同表现形式的图像，覆盖 200 个类别。 | [论文](https://arxiv.org/abs/2006.16241)、[下载页](https://github.com/hendrycks/imagenet-r) |
| ImageNet-Sketch | 覆盖 1000 个类别的手绘草图。 | [项目与下载页](https://github.com/HaohanWang/ImageNet-Sketch) |

使用或再分发数据前，请阅读各数据集来源的许可和使用条款。

## 下载与解压

以下命令均在项目根目录执行。先创建下载目录：

```bash
mkdir -p data/downloads data/raw
```

### ImageNet (ILSVRC 2012)

**下载**

ImageNet 不能匿名直链下载。请在 [ImageNet 下载页](https://image-net.org/download.php) 获取以下两个文件，并放入 `data/downloads/`：

```text
data/downloads/ILSVRC2012_img_train.tar
data/downloads/ILSVRC2012_img_val.tar
```

**解压**

训练集压缩包内通常按类别保存 tar 文件；验证集需要按 WNID 整理。最终目录应为：

```text
data/raw/imagenet/
  train/<wnid>/*.JPEG
  val/<wnid>/*.JPEG
```

训练集需要再展开每个类别 tar；验证集解压后，根据 ImageNet 提供的验证集标签完成分类整理：

```bash
mkdir -p data/raw/imagenet/train data/raw/imagenet/val
tar -xf data/downloads/ILSVRC2012_img_train.tar -C data/raw/imagenet/train
tar -xf data/downloads/ILSVRC2012_img_val.tar -C data/raw/imagenet/val
for archive in data/raw/imagenet/train/*.tar; do
  wnid=$(basename "$archive" .tar)
  mkdir -p "data/raw/imagenet/train/$wnid"
  tar -xf "$archive" -C "data/raw/imagenet/train/$wnid"
done
```

### ImageNet-V2

**下载**

```bash
wget -c -O data/downloads/imagenetv2-matched-frequency.tar \
  'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz'
```

**解压**

```bash
mkdir -p data/raw/imagenetv2
tar -xf data/downloads/imagenetv2-matched-frequency.tar -C data/raw/imagenetv2
```

该文件虽然下载地址以 `.tar.gz` 结尾，实际是 tar 归档，因此使用 `tar -xf`。

### ImageNet-A

**下载**

```bash
wget -c -O data/downloads/imagenet-a.tar \
  'https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar'
```

**解压**

```bash
mkdir -p data/raw/imagenet-a
tar -xf data/downloads/imagenet-a.tar -C data/raw/imagenet-a
```

解压后应能在 `data/raw/imagenet-a/` 下看到按 WNID 命名的类别目录；若压缩包多了一层同名目录，将类别目录移到该根目录下。

### ImageNet-R

**下载**

```bash
wget -c -O data/downloads/imagenet-r.tar \
  'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
```

**解压**

```bash
mkdir -p data/raw/imagenet-r
tar -xf data/downloads/imagenet-r.tar -C data/raw/imagenet-r
```

解压后应能在 `data/raw/imagenet-r/` 下看到按 WNID 命名的类别目录；必要时移除多余的顶层目录。

### ImageNet-Sketch

**下载**

作者通过 [Google Drive](https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA) 提供文件。手动下载后保存为 `data/downloads/imagenet-sketch.zip`；安装 `gdown` 时也可以运行：

```bash
gdown 'https://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA' \
  -O data/downloads/imagenet-sketch.zip
```

**解压**

```bash
mkdir -p data/raw/imagenet-sketch
unzip -q data/downloads/imagenet-sketch.zip -d data/raw/imagenet-sketch
```

若解压后多出 `data/raw/imagenet-sketch/sketch/`，将其中的类别目录移到 `data/raw/imagenet-sketch/`。

最终目录结构如下：

```text
data/raw/
  imagenet/                 # train/<wnid>/ 和 val/<wnid>/
  imagenetv2/               # 数字类别目录
  imagenet-a/               # WNID 类别目录
  imagenet-r/               # WNID 类别目录
  imagenet-sketch/          # WNID 类别目录
```

## 可视化

运行 `scripts/visualize_datasets.py` 可以为每个数据集生成图像样例画廊。脚本会：

1. 从每个数据集随机选择指定数量的类别；
2. 每个类别随机选择一张图像，并保持比例缩放到白色方形画布；
3. 在图片上方显示 ImageNet 类别名称，并按配置添加边框和总标题；
4. 将结果保存为 PNG 和 PDF。

固定 `seed` 后，使用相同数据和配置可以复现抽样结果。ImageNet-V2 的数字目录和其他数据集的 WNID 目录均通过 `configs/visualization/imagenet_class_index.json` 转换为类别名称。

使用默认配置运行：

```bash
python scripts/visualize_datasets.py \
  --config configs/visualization/datasets.yaml
```

也可以临时覆盖配置，例如只处理 ImageNet-A，抽取 10 个类别并排成 2 行：

```bash
python scripts/visualize_datasets.py \
  --config configs/visualization/datasets.yaml \
  --datasets imagenet-a --num-classes 10 --rows 2
```

## 配置文件

`configs/visualization/datasets.yaml` 控制数据路径、抽样数量和画布样式。相对路径以项目根目录为基准。

| 参数 | 作用 |
| --- | --- |
| `data_root` | 数据集根目录，默认 `data/raw`。 |
| `output_root` | 输出目录，默认 `runs/dataset_visualizations`。 |
| `class_index` | ImageNet 类别映射文件，通常无需修改。 |
| `datasets` | 要处理的数据集列表；可选 `imagenet`、`imagenetv2`、`imagenet-a`、`imagenet-r`、`imagenet-sketch`。 |
| `num_classes` | 每个数据集抽取的类别数，每类 1 张图。 |
| `rows` | 画布行数，列数自动计算。 |
| `seed` | 随机种子，用于复现抽样结果。 |
| `figsize`、`dpi` | 画布尺寸和 PNG 分辨率。 |
| `show_title`、`show_figure_title` | 是否显示类别名称和总标题。 |
| `font_size`、`title_font_size` | 类别名称和总标题字号。 |
| `border_width`、`border_color` | 图片边框宽度和颜色；宽度设为 `0` 可隐藏边框。 |
| `title_pad`、`figure_title_pad`、`pad_inches` | 标题间距和保存时的外侧留白。 |

## 输出

每个数据集会在自己的子目录生成 `contact_sheet.png` 和 `contact_sheet.pdf`，例如：

```text
runs/dataset_visualizations/imagenet-a/contact_sheet.png
runs/dataset_visualizations/imagenet-a/contact_sheet.pdf
```

重新运行会覆盖同名文件。`[WARNING]` 表示某张图片无法读取；`[FAIL]` 表示配置、类别映射或数据目录有问题。
