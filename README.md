# VLM CLIP Test-Time Adaptation

本仓库复现 OpenAI CLIP zero-shot 分类和 Test-Time Prompt Tuning（TPT），支持在 ImageNet-V2、ImageNet-A、ImageNet-R、ImageNet-Sketch 上评测标准 TPT，并比较低熵视图筛选数量、learning rate 和每张图像的 prompt 更新步数。

## 从零开始

先克隆仓库并进入项目目录；后续命令均从该目录执行：

```bash
git clone https://github.com/0917Ray/VLM_TTA_Exploration.git
cd VLM_TTA_Exploration
```

### 1. 安装环境

```bash
conda create -n clip python=3.10 -y
conda activate clip
python -m pip install --upgrade pip
```

先按机器的 CUDA 版本安装 PyTorch。CUDA 13.0 的示例为：

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install 'setuptools<81'
python -m pip install --no-build-isolation -r requirements.txt
clip-evaluate --help
```

其他 CUDA 或 CPU 环境请使用 [PyTorch 官方安装命令](https://pytorch.org/get-started/locally/)。完整说明见[准备工作](guide/preparation.md)。

### 2. 准备数据和权重

四个数据集根目录必须直接包含类别目录：

```text
data/raw/imagenetv2/<0-999>/...
data/raw/imagenet-a/<wnid>/...
data/raw/imagenet-r/<wnid>/...
data/raw/imagenet-sketch/<wnid>/...
```

下载与解压命令见[准备工作](guide/preparation.md#数据集准备)。下载 CLIP 权重：

```bash
mkdir -p models
python - <<'PY'
import clip

for name in ("RN50", "ViT-B/16"):
    clip.load(name, device="cpu", download_root="models")
PY
```

最终应存在 `models/RN50.pt` 和 `models/ViT-B-16.pt`。

### 3. 运行 smoke test

先用一张 GPU、少量样本验证完整 TPT 管线：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml \
  --models ViT-B/16 --datasets imagenet-a \
  --devices cuda:0 --seeds 42 --num-workers 0 \
  --views 8 --selection-fraction 0.25 --limit 8 \
  --output-root /tmp/tpt_smoke
```

成功后会生成 `/tmp/tpt_smoke/seed_42/` 和 `/tmp/tpt_smoke/summary/`。

### 4. 运行实验

当前配置概览：

| 工作流 | 当前范围 | 结果组合数 | 输出目录 |
| --- | --- | ---: | --- |
| Pure CLIP zero-shot | 2 模型 × 4 数据集 × 3 seeds | 24 | `runs/clip_zero_shot/` |
| 标准 TPT | 2 模型 × 4 数据集 × 3 seeds | 24 | `runs/tpt_classification/` |
| 低熵视图 count 消融 | ImageNet-A、ViT-B/16、count 1-10、Only origin、seed 42 | 11 | `runs/tpt_selection_fraction_ablation/idea_count_ablation_zero_shot/` |
| 一步 LR 消融 | ImageNet-A、ViT-B/16、10 个 LR、count 1、seed 42 | 10 | `runs/tpt_lr_ablation/idea_one_step_TTA/` |
| LR × Steps | ImageNet-A、ViT-B/16、7 个 LR × 10 个 steps、count 1、seed 42 | 70 | `runs/tpt_lr_steps_ablation/idea_no_weight_decay/` |

三个消融当前都只有一个 seed，适合探索超参数，不能用标准差判断跨 seed 稳定性。LR × Steps 配置当前实际使用 `weight_decay: 0.05`；`idea_no_weight_decay` 只是已有历史目录名，不代表实验参数。

Pure CLIP zero-shot（安装后提供 `clip-evaluate` 命令）：

```bash
clip-evaluate \
  --config configs/evaluation/clip_zero_shot.yaml
```

仓库脚本 `python scripts/evaluate_clip_zeroshot.py ...` 调用同一实现，参数完全一致。

标准 TPT（64 views、最低熵 10%、Top-1 至 Top-5）：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml
```

低熵视图 count 消融、Only origin、CLIP zero-shot 基线和自动绘图：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_selection_fraction_ablation.yaml
```

一步 TPT 的 learning-rate 消融：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_lr_ablation.yaml
```

Learning rate 与更新步数的二维消融（当前 `weight_decay: 0.05`）：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_lr_steps_ablation.yaml
```

低熵视图消融的结果位于：

```text
runs/tpt_selection_fraction_ablation/idea_count_ablation_zero_shot/summary/
```

图中横轴依次包含 `CLIP zero-shot`、`Only origin` 和 count/fraction 筛选设置。详细配置见[TPT Selection Fraction 消融](guide/tpt_selection_fraction_ablation.md)。

## 评测协议

- 文本类别名称来自 `configs/evaluation/imagenet_classes_tpt.json`。
- ImageNet-A/R 只在各自 200 个候选类别中预测。
- ImageNet-V2/Sketch 使用完整 1000 类。
- TPT 只更新 prompt 上下文 token；每张测试图像结束后重置。
- TPT 同时记录更新后的 Top-k 和首次更新前、原图中心裁剪上的 CLIP zero-shot Top-k。

## 已记录的 Zero-shot Top-1

单模板 `a photo of a {}.` 的结果：

| 模型 | ImageNet-V2 | ImageNet-A | ImageNet-R | ImageNet-Sketch |
| --- | ---: | ---: | ---: | ---: |
| CLIP-RN50 | 51.4600% | 21.8533% | 56.1167% | 33.3589% |
| CLIP-ViT-B/16 | 60.8800% | 47.8267% | 73.9833% | 46.1023% |

## 文档

- [准备工作](guide/preparation.md)：环境、数据、权重和数据可视化。
- [Pure CLIP Zero-shot](guide/clip_test.md)：协议、smoke test、正式运行和输出。
- [TPT 图像分类](guide/tpt_classification.md)：标准 TPT、参数和结果结构。
- [TPT Quick Demo](guide/tpt_quick_demo.md)：用较小任务逐步验证多 GPU TPT。
- [TPT Selection Fraction 消融](guide/tpt_selection_fraction_ablation.md)：fraction/count、Only origin、Top-k、zero-shot 类别和重绘。
- [TPT Learning-rate 消融](guide/tpt_lr_ablation.md)：一步 TPT 的 10 个学习率、自动绘图和结果边界。
- [TPT LR × Steps 消融](guide/tpt_lr_steps_ablation.md)：二维扫描、weight decay 语义、最佳步数和热力图。
- [GitHub 同步](guide/github_sync.md)：只同步代码、配置和文档。

## 目录

```text
configs/   # 评测和绘图配置
data/      # 本地数据，不提交 Git
guide/     # 复现文档
models/    # 本地 CLIP 权重，不提交 Git
runs/      # 本地实验输出，不提交 Git
scripts/   # 评测、绘图和可视化入口
```
