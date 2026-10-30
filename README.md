# VLM Test-Time Adaptation Exploration

用于探索视觉语言模型测试时自适应方法的实验项目。

## 当前阶段

项目目前包含数据集准备与可视化、Pure CLIP zero-shot 评测，以及 TPT 在四个 ImageNet Variants 上的图像分类实现。评测支持 YAML 配置、多 GPU 调度和结果汇总。

原始数据、下载压缩包、模型权重和生成结果只保留在本地，不提交到 GitHub。

## Zero-shot 结果

Pure CLIP zero-shot Top-1 准确率如下：

| 模型 | ImageNet-V2 | ImageNet-A | ImageNet-R | ImageNet-Sketch |
| --- | ---: | ---: | ---: | ---: |
| CLIP-RN50 | 51.4600% | 21.8533% | 56.1167% | 33.3589% |
| CLIP-ViT-B/16 | 60.8800% | 47.8267% | 73.9833% | 46.1023% |

评测使用单模板 `a photo of a {}.` 和 TPT 的 ImageNet 类别名称。ImageNet-A/R
按官方协议在各自 200 类候选集合中预测，ImageNet-V2/Sketch 使用完整 1000 类。
完整设置与输出格式见 [Pure CLIP Zero-shot 测试](guide/clip_test.md)。

## 快速开始

首次使用时，按照[环境准备](guide/preparation.md#环境准备)创建 `clip` 环境并安装依赖。以下命令均在项目根目录执行：

### 1. 数据集准备与可视化

按照[准备工作](guide/preparation.md)下载并解压数据集。数据统一放在 `data/raw/`：

```text
data/raw/imagenet/
data/raw/imagenetv2/
data/raw/imagenet-a/
data/raw/imagenet-r/
data/raw/imagenet-sketch/
```

激活环境并生成数据集样例画廊：

```bash
conda activate clip
python scripts/visualize_datasets.py \
  --config configs/visualization/datasets.yaml
```

结果保存在 `runs/dataset_visualizations/<dataset>/`，每个数据集包含 PNG 和 PDF 画廊。

### 2. Pure CLIP Zero-shot 测试

评测需要以下权重：

```text
models/RN50.pt
models/ViT-B-16.pt
```

若本地没有权重，可通过 OpenAI CLIP 下载：

```bash
mkdir -p models
python - <<'PY'
import clip

for model_name in ("RN50", "ViT-B/16"):
    clip.load(model_name, device="cpu", download_root="models")
PY
```

运行两个模型在 ImageNet-V2、ImageNet-A、ImageNet-R 和 ImageNet-Sketch 上的完整评测：

```bash
python scripts/evaluate_clip_zeroshot.py \
  --config configs/evaluation/clip_zero_shot.yaml
```

默认配置使用 4 张 GPU；设备数量、模型、数据集和 batch size 均可在配置文件中调整。结果保存在 `runs/clip_zero_shot/`，其中 `results.csv` 是逐项记录，`average_results.md` 是汇总结果。

ImageNet-A/R 按官方协议使用各自的 200 类候选集合，ImageNet-V2/Sketch 使用完整 1000 类。详细说明见 [Pure CLIP Zero-shot 测试](guide/clip_test.md)。

## 项目结构

```text
configs/evaluation/      # Zero-shot 与 TPT 评测配置
configs/visualization/   # 可视化配置与 ImageNet 类别索引
data/downloads/          # 数据集压缩包，不纳入 Git
data/raw/                # 解压后的数据集，不纳入 Git
guide/                   # 环境、数据、评测和项目同步说明
models/                  # 模型权重，不纳入 Git
runs/                    # 可视化、评测结果和日志，不纳入 Git
scripts/                 # 可执行脚本
```

## 使用指南

- [准备工作](guide/preparation.md)：环境安装、数据集下载与解压、目录要求和可视化。
- [Pure CLIP Zero-shot 测试](guide/clip_test.md)：评测协议、配置、多 GPU 和输出说明。
- [TPT 图像分类](guide/tpt_classification.md)：方法简介、论文设置、运行命令和输出说明。
- [GitHub 同步教程](guide/github_sync.md)：代码提交与远程同步流程。
