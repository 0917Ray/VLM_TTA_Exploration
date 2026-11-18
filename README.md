# VLM CLIP Adaptation Experiments

本仓库复现 OpenAI CLIP zero-shot 分类和 Test-Time Prompt Tuning（TPT），并分析低熵视图筛选、学习率、更新步数以及逐样本学习率选择对结果的影响。

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

其他 CUDA 或 CPU 环境请使用 [PyTorch 官方安装命令](https://pytorch.org/get-started/locally/)。完整说明见[准备工作](guide/0_preparation.md)。

### 2. 准备数据和权重

四个数据集根目录必须直接包含类别目录：

```text
data/raw/imagenetv2/<0-999>/...
data/raw/imagenet-a/<wnid>/...
data/raw/imagenet-r/<wnid>/...
data/raw/imagenet-sketch/<wnid>/...
```

下载与解压命令见[准备工作](guide/0_preparation.md#2-数据集准备)。下载 CLIP 权重：

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
python scripts/2_evaluate_tpt.py \
  --config configs/evaluation/2_tpt_classification.yaml \
  --models ViT-B/16 --datasets imagenet-a \
  --devices cuda:0 --seeds 42 --num-workers 0 \
  --views 8 --selection-fraction 0.25 --limit 8 \
  --output-root /tmp/tpt_smoke
```

成功后会生成 `/tmp/tpt_smoke/seed_42/` 和 `/tmp/tpt_smoke/summary/`。

### 4. 运行实验

工作流映射如下；实际模型、数据集、seeds 和实验网格始终以对应 YAML 与运行生成的 `resolved_config.yaml` 为准。

| 工作流 | 配置 | 执行入口 | 输出根目录 |
| --- | --- | --- | --- |
| Pure CLIP zero-shot | `configs/evaluation/1_clip_zero_shot.yaml` | `clip-evaluate` | `runs/1_clip_zero_shot/` |
| 标准 TPT | `configs/evaluation/2_tpt_classification.yaml` | `scripts/2_evaluate_tpt.py` | `runs/2_tpt_classification/` |
| 低熵视图消融 | `configs/evaluation/3_tpt_selection_fraction_ablation.yaml` | `scripts/3_evaluate_tpt_selection_ablation.py` | `runs/3_tpt_selection_fraction_ablation/` |
| 一步 LR 消融 | `configs/evaluation/4_tpt_lr_ablation.yaml` | `scripts/4_evaluate_tpt_lr_ablation.py` | `runs/4_tpt_lr_ablation/` |
| LR × Steps 消融 | `configs/evaluation/5_tpt_lr_steps_ablation.yaml` | `scripts/5_evaluate_tpt_lr_steps_ablation.py` | `runs/5_tpt_lr_steps_ablation/` |
| Data-wise LR TPT | `configs/evaluation/6_datawise_lr_tpt.yaml` | `scripts/6_evaluate_datawise_lr_tpt.py` | `runs/6_datawise_lr_tpt/` |
| TPT 状态转移统计 | `configs/evaluation/7_tpt_transition_stats.yaml` | `scripts/7_evaluate_tpt_transition_stats.py` | `runs/7_tpt_transition_stats/` |
| TPT 梯度对齐 | `configs/evaluation/8_tpt_gradient_alignment.yaml` | `scripts/8_evaluate_tpt_gradient_alignment.py` | `runs/8_tpt_gradient_alignment/` |

Pure CLIP zero-shot（安装后提供 `clip-evaluate` 命令）：

```bash
clip-evaluate \
  --config configs/evaluation/1_clip_zero_shot.yaml
```

仓库脚本 `python scripts/1_evaluate_clip_zeroshot.py ...` 调用同一实现，参数完全一致。

标准 TPT（64 views、最低熵 10%、Top-1 至 Top-5）：

```bash
python scripts/2_evaluate_tpt.py \
  --config configs/evaluation/2_tpt_classification.yaml
```

低熵视图 count 消融、Only origin、CLIP zero-shot 基线和自动绘图：

```bash
python scripts/3_evaluate_tpt_selection_ablation.py \
  --config configs/evaluation/3_tpt_selection_fraction_ablation.yaml
```

一步 TPT 的 learning-rate 消融（10 个 LR，`weight_decay: 0.01`）：

```bash
python scripts/4_evaluate_tpt_lr_ablation.py \
  --config configs/evaluation/4_tpt_lr_ablation.yaml
```

Learning rate 与更新步数的二维消融（7 个 LR × 10 个步数，`weight_decay: 0.05`）：

```bash
python scripts/5_evaluate_tpt_lr_steps_ablation.py \
  --config configs/evaluation/5_tpt_lr_steps_ablation.yaml
```

逐样本遍历候选 LR、按 post-update entropy 选择并绘制 LR 比例与正确率：

```bash
python scripts/6_evaluate_datawise_lr_tpt.py \
  --config configs/evaluation/6_datawise_lr_tpt.yaml
```

Data-wise LR TPT 将一个完整任务定义为 `seed × model × dataset × selection setting × step setting`，每张 GPU 同时最多运行一个完整任务。每个 seed 会生成独立 summary 和图表，顶层 summary 同时保存逐 seed 结果与跨 seed 平均值；完整说明见 [Data-wise LR TPT](guide/6_datawise_lr_tpt.md)。

统计 TPT 前后的正确/错误转移，以及 entropy 与真实标签 cross-entropy 的升降组合：

```bash
python scripts/7_evaluate_tpt_transition_stats.py \
  --config configs/evaluation/7_tpt_transition_stats.yaml
```

默认运行 ImageNet-A、ViT-B/16、seed 123 和 LR 0.003，并自动生成两张四分类柱状图。多数据集、多 LR 和多 seed 会展开为独立完整任务并轮询分配到配置的 GPU；协议、smoke test 和输出说明见 [TPT 状态转移统计](guide/7_tpt_transition_stats.md)。

比较 one-step TPT entropy 与监督 cross-entropy 的 raw gradient 和 AdamW 实际参数更新：

```bash
python scripts/8_evaluate_tpt_gradient_alignment.py \
  --config configs/evaluation/8_tpt_gradient_alignment.yaml
```

默认设置与状态转移统计一致，单任务运行 ImageNet-A、ViT-B/16、seed 123、LR 0.003；分别导出 raw/update norm、cosine、内积分布及按初始正确/错误分组的统计和图。实验定义、smoke test 与结果解释见 [TPT 梯度对齐](guide/8_tpt_gradient_alignment.md)。

低熵视图消融的结果位于：

```text
runs/3_tpt_selection_fraction_ablation/idea_count_ablation_zero_shot/summary/
```

图中横轴依次包含 `CLIP zero-shot`、`Only origin` 和 count/fraction 筛选设置。详细配置见[TPT Selection Fraction 消融](guide/3_tpt_selection_fraction_ablation.md)。

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

## 已验证的 Data-wise LR TPT 结果

在 ImageNet-A、ViT-B/16、64 views、最低 entropy 的 1 个 view、one-step AdamW、`weight_decay=0.01` 和 seeds `42, 123` 下，逐样本在 9 个候选 LR 中按 post-update entropy 选择，得到：

| 方法 | Top-1 | Top-5 | Seeds | 不确定性 |
| --- | ---: | ---: | ---: | --- |
| CLIP zero-shot | 47.8800% | 78.1600% | 2 | 两个 seed 相同 |
| Data-wise LR TPT | 55.5000% | 81.1333% | 2 | sample standard deviation：Top-1 `0.2734 pp`，Top-5 `0.0754 pp` |

该结果只描述上述已完成协议，不代表其他模型、数据集、selection 设置或多步设置。测试标签不参与 LR 选择。

## 文档

- [准备工作](guide/0_preparation.md)：环境、数据、权重和数据可视化。
- [Pure CLIP Zero-shot](guide/1_clip_zero_test.md)：协议、smoke test、正式运行和输出。
- [TPT 图像分类](guide/2_tpt_classification.md)：标准 TPT、参数和结果结构。
- [TPT Quick Demo](guide/2_tpt_quick_demo.md)：用较小任务逐步验证多 GPU TPT。
- [TPT Selection Fraction 消融](guide/3_tpt_selection_fraction_ablation.md)：fraction/count、Only origin、Top-k、zero-shot 类别和重绘。
- [TPT Learning-rate 消融](guide/4_tpt_lr_ablation.md)：固定一步更新，单独扫描 LR 并生成准确率曲线和柱状图。
- [TPT LR × Steps 消融](guide/5_tpt_lr_steps_ablation.md)：联合扫描 LR 与更新步数，生成逐 LR 曲线、最佳步数和热力图。
- [Data-wise LR TPT](guide/6_datawise_lr_tpt.md)：逐样本 entropy LR 选择、多步轨迹、逐样本记录、LR 比例与正确率绘图。
- [TPT 状态转移统计](guide/7_tpt_transition_stats.md)：一步 TPT 前后正确性转移、entropy/监督 CE 转移、多 GPU 调度和柱状图重绘。
- [TPT 梯度对齐](guide/8_tpt_gradient_alignment.md)：raw gradient 与 AdamW 实际 update 的 T/C 对齐、分组统计和分布图。
- [GitHub 同步](guide/github_sync.md)：只同步代码、配置和文档。

## 目录

```text
configs/   # 评测和绘图配置
data/      # 本地数据，不提交 Git
guide/     # 复现文档
models/    # 本地 CLIP 权重，不提交 Git
runs/      # 本地实验输出，不提交 Git
scripts/   # 评测、绘图和可视化入口
tests/     # 关键配置、调度、汇总和数值一致性测试
```
