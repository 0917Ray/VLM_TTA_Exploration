# Pure CLIP Zero-shot 准确率测试

本文使用冻结的 OpenAI CLIP，对已下载的四个 ImageNet Variants 进行 zero-shot Top-1 准确率测试。测试模型为：

- CLIP-RN50：`models/RN50.pt`
- CLIP-ViT-B/16：`models/ViT-B-16.pt`

评测不进行训练、微调或测试时自适应。类别名称使用 TPT 的 ImageNet 类别表；ImageNet-A/R 按官方协议在各自 200 类中预测，ImageNet-V2/Sketch 在完整 1000 类中预测。

## 特别注意事项

- ImageNet-A/R 按标准协议只在数据集各自包含的 200 类中预测。其余 800 类没有对应测试图片，不应作为候选类别。
- 这两组 200 类由 ImageNet-A/R 数据集作者确定，早于 CLIP 和 TPT，并非 TPT 根据准确率筛选。
- 类别名称使用 TPT 的 1000 类名称表。该名称表与旧版 ImageNet synset 名称有 363 项不同，会直接影响文本特征和分类结果。
- ImageNet-V2 和 ImageNet-Sketch 仍使用完整的 1000 类候选集合。

## 运行环境

先完成 [数据准备](preparation.md) 和 `clip` 环境安装，然后在项目根目录执行：

```bash
conda activate clip
```

脚本默认读取：

```text
data/raw/imagenetv2/
data/raw/imagenet-a/
data/raw/imagenet-r/
data/raw/imagenet-sketch/
models/RN50.pt
models/ViT-B-16.pt
```

## 完整测试

默认配置位于 `configs/evaluation/clip_zero_shot.yaml`，控制数据路径、模型、数据集、GPU、多个 seed、batch size、prompt 和输出目录。运行两个模型和四个数据集：

```bash
python scripts/evaluate_clip_zeroshot.py \
  --config configs/evaluation/clip_zero_shot.yaml
```

命令行参数可以临时覆盖配置文件，例如：

```bash
python scripts/evaluate_clip_zeroshot.py \
  --config configs/evaluation/clip_zero_shot.yaml \
  --models RN50 --datasets imagenet-a imagenet-r \
  --devices cuda:0 cuda:1 --seeds 1 2 3 --batch-size 256
```

脚本中的模型参数名对应如下：

| 参数 | 权重 | 含义 |
| --- | --- | --- |
| `RN50` | `models/RN50.pt` | CLIP-RN50 |
| `ViT-B/16` | `models/ViT-B-16.pt` | CLIP-ViT-B/16 |

只测试一个模型或一个数据集时，减少 `--models` 或 `--datasets` 的值即可。例如：

```bash
python scripts/evaluate_clip_zeroshot.py \
  --config configs/evaluation/clip_zero_shot.yaml \
  --models RN50 --datasets imagenet-a
```

## 评测设置

- `devices` 指定参与调度的设备。每张 GPU 同时只处理一个 `(model, dataset)` 任务，完成后继续处理分配给它的下一个任务。
- 类别名称读取 `configs/evaluation/imagenet_classes_tpt.json`，来源固定为 TPT 官方仓库 commit `63ecbace`。
- 文本模板固定为 `a photo of a {}.`；可用 `--prompt-template` 覆盖。
- ImageNet-A 和 ImageNet-R 必须各包含 200 个类别，脚本只在对应 200 类中计算 logits；类别数不完整时会停止并报错。
- ImageNet-V2 和 ImageNet-Sketch 使用完整 1000 类候选集合。
- ImageNet-V2 的数字目录直接对应类别索引；其他三个数据集使用 WNID 映射。
- 图像预处理使用 OpenAI CLIP 自带的 `preprocess`。
- `seeds` 接受一个或多个整数；每个 seed 都会独立运行并记录结果。命令行可用 `--seeds 1 2 3` 覆盖。
- `batch_size` 和 `num_workers` 对每个并行任务分别生效；增加 GPU 数量时应同时考虑总显存和 CPU worker 数量。
- 可用 `--limit N` 只评测每个数据集的前 `N` 张图，用于快速检查命令。

在配置文件中设置多个 seed：

```yaml
seeds:
  - 42
  - 43
  - 44
```

当前图像预处理和 zero-shot 推理没有随机增强，因此不同 seed 通常会得到相同结果。保留多 seed 记录可以统一后续带随机性的实验流程。

在配置文件中设置多张 GPU：

```yaml
devices:
  - cuda:0
  - cuda:1
  - cuda:2
  - cuda:3
```

共有 2 个模型和 4 个数据集，因此会产生 8 个任务。使用 4 张 GPU 时先并行运行 4 个任务，每张卡完成后继续运行下一个任务。命令行可用 `--devices cuda:0 cuda:1` 临时覆盖。

脚本会在每个模型和数据集完成后打印样本数、正确数和 Top-1 准确率，并持续更新：

```text
runs/clip_zero_shot/results.csv
runs/clip_zero_shot/results.json
runs/clip_zero_shot/average_results.csv
runs/clip_zero_shot/average_results.json
runs/clip_zero_shot/average_results.md
runs/clip_zero_shot/run_<timestamp>.log
```

`results.csv` 和 `results.json` 按 seed 保存明细，并记录每个任务实际使用的 GPU、候选类别数和类别名称来源。`average_results.*` 按模型和数据集保存平均 Top-1 准确率、总体标准差、最小值和最大值，其中 Markdown 文件提供可直接阅读的汇总表。

每次运行都会生成独立日志，记录参数、每个 seed 的结果和 Python warning。诸如 CLIP 依赖发出的 `pkg_resources` 弃用提示只写入该日志，不打印到终端。
