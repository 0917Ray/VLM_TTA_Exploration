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

## 配置参数

`configs/evaluation/clip_zero_shot.yaml` 中的相对路径均相对于项目根目录解析。当前配置的全部参数如下：

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `data_root` | `data/raw` | 数据集根目录；每个任务读取其下的对应数据集子目录。 |
| `class_index` | `configs/visualization/imagenet_class_index.json` | ImageNet 索引与 WNID 的映射，用于从数据集目录名恢复原始类别索引。文件必须包含 1000 类。 |
| `class_names` | `configs/evaluation/imagenet_classes_tpt.json` | 构造 zero-shot 文本 prompt 时使用的 1000 个类别名称；当前文件采用 TPT 类别名称。 |
| `model_root` | `models` | checkpoint 根目录；`RN50` 和 `ViT-B/16` 分别映射到其中的 `RN50.pt` 和 `ViT-B-16.pt`。 |
| `output_root` | `runs/clip_zero_shot` | 结果、跨 seed 汇总和带时间戳日志的写入目录。 |
| `datasets` | 四个 ImageNet Variants | 要评测的数据集列表。每个模型与每个数据集组成一个调度任务；只允许 `imagenetv2`、`imagenet-a`、`imagenet-r`、`imagenet-sketch`。 |
| `models` | `RN50`、`ViT-B/16` | 要加载和评测的 CLIP 模型列表。当前配置与四个数据集组合成 `2 x 4 = 8` 个任务。 |
| `batch_size` | `512` | 每个并行任务单次送入模型的图片数。它分别作用于每张正在运行的 GPU，不是所有 GPU 共用的总 batch size；调高会增加单卡显存占用。 |
| `num_workers` | `12` | 每个并行任务的 DataLoader 子进程数。四个任务同时运行时最多会创建约 `4 x 12` 个加载进程，应结合 CPU 核数和存储速度设置；`0` 表示在任务进程内加载。 |
| `devices` | `cuda:0` 至 `cuda:3` | 可用于任务调度的设备列表。每个设备同一时刻运行一个模型/数据集任务，任务数超过设备数时在该设备上继续排队；设备不能重复。 |
| `prompt_template` | `a photo of a {}.` | 将类别名称插入文本 prompt 的 Python 格式字符串。必须保留一个可由 `format(class_name)` 替换的 `{}`，否则类别名不会进入 prompt 或会触发格式化错误。 |
| `seeds` | `42`、`43`、`44` | 每个模型/数据集任务内依次运行的 seed，并为每个 seed 保存一条明细。当前 zero-shot 流程没有随机增强，因此这些 seed 通常得到相同准确率，但可统一多 seed 实验格式。列表不能为空且不能重复。 |
| `limit` | `null` | 每个数据集最多评测的前 N 张图片。`null` 表示使用完整数据集；正整数适合 smoke test。 |
| `log_file` | `run.log` | 日志基础文件名。相对名称会写入 `output_root`，并自动变成 `run_<timestamp>.log` 以避免不同运行互相覆盖。 |

除 `log_file` 外，上述字段都有同名的命令行覆盖参数（YAML 下划线在命令行中写作连字符，例如 `batch_size` 对应 `--batch-size`）。任务总数由 `len(models) x len(datasets)` 决定；每个任务会对 `seeds` 中的所有值依次评测。

### `num_workers` 如何影响速度

`num_workers` 是每个评测任务的 PyTorch DataLoader 子进程数，不是 GPU 数量，也不会让模型前向传播分散到多个进程。worker 在 CPU 上负责：

1. 从磁盘读取图片并用 Pillow 解码为 RGB；
2. 执行 CLIP 的 resize、crop、归一化等预处理；
3. 把多张图片整理为一个 batch，提前交给主任务进程。

当 `num_workers: 0` 时，这些工作全部由运行模型的任务进程同步完成。GPU 必须等待下一批图片准备好，优点是进程开销小且最容易排查报错。当值大于 0 时，脚本使用 `spawn` 创建子进程，并行准备后续 batch；CUDA 任务还启用 pinned memory 和异步拷贝，使 CPU 预处理与当前 GPU 前向传播尽量重叠。脚本将 worker 保持到当前 DataLoader 结束，但不会让它们跨不同模型/数据集任务复用。

该值对每个并行任务分别生效。若同时有 `J` 个任务在不同 GPU 上运行，每个任务设置 `W` 个 worker，则活跃 DataLoader 子进程约为：

```text
总 DataLoader workers = J x W
```

例如四张 GPU 各运行一个任务且 `num_workers: 12` 时，最多约有 48 个加载子进程。PyTorch 默认让每个 worker 预取 2 个 batch，因此还可能同时准备约 `2 x J x W` 个 batch。增加 worker 能缓解图片读取或 CPU 预处理瓶颈，但过多 worker 会增加进程调度、内存、文件系统、NUMA 和内存带宽竞争，反而让每个任务变慢。

调参时固定模型、数据集、`batch_size` 和 `limit`，分别测试 `0`、`2`、`4`、`8` 等值，并比较完整命令的墙钟时间和 GPU 利用率。GPU 经常空闲而 CPU 尚有余量时可增加；CPU 使用率高、磁盘繁忙、上下文切换多，或增加 GPU 后单任务明显变慢时应降低。最终比较不同方法时应固定该值，避免把数据管线差异混入耗时结果。

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
