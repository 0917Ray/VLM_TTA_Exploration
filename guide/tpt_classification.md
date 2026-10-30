# TPT 图像分类

本文介绍 Test-Time Prompt Tuning（TPT）在 ImageNet Variants 图像分类任务上的实现与运行方法。实现核对 TPT [官方仓库](https://github.com/azshue/TPT) commit `63ecbace` 的手工 prompt 初始化版本，不包含 CoOp、CoCoOp、跨数据集分类或图文检索。

## 方法简介

TPT 在测试时不更新 CLIP 的图像编码器和文本编码器，只更新 prompt 中的上下文 token。每张测试图像独立执行以下步骤：

1. 生成 1 个标准 CLIP 视图和多个随机裁剪、翻转视图；
2. 计算所有视图的类别预测熵，保留熵最低的一部分高置信度视图；
3. 最小化这些视图平均预测分布的熵，更新 prompt 上下文 token；
4. 使用更新后的 prompt 对标准视图分类；
5. 恢复初始 prompt 和优化器状态，再处理下一张图像。

默认设置与官方 image classification 脚本一致：64 个视图、保留最低熵的 10%、1 次更新、AdamW 学习率 `0.005`，prompt 初始化为 `a photo of a`。

## 评测协议

- 模型权重读取 `models/RN50.pt` 或 `models/ViT-B-16.pt`。
- 类别名称读取 `configs/evaluation/imagenet_classes_tpt.json`。
- ImageNet-A/R 只在各自包含的 200 类中预测。
- ImageNet-V2/Sketch 使用完整 1000 类。
- TPT 包含随机增强，因此不同 seed 可能产生不同结果。

## 运行

先完成 [环境与数据准备](preparation.md)，然后在项目根目录激活环境：

```bash
conda activate clip
```

建议先运行小规模检查：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml \
  --models RN50 --datasets imagenet-a \
  --devices cuda:0 --num-workers 0 \
  --views 8 --selection-fraction 0.25 \
  --limit 8 --output-root /tmp/tpt_smoke
```

使用论文设置运行完整评测：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml
```

为某个实验想法设置独立名称时，可以在 YAML 中填写 `idea`，或通过命令行临时指定：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml \
  --idea entropy_v1
```

此时本次运行的输出根目录为 `runs/tpt_classification/idea_entropy_v1/`。`idea: null` 或空字符串不会增加子目录，输出仍直接写入 `runs/tpt_classification/`。名称中的空白会转换为下划线，且名称不能包含 `/` 或 `\`。

默认配置运行 seeds `11, 42, 123`。每个 seed 完成的任务会立即写入对应的 `seed_<seed>/` 目录；全部模型、数据集和 seed 任务完成后，才会生成包含全部明细和跨 seed 均值的 `summary/`。也可以临时指定 seed，例如 `--seeds 10 11 12`。

默认配置同时评测 RN50、ViT-B/16 和四个数据集，使用 `cuda:0`、`cuda:1`、`cuda:2`、`cuda:3` 调度任务；每张 GPU 同时处理一个任务。只有一张 GPU 时可以覆盖设备列表：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml \
  --devices cuda:0
```

测试 ViT-B/16 或同时运行两个模型：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml \
  --models RN50 ViT-B/16
```

## 配置

默认配置位于 `configs/evaluation/tpt_classification.yaml`，相对路径均相对于项目根目录解析。当前配置的全部参数如下：

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `data_root` | `data/raw` | 数据集根目录；任务从其下读取对应的 ImageNet Variant 子目录。 |
| `class_index` | `configs/visualization/imagenet_class_index.json` | 1000 类 ImageNet 索引与 WNID 映射，用于把数据集目录映射回原始类别索引。 |
| `class_names` | `configs/evaluation/imagenet_classes_tpt.json` | TPT prompt 使用的 1000 个类别名称，也是 ImageNet-A/R 候选类别子集的名称来源。 |
| `model_root` | `models` | checkpoint 根目录；模型名分别映射到 `RN50.pt` 和 `ViT-B-16.pt`。 |
| `output_root` | `runs/tpt_classification` | 完整实验的基础输出目录；`idea` 有值时还会在其下增加一层 idea 目录。 |
| `idea` | `null` | 可选实验标识。非空时输出到 `output_root/idea_<idea>/` 并在结果中记录；空白会转为下划线，不能包含路径分隔符。`null` 表示不增加该层。 |
| `datasets` | 四个 ImageNet Variants | 要评测的数据集列表，只允许 `imagenetv2`、`imagenet-a`、`imagenet-r`、`imagenet-sketch`。 |
| `models` | `RN50`、`ViT-B/16` | 要评测的模型列表。它与数据集和 seed 做笛卡尔积，因此当前完整配置共有 `2 x 4 x 3 = 24` 个任务。 |
| `devices` | `cuda:0` 至 `cuda:3` | 参与任务调度的设备。每张 GPU 同时运行一个独立任务，剩余任务按轮转分配后排队；列表不能为空或重复。 |
| `seeds` | `11`、`42`、`123` | 控制每张图片随机裁剪和翻转的随机种子。每个 seed 独立运行，并写入对应的 `seed_<seed>/`；列表不能为空或重复。 |
| `num_workers` | `4` | 每个任务的 DataLoader 子进程数。四张 GPU 同时工作时最多约有 `4 x 4` 个加载进程；`0` 表示在任务进程内加载。 |
| `views` | `64` | 每张图片生成的总视图数，包含 1 个标准中心裁剪和 `63` 个随机增强视图，必须至少为 2。 |
| `selection_fraction` | `0.1` | 按预测熵从低到高保留的视图比例。脚本使用 `int(views x selection_fraction)`，当前保留 `6` 个视图；取值范围为 `(0, 1]` 且至少选中 1 个视图。 |
| `tta_steps` | `1` | 每张图片上执行的 prompt 参数更新次数。增加它会近似成比例增加适配计算量，必须大于 0。 |
| `learning_rate` | `0.005` | 每张图片临时创建的 AdamW 优化器学习率，必须大于 0。 |
| `prompt_init` | `a photo of a` | 可学习上下文 token 的初始化文本。脚本会在其后添加类别名和句点，必须是非空字符串。 |
| `resolution` | `224` | 标准视图和随机视图的最终输入边长。必须与 checkpoint 声明的输入分辨率一致，否则任务直接报错。 |
| `print_frequency` | `100` | 每处理多少张图片向任务日志写一次阶段性 Top-1；最后一张总会记录。它不改变终端进度条、计算或结果。 |
| `limit` | `null` | 每个数据集最多评测的前 N 张图片。`null` 表示完整数据集；只能改为正整数。 |
| `log_file` | `run.log` | 日志基础名称。脚本在实验根目录创建主 `run_<timestamp>.log`，并在每个 seed 目录创建相同命名模式的任务日志。 |

除 `log_file` 外，各字段都可用同名命令行参数临时覆盖（下划线改为连字符，例如 `--selection-fraction`），不会修改 YAML。

### `num_workers` 如何影响 TPT

`num_workers` 是每个独立 `(模型, 数据集, seed)` 任务的 PyTorch DataLoader 子进程数，只负责 CPU 数据管线，不会并行模型的前向、反向或 prompt 更新。每个 worker 负责读取和解码图片，并为一张图片执行以下预处理：

1. 生成 1 个 resize 后中心裁剪的标准视图；
2. 生成 `views - 1` 个随机裁剪、随机翻转视图；
3. 将全部视图转换成 tensor 并执行 CLIP 归一化。

当前 `views: 64` 时，一个样本就包含 64 个 `3 x 224 x 224` 的 float32 tensor，纯像素 tensor 约为 36.8 MiB。TPT 的 DataLoader 固定 `batch_size=1`，因此这里的一个 batch 就是一张原图及其 64 个视图，而不是 64 张原图。

`num_workers: 0` 表示由任务主进程同步执行上述操作，进程开销最低，但 GPU 可能等待下一张图片。大于 0 时，脚本使用 `spawn` 创建 worker，启用 persistent workers，并在 CUDA 上启用 pinned memory 与异步传输。PyTorch 默认每个 worker 预取 2 个 batch；例如一个任务设置 4 个 workers，可能提前准备最多约 8 个 64-view 样本。实际内存还包含解码图片、队列和进程本身，因此不能只按 tensor 大小估算。

该参数按任务而非整次 run 计算。若 `J` 个 GPU 任务同时运行、每个任务设置 `W` 个 workers，则活跃 DataLoader 子进程约为：

```text
总 DataLoader workers = J x W
默认最多预取的 batch 数约为 2 x J x W
```

完整配置四张 GPU 同时运行且 `num_workers: 4` 时，约有 16 个 DataLoader workers，最多可能预取约 32 张原图对应的 64-view batch。增加 worker 可以把图片解码和随机增强与 GPU 计算重叠，但过多时会争抢 CPU 调度、磁盘、内存带宽、pinned memory 和 NUMA 资源。于是 GPU 数量或 worker 数量增加后，单任务反而可能变慢；`num_workers` 是需要实测的吞吐量参数，并非越大越好。

建议先用固定的 `--limit` 分别测试 `--num-workers 0`、`2`、`4`、`8`，比较整次 run 的墙钟时间，而不只看某个任务的耗时。多 GPU 时还要比较总吞吐量。若 GPU 利用率周期性降到很低且 CPU 有余量，可以增加 worker；若 CPU、磁盘或内存带宽繁忙，或并行 GPU 增加后任务明显变慢，应降低 worker。随机增强由 worker 执行，改变 worker 数可能改变同一 seed 下随机数被样本消费的顺序，因此正式比较不同 idea 时应固定 `num_workers`。

`views` 表示测试时增强数量，不是普通的数据加载 batch size。增加 `views`、`tta_steps` 或候选类别数量会明显增加运行时间和显存占用。

`resolution: 224` 对标准视图和随机增强视图同时生效。标准视图先使用 Bicubic 插值保持宽高比，将短边缩放到 224，再中心裁剪为 `224 x 224`；随机视图通过 `RandomResizedCrop(224)` 得到相同尺寸。脚本会核对 checkpoint 声明的输入分辨率，配置与模型不一致时直接报错。

`print_frequency: 100` 只控制文件日志中的阶段性进度记录。每处理 100 张图片，日志写入一次已处理数量和当前 Top-1；数据集最后一张图片无论是否达到 100 的倍数都会记录。终端进度条仍实时更新，此参数不影响模型计算、样本数量或最终准确率。

## 输出

以设置 `idea: entropy_v1` 为例，本次完整实验的输出结构为：

```text
runs/tpt_classification/idea_entropy_v1/
  run_<timestamp>.log
  seed_11/
    results.csv
    results.json
    run_<timestamp>.log
  seed_42/
    results.csv
    results.json
    run_<timestamp>.log
  seed_123/
    results.csv
    results.json
    run_<timestamp>.log
  summary/
    results.csv
    results.json
    average_results.csv
    average_results.json
    average_results.md
```

`idea: null` 或空字符串时不会创建 `idea_<idea>/` 这一层，上述 `seed_11/`、`seed_42/`、`seed_123/` 和 `summary/` 会直接位于 `runs/tpt_classification/` 下。

每个独立任务 `(模型, 数据集, seed)` 完成后，脚本立即更新对应 `seed_<seed>/results.csv` 和 `results.json`。完整配置中每个 seed 包含 2 个模型和 4 个数据集，因此全部完成后，每个 seed 的结果文件包含 8 条任务明细。即使运行中途停止，已经完成并写入的任务结果仍保留在对应 seed 目录中。

全部 24 个任务成功完成后，脚本才生成 `summary/`：

- `summary/results.csv` 和 `results.json` 保存 24 条任务明细。
- `average_results.csv` 和 `average_results.json` 按模型和数据集汇总三个 seeds 的 Top-1、Top-5、总体标准差与平均耗时。
- `average_results.md` 以表格展示不同模型、数据集的平均 Top-1、总体标准差和平均耗时。

每个 seed 的 warning 和运行进度写入该 seed 的 `run_<timestamp>.log`。普通 Python warning 不显示在终端。

运行时每张 GPU 显示独立进度条，并实时更新当前 Top-1 准确率。

结果 CSV/JSON 和跨 seed 汇总文件都会记录 `idea` 字段，便于结果文件离开原目录后仍能识别所属实验。
