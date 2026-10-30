# TPT Quick Demo

本文说明如何使用 ImageNet-A 快速迭代当前 TPT 实现。Quick Demo 直接复用 `scripts/evaluate_tpt.py`，不维护另一套评测代码；它只通过独立配置缩小实验规模，避免快速实验和完整评测出现实现差异。

专用配置位于：

```text
configs/evaluation/tpt_quick_demo.yaml
```

当前配置中的 `idea` 为 `init_test`，因此默认结果写入 `runs/tpt_quick_demo/idea_init_test/`。修改该值后，脚本会自动创建新的 `idea_<名称>/`，无需手工拼接 `output_root`。

## 设计目标

完整配置包含 2 个模型、4 个数据集和 3 个 seeds，共 24 个任务。当前 Quick Demo 启用 ViT-B/16、ImageNet-A 和两个 seeds：

```text
1 个模型 x 1 个数据集 x 2 个 seeds = 2 个任务
```

ImageNet-A 本地数据包含 200 个类别和 7500 张图片。Quick Demo 默认评测全部 7500 张图片，但保留正式配置中的核心 TPT 参数：

- 64 个视图，包含 1 个标准视图和 63 个随机视图。
- 保留预测熵最低的 10% 视图，实际为 6 个视图。
- 每张图片执行 1 次 prompt 更新。
- AdamW 学习率为 `0.005`。
- prompt 初始化为 `a photo of a`。
- 当前测试 ViT-B/16；YAML 中保留了已注释的 RN50 条目，需要时可取消注释。
- 当前使用两个 seeds：`11`、`42`；YAML 中的 `123` 目前被注释。
- 当前使用 `cuda:0`；`cuda:1` 至 `cuda:3` 作为可选设备保留在注释中。

因此，当前默认 Quick Demo 适合先在 ViT-B/16 上快速比较 idea，而不仅是检查程序能否启动。它减少了数据集、模型和 seed 维度，但没有为了提速而改变 TPT 的主要超参数。需要验证跨架构效果时，取消 RN50 的注释；需要更稳健的波动估计时，再启用 seed `123`。

## 运行默认 Quick Demo

先在项目根目录激活环境：

```bash
conda activate clip
```

然后运行：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml
```

默认只使用 `cuda:0`，两个 seed 任务会在该 GPU 上串行运行。若取消 YAML 中其他设备的注释，脚本会把任务轮流分配到多张 GPU；每张 GPU 同一时刻只运行一个任务。实际参与调度的设备数不会超过任务数。

需要改用部分 GPU 时，不必修改配置文件，可以临时覆盖。例如只使用 `cuda:1` 和 `cuda:2`：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --devices cuda:1 cuda:2
```

也可以在命令行明确指定单张 GPU，当前两个任务会在该 GPU 上串行运行：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --devices cuda:0
```

## 超短 Smoke Test

修改代码后，建议先用少量图片检查数据加载、前向传播、prompt 更新和结果写入是否正常：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --num-workers 0 \
  --views 8 \
  --selection-fraction 0.25 \
  --limit 32 \
  --output-root /tmp/tpt_quick_smoke
```

该命令只处理 32 张图片，每张图片使用 8 个视图，并选择其中熵最低的 2 个视图。它适合排查运行错误，但因为样本数和视图参数都发生了变化，准确率不能用于判断 idea 是否优于 baseline。

Smoke Test 通过后，再运行默认 Quick Demo，获得 ViT-B/16 在完整 ImageNet-A 和两个 seeds 上的结果。

## 推荐迭代流程

每个 idea 建议按以下顺序实验：

1. 使用超短 Smoke Test 检查实现是否可以完整运行。
2. 使用默认 Quick Demo 在完整 ImageNet-A 上比较 ViT-B/16、两个 seeds 的结果。
3. 记录 baseline 与新方法的 Top-1、Top-5、跨 seed 波动和耗时。
4. 只有 Quick Demo 表现有改善时，再运行完整的 `tpt_classification.yaml`。

完整评测命令为：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml
```

## 为每个 Idea 使用独立运行目录

Quick Demo 的基础输出目录为：

```text
runs/tpt_quick_demo
```

由于当前 `idea: init_test`，实际默认输出目录是 `runs/tpt_quick_demo/idea_init_test/`。

比较多个 idea 时，应为每次实验设置独立的 `idea`。可以直接修改 YAML：

```yaml
idea: entropy_v1
```

也可以通过命令行覆盖：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --idea entropy_v1
```

该命令会自动使用 `runs/tpt_quick_demo/idea_entropy_v1/`。名称中的空白会转换为下划线，且名称不能包含 `/` 或 `\`。

例如可以采用以下目录结构：

```text
runs/tpt_quick_demo/
  idea_baseline/
  idea_entropy_v1/
  idea_entropy_v2/
```

运行 baseline 时同样指定独立目录：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --idea baseline
```

重复使用同一个 idea 名称会覆盖各 `seed_<seed>/` 中的结果文件和 `summary/` 汇总文件；带时间戳的旧日志仍会保留。因此新实验应使用新名称。`idea: null` 或空字符串不增加 idea 子目录，结果直接写入基础 `output_root`。

`runs/` 已被 Git 忽略，这些实验结果只保留在本地。

## 输出内容

以当前默认输出目录为例，主要结果位于：

```text
runs/tpt_quick_demo/idea_init_test/
  run_<timestamp>.log
  seed_11/
    results.csv
    results.json
    run_<timestamp>.log
  seed_42/
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

每个独立任务 `(模型, ImageNet-A, seed)` 完成后，脚本会立即更新对应 `seed_<seed>/results.csv` 和 `results.json`。当前每个 seed 只有 ViT-B/16，因此各包含一条结果；启用 RN50 后，每个 seed 完成两个模型时会包含两条结果。即使运行中途停止，已经完成的任务结果仍保留在各 seed 目录中。

每条 CSV/JSON 结果和最终平均结果都包含 `idea` 字段，移动或汇总结果文件后仍可识别实验名称。

当前 2 个任务全部成功完成后，脚本才生成 `summary/`：

- `summary/results.csv` 和 `results.json` 保存 2 条任务明细。
- `average_results.csv` 和 `average_results.json` 汇总 ViT-B/16 的两个 seeds。
- `average_results.md` 展示该模型的平均 Top-1、总体标准差和平均耗时。

重点关注单次任务明细中的：

- `top1_accuracy`
- `top5_accuracy`
- `elapsed_minutes`

跨 seed 对比时重点关注 `average_results.*` 中的平均准确率和总体标准差。当前 Quick Demo 只启用 ViT-B/16、两个 seeds 和 ImageNet-A，不能代替两个模型、三个 seeds、四个数据集上的完整结论。

## 常用临时覆盖

只测试前 500 张图片，同时保留正式的 64-view 设置：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --limit 500 \
  --idea name_limit500
```

临时同时测试两个模型：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --models RN50 ViT-B/16 \
  --devices cuda:0 cuda:1 \
  --idea name_both_models
```

只运行一个 seed：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --seeds 11 \
  --idea name_seed11
```

为了保证不同 idea 之间可比较，除了正在研究的变量之外，应保持模型、样本数、seed、views、selection fraction、TTA steps 和学习率一致。

## 配置说明

`configs/evaluation/tpt_quick_demo.yaml` 中的相对路径均相对于项目根目录解析。注释行不参与运行；下表只把当前未注释的值视为当前配置。

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `data_root` | `data/raw` | 数据集根目录；当前任务读取 `data/raw/imagenet-a/`。 |
| `class_index` | `configs/visualization/imagenet_class_index.json` | 1000 类 ImageNet 索引与 WNID 映射，用于把 ImageNet-A 的类别目录映射回原始类别索引。 |
| `class_names` | `configs/evaluation/imagenet_classes_tpt.json` | 构造 TPT prompt 的 1000 个类别名称；ImageNet-A 从中取出其 200 个候选类别。 |
| `model_root` | `models` | checkpoint 根目录；当前 `ViT-B/16` 对应 `models/ViT-B-16.pt`。 |
| `output_root` | `runs/tpt_quick_demo` | Quick Demo 的基础输出目录；当前还会追加 `idea_init_test/`。 |
| `idea` | `init_test` | 本次 idea 的标识，使不同实验写入独立的 `idea_<名称>/` 并在结果中记录。设为 `null` 或空字符串时不增加 idea 子目录；名称不能包含路径分隔符。 |
| `datasets` | `imagenet-a` | 要评测的数据集列表。只保留 ImageNet-A 是 Quick Demo 相比完整实验最主要的范围缩减。 |
| `models` | `ViT-B/16` | 当前启用的模型。取消 `RN50` 行的注释即可同时测试两个架构，任务数也会翻倍。 |
| `devices` | `cuda:0` | 当前参与调度的设备。取消其他 GPU 的注释可并行处理多个任务；每张 GPU 同时只运行一个任务，列表不能重复。 |
| `seeds` | `11`、`42` | 控制随机裁剪和翻转，并分别写入 `seed_11/`、`seed_42/`。取消 `123` 的注释会增加第三个独立任务；列表不能为空或重复。 |
| `num_workers` | `4` | 每个并行任务的 DataLoader 子进程数。启用 N 张 GPU 同时运行时，总加载进程最多约为 `N x 4`；`0` 表示在任务进程内加载。 |
| `views` | `64` | 每张图片的总视图数：1 个标准中心裁剪和 63 个随机增强视图，必须至少为 2。 |
| `selection_fraction` | `0.1` | 按预测熵保留的低熵视图比例。脚本使用 `int(64 x 0.1)`，因此当前选择 6 个视图；取值必须在 `(0, 1]` 且至少选中 1 个。 |
| `tta_steps` | `1` | 每张图片执行的 prompt 参数更新次数。增大它会显著延长运行时间，必须大于 0。 |
| `learning_rate` | `0.005` | 每张图片临时 AdamW 优化器的学习率，必须大于 0。 |
| `prompt_init` | `a photo of a` | 可学习 prompt 上下文 token 的初始化文本；脚本在其后追加类别名和句点，不能为空。 |
| `resolution` | `224` | 标准视图和随机视图的最终输入边长。必须匹配 checkpoint 声明的分辨率，否则任务直接报错。 |
| `print_frequency` | `100` | 每处理 100 张图片向任务日志写一次阶段性 Top-1，最后一张总会记录；不影响终端进度条、计算或准确率。 |
| `limit` | `null` | 每个任务最多处理的前 N 张图片。`null` 表示完整的 7500 张 ImageNet-A 图片；正整数可用于短测试。 |
| `log_file` | `run.log` | 日志基础名称。实验目录和每个 seed 目录都会生成带时间戳的 `run_<timestamp>.log`。 |

除 `log_file` 外，各字段均可用同名命令行参数临时覆盖，下划线在命令行中改为连字符，例如 `--num-workers`。当前任务数是 `len(models) x len(datasets) x len(seeds) = 2`；`devices` 只决定这些任务如何并行，不改变任务数。

### Quick Demo 中的 `num_workers`

`num_workers` 按并行任务分别计算。worker 不执行 CLIP 或 prompt 更新，而是在 CPU 上读取、解码图片，并为每张图片生成 1 个标准视图和 63 个随机增强视图。当前 `num_workers: 4` 的含义如下：

- 使用一张 GPU 时，串行任务在运行期间有 4 个 DataLoader workers；两个 seed 仍然是前后执行。
- 使用两张 GPU 并行两个 seed 时，每个任务各有 4 个 workers，总数约为 8。
- 若以后启用更多模型、seed 和四张 GPU，每张 GPU 各运行一个任务时，总数约为 16，而不是整次 run 共用 4 个。

脚本在 worker 大于 0 时采用 `spawn`，并默认让每个 worker 预取 2 个 batch。TPT 的 `batch_size` 固定为 1，但一个 batch 包含一张图片的 64 个视图，因此 worker 增多会同时提高 CPU 增强并行度、预取内存和 pinned-memory 压力。`0` 表示全部预处理在任务主进程同步完成，通常适合 smoke test 和排错；更大的值只有在 GPU 正在等待数据时才会加速。

结合当前观察，Quick Demo 建议先保持两张 GPU，并对以下组合做一次固定样本数的测速：

```bash
# 示例：两个 seed、两张 GPU，每个任务分别测试 2 或 4 个 workers
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --devices cuda:0 cuda:1 \
  --num-workers 2 \
  --limit 500 \
  --idea benchmark_workers2
```

再用 `--num-workers 4` 重复同样设置，比较从命令启动到汇总完成的总时间。若两者接近，优先选 2 以减少资源竞争；若 4 明显更快则保留 4。四张 GPU 运行完整实验时可从每任务 `2` 个 workers 起测。你的机器虽然有 128 个逻辑 CPU，但 CPU 核数充足并不代表磁盘、NUMA、内存带宽和 pinned-memory 路径不会竞争，因此不能仅按核心数把 worker 调得很高。正式比较 idea 时固定 GPU 数量和 `num_workers`；改变 worker 数还可能改变随机增强的消费顺序。

当前 Quick Demo 与完整配置的差异如下：

| 维度 | 当前 Quick Demo | 完整评测 |
| --- | --- | --- |
| 数据集 | ImageNet-A | 四个 ImageNet Variants |
| 模型 | ViT-B/16 | RN50、ViT-B/16 |
| seeds | `11, 42` | `11, 42, 123` |
| GPU | `cuda:0` | `cuda:0` 至 `cuda:3` |
| 任务数 | 2 | 24 |
| TPT 超参数 | 与完整评测一致 | 正式设置 |

最终报告结果仍应来自完整配置。Quick Demo 中被注释的 RN50、seed `123` 和额外 GPU 是便于逐步扩大验证范围的备选项，并非当前默认运行的一部分。
