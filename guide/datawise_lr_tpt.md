# Data-wise Learning-rate Selection TPT

该工作流为每张测试图像独立选择 learning rate。所有候选使用同一张图像、同一批增强视图和相同的低 entropy 视图索引；选择标准是候选轨迹结束后的 TPT entropy，不使用测试标签。

文件链：

```text
guide/datawise_lr_tpt.md
  -> configs/evaluation/datawise_lr_tpt.yaml
    -> scripts/evaluate_datawise_lr_tpt.py
      -> data/raw + models
        -> runs/datawise_lr_tpt/idea_<name>/run_<id>/
          -> scripts/plot_datawise_lr_tpt.py
            -> summary/plots/
```

后续命令均从仓库根目录执行。

运行前先完成[准备工作](preparation.md)，确保目标数据集位于 `data/raw/`、CLIP checkpoint 位于 `models/`，并已安装 `requirements.txt` 中的评测与绘图依赖。

## 1. 方法与协议

对每张图像生成 `views` 个视图，其中第一个是 canonical center crop，其余为随机增强。视图选择接口与 `tpt_lr_ablation.yaml` 一致：count 模式直接指定最低 entropy view 数，fraction 模式按 `int(views × fraction)` 得到实际 view 数。具体视图数、选择模式和选项均以运行时配置为准。

对配置中的每个候选 LR：

- 候选都从相同的初始 prompt 和空 AdamW state 出发。
- step setting 为 `1` 时，所有候选共享一次初始 gradient；脚本用真实首步 AdamW 公式构造候选 prompt，按 post-update entropy 选择 LR，再执行唯一一次真实 `optimizer.step()`。
- step setting 大于 `1` 时，每个 LR 独立运行固定 LR 的完整多步 AdamW 轨迹；后续 gradient 依赖前一步 prompt，因此不能跨 LR 共享。
- 选出最终 entropy 最低的 LR 后，使用对应 prompt 对 canonical view 分类。
- 下一张图像重新恢复初始 prompt，不保留前一张图像的 LR、prompt 或 optimizer state。

测试标签只用于最后统计 Top-k accuracy，不参与 LR 选择。

## 2. 配置

主配置是 `configs/evaluation/datawise_lr_tpt.yaml`。关键字段：

| 字段 | 含义 |
| --- | --- |
| `datasets` | 一次运行包含的数据集 |
| `models` | 一次运行包含的 CLIP 模型 |
| `devices` | 并行任务使用的设备，每张 GPU 同时最多运行一个完整任务 |
| `seeds` | seed 是任务维度之一，不同 seed 可以分配到不同 GPU 并行运行 |
| `views` | 每张图像的总视图数 |
| `selection_fractions` | 是否启用 fraction 模式 |
| `fraction_options` | fraction 模式的候选比例；实际 count 为 `int(views × fraction)` |
| `selection_counts` | 是否启用 count 模式 |
| `count_options` | count 模式的候选 view 数 |
| `candidate_lrs` | 每张图像都要比较的固定 LR 集合，必须升序、无重复且非负 |
| `tta_step_options` | 要比较的更新步数；只跑一种设置时使用 `[1]` |
| `weight_decay` | 所有候选共同使用的 AdamW weight decay |
| `idea` | 结果变体目录名 |
| `run_name` | 可选的固定执行 ID；未设置时使用时间戳 |
| `limit` | 每个任务最多处理的样本数；`null` 表示完整数据集 |
| `num_workers` | 每个任务内部 DataLoader 使用的 CPU worker 数，不控制 GPU 任务并发数 |

配置优先级为：脚本默认值 `<` YAML `<` 显式 CLI 参数。CLI 覆盖不会回写 YAML。

每次运行都会创建新的 `run_<id>/`。若显式设置的 `run_name` 对应目录已经存在，程序会停止而不会覆盖或混合旧结果。

`selection_fractions` 和 `selection_counts` 必须且只能有一个为 `true`。每个 `fraction_options` 或 `count_options` 项会展开成独立实验任务，并分别生成逐样本记录、准确率汇总和 LR 分布图。CLI 可用 `--selection-fractions 0.1 0.2` 或 `--selection-counts 1 2 3` 临时切换模式和选项。

若 `candidate_lrs` 不包含 `0.0`，每张图像都会执行非零更新。若需要把 no-update 作为合法选择，可将 `0.0` 加到候选列表首位。

## 3. 任务定义与 GPU 调度

一个完整任务定义为固定的：

```text
seed × model × dataset × selection setting × step setting
```

其中 `selection setting` 是一个具体的 count 或 fraction 选项，`step setting` 是一个具体的 `tta_step_options` 值。任务会处理该数据集的完整选定样本范围：`limit: null` 时处理完整数据集，设置 `limit` 时处理该任务的前 `limit` 个样本。

总任务数为：

```text
models × datasets × selection settings × step settings × seeds
```

调度规则如下：

- 完整任务按配置顺序 round-robin 分配到 `devices`；单个任务不会拆分到多张 GPU。
- 每张 GPU 同时最多运行一个完整任务；同一张卡分到的多个任务按队列顺序执行。
- 不同 seed 属于不同任务，因此可以分配到不同 GPU 并行运行。
- `candidate_lrs` 在每张图像内部依次比较，不会展开成独立调度任务。
- `num_workers` 只控制每个任务的 DataLoader CPU worker 数，不增加同一张 GPU 上的任务并发数。
- 若任务数少于 GPU 数，只使用前 `任务数` 张设备，其余 GPU 保持空闲。

例如一个模型、一个数据集、一个 count、一个 step、两个 seeds 和两张 GPU 会生成两个完整任务，分别在 `cuda:0` 和 `cuda:1` 上并行执行。

## 4. Smoke Test

先用一张 GPU、一个模型、一个数据集、一个 seed 和两个样本验证 one-step 链路：

```bash
python scripts/evaluate_datawise_lr_tpt.py \
  --config configs/evaluation/datawise_lr_tpt.yaml \
  --models ViT-B/16 --datasets imagenet-a \
  --devices cuda:0 --seeds 42 --num-workers 0 \
  --views 8 --selection-counts 1 \
  --candidate-lrs 0.001 0.005 --tta-steps 1 \
  --limit 2 --output-root /tmp/datawise_lr_tpt_smoke \
  --idea smoke
```

再验证多步分支：

```bash
python scripts/evaluate_datawise_lr_tpt.py \
  --config configs/evaluation/datawise_lr_tpt.yaml \
  --models ViT-B/16 --datasets imagenet-a \
  --devices cuda:0 --seeds 42 --num-workers 0 \
  --views 8 --selection-counts 1 2 \
  --candidate-lrs 0.001 0.005 --tta-step-options 1 2 \
  --limit 2 --output-root /tmp/datawise_lr_tpt_multistep_smoke \
  --idea smoke
```

验证两个 seed 是否分配到两张 GPU：

```bash
python scripts/evaluate_datawise_lr_tpt.py \
  --config configs/evaluation/datawise_lr_tpt.yaml \
  --models ViT-B/16 --datasets imagenet-a \
  --devices cuda:0 cuda:1 --seeds 42 123 \
  --num-workers 0 --views 8 --selection-counts 1 \
  --candidate-lrs 0.001 0.005 --tta-steps 1 \
  --limit 2 --output-root /tmp/datawise_lr_tpt_multigpu_smoke \
  --idea multigpu_smoke
```

该命令应首先显示：

```text
[INFO] data-wise LR TPT: 2 tasks (2 seeds, 1 tasks/seed), devices=cuda:0, cuda:1
```

随后两张卡各执行一个完整任务。具体设备分配也会记录在本次运行的主日志中。

这类受限运行只验证执行、输出、汇总和绘图，不能作为正式准确率。

## 5. 正式运行

正式运行使用 YAML 中列出的模型、数据集、selection 设置、step 设置和 seeds：

```bash
python scripts/evaluate_datawise_lr_tpt.py \
  --config configs/evaluation/datawise_lr_tpt.yaml
```

多步正式实验建议使用新的 `idea` 隔离：

```bash
python scripts/evaluate_datawise_lr_tpt.py \
  --config configs/evaluation/datawise_lr_tpt.yaml \
  --tta-step-options 1 2 3 --idea entropy_min_steps_1_2_3
```

多步计算量约随 `候选 LR 数 × steps` 增长，正式全数据运行前必须先做 smoke test。

## 6. 输出

每次执行使用独立的 `run_<timestamp-or-run-name>`：

```text
runs/datawise_lr_tpt/idea_<name>/run_<id>/
├── resolved_config.yaml
├── run_<timestamp>.log
├── COMPLETED.json
├── seed_11/
│   ├── results.{json,csv,md}
│   ├── vit-b-16__imagenet-a__count_1__steps_1/
│   │   ├── task.log
│   │   ├── task_result.json
│   │   ├── sample_lr_selection.jsonl
│   │   └── sample_lr_selection.{json,csv}
│   └── summary/
│       ├── summary.md
│       ├── results.{json,csv,md}
│       ├── summary_results.{json,csv,md}
│       ├── lr_distribution.{json,md}
│       ├── accuracy_plot_data.csv
│       ├── lr_distribution_plot_data.csv
│       └── plots/
├── seed_42/
├── seed_123/
└── summary/
    ├── summary.md
    ├── results.{json,csv,md}
    ├── average_results.{json,csv,md}
    ├── lr_distribution.{json,md}
    ├── accuracy_plot_data.csv
    ├── lr_distribution_plot_data.csv
    └── plots/
```

`sample_lr_selection.jsonl` 在长任务中逐样本追加，即使任务中断也能检查已经完成的记录。任务正常结束后，同时导出更方便分析的 JSON 和 CSV。

每个样本记录：样本相对路径、候选 LR、各候选最终 entropy、被选 LR、初始/最终 entropy、真实标签、zero-shot 与更新后预测及正确性，以及 AMP 使用的最终 scale 和 overflow retry 次数。

并行任务只写各自的 task 目录。每个任务完成后，父进程串行刷新对应 `seed_<seed>/results.{json,csv,md}`，避免多个 worker 同时改写 seed 结果文件。全部任务成功后，为每个 seed 生成独立的 `seed_<seed>/summary/`，其中 `summary.md`、统计表、LR 分布和图表都只包含该 seed，并在标题中明确 seed 编号。

顶层 `summary/` 是跨 seed 总结：`results.*` 保留各个 seed 的逐任务结果，`average_results.*` 保存跨 seed 平均值和 sample standard deviation，`summary.md` 把逐 seed 结果与平均结果放在同一个文档中。顶层图表标题标记参与平均的 seeds。只有这些内容全部生成后才写入 `COMPLETED.json`；任务失败时，已写入的 task 结果和逐样本 JSONL 会保留，但不能视为完整实验。

跨 seed 的误差使用 sample standard deviation（`ddof=1`）。只有一个 seed 时不确定性为 undefined，在 JSON 中保存为 `null`，CSV/Markdown 中留空或显示 `n/a`。

## 7. 绘图

评测成功后默认自动调用 `scripts/plot_datawise_lr_tpt.py`。每个 seed 的 `seed_<seed>/summary/plots/` 和顶层跨 seed 的 `summary/plots/` 都会为每个 `model × dataset × selection setting` 生成：

```text
summary/plots/<model>_<dataset>/<count_1-or-fraction_0p1>/
├── lr_selection_distribution.{png,pdf}
├── accuracy_bar.{png,pdf}
└── accuracy_line.{png,pdf}
```

LR 分布图的横轴是候选 LR，纵轴是被选择样本比例；存在多个 `tta_step_options` 时按更新步数分组。正确率图包含 CLIP zero-shot 和各个 data-wise LR step 设置的 Top-1/Top-5。

独立重绘最近一个完成的默认配置运行：

```bash
python scripts/plot_datawise_lr_tpt.py \
  --config configs/visualization/datawise_lr_tpt_plots.json
```

重绘指定运行：

```bash
python scripts/plot_datawise_lr_tpt.py \
  --config configs/visualization/datawise_lr_tpt_plots.json \
  --accuracy-data /path/to/run/summary/accuracy_plot_data.csv \
  --distribution-data /path/to/run/summary/lr_distribution_plot_data.csv \
  --output-dir /path/to/run/summary/plots
```

重绘某个 seed 的结果时，将三个路径改为该 seed 的 summary：

```bash
python scripts/plot_datawise_lr_tpt.py \
  --config configs/visualization/datawise_lr_tpt_plots.json \
  --accuracy-data /path/to/run/seed_42/summary/accuracy_plot_data.csv \
  --distribution-data /path/to/run/seed_42/summary/lr_distribution_plot_data.csv \
  --output-dir /path/to/run/seed_42/summary/plots
```

`summary/plots/` 会保存图表使用的数据、有效绘图配置和绘图脚本快照，可以脱离数据集、checkpoint 和评测代码重新作图。

## 8. 计算边界与解释

- one-step 为每张图像增加多个无梯度候选 forward，但仍只有一次 backward 和一次真实 AdamW update。
- multi-step 必须为每个 LR 运行完整轨迹；例如 9 个 LR、3 steps 会产生 27 次 candidate update backward，再进行候选比较。
- 总任务数还会乘以启用模式中的 selection option 数量；例如从一个 count 选项扩展到三个 count 选项，会使其他维度相同时的任务数变为 3 倍。
- 所有 LR 都在同一批固定的 selected views 上计算，避免候选之间的随机增强和视图集合不同。
- CUDA AMP gradient overflow 时会降低 GradScaler scale，并在相同原始 prompt 和 views 上重算该 update；恢复事件写入 task log 和逐样本记录，不会静默少执行一个 step。
- 最低 entropy 不保证分类正确。必须同时查看 LR 分布、entropy reduction、prediction flip 和 accuracy。
- 若大量样本选择候选边界，应先扩展或移动候选范围，再讨论“最优 LR”分布。

## 9. 运行注意事项

- 修改调度脚本后，已经启动的 Python 进程不会自动加载新代码；必须停止旧实验并重新启动，新的任务划分才会生效。
- 启动日志中的 `tasks` 是包含 seeds 在内的完整任务总数；`tasks/seed` 不包含 seed 维度。
- GPU 利用率应结合主日志中的 `device assignment` 检查。某张卡空闲不一定是调度失败，也可能是完整任务数少于设备数，或该卡队列已经提前完成。
