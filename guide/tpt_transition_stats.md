# TPT Prediction and Loss Transition Statistics

该工作流统计 one-step TPT 前后的样本状态转移，并生成两张四分类柱状图。所有命令从仓库根目录执行。

工作流链路：

```text
guide/tpt_transition_stats.md
  -> configs/evaluation/tpt_transition_stats.yaml
  -> scripts/evaluate_tpt_transition_stats.py
  -> data/raw + models
  -> runs/tpt_transition_stats/run_<id>/
  -> summary CSV/Markdown + reproducible plots
```

## 1. 实验协议

每个样本独立执行：

1. 恢复初始化 prompt，并清空 optimizer state。
2. 生成一个 canonical center crop 和 `views - 1` 个随机增强视图。
3. 在更新前按单视图 entropy 选择最低的 `int(views × selection_fraction)` 个视图。
4. 使用所选视图平均预测分布的 marginal entropy 做一步 AdamW TPT。
5. 使用更新后的 prompt 重新计算 canonical prediction、所选视图 entropy 和监督 cross-entropy。
6. 丢弃当前 prompt update，下一张图像重新开始。

准确率状态只比较 canonical view：

- `correct_to_correct`：更新前正确，TPT 后仍正确；
- `correct_to_wrong`：更新前正确，TPT 后错误；
- `wrong_to_correct`：更新前错误，TPT 后正确；
- `wrong_to_wrong`：更新前错误，TPT 后仍错误。

这些 transition 的具体含义为：

| Transition | 含义 | 对 TPT 效果的解释 |
| --- | --- | --- |
| `correct_to_correct` | 更新前后的 top-1 预测均等于真实标签 | TPT 保持了原本正确的决策。 |
| `correct_to_wrong` | 更新前 top-1 预测正确，更新后变为错误 | TPT 引入了有害退化。 |
| `wrong_to_correct` | 更新前 top-1 预测错误，更新后变为正确 | TPT 成功纠正了原本的错误。 |
| `wrong_to_wrong` | 更新前后的 top-1 预测均不等于真实标签 | TPT 未能纠正错误；更新前后的错误类别可能不同，因此不表示模型输出没有变化。 |

loss 状态在 **同一组由初始 prompt 选出的视图** 上比较。设所选视图的平均预测分布为：

$$
\bar p = \frac{1}{M}\sum_{j=1}^{M}\operatorname{softmax}(z_j).
$$

TPT entropy 和有标签 marginal cross-entropy 分别为：

$$
\mathcal L_{\mathrm{ent}} = -\sum_c \bar p_c\log\bar p_c,
$$

$$
\mathcal L_{\mathrm{CE}} = -\log\bar p_y.
$$

真实标签只用于计算 $\mathcal L_{\mathrm{CE}}$ 和最终统计，不参与视图选择、entropy backward 或 optimizer step。这里的 CE 是 one-hot label 与平均预测分布之间的 cross-entropy，因而和 TPT entropy 使用相同的视图及概率聚合方式。

四个 loss 桶为：

- `entropy_down_ce_down`：预测分布变得更集中，同时真实类别的概率上升；entropy 目标与监督目标在这一步的结果一致；
- `entropy_down_ce_up_or_equal`：预测分布变得更集中，但真实类别的概率没有提高；entropy minimization 可能强化了错误方向；
- `entropy_up_or_equal_ce_up_or_equal`：预测置信度和监督目标均未改善；
- `entropy_up_or_equal_ce_down`：真实类别的概率提高，但 entropy 没有下降；监督意义上的改善没有被 entropy 指标捕获。

这里 entropy 只衡量预测分布的集中程度，不衡量预测是否正确；entropy 下降可能来自模型更加确信一个错误类别。相对地，CE 下降等价于平均预测分布赋给真实类别的概率上升，但也不保证 top-1 类别已经变为真实类别。

当 `after < before - loss_tolerance` 时记为下降，否则进入 `up_or_equal` 桶。这样四个桶覆盖所有样本。脚本还单独保存 entropy/CE 在 tolerance 内持平的数量，避免把持平误读为严格上升。

上述 loss transition 是执行一次有限步长 AdamW 更新后观察到的结果统计，反映两个目标在该更新结果上的一致或冲突，不等价于原始梯度的 cosine similarity，也不能单独作为二者存在因果关系的证据。

## 2. 默认配置

主配置为 `configs/evaluation/tpt_transition_stats.yaml`：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `datasets` | `[imagenet-a]` | 要运行的数据集；可指定多个。 |
| `models` | `[ViT-B/16]` | CLIP checkpoint。 |
| `learning_rates` | `[0.003]` | 每个 LR 展开为独立任务。 |
| `seeds` | `[123]` | 每个 seed 展开为独立任务。 |
| `devices` | `cuda:0` 至 `cuda:3` | 任务轮询分配，每张 GPU 同时最多运行一个任务。 |
| `views` | `64` | 一个 canonical view 加 63 个随机增强。 |
| `selection_fraction` | `0.1` | 当前实际选择 6 个最低 entropy 视图。 |
| `weight_decay` | `0.01` | AdamW weight decay。 |
| `loss_tolerance` | `1e-8` | 判断 loss 是否下降/持平的绝对 tolerance。 |
| `limit` | `null` | 每个任务的样本上限；`null` 表示完整数据集。 |
| `num_workers` | `2` | 每个完整任务内部的 DataLoader workers。 |
| `run_name` | `null` | 空值使用时间戳；显式值用于固定执行目录名。 |

配置优先级为脚本默认值 `<` YAML `<` CLI。相对路径以仓库根目录解析，CLI 覆盖不会修改 YAML。

一个完整任务是：

```text
model × dataset × learning_rate × seed
```

完整任务按顺序 round-robin 分配到 `devices`。单个任务不会切分到多卡，每张卡同一时刻最多执行一个任务。任务数少于 GPU 数时只使用前面的设备；因此默认的一个数据集、一个模型、一个 LR、一个 seed 只使用 `cuda:0`。

## 3. Smoke Test

先用 8 个样本、8 views 验证数据、模型、一步更新、汇总和自动绘图：

```bash
python scripts/evaluate_tpt_transition_stats.py \
  --config configs/evaluation/tpt_transition_stats.yaml \
  --datasets imagenet-a --models ViT-B/16 \
  --learning-rates 0.003 --seeds 123 --devices cuda:0 \
  --views 8 --selection-fraction 0.25 \
  --num-workers 0 --limit 8 \
  --output-root /tmp/tpt_transition_stats_smoke
```

该运行选择 2 个最低 entropy 视图，并应生成：

```text
/tmp/tpt_transition_stats_smoke/run_<id>/
├── COMPLETED.json
├── resolved_config.yaml
├── seed_123/
│   ├── results.{json,csv,md}
│   └── vit-b-16__imagenet-a__lr_0p003/
│       ├── sample_transitions.{json,csv,jsonl}
│       └── task_result.json
└── summary/
    ├── summary.md
    ├── accuracy_transition_plot_data.csv
    ├── loss_transition_plot_data.csv
    └── plots/
```

Smoke test 的比例只验证执行链路，不用于研究结论。

## 4. 正式运行

运行默认的 ImageNet-A、seed 123、LR 0.003：

```bash
python scripts/evaluate_tpt_transition_stats.py \
  --config configs/evaluation/tpt_transition_stats.yaml
```

例如同时运行两个数据集、两个 LR 和两个 seeds：

```bash
python scripts/evaluate_tpt_transition_stats.py \
  --config configs/evaluation/tpt_transition_stats.yaml \
  --datasets imagenet-a imagenet-r \
  --learning-rates 0.001 0.003 \
  --seeds 42 123 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

该命令生成 `1 model × 2 datasets × 2 LRs × 2 seeds = 8` 个任务；四张卡各自串行处理两个任务。

## 5. 输出

每次执行使用独立目录：

```text
runs/tpt_transition_stats/run_<id>/
├── resolved_config.yaml
├── run_<timestamp>.log
├── COMPLETED.json
├── seed_123/
│   ├── results.{json,csv,md}
│   ├── <model>__<dataset>__lr_<lr>/
│   │   ├── task.log
│   │   ├── task_result.json
│   │   └── sample_transitions.{json,csv,jsonl}
│   └── summary/
│       ├── summary.md
│       ├── average_results.{json,csv}
│       ├── accuracy_transition_plot_data.csv
│       ├── loss_transition_plot_data.csv
│       └── plots/
└── summary/
    ├── results.{json,csv,md}
    ├── average_results.{json,csv}
    ├── summary.md
    ├── accuracy_transition_plot_data.csv
    ├── loss_transition_plot_data.csv
    └── plots/
```

`sample_transitions.jsonl` 每处理一个样本立即追加，长任务中断时仍可检查已有记录。任务完成后导出 JSON 和 CSV。每个样本包含前后 prediction、正确性、两种 loss、delta、四分类名称、视图索引和 AMP 状态。

跨 seed 比例先在每个 seed 内计算，再对 seed-level proportions 求平均。误差条使用 sample standard deviation；只有一个 seed 时不确定性为 undefined，机器结果保存为 `null`，图中不显示误差条。只有任务、汇总和绘图全部成功后才写 `COMPLETED.json`。

## 6. 绘图与独立重绘

评测成功后自动生成：

```text
summary/plots/<model>__<dataset>__lr_<lr>/
├── accuracy_transitions.{png,pdf}
└── loss_transitions.{png,pdf}
```

`summary/plots/` 同时保存实际绘图 CSV、有效配置和绘图脚本副本。修改展示参数后可独立重绘最近一个完成的默认运行：

accuracy 图默认在 `Correct -> Wrong` 上方绘制红色叉号，在 `Wrong -> Correct` 上方绘制绿色勾选框。loss 图在 `Entropy down / CE down` 上方绘制绿色勾选框，在 `Entropy down / CE up or equal` 上方绘制红色叉号。标记由矢量图元生成，避免 PDF 依赖彩色 Emoji 字体；可通过绘图配置中的 `annotations` 调整开关、尺寸、偏移和颜色。

```bash
python scripts/plot_tpt_transition_stats.py \
  --config configs/visualization/tpt_transition_stats.yaml
```

重绘指定结果：

```bash
python scripts/plot_tpt_transition_stats.py \
  --config configs/visualization/tpt_transition_stats.yaml \
  --accuracy-data /path/to/run/summary/accuracy_transition_plot_data.csv \
  --loss-data /path/to/run/summary/loss_transition_plot_data.csv \
  --output-dir /path/to/run/summary/replotted
```

独立重绘不加载数据集或模型 checkpoint。

## 7. 解释边界

- 这是使用测试标签进行统计的 oracle analysis；不能把 loss 转移图描述为完全无标签的 TTA 评测。
- CE 不参与 TPT 更新，因此图回答的是 entropy step 是否同时改善监督目标，而不是 supervised prompt tuning 本身的准确率。
- 正确到错误与错误到正确的比例比净准确率变化提供更多信息，但不能单独解释为何某个样本发生转移。
- `up_or_equal` 含持平样本；严格持平数量应从 `task_result.json` 的 `entropy_equal_count` 和 `cross_entropy_equal_count` 核对。
- 不同 seed 改变随机增强；正式比较 LR 或数据集时应保持 views、selection、模型和 seed 列表一致。
