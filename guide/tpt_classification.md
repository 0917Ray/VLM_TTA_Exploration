# TPT 图像分类

该流程实现 Test-Time Prompt Tuning（TPT）的手工 prompt 初始化版本。图像编码器和文本编码器保持冻结，每张测试图像只临时更新 prompt 上下文 token。

## 方法与协议

每张图像独立执行：

1. 生成 1 个标准中心裁剪视图和 `views - 1` 个随机裁剪/翻转视图。
2. 保存首次 prompt 更新前、标准视图上的 CLIP zero-shot logits。
3. 按预测熵从低到高选择视图，最小化其平均预测分布熵。
4. 使用更新后的 prompt 对标准视图分类。
5. 重置 prompt 和优化器状态，再处理下一张图像。

- ImageNet-A/R：各自在 200 个数据集候选类别中预测。
- ImageNet-V2/Sketch：在完整 1000 类中预测。
- 类别名称：`configs/evaluation/imagenet_classes_tpt.json`。
- 指标：TPT 和更新前 CLIP zero-shot 的 Top-1 至 Top-5。
- 随机增强受 seed 影响，因此正式比较必须固定 seeds、views 和 worker 设置。

## 1. Smoke test

完成[准备工作](preparation.md)后运行：

```bash
conda activate clip
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml \
  --models ViT-B/16 --datasets imagenet-a \
  --devices cuda:0 --seeds 42 --num-workers 0 \
  --views 8 --selection-fraction 0.25 --limit 8 \
  --output-root /tmp/tpt_smoke
```

该命令处理 8 张图片，每张使用 8 个视图并选择最低熵的 2 个。成功后检查：

```text
/tmp/tpt_smoke/seed_42/results.csv
/tmp/tpt_smoke/summary/average_results.csv
/tmp/tpt_smoke/summary/selection_fraction_chart_data.csv
```

smoke test 只验证流程，不用于报告准确率。

## 2. 正式运行

正式实验的范围和参数由 `configs/evaluation/tpt_classification.yaml` 定义：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml
```

只有一张 GPU 时：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml \
  --devices cuda:0
```

为实验设置独立输出目录：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_classification.yaml \
  --idea entropy_v1
```

输出会写到 `<output_root>/idea_entropy_v1/`。`idea: null` 时直接写入 `<output_root>/`；空白会转为下划线，且不能包含路径分隔符。

## 配置

`configs/evaluation/tpt_classification.yaml` 定义实验范围和运行参数：

| 参数 | 含义 |
| --- | --- |
| `data_root` | 数据集父目录。 |
| `class_index` | ImageNet index、WNID 和类别名映射文件。 |
| `class_names` | prompt 使用的类别名称文件。 |
| `model_root` | CLIP checkpoint 目录。 |
| `output_root` | 基础输出目录。 |
| `idea` | 可选实验名；非空时追加 `idea_<name>/`。 |
| `datasets` | 评测数据集列表。 |
| `models` | 评测模型列表。 |
| `devices` | 任务调度设备列表，每张卡同时运行一个任务。 |
| `seeds` | 随机增强 seeds。 |
| `num_workers` | 每个并行任务的数据加载进程数。 |
| `views` | 总视图数，包含标准视图。 |
| `selection_fractions` | 是否启用 fraction 模式。 |
| `fraction_options` | fraction 候选；实际数量为 `max(1, int(views × fraction))`。 |
| `selection_counts` | 是否启用 count 模式；与 fraction 模式必须一真一假。 |
| `count_options` | count 模式的候选视图数。 |
| `augment` | 是否生成随机增强视图。 |
| `only_origin` | 是否在扫描之外额外加入只用标准原图的 TPT 任务。 |
| `top_k` | 记录 Top-1 至 Top-k；兼容字段 Top-1/Top-5 始终保留。 |
| `tta_steps` | 每张图的 prompt 更新次数。 |
| `tta_step_options` | 更新步数候选列表；设置后每个值生成独立任务。 |
| `learning_rate` | 单点 AdamW 学习率。 |
| `learning_rates` | 学习率候选列表；设置后每个值生成独立任务。 |
| `weight_decay` | AdamW weight decay；设为 `0.0` 可关闭。 |
| `prompt_init` | prompt 上下文初始化文本。 |
| `resolution` | 输入分辨率，必须匹配 checkpoint。 |
| `print_frequency` | 文件日志的阶段性记录间隔。 |
| `limit` | 每个任务最多处理的样本数；空值表示不限制。 |
| `plot.enabled` | 是否在完整评测结束后自动绘图。 |
| `log_file` | 带时间戳日志的基础名称。 |

CLI 可覆盖常用字段，例如 `--models`、`--datasets`、`--devices`、`--seeds`、`--selection-fractions`、`--selection-counts`、`--learning-rates`、`--tta-step-options`、`--weight-decay`、`--top-k` 和 `--limit`。CLI 中用连字符代替下划线。单值和列表形式互斥，例如 `--tta-steps 1` 与 `--tta-step-options 1 2 4` 不能同时使用。

`num_workers` 按并行任务计算，总 DataLoader worker 数约为并行任务数乘以 `num_workers`。显存不足先减少 `views`；CPU 或内存压力高则减少 workers。正式比较不同实验时固定这两个值。

## 输出

```text
<run_root>/
  run_<timestamp>.log
  seed_<seed>/
    results.csv
    results.json
    run_<timestamp>.log
  summary/
    results.csv
    results.json
    average_results.csv
    average_results.json
    average_results.md
    selection_fraction_results.csv
    selection_fraction_results.json
    selection_fraction_results.md
    selection_fraction_chart_data.csv
```

- 每个任务完成后立即更新对应 `seed_<seed>/results.*`。
- 全部任务完成后生成 `summary/` 和跨 seed 均值/总体标准差。
- `topk_accuracy`、`zero_shot_topk_accuracy` 分别保存 TPT 与更新前 CLIP zero-shot 的 Top-k 列表。
- 每条逐任务和汇总记录都包含 `learning_rate`、`tta_steps` 与 `weight_decay`；不同 LR/steps 不会在跨 seed 汇总时混合。
- `selection_fraction_chart_data.csv` 是可重复绘图的完整精度输入。

fraction/count 消融、Only origin、自动绘图和独立重绘见[TPT Selection Fraction 消融](tpt_selection_fraction_ablation.md)。
学习率消融的推荐配置与解释见[TPT Learning-rate 消融](tpt_lr_ablation.md)。
学习率与更新步数的联合消融见[TPT LR × Steps 消融](tpt_lr_steps_ablation.md)。
