# TPT Learning Rate × Steps 消融

该实验联合扫描 learning rate 和每张测试图像的 prompt 更新步数。优化器仍是 AdamW；这里的 “GD steps” 指梯度更新次数，不是把优化器替换为原始 SGD。

## 1. 配置与运行

配置文件：`configs/evaluation/tpt_lr_steps_ablation.yaml`。实验范围、候选 LR、候选更新步数、其他超参数、设备、输出位置和绘图选项均以该文件为准。总任务数为：

```text
模型数 × 数据集数 × selection 设置数 × LR 数 × step 数 × seed 数
```

每张 GPU 同时执行一个任务，设备按任务队列依次消费。目录名和 `idea` 只用于组织输出，不能代表实验参数；判断协议时应查看配置或输出记录中的 `learning_rate`、`tta_steps` 和 `weight_decay` 字段。

正式运行：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_lr_steps_ablation.yaml
```

结果写入配置指定的 `output_root`；`idea` 非空时使用 `<output_root>/idea_<idea>/`。

如果要真正运行无 weight decay 协议，使用新的 idea 隔离输出：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_lr_steps_ablation.yaml \
  --weight-decay 0 --idea weight_decay_0
```

该命令将结果隔离到 `<output_root>/idea_weight_decay_0/`。正式报告应使用足够的 seeds 评估稳定性。

## 2. Smoke test

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_lr_steps_ablation.yaml \
  --learning-rates 0.002 0.003 \
  --tta-step-options 1 2 \
  --devices cuda:0 --seeds 42 --num-workers 0 \
  --views 8 --selection-counts 1 --limit 2 \
  --output-root /tmp/tpt_lr_steps_smoke --idea smoke
```

该命令运行 4 个任务。它只验证执行、汇总和绘图链路，不能用于报告准确率。

单点覆盖可使用 `--learning-rate 0.003 --tta-steps 4`。列表字段必须非空、无重复且为正数；`weight_decay` 必须为有限非负数。

## 3. 自动绘图

评测结束后，`plot.kind: lr_steps` 会调用 `scripts/plot_tpt_lr_steps_ablation.py`，从 `summary/selection_fraction_chart_data.csv` 生成以下结果。

绘图配置使用 `weight_decay: null` 时，会自动接受结果中唯一的 weight decay 值。若同一汇总文件包含多个 weight decay，脚本会停止并提示在绘图配置中显式指定 `data.weight_decay`，避免不同正则化设置混入同一个二维网格。

### 每个 LR 的步数曲线

目录：`summary/lr_steps_plots/per_learning_rate/`。

每个 LR 各生成一张折线图和柱状图，横轴依次为 `CLIP zero-shot` 和配置中的各个候选步数，包含 Top-1 和 Top-5、总体标准差、点值标注和各指标最优五角星。只有一个 seed 时误差条为 0。

```text
lr_<lr>_steps_line.{png,pdf}
lr_<lr>_steps_bar.{png,pdf}
```

### 每个 LR 的最佳步数

```text
best_steps_by_lr_line.{png,pdf}
best_steps_by_lr_bar.{png,pdf}
```

Top-1 和 Top-5 分别选择最佳步数，因为二者可能不同。准确率并列时选择较少的步数，表示相同精度下优先较低计算量。zero-shot 没有更新步数，因此不放入这组图。

### 最佳步数对应的准确率

```text
best_accuracy_by_lr_line.{png,pdf}
best_accuracy_by_lr_bar.{png,pdf}
best_steps_and_accuracy.csv
```

横轴包含 `CLIP zero-shot` 和所有 LR。每个 LR 的 Top-1/Top-5 使用各自最佳步数的准确率；CSV 同时保存选中的步数、均值和标准差。

### LR × Steps 热力图

```text
top1_lr_steps_heatmap.{png,pdf}
top5_lr_steps_heatmap.{png,pdf}
```

这是建议额外保留的核心图。逐 LR 最优曲线会隐藏次优区域，而热力图能直接显示 LR 与步数的交互、稳定平台和对超参数敏感的尖锐最优点。若热力图只有孤立单元很高，不应仅凭该点下结论，应增加 seeds 或在邻近 LR/steps 上复核。

热力图颜色直接取自绘图配置中各系列的 `color`：Top-1 使用自定义绿色，Top-5 使用自定义红色；顺序色图从近白色过渡到该自定义颜色。修改 `series.color` 后，折线图、柱状图和热力图会一起更新。每张热力图的全局最高 accuracy 单元格使用与折线图相同样式的小五角星标出；星号大小由 `figure.heatmap_best_marker_size` 控制，若最高值并列则全部标出。

## 4. 独立重绘

只需运行 LR × steps 绘图脚本；不要将它直接粘在另一个 `python` 命令末尾：

```bash
python scripts/plot_tpt_lr_steps_ablation.py \
  --config configs/visualization/tpt_lr_steps_ablation_plots.json
```

CLI 临时改变输出目录或 idea 后，应显式指定数据与输出：

```bash
python scripts/plot_tpt_lr_steps_ablation.py \
  --config configs/visualization/tpt_lr_steps_ablation_plots.json \
  --data /path/to/run/summary/selection_fraction_chart_data.csv \
  --output-dir /path/to/run/summary/lr_steps_plots
```

## 5. 结果解释

“每个 LR 的最佳步数”和“最佳步数准确率”是从候选网格中选择后的结果，会有选择偏差。正式报告应同时给出热力图或完整逐 LR 曲线，并说明候选网格。更严格的协议是在开发集上选择 LR/steps，然后固定超参数到最终测试集。

除热力图外，可在运行成本成为重点时增加“准确率－平均耗时”的 Pareto 图。不过 `average_elapsed_minutes` 包含模型加载和数据管线开销，小规模任务尤其容易失真，因此本流程不自动生成该图；应先采用足够大的样本量并确认计时口径。
