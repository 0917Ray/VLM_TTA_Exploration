# TPT Learning-rate 消融

该实验固定模型、数据集、低熵视图数量和 TTA 更新步数，只改变 AdamW learning rate，用于比较一步 prompt 更新的步长。它不能单独回答“一步更新是否足够”；learning rate 与更新步数的联合分析见[TPT LR × Steps 消融](tpt_lr_steps_ablation.md)。

## 1. 配置与运行

配置文件：`configs/evaluation/tpt_lr_ablation.yaml`。实验范围、超参数、设备、输出位置和绘图选项均以该文件为准。

运行：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_lr_ablation.yaml
```

结果写入配置指定的 `output_root`；`idea` 非空时使用 `<output_root>/idea_<idea>/`。

## 2. Smoke test

先用两个 LR 和少量样本验证调度与汇总：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_lr_ablation.yaml \
  --learning-rates 0.001 0.005 \
  --devices cuda:0 --seeds 42 --num-workers 0 --limit 2 \
  --output-root /tmp/tpt_lr_smoke --idea one_step
```

该命令生成 2 个任务，即 `1 model × 1 dataset × 1 selection setting × 2 LR × 1 seed`。YAML 中的 CLI 覆盖不会修改配置文件。

## 3. 配置与任务展开

`learning_rates` 定义待比较的 AdamW 学习率列表，必须非空、无重复、有限且大于零。selection 相关字段和 `tta_steps` 应在这组实验中保持不变，避免把其他因素混入 LR 比较。

临时扫描：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_lr_ablation.yaml \
  --learning-rates 0.001 0.005 0.01
```

临时恢复单点：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_lr_ablation.yaml \
  --learning-rate 0.005
```

单值字段 `learning_rate` 继续有效；新实验推荐统一使用列表字段 `learning_rates`，这样单点与扫描的含义一致。

一般情况下，总任务数为：

```text
模型数 × 数据集数 × selection 设置数 × learning-rate 数 × step 数 × seed 数
```

启用 `only_origin: true` 时，每个模型、数据集、LR 和 seed 还会增加一个 Only origin 任务。多 GPU 仍按“一张卡同时运行一个任务”调度。

## 4. 输出与绘图

逐任务 `results.csv/json` 和所有汇总文件都记录：

- `learning_rate`：该任务实际使用的 AdamW 学习率。
- `tta_steps`：每张图像的更新次数。
- selection 设置、seed、TPT Top-k 和更新前 zero-shot Top-k。

`average_results.*` 按模型、数据集、selection 设置、LR 和 steps 分组，只在同一组合内跨 seed 求均值与总体标准差。一次运行使用一个 weight decay，该值也会写入逐任务、汇总和绘图数据。`selection_fraction_chart_data.csv` 保留所有这些字段。

自动绘图配置是 `configs/visualization/tpt_learning_rate_plots.json`，其中：

```json
"learning_rate": null,
"x_column": "learning_rate"
```

`null` 表示不按 LR 过滤，并把所有 LR 放在横轴上。图中 `CLIP zero-shot` 是首次更新前的标准视图结果，不执行 prompt 更新；脚本会从各 LR 记录汇总该基线后显示一次。

启用 `value_labels` 后，折线图会在每个点旁标注准确率数值。启用 `highlight_best` 后，指定指标会在 `CLIP zero-shot` 与全部 LR 点中寻找全局最高点，并使用对应曲线颜色的 marker 标记；并列最高时全部标记。例如：

```json
"value_labels": true,
"highlight_best": {
  "enabled": true,
  "columns": ["top1_accuracy_percent", "top5_accuracy_percent"],
  "marker": "*"
}
```

将 `value_labels` 或 `highlight_best.enabled` 设为 `false` 可分别关闭数值或五角星。

评测完成后可独立重绘：

```bash
python scripts/plot_tpt_selection_ablation.py \
  --config configs/visualization/tpt_learning_rate_plots.json
```

输出结构为：

```text
<run_root>/
  seed_<seed>/results.{csv,json}
  summary/
    results.{csv,json}
    average_results.{csv,json,md}
    selection_fraction_chart_data.csv
    plots/
      learning_rate_bar.{png,pdf}
      learning_rate_line.{png,pdf}
      selection_fraction_plot_data.csv
      selection_fraction_plots_config.json
      plot_tpt_selection_ablation.py
```

## 5. 实验建议

先在一个代表性数据集上用对数尺度的较宽 LR 范围定位有效区间，再缩小范围；确定 LR 后再扩展到所有数据集。不要在同一测试集上选择最优 LR 后把该点当作无偏测试结果：应报告完整消融曲线，或在预先指定的开发集/数据集上选 LR，再固定到最终测试集。

比较多步 TTA 时，每个 `tta_steps` 应使用相同 seeds、views、selection 设置和候选 LR，或者先为每个步数各自选择 LR，再明确报告调参协议。较大 LR 可能使单步结果变好，也可能造成 prompt 更新不稳定，因此不要只看单个 seed。
