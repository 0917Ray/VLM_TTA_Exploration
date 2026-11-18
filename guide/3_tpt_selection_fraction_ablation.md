# TPT 样本数量消融

该流程比较同一张测试图像的低熵视图筛选策略.

## 1. 配置与准备

运行前先完成[准备工作](0_preparation.md)
配置文件：`configs/evaluation/3_tpt_selection_fraction_ablation.yaml`

## 2. Smoke test

```bash
python scripts/3_evaluate_tpt_selection_ablation.py \
  --config configs/evaluation/3_tpt_selection_fraction_ablation.yaml \
  --selection-counts 1 2 \
  --devices cuda:0 --num-workers 0 --limit 2 \
  --output-root /tmp/tpt_selection_smoke --idea smoke
```

## 3. 正式运行

```bash
python scripts/3_evaluate_tpt_selection_ablation.py \
  --config configs/evaluation/3_tpt_selection_fraction_ablation.yaml
```

## fraction 与 count 模式

两个开关必须一 `True` 一 `False`

### Count 模式

设置 `selection_counts: true` 和 `selection_fractions: false` 后，只读取 `count_options`。每个 count 必须唯一且位于 `[1, views]`。命令行可临时覆盖：

```bash
python scripts/3_evaluate_tpt_selection_ablation.py \
  --config configs/evaluation/3_tpt_selection_fraction_ablation.yaml \
  --selection-counts 1 2 3 4 5
```

### Fraction 模式

设置 `selection_fractions: true` 和 `selection_counts: false` 后，只读取 `fraction_options`。每个 fraction 必须唯一且位于 `(0, 1]`，实际选择数量为：

```text
max(1, int(views × fraction))
```


```bash
python scripts/3_evaluate_tpt_selection_ablation.py \
  --config configs/evaluation/3_tpt_selection_fraction_ablation.yaml \
  --selection-fractions 0.02 0.04 0.06 0.08 0.1
```

## Only origin 与无增广

推荐的对比设置是：

```yaml
augment: true     # 保留 count/fraction 增广扫描
only_origin: true # 添加 Only origin 任务
```


`augment: false` 表示整个运行只生成原图标准视图. 此时 count 模式只能使用 `[1]`，fraction 模式只能使用 `[1.0]`

## Top-k 记录与选择性绘图

`top_k: K` 会记录 TPT 和 CLIP zero-shot 的 Top-1 至 Top-K：

- 逐任务：`topk_correct`、`topk_accuracy`、`zero_shot_topk_correct`、`zero_shot_topk_accuracy`。
- 汇总 CSV：`topN_accuracy_percent`、`topN_std_percent` 及对应 `zero_shot_` 字段。


`CLIP zero-shot` 横轴类别会自动使用相同的 Top-N 选择和颜色。设置 `zero_shot.enabled: false` 可隐藏该类别；`only_origin: false` 则不会生成 Only origin 数据。

## 绘图配置

自动绘图由评测 YAML 控制：

```yaml
plot:
  enabled: true
  config: configs/visualization/3_tpt_selection_fraction_plots.json
```

绘制范围由绘图配置的 `data.series` 独立决定.
要只绘制 Top-1、Top-3、Top-5，可在 `3_tpt_selection_fraction_plots.json` 中设置为：

```json
"series": [
  {"column": "top1_accuracy_percent", "error_column": "top1_std_percent", "label": "Top-1", "color": "#4F7C65", "marker": "o"},
  {"column": "top3_accuracy_percent", "error_column": "top3_std_percent", "label": "Top-3", "color": "#B5823A", "marker": "D"},
  {"column": "top5_accuracy_percent", "error_column": "top5_std_percent", "label": "Top-5", "color": "#A75B73", "marker": "s"}
]
```

绘图 JSON 的关键字段：

| 字段 | 作用 |
| --- | --- |
| `data.evaluation_config` | 用评测 YAML 自动推导数据和输出路径。 |
| `data.model`, `data.dataset` | 从汇总 CSV 选择一组结果。 |
| `data.learning_rate` | 视图筛选图所用的学习率；LR 消融图设为 `null`。 |
| `data.x_column` | count 模式用 `selection_count`；fraction 模式用 `selection_percent`。 |
| `data.series` | 选择要绘制的 Top-N、颜色和 marker。 |
| `zero_shot.enabled`, `label` | 显示和命名独立 zero-shot 横轴类别。 |
| `bar.enabled`, `line.enabled` | 启用柱状图或折线图。 |
| `bar.value_format` | 柱顶数值格式，例如 `{:.1f}` 表示保留一位小数。 |
| `line.value_labels` | 是否在折线点旁标注数值。 |
| `line.highlight_best` | 是否用 marker 标出指定指标最优点。 |
| `legend.ncol` | 图例列数。 |
| `axis.ylim` | 统一 y 轴范围。 |

切换到 fraction 模式时，同时修改绘图 JSON：

```json
"x_column": "selection_percent"
```

并把标题和 `axis.xlabel` 中的 count 改为百分比描述。

## 独立重绘

完成评测后，只需运行：

```bash
python scripts/3_plot_tpt_selection_ablation.py \
  --config configs/visualization/3_tpt_selection_fraction_plots.json
```

绘图脚本读取 `data.evaluation_config` 指向的评测 YAML，并根据其中的 `output_root` 和 `idea` 自动推导。`idea` 非空时路径为：

```text
输入：<output_root>/idea_<idea>/summary/selection_fraction_chart_data.csv
输出：<output_root>/idea_<idea>/summary/plots/
```


绘图目录还会保存 `selection_fraction_plot_data.csv`、绘图配置副本和脚本副本，可在不重新评测的情况下复现图片：

```bash
cd /path/to/your/run/summary/plots
python 3_plot_tpt_selection_ablation.py --config selection_fraction_plots_config.json
```

## 输出

```text
<run_root>/
  seed_<seed>/results.csv
  seed_<seed>/results.json
  summary/
    results.csv
    results.json
    average_results.csv
    average_results.json
    selection_fraction_results.csv
    selection_fraction_results.json
    selection_fraction_results.md
    selection_fraction_chart_data.csv
    plots/
      selection_fraction_bar.png
      selection_fraction_bar.pdf
      selection_fraction_line.png
      selection_fraction_line.pdf
      selection_fraction_plot_data.csv
      selection_fraction_plots_config.json
      3_plot_tpt_selection_ablation.py
```
