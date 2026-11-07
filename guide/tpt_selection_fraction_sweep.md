# TPT 筛选扫描与绘图

该流程比较同一张测试图像的低熵视图筛选策略。它筛选的是该图像的增强视图，不是从数据集中筛选图片。

## 三类横轴结果

默认图从左到右包含：

1. `CLIP zero-shot`：原图标准中心裁剪，读取首次 TPT 更新前的 logits，不进行适配。
2. `Only origin`：只用原图标准中心裁剪执行 TPT，不生成随机增强。
3. count/fraction：生成 `views` 个视图，按预测熵升序选择后执行 TPT。

因此 `count=1` 不等于 `Only origin`：熵最低的视图可能是随机增强视图。

## 1. 当前配置

`configs/evaluation/tpt_selection_fraction_sweep.yaml` 当前运行：

- ImageNet-A、ViT-B/16、seed 42、64 views。
- count 模式，扫描最低熵的 1 至 10 个视图。
- `only_origin: true`，额外加入 Only origin TPT。
- `top_k: 10`，记录 Top-1 至 Top-10。
- 完成后自动生成柱状图和折线图。
- 绘图 JSON 当前选择 Top-1 至 Top-5；记录与绘制范围彼此独立。

运行前先完成[准备工作](preparation.md)。

## 2. Smoke test

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_selection_fraction_sweep.yaml \
  --selection-counts 1 2 \
  --devices cuda:0 --num-workers 0 --limit 2 \
  --output-root /tmp/tpt_selection_smoke --idea smoke
```

YAML 的 `only_origin: true` 仍然生效，因此该命令运行 `count=1`、`count=2` 和 `Only origin` 三个任务。结果位于 `/tmp/tpt_selection_smoke/idea_smoke/`。

## 3. 正式运行

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_selection_fraction_sweep.yaml
```

当前输出目录由以下 YAML 字段决定：

```yaml
output_root: runs/tpt_selection_fraction_sweep
idea: count_sweep_zero_shot
```

因此结果写入：

```text
runs/tpt_selection_fraction_sweep/idea_count_sweep_zero_shot/
```

## fraction 与 count 模式

两个开关必须一真一假。

### Count 模式

```yaml
selection_fractions: false
fraction_options: [0.02, 0.04, 0.06, 0.08, 0.1]
selection_counts: true
count_options: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

启用后只读取 `count_options`。每个 count 必须唯一且位于 `[1, views]`。命令行可临时覆盖：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_selection_fraction_sweep.yaml \
  --selection-counts 1 2 3 4 5
```

### Fraction 模式

```yaml
selection_fractions: true
fraction_options: [0.02, 0.04, 0.06, 0.08, 0.1]
selection_counts: false
count_options: [1, 2, 3, 4, 5]
```

启用后只读取 `fraction_options`。每个 fraction 必须唯一且位于 `(0, 1]`，实际选择数量为：

```text
max(1, int(views × fraction))
```

所以 64 views、fraction `0.04` 实际选择 2 个视图。CLI 覆盖示例：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_selection_fraction_sweep.yaml \
  --selection-fractions 0.02 0.04 0.06 0.08 0.1
```

## Only origin 与无增广

推荐的对比设置是：

```yaml
augment: true
only_origin: true
```

这会保留 count/fraction 增广扫描，并额外添加 Only origin 任务。

`augment: false` 表示整个运行只生成原图标准视图。此时 count 模式只能使用 `[1]`，fraction 模式只能使用 `[1.0]`；它不适合同时比较增强扫描。CLI 必须成对覆盖，例如 `--no-augment --selection-counts 1` 或 `--no-augment --selection-fractions 1.0`，否则原 YAML 中大于 1 的 count 或小于 1 的 fraction 会被拒绝。

## Top-k 记录与选择性绘图

`top_k: K` 会记录 TPT 和 CLIP zero-shot 的 Top-1 至 Top-K：

- 逐任务：`topk_correct`、`topk_accuracy`、`zero_shot_topk_correct`、`zero_shot_topk_accuracy`。
- 汇总 CSV：`topN_accuracy_percent`、`topN_std_percent` 及对应 `zero_shot_` 字段。

兼容字段 Top-1 和 Top-5 始终保留。当前 sweep 设置 `top_k: 10`，但绘图配置的 `data.series` 默认只列出 Top-1 至 Top-5。要只绘制 Top-1、Top-3、Top-5，改为：

```json
"series": [
  {"column": "top1_accuracy_percent", "error_column": "top1_std_percent", "label": "Top-1", "color": "#4F7C65", "marker": "o"},
  {"column": "top3_accuracy_percent", "error_column": "top3_std_percent", "label": "Top-3", "color": "#B5823A", "marker": "D"},
  {"column": "top5_accuracy_percent", "error_column": "top5_std_percent", "label": "Top-5", "color": "#A75B73", "marker": "s"}
]
```

`CLIP zero-shot` 横轴类别会自动使用相同的 Top-N 选择和颜色。设置 `zero_shot.enabled: false` 可隐藏该类别；`only_origin: false` 则不会生成 Only origin 数据。

## 绘图配置

自动绘图由评测 YAML 控制：

```yaml
plot:
  enabled: true
  config: configs/visualization/tpt_selection_fraction_plots.json
```

绘图 JSON 的关键字段：

| 字段 | 作用 |
| --- | --- |
| `data.evaluation_config` | 用评测 YAML 自动推导数据和输出路径。 |
| `data.model`, `data.dataset` | 从汇总 CSV 选择一组结果。 |
| `data.x_column` | count 模式用 `selection_count`；fraction 模式用 `selection_percent`。 |
| `data.series` | 选择要绘制的 Top-N、颜色和 marker。 |
| `zero_shot.enabled`, `label` | 显示和命名独立 zero-shot 横轴类别。 |
| `bar.enabled`, `line.enabled` | 启用柱状图或折线图。 |
| `bar.value_format` | 柱顶数值格式；当前 `{:.1f}` 保留一位小数。 |
| `legend.ncol` | 图例列数；当前五个 Top-N 单行显示。 |
| `axis.ylim` | 统一 y 轴范围。 |

切换到 fraction 模式时，同时修改绘图 JSON：

```json
"x_column": "selection_percent"
```

并把标题和 `axis.xlabel` 中的 count 改为百分比描述。

## 独立重绘

完成评测后，只需运行：

```bash
python scripts/plot_tpt_selection_sweep.py \
  --config configs/visualization/tpt_selection_fraction_plots.json
```

绘图脚本读取 `data.evaluation_config` 指向的评测 YAML，并根据其中的 `output_root` 和 `idea` 自动推导：

```text
输入：<output_root>/idea_<idea>/summary/selection_fraction_chart_data.csv
输出：<output_root>/idea_<idea>/summary/plots/
```

如果评测时用 CLI 临时覆盖了 `--output-root` 或 `--idea`，YAML 不会自动更新。此时显式指定：

```bash
RUN_DIR=/path/to/your/run
python scripts/plot_tpt_selection_sweep.py \
  --config configs/visualization/tpt_selection_fraction_plots.json \
  --data "$RUN_DIR/summary/selection_fraction_chart_data.csv" \
  --output-dir "$RUN_DIR/summary/plots"
```

绘图目录还会保存 `selection_fraction_plot_data.csv`、绘图配置副本和脚本副本，可在不重新评测的情况下复现图片：

```bash
cd /path/to/your/run/summary/plots
python plot_tpt_selection_sweep.py --config selection_fraction_plots_config.json
```

旧 CSV 如果没有 `zero_shot_topN_*` 字段，仍能绘图，但不会出现 `CLIP zero-shot` 类别。

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
      plot_tpt_selection_sweep.py
```

图表使用汇总均值；开启 `errorbar.enabled` 后，误差条使用配置中声明的跨 seed population standard deviation。
