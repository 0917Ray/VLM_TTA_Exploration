# TPT Quick Demo

`configs/evaluation/tpt_quick_demo.yaml` 用于验证完整 TPT、GPU 调度、跨 seed 汇总和输出目录。实验范围与运行参数以该配置文件为准；若只想检查代码是否可运行，请使用下面的 smoke test。

## 1. Smoke test

```bash
conda activate clip
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --devices cuda:0 --seeds 42 \
  --num-workers 0 --views 8 \
  --selection-fraction 0.25 --limit 8 \
  --output-root /tmp/tpt_quick_smoke --idea smoke
```

注意：CLI 中的 `--idea smoke` 会追加到 `--output-root`，所以上述结果位于 `/tmp/tpt_quick_smoke/idea_smoke/`。

## 2. 运行 Demo

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml
```

若机器只有一张 GPU：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --devices cuda:0
```

这只改变调度设备，不改变配置中定义的实验任务；任务会在同一张 GPU 上依次运行。

## 3. 逐步扩大实验

先限制样本数测试吞吐：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --limit 500 --idea benchmark_workers2
```

再固定其余设置，仅比较 `--num-workers 0`、`2`、`4` 的整次运行时间。选定后，正式实验保持 workers、GPU 数、views 和 seeds 不变。

扩展到 RN50 和 ViT-B/16：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --models RN50 ViT-B/16 --idea two_models
```

下面是临时扩展模型、数据集、seeds 和设备的 CLI 示例：

```bash
python scripts/evaluate_tpt.py \
  --config configs/evaluation/tpt_quick_demo.yaml \
  --models RN50 ViT-B/16 \
  --datasets imagenetv2 imagenet-a imagenet-r imagenet-sketch \
  --seeds 11 42 123 \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --idea full_variants
```

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

配置字段与标准 TPT 相同，详见[TPT 图像分类](tpt_classification.md#配置)。是否自动绘图由 `plot.enabled` 控制；筛选消融与图表见[TPT Selection Fraction 消融](tpt_selection_fraction_ablation.md)。
