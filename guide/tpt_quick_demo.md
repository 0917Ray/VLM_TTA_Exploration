# TPT Quick Demo

`configs/evaluation/tpt_quick_demo.yaml` 用较小的实验组合验证完整 TPT、多 GPU 调度、跨 seed 汇总和输出目录。它仍会处理完整 ImageNet-A，若只想检查代码是否可运行，请使用下面的 smoke test。

## 当前配置

| 项目 | 当前值 |
| --- | --- |
| 数据集 | ImageNet-A |
| 模型 | ViT-B/16 |
| seeds | `11, 42` |
| GPU | `cuda:0, cuda:2` |
| workers | 每个任务 `2` |
| TPT | 64 views、最低熵 10%、1 step、Top-1 至 Top-5 |
| 输出 | `runs/tpt_quick_demo/idea_init_test_num_work_2/` |

两个 seed 是两个独立任务；当前两张 GPU 会各运行一个任务。

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

注意：YAML 中的 `idea` 仍会追加到 CLI `--output-root`，所以上述结果位于 `/tmp/tpt_quick_smoke/idea_smoke/`。

## 2. 运行当前 Demo

确保 `cuda:0` 和 `cuda:2` 可用：

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

这只改变调度方式，不改变两个 seed 任务；它们会在同一张 GPU 上依次运行。

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

扩展到四个数据集和三个 seed 后，等价于标准完整实验组合：

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
runs/tpt_quick_demo/idea_init_test_num_work_2/
  run_<timestamp>.log
  seed_11/
    results.csv
    results.json
    run_<timestamp>.log
  seed_42/...
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

配置字段与标准 TPT 相同，详见[TPT 图像分类](tpt_classification.md#配置)。Quick Demo 默认不自动绘图；需要筛选扫描与图表时使用[TPT 筛选与绘图](tpt_selection_fraction_sweep.md)。
