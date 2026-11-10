# Pure CLIP Zero-shot 评测

该流程使用冻结的 OpenAI CLIP 和单模板 `a photo of a {}.`，不进行训练、微调或测试时自适应。

安装 `requirements.txt` 后可使用 `clip-evaluate`。它和 `python scripts/evaluate_clip_zeroshot.py` 指向同一个 `main()`，下文保留脚本形式以明确源码入口；两者的参数完全相同。

## 评测协议

- 模型：`RN50`、`ViT-B/16`。
- 类别名称：`configs/evaluation/imagenet_classes_tpt.json`。
- ImageNet-A/R：仅在数据集各自包含的 200 类中预测。
- ImageNet-V2/Sketch：在完整 1000 类中预测。
- 图像预处理：OpenAI CLIP checkpoint 自带的 preprocess。
- 指标：Top-1 accuracy。

ImageNet-A/R 的 200 类是数据集定义的候选集合，不是按模型结果筛选的。

## 1. 前置检查

完成[准备工作](preparation.md)并激活环境：

```bash
conda activate clip
ls models/RN50.pt models/ViT-B-16.pt
```

配置中启用的数据集目录必须存在，例如：

```text
data/raw/imagenetv2/
data/raw/imagenet-a/
data/raw/imagenet-r/
data/raw/imagenet-sketch/
```

## 2. Smoke test

```bash
python scripts/evaluate_clip_zeroshot.py \
  --config configs/evaluation/clip_zero_shot.yaml \
  --models ViT-B/16 --datasets imagenet-a \
  --devices cuda:0 --seeds 42 \
  --batch-size 4 --num-workers 0 --limit 8 \
  --output-root /tmp/clip_zero_shot_smoke
```

成功后应生成 `/tmp/clip_zero_shot_smoke/results.csv` 和 `average_results.md`。`limit` 结果只用于检查流程，不能作为正式准确率。

## 3. 正式评测

正式评测的范围和参数由 `configs/evaluation/clip_zero_shot.yaml` 定义：

```bash
python scripts/evaluate_clip_zeroshot.py \
  --config configs/evaluation/clip_zero_shot.yaml
```

只有一张 GPU 时：

```bash
python scripts/evaluate_clip_zeroshot.py \
  --config configs/evaluation/clip_zero_shot.yaml \
  --devices cuda:0
```

只评测一个模型和数据集：

```bash
python scripts/evaluate_clip_zeroshot.py \
  --config configs/evaluation/clip_zero_shot.yaml \
  --models RN50 --datasets imagenet-a
```

CLI 覆盖只影响当前运行，不修改 YAML。

## 配置

`configs/evaluation/clip_zero_shot.yaml` 的路径相对于仓库根目录解析。

| 参数 | 含义 |
| --- | --- |
| `data_root` | 数据集父目录。 |
| `class_index` | ImageNet index、WNID 和名称映射文件。 |
| `class_names` | 构造 prompt 的类别名称文件。 |
| `model_root` | CLIP checkpoint 所在目录。 |
| `output_root` | 结果和日志目录。 |
| `datasets` | 评测数据集列表；支持 `imagenetv2`、`imagenet-a`、`imagenet-r`、`imagenet-sketch`。 |
| `models` | 评测模型列表；支持 `RN50`、`ViT-B/16`。 |
| `batch_size` | 每个并行任务的图像 batch size；显存不足时降低。 |
| `num_workers` | 每个并行任务的数据加载进程数；排错时可使用 `0`。 |
| `devices` | 调度设备列表；每张 GPU 同时运行一个任务。 |
| `prompt_template` | prompt 模板，必须包含可插入类别名称的 `{}`。 |
| `seeds` | 每个任务依次记录的 seeds。 |
| `limit` | 每个数据集最多处理的样本数；空值表示全部。 |
| `log_file` | 日志文件的基础名称；实际文件名带时间戳。 |

同名 CLI 参数使用连字符，例如 `batch_size` 对应 `--batch-size`。`batch_size` 和 `num_workers` 都是每个并行任务的值，总 DataLoader worker 数约为并行任务数乘以 `num_workers`。

zero-shot 预处理没有随机增强，所以多个 seed 通常得到相同结果。保留 seeds 是为了统一实验记录格式。

## 输出

```text
<output_root>/
  results.csv
  results.json
  average_results.csv
  average_results.json
  average_results.md
  run_<timestamp>.log
```

- `results.*`：每个模型、数据集、seed 的样本数、正确数、Top-1、候选类别数和设备。
- `average_results.*`：按模型和数据集汇总 seed 均值、总体标准差、最小值和最大值。
- `average_results.md`：可直接阅读的结果表。

任务完成后会持续更新 `results.*`；全部任务完成后生成平均汇总。重新使用同一 `output_root` 会覆盖结果文件，但日志文件带时间戳，不会覆盖旧日志。
