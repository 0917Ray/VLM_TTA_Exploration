# Pure CLIP Zero-shot 评测

该流程使用冻结的 OpenAI CLIP 和单模板 `a photo of a {}.`，不进行训练、微调或测试时自适应。

安装 `requirements.txt` 后可使用 `clip-evaluate`。它和 `python scripts/1_evaluate_clip_zeroshot.py` 指向同一个 `main()`，下文保留脚本形式以明确源码入口；两者的参数完全相同。

## 评测协议

- 模型：`RN50`、`ViT-B/16`
- 类别名称：`configs/evaluation/imagenet_classes_tpt.json`
- ImageNet-A/R：仅在数据集各自包含的 200 类中预测
- ImageNet-V2/Sketch：在完整 1000 类中预测
- 图像预处理：OpenAI CLIP checkpoint 自带的 preprocess
- 指标：Top-1 accuracy


## 1. 前置检查

完成 [准备工作](0_preparation.md) 并激活环境：

```bash
conda activate clip
ls models/RN50.pt models/ViT-B-16.pt
```

确保配置中启用的数据集目录必须存在，例如：

```text
data/raw/imagenetv2/
data/raw/imagenet-a/
data/raw/imagenet-r/
data/raw/imagenet-sketch/
```

## 2. Smoke test

```bash
python scripts/1_evaluate_clip_zeroshot.py \
  --config configs/evaluation/1_clip_zero_shot.yaml \
  --models ViT-B/16 --datasets imagenet-a \
  --devices cuda:0 --seeds 42 \
  --batch-size 4 --num-workers 0 --limit 8 \
  --output-root /tmp/clip_zero_shot_smoke
```

预期结果：
- 生成 `/tmp/clip_zero_shot_smoke/results.csv` 和 `average_results.md`.

## 3. 正式评测

正式评测的范围和参数由 `configs/evaluation/1_clip_zero_shot.yaml` 定义，需要修改参数则直接在其中修改即可.

正式评测：

```bash
python scripts/1_evaluate_clip_zeroshot.py \
  --config configs/evaluation/1_clip_zero_shot.yaml
```

指定两张 GPU、单个模型和部分数据集评测：

```bash
python scripts/1_evaluate_clip_zeroshot.py \
  --config configs/evaluation/1_clip_zero_shot.yaml \
  --devices cuda:0 cuda:1 \
  --models RN50 \
  --datasets imagenet-a imagenet-r
```


## 配置说明

配置文件位于 `configs/evaluation/1_clip_zero_shot.yaml`.

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

注：zero-shot 预处理没有随机增强，所以多个 seed 会得到相同结果. 保留 seeds 是为了统一实验记录格式.

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
