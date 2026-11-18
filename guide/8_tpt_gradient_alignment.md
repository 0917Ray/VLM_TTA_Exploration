# 8. TPT Entropy 与监督 Cross-Entropy 梯度对齐实验设计

本工作流比较 one-step TPT 中无标签 entropy minimization 与有标签 cross-entropy 所产生的方向是否一致，以及 AdamW 的实际参数更新是否改变这种对齐关系。Raw gradient 和 optimizer-applied update 分成两个独立比较设定。

## 1. 术语和比较边界

本文将：

- `T` 定义为 TPT 使用的 marginal entropy minimization；
- `C` 定义为使用真实标签的 marginal cross-entropy；
- 用户描述中的“`T` 与 `A`”按上下文理解为“`T` 与 `C`”。如果 `A` 另有特定含义，需要在实现前修正术语。

实验有两个**互相独立的比较设定**：

| 比较设定 | 比较的两个向量 | 向量含义 |
| --- | --- | --- |
| raw-gradient 对齐 | `g_T` 与 `g_C` | 在同一个初始 prompt、同一组视图上，分别对 entropy 和 CE 求导得到的纯梯度 |
| optimizer-update 对齐 | `u_T` 与 `u_C` | 分别将 `g_T`、`g_C` 送入独立且状态相同的 AdamW 后，实际作用于参数的单步更新 |

因此，本实验**不**计算以下跨设定比较：

- `g_T` 与 `u_T` 的相似度；
- `g_C` 与 `u_C` 的相似度；
- raw-gradient 与 optimizer-update 混合后的 cosine similarity 或内积。

这样可以回答两个分开的问题：目标函数本身的下降方向是否一致，以及相同优化器的预处理、动量初始化和 weight decay 是否会改变 T/C 之间的对齐程度。

## 2. 单样本实验协议

每个样本独立执行，T 分支和 C 分支都从同一个初始 prompt 参数 `theta_0` 开始。

1. 固定随机种子，生成一个 canonical center crop 和 `views - 1` 个随机增强视图。
2. 用初始 prompt 对所有视图计算 logits 和 softmax 概率。
3. 按现有 TPT 协议，用初始单视图 entropy 选出最低的 `int(views * selection_fraction)` 个视图。T 分支和 C 分支必须使用**同一组已选视图**，否则差异同时包含视图采样/筛选差异，无法归因于 loss。
4. 记选中视图的平均预测分布为：

   $$
   \bar p(\theta)=\frac{1}{M}\sum_{j=1}^{M}\operatorname{softmax}(z_j(\theta)).
   $$

   T 分支的 loss 为：

   $$
   L_T(\theta)=-\sum_c \bar p_c(\theta)\log \bar p_c(\theta).
   $$

   C 分支的 loss 为：

   $$
   L_C(\theta)=-\log \bar p_y(\theta),
   $$

   其中 `y` 是 ImageNet-A 的真实标签。标签只进入 C 分支的 loss，不参与视图选择或 T 分支。
5. 在完全相同的 `theta_0` 上分别反向传播：

   $$
   g_T=\nabla_\theta L_T(\theta_0),\qquad
   g_C=\nabla_\theta L_C(\theta_0).
   $$

6. 记录 raw gradient 后，分别恢复到 `theta_0`，建立两个独立的 AdamW optimizer。两个 optimizer 使用相同的 `lr`、`betas`、`eps`、weight decay，并且每个样本开始时 optimizer state 都为空。T 分支只执行 `L_T` 的一步，C 分支只执行 `L_C` 的一步。
7. 记录实际的参数变化：

   $$
   u_T=\theta_T^{+}-\theta_0,\qquad
   u_C=\theta_C^{+}-\theta_0.
   $$

   `u_T` 和 `u_C` 是 optimizer-applied parameter updates，不再称作 raw gradient。它们包含 AdamW 的逐元素预处理和配置中的 weight decay；如果实现使用 AMP，记录前应先 unscale，或直接在用于统计的精度下计算，以免把 loss scaler 当作方向差异。
8. T 分支和 C 分支的参数变化都在单样本结束后丢弃，下一样本重新从同一个初始化 prompt 开始。

默认只更新 TPT 的 prompt context 参数。若将来扩展到其他可训练参数，应固定参数集合和 flatten 顺序，并把参数集合写入结果元数据。

## 3. 每个样本记录的指标

所有向量先按固定参数顺序 flatten。设 `dot(a,b)` 为欧氏内积，`||a||_2` 为 L2 norm，`eps` 为防止零范数除零的数值常数。

### 3.1 Raw gradient 设定

记录：

1. `raw_norm_T = ||g_T||_2`；
2. `raw_norm_C = ||g_C||_2`；
3. `raw_cosine = dot(g_T, g_C) / (||g_T||_2 ||g_C||_2 + eps)`；
4. `raw_dot = dot(g_T, g_C)`。

其中 cosine 接近 `1` 表示 T/C 的 raw gradient 方向一致，接近 `0` 表示近似正交，小于 `0` 表示方向冲突。内积同时受到方向和梯度大小影响，因此不能把内积直接当作纯方向指标。

严格来说，loss 的下降方向是 `-g_T` 和 `-g_C`。由于两个向量同时乘以 `-1` 不会改变各自 norm、二者的 cosine 或二者的内积，本文直接统计框架产生的 raw gradient `g_T` 和 `g_C`，所得对齐指标与比较下降方向完全相同。

### 3.2 Optimizer update 设定

记录：

1. `update_norm_T = ||u_T||_2`；
2. `update_norm_C = ||u_C||_2`；
3. `update_cosine = dot(u_T, u_C) / (||u_T||_2 ||u_C||_2 + eps)`；
4. `update_dot = dot(u_T, u_C)`。

由于 T 和 C 两个向量都采用同一“参数变化”符号，若另定义下降方向为 `-u_T` 和 `-u_C`，二者之间的 cosine 和内积数值不变。实现中仍应固定保存 `u = theta_after - theta_before`，避免把符号约定混入不同实验。

对 norm 只统计非负的尺度指标；“负值比例”只适用于 raw/update cosine 和 dot。每个统计值都应同时保存有效样本数和非有限值数量。

## 4. 分布、分层和汇总统计

### 4.1 总体分布

对每个实验任务分别绘制以下分布：

- raw gradient：`raw_norm_T` 与 `raw_norm_C` 的 norm 分布、`raw_cosine` 分布、`raw_dot` 分布；
- optimizer update：`update_norm_T` 与 `update_norm_C` 的 norm 分布、`update_cosine` 分布、`update_dot` 分布。

建议每种设定使用一个三联图：左侧叠加 T/C norm 分布，中间 cosine 分布，右侧 dot 分布。norm 可以使用 log 横轴或明确的稳健 x 轴范围；cosine 图应标出 `0` 参考线，dot 图也应标出 `0` 参考线。应优先保存直方图的固定 bins 和完整数据 CSV，避免只保存图片后无法复现。

### 4.2 按初始预测正确/错误分层

使用 canonical center crop 在 TPT 更新前的 top-1 prediction 与真实标签比较：

- `initial_correct`：初始 canonical prediction 等于真实标签；
- `initial_wrong`：初始 canonical prediction 不等于真实标签。

对 `all`、`initial_correct`、`initial_wrong` 三个群体分别计算并绘制上述全部 raw/update 分布。这个分层只使用更新前状态，不使用 TPT 后预测，避免把结果变量泄漏进分组定义。

### 4.3 数值汇总

对以下 8 个样本级指标分别计算：

```text
raw_norm_T, raw_norm_C,
raw_cosine, raw_dot,
update_norm_T, update_norm_C,
update_cosine, update_dot
```

每个指标、每个群体（`all` / `initial_correct` / `initial_wrong`）保存：

- `mean`；
- `median`；
- `standard_deviation`，默认使用样本标准差 `ddof=1`；
- `IQR = Q3 - Q1`；
- `valid_count` 和 `nonfinite_count`。

对 `raw_cosine`、`raw_dot`、`update_cosine`、`update_dot` 额外保存：

```text
negative_count = count(metric < 0)
negative_ratio = negative_count / valid_count
```

严格等于 `0` 的值不计入负值比例。若需要消除浮点噪声影响，可另加 `near_zero_ratio`，但不能未经说明把近零值并入负值比例。

多 seed、多数据集和多 LR 时，先保存每个 `model × dataset × learning_rate × seed` 任务的样本级记录和汇总，再对同一 model/dataset/LR 的 seed 结果进行汇总。默认建议保留两种视图：

- `per_seed_summary`：每个 seed 独立统计，便于检查随机增强带来的波动；
- `pooled_summary`：将同一 model/dataset/LR 下各 seed 的有效样本记录拼接后统计，作为总体分布。

不能把不同参数维度的梯度向量跨样本平均后再计算 cosine；alignment 必须先按样本计算，再汇总标量。

## 5. 默认实验配置

默认配置应与 [状态转移统计](7_tpt_transition_stats.md) 保持一致，并且本次默认只展开一个任务：

| 字段 | 默认值 |
| --- | --- |
| dataset | `imagenet-a` |
| model | `ViT-B/16` |
| learning rate | `0.003` |
| seed | `123` |
| devices | `cuda:0`, `cuda:1`, `cuda:2`, `cuda:3` |
| views | `64` |
| selection fraction | `0.1`，实际选择 6 个视图 |
| optimizer | `AdamW` |
| weight decay | `0.01` |
| prompt init | `a photo of a` |

默认网格只有一个完整任务，因此虽配置了四张 GPU，任务只会分配到 `cuda:0`。指定多数据集、LR 或 seed 后，再按任务轮询分配其他 devices。

多任务定义为：

```text
model × dataset × learning_rate × seed
```

任务按 devices round-robin 分配，每张 GPU 同时最多运行一个任务。CLI 覆盖顺序为：脚本默认值 `<` YAML `<` CLI。由于 T/C 每个样本都各执行一次 FP32 backward 和一次 AdamW step，运行成本约为现有单分支 TPT 的两倍以上，正式运行前应先做小规模 smoke test。

## 6. Smoke test 和正式运行边界

在实现后，先使用与现有 transition workflow 相同的受限设置验证链路：

```text
dataset: imagenet-a
model: ViT-B/16
learning_rate: 0.003
seed: 123
device: cuda:0
limit: 8
views: 8
selection_fraction: 0.25
num_workers: 0
```

Smoke test 至少应验证：

- T/C 使用相同的初始 prompt、视图和 selected view indices；
- raw 和 optimizer-update 两种设定都有 8 个样本级记录；
- norm、cosine、dot 都是有限值，零梯度时有明确处理；
- `all_count = initial_correct_count + initial_wrong_count`；
- 统计表的 mean、median、sample standard deviation、IQR 和负值比例可由保存的 sample metrics 独立重算；
- 绘图脚本只读取保存的标量 CSV，不重新加载模型或数据集。

Smoke test 的数值不用于正式结论。正式默认运行再将 `limit` 恢复为 `null`，其他配置保持一致。

Smoke test 命令：

```bash
python scripts/8_evaluate_tpt_gradient_alignment.py \
  --config configs/evaluation/8_tpt_gradient_alignment.yaml \
  --datasets imagenet-a --models ViT-B/16 \
  --learning-rates 0.003 --seeds 123 --devices cuda:0 \
  --views 8 --selection-fraction 0.25 \
  --num-workers 0 --limit 8 \
  --output-root /tmp/tpt_gradient_alignment_smoke
```

默认全量命令：

```bash
python scripts/8_evaluate_tpt_gradient_alignment.py \
  --config configs/evaluation/8_tpt_gradient_alignment.yaml
```

CLI 覆盖 YAML，不会改写配置。视图选择和初始 canonical prediction 沿用 TPT 的 autocast 协议；随后固定 selected views，梯度和 optimizer update 使用 FP32 计算。完整向量默认不落盘；需保存向量时显式指定 `--save-vectors`。

## 7. 输出契约

建议输出结构如下：

```text
runs/8_tpt_gradient_alignment/run_<id>/
├── resolved_config.yaml
├── run_<timestamp>.log
├── COMPLETED.json
├── seed_123/
│   ├── summary/
│   │   ├── results.{json,csv}
│   │   ├── gradient_summary.{json,csv}
│   │   ├── distribution_data.csv
│   │   ├── summary.md
│   │   └── plots/
│   ├── vit-b-16__imagenet-a__lr_0p003/
│   │   ├── sample_gradient_metrics.{csv,jsonl}
│   │   ├── task_result.json
│   │   └── task.log
└── summary/
    ├── results.{json,csv}
    ├── gradient_summary.{json,csv}
    ├── distribution_data.csv
    ├── summary.md
    └── plots/
```

每个 summary 中的 `gradient_summary` 按 model、dataset、learning rate 和 `all`/正确性分组列出统计；`distribution_data.csv` 保存图表使用的逐样本长表。图表再按 model/dataset/LR 写入独立子目录，避免多任务结果混画。

`sample_gradient_metrics` 默认只保存每个样本的标量指标、初始正确性、任务元数据、selected view indices 和数值状态，不保存完整梯度向量。完整向量会使存储量约等于“样本数 × prompt 参数量 × 4 bytes × T/C × raw/update”，只有在需要事后检查某个样本或重新定义 alignment 时才通过 `save_vectors: true` 显式开启，并单独存储压缩文件。

结果中必须区分：

- raw gradient 的统计；
- optimizer update 的统计；
- 初始正确/错误分组；
- seed-level 与 pooled 汇总；
- 标准差的定义、IQR 的分位数方法和负值比例的分母。

## 8. 解释边界和需要确认的决定

1. 本实验比较的是同一初始点、同一 selected views 上的局部一步方向，不是完整监督 prompt tuning 的训练轨迹，也不是 TTA 多步适应后的全局关系。
2. `u_T`/`u_C` 包含 AdamW 的实现细节和 weight decay。若研究问题只关心 Adam 的梯度预处理，应额外保存去掉 weight decay 的 update 分量，但不能用它替代“实际参数变化”的主结果。
3. CE 分支需要 ImageNet-A 的真实标签；标签只用于 C 的 backward 和统计，不会进入 T 分支或视图选择。
4. cosine 描述方向相似度，dot 同时受 norm 影响；两者必须同时报告，不能只用其中一个替代另一个。
5. 正确/错误分层按 canonical view 定义，而梯度来自 selected augmented views；这是有意保持与现有 TPT transition protocol 一致的两种视角，结果解释时需要明确。
6. 在开始实现前需要确认两点：
   - `A` 是否确实是 `C`（Cross-Entropy）的笔误；
   - optimizer-update 主指标是否采用实际参数变化 `u = theta_after - theta_before`，并将 weight decay 保留在主结果中。
