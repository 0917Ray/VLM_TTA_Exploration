# VLM Test-Time Adaptation Exploration

用于探索视觉语言模型测试时自适应方法的实验项目。

## 当前阶段

当前已完成 ImageNet-V2、ImageNet-A、ImageNet-R 和 ImageNet-Sketch 的数据集下载、
目录整理与可视化审阅。可视化脚本会从每个数据集随机抽取类别，生成带类别名称的
PNG 和 PDF 样例画廊。

原始数据、下载压缩包、模型权重和生成结果只保留在本地，不提交到 GitHub。

## 快速开始

安装当前可视化阶段所需的依赖：

```bash
python -m pip install Pillow PyYAML matplotlib
```

使用默认配置生成数据集画廊：

```bash
python scripts/visualize_datasets.py \
  --config configs/visualization/datasets.yaml
```

默认结果保存在 `runs/dataset_visualizations/<dataset>/`。

## 项目结构

```text
configs/visualization/   # 可视化配置与 ImageNet 类别映射
data/downloads/          # 数据集压缩包，不纳入 Git
data/raw/                # 解压后的数据集，不纳入 Git
guide/                   # 数据准备和项目同步说明
models/                  # 模型权重，不纳入 Git
runs/                    # 可视化及实验输出，不纳入 Git
scripts/                 # 可执行脚本
```

数据来源、目录要求和可视化参数见
[数据集准备](guide/preparation.md)。代码提交与 GitHub 同步流程见
[GitHub 同步教程](guide/github_sync.md)。
