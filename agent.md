# m6A Dataset 仓库整理说明（Agent 指南）

本文用于指导后续整理与新增文件，避免原始数据、模型与输出混放，并记录当前已完成的归档结构与命名规范。

## 1. 仓库用途
- 存放 m6A/YTH 相关数据、特征表示、模型权重与训练脚本。
- 既包含原始数据（FA/Bed/Tar.gz），也包含处理后的数据集与模型输出。

## 2. 当前主要内容（现状概览）
- 数据与权重已归档到：`data/`、`models/`、`outputs/`。
- 数据集构建与可视化脚本在：`scripts/dataset/`。
- 目录：
  - `rnafm_mlp/`：RNA-FM 相关脚本、数据处理、训练输出。
  - `RNA-FM/`、`RiNALMo/`：模型相关资源。
  - `representations/`、`weights/`、`yth_rbp/`：特征/权重/其它数据。

## 3. 目录结构（新增文件请按此放置）

```
./data/
  raw/                # 原始数据（fa/bed/tar.gz）
  processed/          # 处理后数据（csv、窗口化数据等）
./models/
  pretrained/         # 预训练权重（*.pt）
  finetuned/          # 训练得到的权重与checkpoint
./outputs/
  logs/               # nohup.out、run.log
  reports/            # html/figures
./scripts/
  dataset/            # 数据集构建脚本
  training/           # 训练与推理脚本
```

### 归档约定（当前已执行）
- `data/raw/`：原始数据（fa/bed/tar.gz）
- `data/processed/`：处理后数据（csv、窗口化数据等）
- `models/pretrained/`：预训练权重（*.pt）
- `outputs/logs/`：nohup/run 日志
- `outputs/reports/`：html/figures
- `scripts/dataset/`：数据集构建与可视化脚本

## 4. 命名规范
- 数据文件：`{species}.{assembly}.{datatype}.{ext}` 或 `m6A_{source}_{version}.csv`
- 模型权重：`{model}_{size}_{date}.pt`（如 `rinalmo_giga_20240115.pt`）
- 输出：`{task}_{date}_{note}.log/.html/.png`

## 5. 清理与维护建议
- 不删除原始数据，只归档。
- 大文件建议统一放入 `data/` 或 `models/`，避免顶层混杂。
- 若使用版本控制，考虑将大模型/数据加入 `.gitignore` 并在 README 说明下载来源。

---
如需进一步整理（例如迁移 `weights/` 或清理历史输出），请指定保留/迁移策略。
