# m6A Dataset / ETD 多任务建模代码仓库（最小可复现）

## 项目简介
本仓库用于 m6A 多任务建模与实验脚本管理，核心是基于 ETD（Encoder-Transformer-Decoder）主干的多任务模型实现，支持：

- 可修饰性预测（PU）
- RBP role 结合预测（PU + Dirichlet 不确定性）
- RNA 二级结构矩阵预测（RNAfold BPP 监督）
- mask 补全（MLM）

当前仓库按“代码优先最小可复现”策略发布，仅包含源码、脚本、测试和说明文档，不包含原始数据、预训练权重、训练输出结果。

## 当前仓库范围（包含 / 不包含）

### 包含
- `models/etd_multitask/`：ETD 多任务模型、loss、metrics、RNAfold 解析工具
- `scripts/`：数据构建、RNAfold 缓存、训练、评估、分析脚本
- `tests/`：核心单元测试与 shape/loss 校验

### 不包含（需本地准备）
- `data/`：RMBase BED、FASTA、处理中间文件、RNAfold 缓存
- `outputs/`：训练日志、checkpoint、评估结果、分析报告
- `weights/`、`models/pretrained/`：大模型权重/预训练权重
- `RNA-FM/`、`RiNALMo/`、`rnafm_mlp/` 等外部/嵌套仓库目录

## 目录结构

```text
models/
  etd_multitask/
scripts/
  dataset/
  training/
  analysis/
tests/
README.md
```

## 环境依赖

### 基础依赖
- Python 3.10+（当前环境示例为 3.12）
- PyTorch（GPU 可选）
- `numpy`, `pandas`
- `pyarrow`（读写 parquet）
- ViennaRNA / `RNAfold`

### 可选 GPU 依赖说明
训练脚本中包含对 `libcusparseLt` 的运行时路径引导逻辑（适配部分 pip 安装的 CUDA 运行时布局）。如果出现 `libcusparseLt.so.0` 相关错误，请确认 CUDA 运行时库可见。

## 数据准备（本地）
本项目默认使用以下输入文件路径（未随仓库上传）：

- `data/raw/human.hg38.m6A.result.col29.bed`
- `data/raw/human.hg38.modrbp.m6A.writer.bed`
- `data/raw/human.hg38.modrbp.m6A.reader.bed`
- `data/raw/human.hg38.modrbp.m6A.eraser.bed`
- `data/raw/Homo_sapiens.GRCh38.cdna.all.fa.gz`

## 典型流程

### 1) 构建多任务数据表
```bash
python scripts/dataset/build_m6a_multitask_dataset.py
```

输出（本地）：
- `data/processed/m6a_multitask_sites.parquet`
- `data/processed/m6a_multitask_transcripts.parquet`
- `data/processed/m6a_multitask_splits.json`

### 2) 生成 RNAfold BPP 缓存
```bash
python scripts/dataset/generate_rnafold_bpp_cache.py \
  --include-splits train,val,test \
  --max-len 12000 \
  --jobs 4
```

说明：
- 使用 `RNAfold -p --noLP --modifications -d2`
- 解析 `*_dp.ps` 中 `%start of base pair probability data` 后的 `ubox` 行
- 缓存输出到 `data/processed/rnafold_bpp/*.npz`

### 2.1) 单点替换 RNAfold 稠密缓存（分片 + 可续跑）
```bash
# 4 个分片并行（tmux）
tmux new-session -d -s fold0 "bash scripts/dataset/run_rnafold_dense_shard0.sh"
tmux new-session -d -s fold1 "bash scripts/dataset/run_rnafold_dense_shard1.sh"
tmux new-session -d -s fold2 "bash scripts/dataset/run_rnafold_dense_shard2.sh"
tmux new-session -d -s fold3 "bash scripts/dataset/run_rnafold_dense_shard3.sh"
```

```bash
# 中途停止（主进程 + 子进程 RNAfold）
pkill -f "build_rnafold_single_site_dense_cache.py --max-len 12000"
pkill -f "/root/miniconda3/envs/m6a/bin/RNAfold -p --noLP -d2 --modifications"
```

```bash
# 续跑（不要加 --overwrite）
bash scripts/dataset/run_rnafold_dense_shard0.sh
bash scripts/dataset/run_rnafold_dense_shard1.sh
bash scripts/dataset/run_rnafold_dense_shard2.sh
bash scripts/dataset/run_rnafold_dense_shard3.sh
```

续跑机制说明：
- 输出按 transcript 写入 `data/processed/rnafold_single_site_dense/<transcript_id>.npz`。
- 已存在的 `.npz` 会自动 `skipped`，因此中断后可直接续跑。
- 只有显式传 `--overwrite` 才会重算并覆盖。

### 3) 训练（smoke 预设，推荐）
```bash
python scripts/training/train_etd_multitask.py --preset smoke_gpu
```

`smoke_gpu` 预设当前默认：
- `ablate_no_struct=True`（不训练结构任务，不初始化/使用 RNAfold provider）
- `loss_w_struct=0.0`
- `disable_cudnn=True`（规避部分环境下的 CUDA 非法访存）
- 默认不开 AMP（仅在显式传 `--amp` 时开启）

如需手工指定参数，示例如下：

### 3.1) 训练（手工 smoke 示例）
```bash
python scripts/training/train_etd_multitask.py \
  --smoke-ratio 0.1 \
  --max-len 4096 \
  --epochs 1 \
  --batch-token-budget 12000 \
  --device cuda \
  --amp \
  --output-dir outputs/etd_multitask/smoke_run
```

### 4) 全量训练（示例）
```bash
python scripts/training/train_etd_multitask.py \
  --smoke-ratio 1.0 \
  --max-len 12000 \
  --epochs 6 \
  --batch-token-budget 24000 \
  --device cuda \
  --amp \
  --output-dir outputs/etd_multitask/full_run
```

也可以将训练参数集中写入 Python 配置文件（示例：`scripts/training/configs/train_etd_multitask_full.py`）：

```bash
python scripts/training/train_etd_multitask.py \
  --config scripts/training/configs/train_etd_multitask_full.py
```

命令行参数会覆盖配置文件中的同名项，例如：

```bash
python scripts/training/train_etd_multitask.py \
  --config scripts/training/configs/train_etd_multitask_full.py \
  --output-dir outputs/etd_multitask/full_run_alt \
  --epochs 3
```

启用 TensorBoard（脚本支持写入 `events`）：

```bash
python scripts/training/train_etd_multitask.py \
  --config scripts/training/configs/train_etd_multitask_full.py \
  --tensorboard \
  --tb-dir outputs/etd_multitask/full_run/tb
```

```bash
tensorboard --logdir outputs/etd_multitask/full_run/tb --port 6006
```

### 5) 评估
```bash
python scripts/training/eval_etd_multitask.py \
  --checkpoint outputs/etd_multitask/full_run/best.pt \
  --split test \
  --device cuda \
  --output outputs/etd_multitask/full_run/eval_test.json
```

### 6) 应用分析（选择性结合 / 弱结合候选）
```bash
python scripts/analysis/analyze_selective_binding.py \
  --checkpoint outputs/etd_multitask/full_run/best.pt \
  --split test \
  --role reader \
  --output-prefix outputs/analysis/selective_binding_report
```

```bash
python scripts/analysis/mine_weak_binding_candidates.py \
  --checkpoint outputs/etd_multitask/full_run/best.pt \
  --split test \
  --role reader \
  --output-prefix outputs/analysis/weak_binding_candidates
```

## 复现注意事项
- `max_len=12000` 的全量训练对显存要求较高，建议先跑 smoke（如 `4096`）验证链路。
- `RNAfold` 全量缓存生成耗时较长，建议分批跑并保留 manifest。
- 当前仓库不含数据与权重，首次复现前需要准备 RMBase/FASTA 输入和本地缓存目录。

## 测试
仓库包含 `tests/` 下的单元测试（RNAfold parser、数据拼 batch、模型 shape、PU loss、条件 mask、分析脚本函数等）。  
如果本地未安装 `pytest`，也可以通过最小脚本逐个导入并调用测试函数做快速检查。

## 许可证与说明
本仓库主要用于研究开发与复现实验流程管理。请遵循相关数据源（如 RMBase、ViennaRNA）的许可证与使用条款。
