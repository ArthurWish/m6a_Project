# Training Experiments

本目录下的训练脚本可以分成 4 条线。

## 1. 单任务 Mask

- 脚本: `train_etd_only.py`
- 任务: mask-only
- 输入: 整条 transcript
- 配置: `configs/etd_mask_config.py`
- 输出目录: `outputs/etd_only/...`

## 2. 单任务 Bind 主线

- 脚本: `train_etd_bind.py`
- 任务: bind-only
- 输入: 整条 transcript + `site_positions`
- 监督: m6A `reader` 的 `RBP family multi-label`
- 配置: `configs/etd_bind_config.py`
- 说明: 当前主线，不是窗口版
- 输出目录: `outputs/etd_only/...`

## 3. 窗口版 Bind

### v2

- 脚本: `train_etd_bind_short.py`
- 模型: `models/etd_only/etd_bind_v2.py`
- 数据: `models/etd_only/bind_dataloader_v2.py`
- 输入: 以中心位点截取的固定窗口
- 默认窗口: `half_window=512`，即 `1025 nt`
- 监督: 中心位点的 individual RBP multi-label
- 输出目录: `outputs/etd_bind_v2/...`

### v3

- 脚本: `train_etd_bind_short-v2.py`
- 模型: `models/etd_only/etd_bind_v3.py`
- 数据: `models/etd_only/bind_dataloader_v3.py`
- 输入: 固定窗口
- 相比 v2: 增加窗口内 `hard negative A`
- 输出目录: `outputs/etd_bind_v2/...`

## 4. 窗口版 m6A + Bind 联合训练

### Current

- 脚本: `train_etd_together.py`
- 模型:
  - baseline: `models/etd_only/etd_bind_v4.py`
  - bias: `models/etd_only/etd_bind_bias.py`
- 数据: `models/etd_only/bind_dataloader_v5.py`
- 输入: 固定窗口
- 任务: `m6A detection + RBP binding`
- 特点:
  - baseline / offline-bias 共用一套 current 数据管线
  - bias 版支持离线结构 cache
- 当前配置:
  - baseline: `configs/etd_together_baseline.py`
  - bias: `configs/etd_together_offline_bias.py`

### Legacy

- 历史 v5 入口已归档到: `legacy/train_etd_together_v5.py`
- 历史 v5 bias 配置已归档到: `configs/legacy/train_etd_together_bias_v5.py`
- 原路径保留兼容包装，但不再作为主线维护

## 5. 多任务主线

- 脚本: `train_etd_multitask.py`
- 任务: `mod / bind / struct / mask`
- 输入: 整条 transcript
- 配置: `configs/experiment_config.py` 或 `configs/train_etd_multitask_full.py`
- 说明: 与单任务 bind 线分开维护
- 输出目录: `outputs/etd_multitask/...`

## 常用判断

- 看到 `outputs/etd_only/...`: 通常是整条序列单任务线
- 看到 `outputs/etd_bind_v2/...`: 通常是窗口 bind v2/v3
- 看到 `outputs/etd_bind_v4/...`: 通常是窗口 m6A + bind 联合训练
- 看到 `outputs/etd_multitask/...`: 通常是整条序列多任务主线
