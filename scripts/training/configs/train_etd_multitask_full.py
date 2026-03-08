"""全量训练配置（Python 版，可写详尽注释）。

使用方式：
    python scripts/training/train_etd_multitask.py \
      --config scripts/training/configs/train_etd_multitask_full.py

说明：
1) 本文件通过 `CONFIG` 字典向训练脚本提供参数默认值。
2) 任何命令行参数都会覆盖这里的同名项。
3) 你可以在这里写任意 Python 注释，比 JSON 更适合记录实验意图。
"""

from __future__ import annotations


CONFIG = {
    # ==================== 数据路径与规模 ====================
    # 位点表（每条 m6A 位点 + writer/reader/eraser 标签与支持度）
    "sites": "data/processed/m6a_multitask_sites.parquet",
    # 转录本表（full_sequence + m6a_positions）
    "transcripts": "data/processed/m6a_multitask_transcripts.parquet",
    # 数据划分文件（train/val/test transcript_id 列表）
    "splits": "data/processed/m6a_multitask_splits.json",
    # RNAfold 稀疏 BPP 缓存目录
    "rnafold_cache": "data/processed/rnafold_bpp",
    # 预留：单位点替换(A->6)离线 RNAfold 稠密缓存目录（当前训练尚未接入）
    "rnafold_single_site_cache": "data/processed/rnafold_single_site_dense",
    # 训练允许的最大序列长度（超出会在加载阶段过滤）
    "max_len": 12000,
    # smoke 联调用采样比例；1.0 表示全量
    "smoke_ratio": 1.0,
    # 动态 batch 的 token 预算（不是固定 batch_size）
    "batch_token_budget": 24000,
    # 输出目录（ckpt、config、metrics、tensorboard）
    "output_dir": "outputs/etd_multitask/full_run",

    # ==================== 训练与优化超参数 ====================
    "epochs": 6,
    # 梯度累积步数：等效总 batch 提升约为 grad_accum 倍
    "grad_accum": 4,
    "lr": 2e-4,
    "min_lr": 2e-5,
    "weight_decay": 1e-2,
    "warmup_steps": 2000,
    "grad_clip": 1.0,

    # ==================== 多任务与输入增强 ====================
    # MLM 的 token 采样遮蔽概率
    "mask_prob": 0.15,
    # mod 任务中 unlabeled A 与 positive m6A 的采样比例
    "mod_unlabeled_ratio": 1.0,
    # A' 输入增强：是否开启
    "aprime_enable": True,
    # 每个 m6A 位点被替换为 A' 的概率
    "aprime_prob": 0.1,
    # 每条序列最多替换多少个 m6A；-1 表示不设上限
    "aprime_max_per_seq": -1,

    # ==================== bind 分支损失策略 ====================
    # True: 启用 G1~G5 grouped loss；False: 回退 legacy loss
    "bind_grouped_loss": True,
    # G3（A + 正样本）概率上限约束
    "bind_g3_prob_max": 0.3,
    # G5（普通 A）概率上限约束
    "bind_g5_prob_max": 0.2,
    # G5 最小不确定度约束（鼓励普通 A 不要过度自信）
    "bind_g5_unc_min": 0.6,
    # G1（A' + 正样本）最大不确定度约束
    "bind_g1_unc_max": 0.2,
    # legacy bind: supervised 路径不确定度上限
    "bind_legacy_supervised_unc_max": 0.2,
    # legacy bind: supervised 路径概率下限
    "bind_legacy_supervised_prob_min": 0.8,
    # legacy bind: unsupervised 路径不确定度下限
    "bind_legacy_unsupervised_unc_min": 0.6,

    # ==================== 总损失权重 ====================
    "loss_w_mod": 1.0,
    "loss_w_bind": 1.2,
    # bind 内部 dirichlet 分量权重
    "loss_w_dir_in_bind": 0.2,
    "loss_w_struct": 0.8,
    "loss_w_mlm": 0.2,
    "loss_w_unc": 0.2,

    # ==================== 条件采样/结构任务 ====================
    # 条件 role 随机置空概率（none）
    "cond_mask_role_prob": 0.3,
    # 条件 base 随机置空概率（mask）
    "cond_mask_base_prob": 0.3,
    # 结构损失最小碱基距离约束
    "struct_min_sep": 4,
    # 动态 batch 长度分桶边界
    "bucket_boundaries": "1024,2048,4096,8192,12000",
    # 结构来源：precomputed=读取缓存；online=按当前序列实时 RNAfold
    "struct_source": "online",
    "online_rnafold_bin": "RNAfold",
    "online_rnafold_timeout_seconds": 240,
    "online_rnafold_cache_size": 2048,

    # ==================== 消融开关 ====================
    "ablate_no_condition": False,
    "ablate_no_dirichlet": False,
    "ablate_no_struct": False,

    # ==================== 运行环境与日志 ====================
    "seed": 42,
    "device": "cuda",
    "amp": True,
    "no_amp": False,
    "tensorboard": False,
    "tb_dir": "outputs/etd_multitask/full_run/tb",
    "tb_log_steps": 20,
}
