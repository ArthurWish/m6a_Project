"""训练参数定义模块。

当前约定：
1) 所有训练参数只维护在这一个 `ArgumentParser` 中。
2) 训练主脚本只负责调用这里的 parser，并读取规范化后的 `cfg`。
- 每个参数的默认值、作用和实验语义都放在同一处维护。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

def build_train_arg_parser(repo_root: Path) -> argparse.ArgumentParser:
    """构建训练实验参数解析器。

    设计原则：
    - 所有参数都能直接从命令行覆盖。
    - 默认值应尽量对应当前主实验，而不是历史兼容值。
    - 参数帮助文字尽量写明“它影响哪一段训练逻辑”。
    """
    parser = argparse.ArgumentParser(description="Train ETD multitask model.")

    # ---- 数据路径与规模 ----
    parser.add_argument(
        "--experiment-name",
        default="default_run",
        help="实验名称；用于派生最终输出目录与默认 TensorBoard 目录，建议每次实验显式指定。",
    )
    parser.add_argument(
        "--sites",
        default=str(repo_root / "data/processed/all_multitask_sites.parquet"),
        help="位点级标注表（修饰位点、role标签、support等）的 parquet 路径。",
    )
    parser.add_argument(
        "--transcripts",
        default=str(repo_root / "data/processed/all_multitask_transcripts.parquet"),
        help="转录本序列表的 parquet 路径；collate 时从这里读取完整 RNA 序列。",
    )
    parser.add_argument(
        "--splits",
        default=str(repo_root / "data/processed/all_multitask_splits.json"),
        help="train/val/test 划分文件路径。",
    )
    parser.add_argument(
        "--rnafold-cache",
        default=str(repo_root / "data/processed/rnafold_bpp"),
        help="旧版预计算 RNAfold cache 路径；仅在 struct_source=precomputed 时使用。",
    )
    parser.add_argument(
        "--rnafold-single-site-cache",
        default=str(repo_root / "data/processed/rnafold_single_site_dense"),
        help="离线单点替换 RNAfold dense cache 路径；当前训练主流程预留，供后续查表结构监督使用。",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=12000,
        help="样本最大序列长度；超过该长度的转录本会在加载阶段被过滤掉。",
    )
    parser.add_argument(
        "--smoke-ratio",
        type=float,
        default=1.0,
        help="按比例抽样 train/val 数据，1.0 表示全量；小于 1 时用于冒烟或快速实验。",
    )
    parser.add_argument(
        "--batch-token-budget",
        type=int,
        default=36000,
        help="动态分桶后每个 batch 的 token 预算上限；控制吞吐和显存占用。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "outputs/etd_multitask/no-struct312"),
        help="训练输出根目录；最终实验目录会解析为 output_dir/experiment_name。",
    )

    # ---- 训练与优化超参数 ----
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数。")
    parser.add_argument("--grad-accum", type=int, default=4, help="梯度累积步数；每累计这么多 batch 才执行一次 optimizer.step。")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW 初始学习率。")
    parser.add_argument("--min-lr", type=float, default=2e-5, help="余弦退火到末尾时的最小学习率。")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW 权重衰减系数。")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="学习率 warmup 步数。")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="每次优化步前做梯度裁剪的 max norm。")

    # ---- 多任务采样与损失相关 ----
    parser.add_argument("--mask-prob", type=float, default=0.15, help="MLM 任务中标准 AUCG token 被选中做 mask 的概率。")
    parser.add_argument("--mod-unlabeled-ratio", type=float, default=1.0, help="mod 任务中未标注 A 的采样倍率，相对正样本数计算。")
    parser.add_argument(
        "--aprime-enable",
        action=argparse.BooleanOptionalAction,
        dest="aprime_enable",
        default=True,
        help="是否启用 A' 输入增强；开启后会在可修饰位点中随机做 A->A' 替换。",
    )
    parser.add_argument(
        "--aprime-prob",
        type=float,
        dest="aprime_prob",
        default=1.0,
        help="每条样本触发 A' 替换的概率；是否替换先过这一层 Bernoulli 门。",
    )
    parser.add_argument(
        "--aprime-max-per-seq",
        type=int,
        dest="aprime_max_per_seq",
        default=-1,
        help="单条样本最多替换多少个 A 为 A'；-1 交由实现侧解释，当前主逻辑一般限制为 1。",
    )

    parser.add_argument(
        "--bind-grouped-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 bind 的 G1~G5 分组损失；关闭则退回 legacy bind loss。",
    )
    parser.add_argument("--bind-g3-prob-max", type=float, default=0.3, help="G3 组概率上界；超过该值会被惩罚。")
    parser.add_argument("--bind-g5-prob-max", type=float, default=0.2, help="G5 普通 A 锚点的概率上界。")
    parser.add_argument("--bind-g5-unc-min", type=float, default=0.6, help="G5 普通 A 锚点的不确定度下界。")
    parser.add_argument("--bind-g1-unc-max", type=float, default=0.2, help="G1 正样本的不确定度上界。")
    parser.add_argument("--bind-legacy-supervised-unc-max", type=float, default=0.2, help="legacy bind loss 下，监督位点允许的不确定度上界。")
    parser.add_argument("--bind-legacy-supervised-prob-min", type=float, default=0.8, help="legacy bind loss 下，监督位点期望达到的最小概率。")
    parser.add_argument("--bind-legacy-unsupervised-unc-min", type=float, default=0.6, help="legacy bind loss 下，未监督位点的不确定度下界。")

    # 总损失权重（由 task_loss_composer 使用）。
    parser.add_argument("--loss-w-mod", type=float, default=1.0, help="mod 主损失权重。")
    parser.add_argument("--loss-w-bind", type=float, default=1.2, help="bind 主损失权重。")
    parser.add_argument("--loss-w-dir-in-bind", type=float, default=0.2, help="bind 中 Dirichlet/evidential 子项的附加权重。")
    parser.add_argument("--loss-w-struct", type=float, default=0.8, help="struct 结构矩阵监督损失权重。")
    parser.add_argument("--loss-w-mlm", type=float, default=0.2, help="MLM 损失权重。")
    parser.add_argument("--loss-w-unc", type=float, default=0.2, help="bind 中不确定度约束损失权重。")

    parser.add_argument("--cond-mask-role-prob", type=float, default=0.3, help="条件注入中 role 条件被 mask 掉的概率。")
    parser.add_argument("--cond-mask-base-prob", type=float, default=0.3, help="条件注入中 base 条件被 mask 掉的概率。")
    parser.add_argument("--cond-mask-mod-type-prob", type=float, default=0.3, help="条件注入中 mod_type 条件被 mask 掉的概率。")
    parser.add_argument("--struct-min-sep", type=int, default=4, help="结构矩阵监督时允许配对的最小线性间隔。")
    parser.add_argument(
        "--bucket-boundaries",
        default="1024,2048,4096,8192,12000",
        help="动态分桶边界，逗号分隔；按序列长度划分 batch，减少 padding 浪费。",
    )
    parser.add_argument(
        "--struct-source",
        choices=["precomputed", "online"],
        default="precomputed",
        help="结构监督来源：precomputed 使用已有 cache，online 运行时调用 RNAfold。",
    )
    parser.add_argument("--online-rnafold-bin", default="RNAfold", help="在线 RNAfold 模式下可执行文件路径。")
    parser.add_argument("--online-rnafold-timeout-seconds", type=int, default=240, help="单条 RNAfold 在线计算超时时间。")
    parser.add_argument("--online-rnafold-cache-size", type=int, default=2048, help="在线 RNAfold provider 的内存缓存条目数。")

    # ---- 消融开关 ----
    parser.add_argument("--ablate-no-condition", action="store_true", help="条件注入消融：task/role/base/mod_type 条件全部退化为无信息状态。")
    parser.add_argument("--ablate-no-dirichlet", action="store_true", help="关闭 bind 中的 Dirichlet/evidential 相关损失。")
    parser.add_argument("--ablate-no-struct", action="store_true", help="关闭 struct 分支训练，并跳过结构 provider 初始化。")

    # ---- 运行时与日志 ----
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子。")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="单卡运行时使用的 device；DDP 下会被 local_rank 覆盖。")
    parser.add_argument(
        "--disable-cudnn",
        action=argparse.BooleanOptionalAction,
        dest="disable_cudnn",
        default=False,
        help="是否禁用 cuDNN；某些环境里可规避少见的 illegal-memory / 内核不稳定问题。",
    )
    parser.add_argument("--amp", action="store_true", help="启用 AMP 混合精度训练。")
    parser.add_argument("--no-amp", action="store_true", help="显式关闭 AMP；优先级高于 --amp。")
    parser.add_argument("--tensorboard", action="store_true", help="是否写 TensorBoard 日志。")
    parser.add_argument("--tb-dir", default="", help="TensorBoard 根目录；为空时默认写到最终实验目录下的 tb 子目录。")
    parser.add_argument("--tb-log-steps", type=int, default=20, help="按 step 写 TensorBoard 的频率。")
    parser.add_argument("--batch-log-interval", type=int, default=10, help="终端 batch 日志打印间隔。")
    return parser


def parse_train_args(repo_root: Path) -> argparse.Namespace:
    """解析训练参数。

    当前不再做任何“二次来源覆盖”：
    - 没有 preset
    - 没有外部配置文件
    - 所有实验变体都通过命令行参数直接表达
    """
    parser = build_train_arg_parser(repo_root)
    return parser.parse_args()


def resolve_train_config(args: argparse.Namespace) -> dict:
    """将 argparse Namespace 规范化为训练配置字典。

    目标：
    1) 提供单一入口，让训练代码统一通过 `cfg[...]` 读取参数。
    2) 在这里集中处理“派生参数”，避免主循环重复写样板逻辑。

    当前会补充的派生字段：
    - `use_amp`: 是否启用混合精度（`amp` 且非 `no_amp`）
    - `bucket_boundaries_list`: 解析后的长度分桶边界（整数列表）
    """
    cfg = dict(vars(args))
    cfg["use_amp"] = bool(cfg.get("amp", False)) and not bool(cfg.get("no_amp", False))

    experiment_name = str(cfg.get("experiment_name", "default_run")).strip()
    if not experiment_name:
        experiment_name = "default_run"
    cfg["experiment_name"] = experiment_name

    # 所有实验输出统一落到 output_dir/experiment_name 下，避免不同运行互相覆盖。
    output_root = Path(str(cfg.get("output_dir", "")))
    cfg["output_dir"] = str(output_root / experiment_name)

    # TensorBoard 若未单独指定，则默认跟随实验目录；若指定的是相对路径，也按实验名分目录。
    tb_dir_raw = str(cfg.get("tb_dir", "")).strip()
    if tb_dir_raw:
        tb_root = Path(tb_dir_raw)
        cfg["tb_dir"] = str(tb_root / experiment_name)
    else:
        cfg["tb_dir"] = ""

    boundaries_raw = str(cfg.get("bucket_boundaries", ""))
    boundaries = [int(x.strip()) for x in boundaries_raw.split(",") if x.strip()]
    if not boundaries:
        boundaries = [1024, 2048, 4096, 8192, 12000]
    cfg["bucket_boundaries_list"] = boundaries
    return cfg
