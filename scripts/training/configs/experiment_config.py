"""训练实验配置解析模块。

职责：
1) 定义训练脚本全部 CLI 参数（arg parser）。
2) 支持 `--config` JSON 配置文件，并允许命令行覆盖同名项。
3) 允许配置文件里使用 `_...` 开头的“伪注释键”，解析时自动忽略。

设计动机：
- 将训练主脚本中的参数定义与配置合并逻辑解耦，避免 `train_etd_multitask.py`
  因参数区块过长而影响可读性。
"""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import torch


def _builtin_preset(name: str) -> dict:
    """返回内置实验预设。

    当前支持：
    - default: 不覆盖 parser 默认值
    - smoke_gpu: 在线结构小样本冒烟（按当前联调命令固化）
    """
    if name == "default":
        return {}
    if name == "smoke_gpu": # /root/miniconda3/envs/m6a/bin/python scripts/training/train_etd_multitask.py --preset smoke_gpu

        return {
            "epochs": 6,
            "smoke_ratio": 0.5,
            "batch_token_budget": 4096,
            "max_len": 2000,
            "device": "cuda",
            "struct_source": "online",
            "online_rnafold_bin": "/root/miniconda3/envs/m6a/bin/RNAfold",
            "online_rnafold_timeout_seconds": 900,
            "batch_log_interval": 1,
            "output_dir": "outputs/etd_multitask/smoke_run_gpu_online_0005",
            "tensorboard": False,
            "disable_cudnn": True,
            "ablate_no_struct": True,
            "loss_w_struct": 0.0,
        }
    raise ValueError(f"Unknown preset: {name}")


def build_train_arg_parser(repo_root: Path) -> argparse.ArgumentParser:
    """构建训练实验参数解析器。

    参数按功能分组：
    - 数据路径与规模
    - 优化器/学习率
    - 多任务与损失相关
    - 消融开关
    - 运行时与日志
    """
    parser = argparse.ArgumentParser(description="Train ETD multitask model.")

    # ---- 配置文件入口 ----
    # 说明：若提供 --config，先加载 JSON 再由命令行覆盖。
    parser.add_argument(
        "--config",
        default="",
        help="Path to config file (.json/.py). CLI args override config values.",
    )
    parser.add_argument(
        "--preset",
        default="default",
        choices=["default", "smoke_gpu"],
        help="Built-in experiment preset. Applied before --config and CLI overrides.",
    )

    # ---- 数据路径与规模 ----
    parser.add_argument("--sites", default=str(repo_root / "data/processed/m6a_multitask_sites.parquet"))
    parser.add_argument("--transcripts", default=str(repo_root / "data/processed/m6a_multitask_transcripts.parquet"))
    parser.add_argument("--splits", default=str(repo_root / "data/processed/m6a_multitask_splits.json"))
    parser.add_argument("--rnafold-cache", default=str(repo_root / "data/processed/rnafold_bpp"))
    parser.add_argument(
        "--rnafold-single-site-cache",
        default=str(repo_root / "data/processed/rnafold_single_site_dense"),
        help="Reserved path for offline single-site replacement RNAfold dense cache.",
    )
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--smoke-ratio", type=float, default=1.0)
    parser.add_argument("--batch-token-budget", type=int, default=24000)
    parser.add_argument("--output-dir", default=str(repo_root / "outputs/etd_multitask"))

    # ---- 训练与优化超参数 ----
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # ---- 多任务采样与损失相关 ----
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mod-unlabeled-ratio", type=float, default=1.0)
    parser.add_argument(
        "--aprime-enable",
        action=argparse.BooleanOptionalAction,
        dest="aprime_enable",
        default=True,
    )
    parser.add_argument("--aprime-prob", type=float, dest="aprime_prob", default=0.7)
    parser.add_argument("--aprime-max-per-seq", type=int, dest="aprime_max_per_seq", default=-1)

    parser.add_argument(
        "--bind-grouped-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bind-g3-prob-max", type=float, default=0.3)
    parser.add_argument("--bind-g5-prob-max", type=float, default=0.2)
    parser.add_argument("--bind-g5-unc-min", type=float, default=0.6)
    parser.add_argument("--bind-g1-unc-max", type=float, default=0.2)
    parser.add_argument("--bind-legacy-supervised-unc-max", type=float, default=0.2)
    parser.add_argument("--bind-legacy-supervised-prob-min", type=float, default=0.8)
    parser.add_argument("--bind-legacy-unsupervised-unc-min", type=float, default=0.6)

    # 总损失权重（由 task_loss_composer 使用）。
    parser.add_argument("--loss-w-mod", type=float, default=1.0)
    parser.add_argument("--loss-w-bind", type=float, default=1.2)
    parser.add_argument("--loss-w-dir-in-bind", type=float, default=0.2)
    parser.add_argument("--loss-w-struct", type=float, default=0.8)
    parser.add_argument("--loss-w-mlm", type=float, default=0.2)
    parser.add_argument("--loss-w-unc", type=float, default=0.2)

    parser.add_argument("--cond-mask-role-prob", type=float, default=0.3)
    parser.add_argument("--cond-mask-base-prob", type=float, default=0.3)
    parser.add_argument("--struct-min-sep", type=int, default=4)
    parser.add_argument("--bucket-boundaries", default="1024,2048,4096,8192,12000")
    parser.add_argument("--struct-source", choices=["precomputed", "online"], default="precomputed")
    parser.add_argument("--online-rnafold-bin", default="RNAfold")
    parser.add_argument("--online-rnafold-timeout-seconds", type=int, default=240)
    parser.add_argument("--online-rnafold-cache-size", type=int, default=2048)

    # ---- 消融开关 ----
    parser.add_argument("--ablate-no-condition", action="store_true")
    parser.add_argument("--ablate-no-dirichlet", action="store_true")
    parser.add_argument("--ablate-no-struct", action="store_true")

    # ---- 运行时与日志 ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--disable-cudnn",
        action=argparse.BooleanOptionalAction,
        dest="disable_cudnn",
        default=False,
        help="Disable cuDNN kernels (useful to avoid rare illegal-memory issues on some CUDA stacks).",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tb-dir", default="")
    parser.add_argument("--tb-log-steps", type=int, default=20)
    parser.add_argument("--batch-log-interval", type=int, default=10)
    return parser


def _load_config_payload(config_path: Path) -> dict:
    """从配置文件加载参数字典。

    支持格式：
    - `.json`：标准 JSON 对象
    - `.py`：Python 脚本，需暴露 `CONFIG`（dict）

    说明：
    - 无论哪种格式，都会忽略以 `_` 开头的键，便于写分段标题或注释字段。
    """
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif suffix == ".py":
        namespace = runpy.run_path(str(config_path))
        payload = namespace.get("CONFIG", None)
        if payload is None:
            raise ValueError(f"Python config must define `CONFIG` dict: {config_path}")
    else:
        raise ValueError(f"Unsupported config format: {config_path} (expect .json or .py)")

    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a dict object: {config_path}")
    return {k: v for k, v in payload.items() if not str(k).startswith("_")}


def parse_train_args(repo_root: Path) -> argparse.Namespace:
    """解析训练参数，并处理配置文件覆盖逻辑（支持 .json/.py）。

    行为：
    1) 先用 bootstrap 解析 `--config` 路径（若有）。
    2) 基于默认 parser 加载配置文件默认值。
    3) 命令行参数最终覆盖配置文件中的同名键。
    """
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="")
    bootstrap.add_argument("--preset", default="default")
    known, _ = bootstrap.parse_known_args()

    parser = build_train_arg_parser(repo_root)

    # 先应用内置预设，再叠加配置文件，最后由 CLI 覆盖。
    preset_payload = _builtin_preset(str(known.preset))
    if preset_payload:
        parser.set_defaults(**preset_payload)

    if known.config:
        config_path = Path(known.config)
        config = _load_config_payload(config_path)

        # 向后兼容旧键名：mod_aprime_* -> aprime_*
        legacy_key_map = {
            "mod_aprime_enable": "aprime_enable",
            "mod_aprime_prob": "aprime_prob",
            "mod_aprime_max_per_seq": "aprime_max_per_seq",
        }
        for old_key, new_key in legacy_key_map.items():
            if old_key in config and new_key not in config:
                config[new_key] = config.pop(old_key)
            elif old_key in config and new_key in config:
                config.pop(old_key)

        # 限制配置键必须在 parser 已注册参数中，防止拼写错误被静默接受。
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(k for k in config if k not in valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown config keys in {config_path}: {unknown_keys}")

        parser.set_defaults(**config)
        parser.set_defaults(config=str(config_path))

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

    boundaries_raw = str(cfg.get("bucket_boundaries", ""))
    boundaries = [int(x.strip()) for x in boundaries_raw.split(",") if x.strip()]
    if not boundaries:
        boundaries = [1024, 2048, 4096, 8192, 12000]
    cfg["bucket_boundaries_list"] = boundaries
    return cfg
