#!/usr/bin/env python3
"""Train ETD multi-task model for m6A multitask objectives."""

from __future__ import annotations

import argparse
import atexit
import ctypes
import datetime
import math
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _bootstrap_cuda_runtime_paths() -> None:
    """Add pip-installed NVIDIA runtime paths if present."""
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidate = (
        Path(sys.prefix)
        / "lib"
        / f"python{pyver}"
        / "site-packages"
        / "nvidia"
        / "cusparselt"
        / "lib"
    )
    if candidate.exists():
        curr = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [str(candidate)]
        if curr:
            parts.append(curr)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        lib_path = candidate / "libcusparseLt.so.0"
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


_bootstrap_cuda_runtime_paths()

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from models.etd_multitask.constants import COND_BASE_IDS, MOD_TYPE_IDS, MOD_TYPE_NAMES, ROLE_IDS, ROLE_NAMES, TASK_IDS, TASK_PROBS
from models.etd_multitask.data import (
    BPPCache,
    apply_condition_mask,
    build_length_bucketed_batches,
    collate_batch,
    estimate_binding_priors,
    estimate_mod_prior,
    estimate_strong_binding_thresholds,
    load_examples,
    sample_task_condition,
    sample_task_name,
)
from models.etd_multitask.losses import (
    NonNegativePULoss,
)
from models.etd_multitask.metrics import (
    binary_auprc,
    binary_auroc,
    binary_f1,
    expected_calibration_error,
)
from models.etd_multitask.model import ETDMultiTaskModel
from models.etd_multitask.rnafold_online import OnlineRNAfoldProvider
from models.etd_multitask.task_loss_composer import compute_multitask_losses
from models.etd_multitask.utils import save_json, set_seed
from scripts.training.configs.experiment_config import parse_train_args, resolve_train_config
from models.etd_multitask.evaluate import evaluate_mod_all_types, evaluate_bind_all_types



def _setup_distributed() -> tuple[int, int, int, bool]:
    if "RANK" not in os.environ:
        return 0, 0, 1, False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)                          # 先 set device
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"), timeout=datetime.timedelta(minutes=120))  # ← 加上 device_id
    return rank, local_rank, world_size, True


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _is_main(rank: int) -> bool:
    return rank == 0


def _shard_batches(
    batches: list,
    rank: int,
    world_size: int,
) -> list:

    if world_size <= 1:
        return batches
    n = len(batches)
    usable = n - (n % world_size)
    batches = batches[:usable]
    return batches[rank::world_size]


def parse_args() -> argparse.Namespace:
    """训练脚本参数入口（具体定义与配置覆盖逻辑在 experiment_config 模块）。"""
    return parse_train_args(REPO_ROOT)


class _TeeStream:
    """Mirror writes to terminal and a log file."""

    def __init__(self, terminal, file_handle):
        self._terminal = terminal
        self._file = file_handle

    def write(self, data):
        try:
            self._terminal.write(data)
        except Exception:
            pass
        try:
            self._file.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self._terminal.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass


def _setup_train_log_file(out_dir: Path):
    """Route stdout/stderr to both terminal and output_dir/train.log."""
    log_path = out_dir / "train.log"
    fh = log_path.open("a", encoding="utf-8")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _TeeStream(orig_stdout, fh)
    sys.stderr = _TeeStream(orig_stderr, fh)

    def _cleanup():
        try:
            # 先恢复原始流，避免解释器退出阶段再次通过已关闭文件 flush。
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
        except Exception:
            pass
        try:
            fh.flush()
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass

    atexit.register(_cleanup)
    print(f"[log] writing training logs to {log_path}", flush=True)


def to_device(batch: dict, device: torch.device) -> dict:
    """将 batch 内所有张量移动到指定 device。

    这里不直接假设 batch 全是 tensor，因为 `collate_batch` 还会返回
    `transcript_ids` 等 Python 对象；这些字段应原样保留。
    """
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


# def build_scheduler(
#     optimizer: torch.optim.Optimizer,
#     total_steps: int,
#     warmup_steps: int,
#     min_lr: float,
#     base_lr: float,
# ):
#     """构建 warmup + cosine 学习率调度器。

#     训练步长定义：
#     - step < warmup_steps: 线性升温
#     - 其余阶段: 从 base_lr 余弦衰减到 min_lr

#     返回：
#     - `torch.optim.lr_scheduler.LambdaLR`
#     """
#     warmup_steps = max(1, int(warmup_steps))
#     total_steps = max(warmup_steps + 1, int(total_steps))
#     min_ratio = float(min_lr / base_lr) if base_lr > 0 else 0.1

#     def lr_lambda(step: int) -> float:
#         if step < warmup_steps:
#             return float(step + 1) / float(warmup_steps)
#         progress = (step - warmup_steps) / float(total_steps - warmup_steps)
#         progress = min(max(progress, 0.0), 1.0)
#         cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
#         return min_ratio + (1.0 - min_ratio) * cosine

#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
    base_lr: float,
):
    """构建 warmup + constant 学习率调度器。

    训练步长定义：
    - step < warmup_steps: 线性升温
    - 其余阶段: 保持 base_lr 不变

    返回：
    - `torch.optim.lr_scheduler.LambdaLR`
    """
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate_reader_binding(
    model: ETDMultiTaskModel,
    examples,
    struct_provider: Any,
    strong_thresholds: dict[str, float],
    batch_token_budget: int,
    boundaries: list[int],
    device: torch.device,
) -> dict[str, float]:
    """在验证集评估 `reader` 绑定分支。

    说明：
    - 训练期间仅做 reader 验证，是为了控制验证成本并与 best 模型选择标准一致。
    - 这里只做推理与统计，不更新参数（`model.eval()` + `torch.no_grad()`）。
    todo：后续可考虑增加 mod/struct 的验证，或者在训练结束后做全量评估。
    """
    model.eval()
    batches = build_length_bucketed_batches(
        examples=examples,
        batch_token_budget=batch_token_budget,
        boundaries=boundaries,
        shuffle=False,
        seed=0,
    )

    y_true_all = []
    y_prob_all = []
    unc_all = []
    strong_all = []
    total_g1 = 0
    total_g5 = 0
    total_g5_sampled = 0
    rng_eval = np.random.default_rng(0)

    with torch.no_grad():
        for batch_examples in batches:
            # 评估阶段固定随机种子，避免 collate 内随机采样导致指标抖动。
            rng = random.Random(0)
            # 对每个验证 batch 做一次 bind-reader 前向
            batch = collate_batch(
                # 条件 token 固定成 bind/reader/A。
                examples=batch_examples,
                task_name="bind",
                role_name="reader",
                cond_base="A",
                sampled_mod_type="m6A",
                struct_provider=struct_provider,
                strong_binding_threshold=strong_thresholds["m6A"]['reader'],
                rng=rng,
                mod_unlabeled_ratio=1.0,
                mask_prob=0.15,
            )
            batch = to_device(batch, device)

            bsz = batch["tokens"].shape[0]
            cond_task = torch.full((bsz,), TASK_IDS["bind"], device=device, dtype=torch.long)
            cond_role = torch.full((bsz,), ROLE_IDS["reader"], device=device, dtype=torch.long)
            cond_base = torch.full((bsz,), COND_BASE_IDS["A"], device=device, dtype=torch.long)
            cond_mod_type = torch.full((bsz,), MOD_TYPE_IDS["m6A"], device=device, dtype=torch.long)
            # 拿到模型输出 bind_logits、bind_uncertainty，以及 site_mask 用于后续统计。
            outputs = model(
                tokens=batch["tokens"],
                cond_task=cond_task,
                cond_role=cond_role,
                cond_base=cond_base,
                cond_mod_type=cond_mod_type,
                attn_mask=batch["attn_mask"],
                site_positions=batch["site_positions"],
                site_mask=batch["site_mask"],
                compute_struct=False,
            )

            # 评估口径（G1 vs G5）：
            # - 正样本：G1（m6A 且本次 reveal 为 A' 且 role=1）
            # - 真负样本：G5（非 m6A 的普通 A）
            site_mask = batch["site_mask"].detach().cpu().numpy().astype(bool)
            g1 = (batch["g1_mask"].detach().cpu().numpy().astype(bool)) & site_mask
            g5 = (batch["g5_mask"].detach().cpu().numpy().astype(bool)) & site_mask

            # 评估集平衡采样：每个 batch 里按 1:1 采样 G5 真负样本到与 G1 正样本同量。
            g1_idx = np.flatnonzero(g1.reshape(-1))
            g5_idx = np.flatnonzero(g5.reshape(-1))
            if g1_idx.size == 0 or g5_idx.size == 0:
                continue
            k = min(int(g1_idx.size), int(g5_idx.size))
            sampled_g5_idx = rng_eval.choice(g5_idx, size=k, replace=False)

            eval_mask = np.zeros_like(site_mask, dtype=bool)
            eval_mask_flat = eval_mask.reshape(-1)
            eval_mask_flat[g1_idx] = True
            eval_mask_flat[sampled_g5_idx] = True

            y_true = g1.astype(np.int64)
            y_prob = torch.sigmoid(outputs["bind_logits"]).detach().cpu().numpy()
            unc = outputs["bind_uncertainty"].detach().cpu().numpy()
            strong = batch["strong_binding_mask"].detach().cpu().numpy().astype(bool)

            y_true_all.append(y_true[eval_mask])
            y_prob_all.append(y_prob[eval_mask])
            unc_all.append(unc[eval_mask])
            strong_all.append(strong[eval_mask])
            total_g1 += int(g1.sum())
            total_g5 += int(g5.sum())
            total_g5_sampled += int(k)

    if not y_true_all:
        return {
            "reader_auroc": float("nan"),
            "reader_auprc": float("nan"),
            "reader_f1": float("nan"),
            "reader_ece": float("nan"),
            "reader_unc_strong": float("nan"),
            "reader_unc_weak": float("nan"),
            "reader_unc_gap": float("nan"),
            "reader_eval_g1": 0.0,
            "reader_eval_g5": 0.0,
            "reader_eval_g5_sampled": 0.0,
            "reader_prob_g1": float("nan"),
            "reader_prob_g5": float("nan"),
            "reader_unc_g1": float("nan"),
            "reader_unc_g5": float("nan"),
        }

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    unc = np.concatenate(unc_all)
    strong = np.concatenate(strong_all)
    g1_mask_eval = y_true == 1
    g5_mask_eval = y_true == 0

    strong_mean = float(unc[strong].mean()) if np.any(strong) else float("nan")
    weak_mean = float(unc[~strong].mean()) if np.any(~strong) else float("nan")
    gap = float((weak_mean - strong_mean) / max(weak_mean, 1e-6)) if np.isfinite(strong_mean) and np.isfinite(weak_mean) else float("nan")
    prob_g1 = float(y_prob[g1_mask_eval].mean()) if np.any(g1_mask_eval) else float("nan")
    prob_g5 = float(y_prob[g5_mask_eval].mean()) if np.any(g5_mask_eval) else float("nan")
    unc_g1 = float(unc[g1_mask_eval].mean()) if np.any(g1_mask_eval) else float("nan")
    unc_g5 = float(unc[g5_mask_eval].mean()) if np.any(g5_mask_eval) else float("nan")

    print(
        f"[评估] reader样本统计：G1正样本={total_g1}，G5真负样本={total_g5}，G5采样后={total_g5_sampled}；"
        f"G1平均预测概率={prob_g1:.4f}，G5平均预测概率={prob_g5:.4f}；"
        f"G1平均不确定度={unc_g1:.4f}，G5平均不确定度={unc_g5:.4f}",
        flush=True,
    )

    return {
        "reader_auroc": binary_auroc(y_true, y_prob),
        "reader_auprc": binary_auprc(y_true, y_prob),
        "reader_f1": binary_f1(y_true, y_prob),
        "reader_ece": expected_calibration_error(y_true, y_prob),
        "reader_unc_strong": strong_mean,
        "reader_unc_weak": weak_mean,
        "reader_unc_gap": gap,
        "reader_eval_g1": float(total_g1),
        "reader_eval_g5": float(total_g5),
        "reader_eval_g5_sampled": float(total_g5_sampled),
        "reader_prob_g1": prob_g1,
        "reader_prob_g5": prob_g5,
        "reader_unc_g1": unc_g1,
        "reader_unc_g5": unc_g5,
    }


def evaluate_mod_task(
    model: ETDMultiTaskModel,
    examples,
    struct_provider: Any,
    batch_token_budget: int,
    boundaries: list[int],
    device: torch.device,
) -> dict[str, float]:
    """在验证集评估 mod 分支（m6A 正样本 vs 采样同量未标注 A）。"""
    model.eval()
    batches = build_length_bucketed_batches(
        examples=examples,
        batch_token_budget=batch_token_budget,
        boundaries=boundaries,
        shuffle=False,
        seed=0,
    )

    y_true_all = []
    y_prob_all = []
    total_pos = 0
    total_unl = 0
    total_unl_sampled = 0
    rng_eval = np.random.default_rng(0)

    with torch.no_grad():
        for batch_examples in batches:
            rng = random.Random(0)
            batch = collate_batch(
                examples=batch_examples,
                task_name="mod",
                role_name="reader",
                cond_base="A",
                sampled_mod_type="m6A",
                struct_provider=struct_provider,
                strong_binding_threshold=1.0,
                rng=rng,
                mod_unlabeled_ratio=1.0,
                mask_prob=0.15,
            )
            batch = to_device(batch, device)

            bsz = batch["tokens"].shape[0]
            cond_task = torch.full((bsz,), TASK_IDS["mod"], device=device, dtype=torch.long)
            cond_role = torch.full((bsz,), ROLE_IDS["none"], device=device, dtype=torch.long)
            cond_base = torch.full((bsz,), COND_BASE_IDS["A"], device=device, dtype=torch.long)
            cond_mod_type = torch.full((bsz,), MOD_TYPE_IDS["m6A"], device=device, dtype=torch.long)
            outputs = model(
                tokens=batch["tokens"],
                cond_task=cond_task,
                cond_role=cond_role,
                cond_base=cond_base,
                cond_mod_type = cond_mod_type,
                attn_mask=batch["attn_mask"],
                site_positions=batch["site_positions"],
                site_mask=batch["site_mask"],
                compute_struct=False,
            )

            pu_mask = batch["mod_pu_mask"].detach().cpu().numpy().astype(bool)
            labels = batch["mod_pu_labels"].detach().cpu().numpy()
            pos = (labels == 1) & pu_mask
            unl = (labels == -1) & pu_mask

            pos_idx = np.flatnonzero(pos.reshape(-1))
            unl_idx = np.flatnonzero(unl.reshape(-1))
            if pos_idx.size == 0 or unl_idx.size == 0:
                continue

            k = min(int(pos_idx.size), int(unl_idx.size))
            sampled_unl_idx = rng_eval.choice(unl_idx, size=k, replace=False)

            eval_mask = np.zeros_like(pu_mask, dtype=bool)
            eval_mask_flat = eval_mask.reshape(-1)
            eval_mask_flat[pos_idx] = True
            eval_mask_flat[sampled_unl_idx] = True

            y_true = pos.astype(np.int64)
            y_prob = torch.sigmoid(outputs["mod_logits"]).detach().cpu().numpy()

            y_true_all.append(y_true[eval_mask])
            y_prob_all.append(y_prob[eval_mask])
            total_pos += int(pos.sum())
            total_unl += int(unl.sum())
            total_unl_sampled += int(k)

    if not y_true_all:
        return {
            "mod_auroc": float("nan"),
            "mod_auprc": float("nan"),
            "mod_f1": float("nan"),
            "mod_ece": float("nan"),
            "mod_eval_pos": 0.0,
            "mod_eval_unl": 0.0,
            "mod_eval_unl_sampled": 0.0,
        }

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)

    print(
        f"[评估] mod样本统计：正样本={total_pos}，未标注A={total_unl}，未标注采样后={total_unl_sampled}",
        flush=True,
    )

    return {
        "mod_auroc": binary_auroc(y_true, y_prob),
        "mod_auprc": binary_auprc(y_true, y_prob),
        "mod_f1": binary_f1(y_true, y_prob),
        "mod_ece": expected_calibration_error(y_true, y_prob),
        "mod_eval_pos": float(total_pos),
        "mod_eval_unl": float(total_unl),
        "mod_eval_unl_sampled": float(total_unl_sampled),
    }


def _prepare_data_and_priors(cfg: dict,  rank: int = 0):
    """加载 train/val 样本，并计算训练依赖的先验统计。
    返回顺序：
    1) train_examples
    2) val_examples
    3) struct_provider
    4) mod_priors
    5) bind_priors
    6) strong_thresholds
    """
    t0 = time.perf_counter()
    if _is_main(rank):
        print("[data] loading train split ...", flush=True)
    train_examples = load_examples(
        sites_path=cfg["sites"],
        transcripts_path=cfg["transcripts"],
        splits_path=cfg["splits"],
        split_names=["train"],
        max_len=int(cfg["max_len"]),
        smoke_ratio=float(cfg["smoke_ratio"]),
        seed=int(cfg["seed"]),
    )
    if _is_main(rank):
        print(
            f"[data] train loaded: n={len(train_examples)} elapsed={time.perf_counter() - t0:.2f}s",
            flush=True,
        )
    t1 = time.perf_counter()
    if _is_main(rank):
        print("[data] loading val split ...", flush=True)
    # 默认让 train/val 共享 smoke_ratio，便于一次参数同时缩放训练与验证规模。
    val_smoke_ratio = float(cfg["smoke_ratio"])
    val_examples = load_examples(
        sites_path=cfg["sites"],
        transcripts_path=cfg["transcripts"],
        splits_path=cfg["splits"],
        split_names=["val"],
        max_len=int(cfg["max_len"]),
        smoke_ratio=val_smoke_ratio,
        seed=int(cfg["seed"]),
    )
    if _is_main(rank):
        print(
            f"[data] val loaded: n={len(val_examples)} elapsed={time.perf_counter() - t1:.2f}s",
            flush=True,
        )

    t2 = time.perf_counter()
    if bool(cfg.get("ablate_no_struct", False)):
        # 结构任务消融时，不再初始化任何 RNAfold provider。
        # 后续 collate_batch 会显式关闭 use_rnafold_struct_feats，从而不触发结构统计计算。
        struct_provider = None
        if _is_main(rank):
            print(
                f"[data] struct provider skipped: ablate_no_struct=True elapsed={time.perf_counter() - t2:.2f}s",
                flush=True,
            )
    elif str(cfg.get("struct_source", "precomputed")) == "online":
        struct_provider = OnlineRNAfoldProvider(
            rnafold_bin=str(cfg.get("online_rnafold_bin", "RNAfold")),
            timeout_seconds=int(cfg.get("online_rnafold_timeout_seconds", 240)),
            cache_size=int(cfg.get("online_rnafold_cache_size", 2048)),
        )
        if _is_main(rank):
            print(
                f"[data] struct provider ready: source={cfg.get('struct_source', 'precomputed')} "
                f"elapsed={time.perf_counter() - t2:.2f}s",
                flush=True,
            )
    else:
        struct_provider = BPPCache(cfg["rnafold_cache"])
        if _is_main(rank):
            print(
                f"[data] struct provider ready: source={cfg.get('struct_source', 'precomputed')} "
                f"elapsed={time.perf_counter() - t2:.2f}s",
                flush=True,
            )
    t3 = time.perf_counter()
    mod_priors = estimate_mod_prior(train_examples)
    bind_priors = estimate_binding_priors(train_examples)
    strong_thresholds = estimate_strong_binding_thresholds(train_examples, q=0.75)
    if _is_main(rank):
        print(
        f"[data] priors ready: mod_priors={mod_priors} "
        f"bind_priors={bind_priors} elapsed={time.perf_counter() - t3:.2f}s",
        flush=True,
    )
    return train_examples, val_examples, struct_provider, mod_priors, bind_priors, strong_thresholds


def _build_train_components(
    cfg: dict,
    device: torch.device,
    train_examples,
    boundaries: list[int],
    mod_priors: dict[str, float],
    bind_priors: dict[str, dict[str, float]],
    use_amp: bool,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
):
    """构建训练核心组件。

    包含：
    - 模型与损失对象（mod/bind 的 nnPU）
    - 优化器（AdamW）
    - 学习率调度器（warmup + cosine）
    - AMP GradScaler
    - 以及调度器所需的 `steps_per_epoch/total_steps`
    """
    model = ETDMultiTaskModel().to(device)
    num_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])


    if _is_main(rank):
        print(f"Model initialized with {num_parameters:,} trainable parameters.")

    if world_size > 1:
        has_unused = bool(cfg.get("ablate_no_struct", False))
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=has_unused)

    mod_pu_loss = {mt: NonNegativePULoss(prior=mod_priors.get(mt, 0.5)) for mt in MOD_TYPE_NAMES}
    bind_pu_loss = {
        mt: {role: NonNegativePULoss(prior=bind_priors.get(mt, {}).get(role, 0.5)) for role in ROLE_NAMES}
        for mt in MOD_TYPE_NAMES
    }


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        betas=(0.9, 0.98),
        weight_decay=float(cfg["weight_decay"]),
    )

    # 先做一次 dry-run 分批，用于估算真实优化步数（考虑 grad_accum）。
    dry_batches = build_length_bucketed_batches(
        examples=train_examples,
        batch_token_budget=int(cfg["batch_token_budget"]),
        boundaries=boundaries,
        shuffle=False,
        seed=int(cfg["seed"]),
    )
    n_local_batches = len(_shard_batches(dry_batches, rank=0, world_size=world_size))
    # steps_per_epoch = max(1, math.ceil(len(dry_batches) / int(cfg["grad_accum"])))
    steps_per_epoch = max(1, math.ceil(n_local_batches / int(cfg["grad_accum"])))
    total_steps = max(1, steps_per_epoch * int(cfg["epochs"]))
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(cfg["warmup_steps"]),
        min_lr=float(cfg["min_lr"]),
        base_lr=float(cfg["lr"]),
    )

    try:
        scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and device.type == "cuda")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    return model, mod_pu_loss, bind_pu_loss, optimizer, scheduler, scaler, steps_per_epoch, total_steps


def _run_train_epoch(
    *,
    epoch: int,
    model: ETDMultiTaskModel,
    train_examples,
    task_cycle: list[str],
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    boundaries: list[int],
    struct_provider: Any,
    strong_thresholds: dict[str, float],
    mod_pu_loss: NonNegativePULoss,
    bind_pu_loss: dict[str, NonNegativePULoss],
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    writer,
    global_step: int,
    optim_step: int,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[dict[str, float], int, int]:
    """执行单个 epoch 训练。

    返回：
    - agg: 本 epoch 平均损失统计
    - global_step: 更新后的全局 batch 计数
    - optim_step: 更新后的优化器步数（受 grad_accum 影响）
    """
    batch_log_interval = int(cfg.get("batch_log_interval", 10))
    batch_log_interval = max(1, batch_log_interval)
    amp_enabled = bool(use_amp and device.type == "cuda")

    grad_accum = int(cfg["grad_accum"])
    is_main = _is_main(rank)
    is_ddp = world_size > 1


    model.train()
    t_build_batches = time.perf_counter()

    batches = build_length_bucketed_batches(
        examples=train_examples,
        batch_token_budget=int(cfg["batch_token_budget"]),
        boundaries=boundaries,
        shuffle=True,
        seed=int(cfg["seed"]) + epoch,
    )

    n_total_batches = len(batches)
    batches = _shard_batches(batches, rank=rank, world_size=world_size)

    if is_main:
        print(
            f"[epoch {epoch:03d}] built {n_total_batches} batches, "
            f"{len(batches)} per rank (world_size={world_size}) "
            f"in {time.perf_counter() - t_build_batches:.2f}s",
            flush=True,
        )

    optimizer.zero_grad(set_to_none=True)

    agg = {
        "loss_total": 0.0,
        "loss_bind": 0.0,
        "loss_mod": 0.0,
        "loss_struct": 0.0,
        "loss_mlm": 0.0,
        "loss_unc": 0.0,
        "loss_dir": 0.0,
        "steps": 0,
    }

    n_local = len(batches)

    for batch_idx, batch_examples in enumerate(batches, start=1):
        t_batch = time.perf_counter()
        # 多任务轮转：按全局步在 task_cycle 中循环。
        # task_name = task_cycle[global_step % len(task_cycle)]
        
        # 每个 batch 构造独立 RNG，保证同一 seed 可复现。
        rng = random.Random(int(cfg["seed"]) + epoch * 100000 + batch_idx * world_size + rank)
        task_name = sample_task_name(rng)
        
        # 两段条件机制：先采样任务条件，再按概率做 mask。
        sampled_role, sampled_base, sampled_mod_type = sample_task_condition(task_name, rng)
        cond_role, cond_base, cond_mod_type = sampled_role, sampled_base, sampled_mod_type
        # cond_role, cond_base, cond_mod_type = apply_condition_mask(
        #     task_name=task_name,
        #     sampled_role=sampled_role,
        #     sampled_base=sampled_base,
        #     sampled_mod_type=sampled_mod_type,
        #     rng=rng,
        #     role_mask_prob=float(cfg["cond_mask_role_prob"]),
        #     base_mask_prob=float(cfg["cond_mask_base_prob"]),
        #     mod_type_mask_prob=float(cfg["cond_mask_mod_type_prob"]),
        # )

        # bind 的监督标签随采样 role 切换；其它任务只占位用 reader。
        role_for_labels = sampled_role if task_name == "bind" else "reader"
        # cond_base 做保护性兜底，防止非法值破坏映射。
        struct_base = cond_base if cond_base in COND_BASE_IDS else "mask"

        t_collate = time.perf_counter()
        batch = collate_batch(
            examples=batch_examples,
            task_name=task_name,
            role_name=role_for_labels,
            cond_base=struct_base,
            sampled_mod_type=sampled_mod_type,
            struct_provider=struct_provider,
            # strong_binding_threshold=strong_thresholds.get(sampled_mod_type, 1.0),
            strong_binding_threshold=strong_thresholds.get(sampled_mod_type, {}).get(role_for_labels, 1.0),
            rng=rng,
            mod_unlabeled_ratio=float(cfg["mod_unlabeled_ratio"]),
            mask_prob=float(cfg["mask_prob"]),
            aprime_enable=bool(cfg["aprime_enable"]),
            aprime_prob=float(cfg["aprime_prob"]),
            aprime_max_per_seq=int(cfg["aprime_max_per_seq"]),
        )
        collate_elapsed = time.perf_counter() - t_collate
        batch = to_device(batch, device)

        # 组装条件 token（task/role/base）送入模型条件分支。
        bsz = batch["tokens"].shape[0]
        cond_task_tensor = torch.full((bsz,), TASK_IDS[task_name], dtype=torch.long, device=device)
        cond_role_tensor = torch.full((bsz,), ROLE_IDS.get(cond_role, ROLE_IDS["none"]), dtype=torch.long, device=device)
        cond_base_tensor = torch.full((bsz,), COND_BASE_IDS.get(cond_base, COND_BASE_IDS["mask"]), dtype=torch.long, device=device)
        cond_mod_type_tensor = torch.full((bsz,), MOD_TYPE_IDS.get(cond_mod_type, MOD_TYPE_IDS["none"]), dtype=torch.long, device=device)

        if bool(cfg["ablate_no_condition"]):
            # 条件消融：三路条件全部置为无信息状态。
            cond_task_tensor = torch.zeros_like(cond_task_tensor)
            cond_role_tensor = torch.zeros_like(cond_role_tensor)
            cond_base_tensor = torch.full_like(cond_base_tensor, COND_BASE_IDS["mask"])
            cond_mod_type_tensor = torch.full_like(cond_mod_type_tensor, MOD_TYPE_IDS["none"])

        # mask 任务使用 MLM 输入，其余任务使用原 token 输入。
        tokens_in = batch["mlm_input"] if task_name == "mask" else batch["tokens"]
        # legacy 策略下，base=A 视为监督更强的条件。


        t_forward = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(
                tokens=tokens_in,
                cond_task=cond_task_tensor,
                cond_role=cond_role_tensor,
                cond_base=cond_base_tensor,
                cond_mod_type=cond_mod_type_tensor,
                attn_mask=batch["attn_mask"],
                site_positions=batch["site_positions"],
                site_mask=batch["site_mask"],
            )

            losses = compute_multitask_losses(
                task_name=task_name,
                outputs=outputs,
                batch=batch,
                args=args,
                sampled_role=sampled_role,
                sampled_mod_type=sampled_mod_type,
                mod_pu_loss=mod_pu_loss,
                bind_pu_loss=bind_pu_loss,
            )
            loss_total = losses["loss_total"]
            loss_bind = losses["loss_bind"]
            loss_mod = losses["loss_mod"]
            loss_struct = losses["loss_struct"]
            loss_mlm = losses["loss_mlm"]
            loss_unc = losses["loss_unc"]
            loss_dir = losses["loss_dir"]
        forward_elapsed = time.perf_counter() - t_forward

        # 梯度累积：反传前先除以累积步数，保证等效梯度尺度稳定。
        t_backward = time.perf_counter()
        loss_for_step = loss_total / int(cfg["grad_accum"])
        if not loss_for_step.requires_grad:
            loss_for_step = loss_for_step + 0.0 * outputs["mod_logits_acu"].sum()

        is_sync_step = (batch_idx % grad_accum == 0) or (batch_idx == n_local)
        no_sync_ctx = model.no_sync() if (is_ddp and not is_sync_step) else nullcontext()
        with no_sync_ctx:
            if amp_enabled:
                scaler.scale(loss_for_step).backward()
            else:
                loss_for_step.backward()

        did_optim_step = False
        if is_sync_step:
            if amp_enabled:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["grad_clip"]))
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optim_step += 1
            did_optim_step = True
        backward_elapsed = time.perf_counter() - t_backward

        agg["loss_total"] += float(loss_total.detach().cpu())
        agg["loss_bind"] += float(loss_bind.detach().cpu())
        agg["loss_mod"] += float(loss_mod.detach().cpu())
        agg["loss_struct"] += float(loss_struct.detach().cpu())
        agg["loss_mlm"] += float(loss_mlm.detach().cpu())
        agg["loss_unc"] += float(loss_unc.detach().cpu())
        agg["loss_dir"] += float(loss_dir.detach().cpu())
        agg["steps"] += 1

        if is_main and writer is not None and (global_step % max(1, int(cfg["tb_log_steps"])) == 0):
            # step 级日志：用于实时观察训练是否稳定。
            writer.add_scalar("train/loss_total_step", float(loss_total.detach().cpu()), global_step)
            writer.add_scalar("train/loss_bind_step", float(loss_bind.detach().cpu()), global_step)
            writer.add_scalar("train/loss_mod_step", float(loss_mod.detach().cpu()), global_step)
            writer.add_scalar("train/loss_struct_step", float(loss_struct.detach().cpu()), global_step)
            writer.add_scalar("train/loss_mlm_step", float(loss_mlm.detach().cpu()), global_step)
            writer.add_scalar("train/loss_unc_step", float(loss_unc.detach().cpu()), global_step)
            writer.add_scalar("train/loss_dir_step", float(loss_dir.detach().cpu()), global_step)
            writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)

        global_step += 1

        if is_main and (
            batch_idx == 1
            or batch_idx == n_local
            or (batch_idx % batch_log_interval == 0)
            or (collate_elapsed >= 30.0)
        ):
            tokens_count = int(sum(x.seq_len for x in batch_examples))
            max_seq_len = int(max(x.seq_len for x in batch_examples))
            print(
                f"[epoch {epoch:03d}][{batch_idx:04d}/{len(batches):04d}] "
                f"task={task_name} role={sampled_role} mod_type={sampled_mod_type} cond=({cond_role},{cond_base},{cond_mod_type}) "
                f"bsz={len(batch_examples)} tokens={tokens_count} max_len={max_seq_len} "
                f"collate={collate_elapsed:.2f}s forward={forward_elapsed:.2f}s "
                f"backward={backward_elapsed:.2f}s step={'Y' if did_optim_step else 'N'} "
                f"loss={float(loss_total.detach().cpu()):.4f} "
                f"batch_total={time.perf_counter() - t_batch:.2f}s",
                flush=True,
            )

    for key in list(agg.keys()):
        if key.startswith("loss_"):
            agg[key] = agg[key] / max(1, agg["steps"])
    return agg, global_step, optim_step


def main() -> None:
    """训练脚本主入口。"""

    rank, local_rank, world_size, distributed = _setup_distributed()
    is_main = _is_main(rank)

    args = parse_args()
    # 统一配置入口：后续训练逻辑尽量只读 cfg，不直接散读 args。
    cfg = resolve_train_config(args)

    if distributed:
        cfg["device"] = f"cuda:{local_rank}"
    
    set_seed(int(cfg["seed"]) + rank)
    use_amp = bool(cfg["use_amp"])
    device = torch.device(cfg["device"])
    if bool(cfg.get("disable_cudnn", False)):
        torch.backends.cudnn.enabled = False
        if is_main:
            print("[runtime] cuDNN disabled by config (disable_cudnn=True)", flush=True)
    boundaries = list(cfg["bucket_boundaries_list"])
    out_dir = Path(cfg["output_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    _barrier()
    if is_main:
        _setup_train_log_file(out_dir)

    if is_main:
        print(f"[runtime] experiment_name={cfg['experiment_name']} output_dir={out_dir}", flush=True)
        print(f"[runtime] world_size={world_size} rank={rank} device={device}", flush=True)


    train_examples, val_examples, struct_provider, mod_priors, bind_priors, strong_thresholds = _prepare_data_and_priors(cfg, rank=rank )
    
    if is_main: 
        if bool(cfg.get("ablate_no_struct", False)):
            print("Structure source: disabled (ablate_no_struct=True)")
        else:
            print(f"Structure source: {cfg.get('struct_source', 'precomputed')}")
    (
        model,
        mod_pu_loss,
        bind_pu_loss,
        optimizer,
        scheduler,
        scaler,
        steps_per_epoch,
        total_steps,
    ) = _build_train_components(
        cfg=cfg,
        device=device,
        train_examples=train_examples,
        boundaries=boundaries,
        mod_priors=mod_priors,
        bind_priors=bind_priors,
        use_amp=use_amp,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )

    # 输出目录统一管理：checkpoint、metrics、run_config、TensorBoard。
    writer = None
    if is_main and bool(cfg["tensorboard"]):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard logging requested, but tensorboard is not available. "
                "Install with `pip install tensorboard`."
            ) from exc
        tb_dir = Path(cfg["tb_dir"]) if cfg["tb_dir"] else (out_dir / "tb")
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    # 保存一次“可复现快照”：配置 + 数据规模 + 先验 + 调度步数。
    run_config = {
        "args": dict(cfg),
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "mod_priors": mod_priors,
        "binding_priors": bind_priors,
        "strong_thresholds": strong_thresholds,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "world_size": world_size,
    }
    if is_main:
        save_json(out_dir / "config.json", run_config)

    # 多任务训练默认四任务轮转；结构消融时去掉 struct。
    task_cycle = ["bind", "mod", "struct", "mask"]
    if bool(cfg["ablate_no_struct"]):
        task_cycle = ["bind", "mod", "mask"]

    history = []
    best = {"epoch": 0, "reader_auprc": -1.0}
    global_step = 0
    optim_step = 0

    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_t0 = time.perf_counter()
        train_t0 = time.perf_counter()
        agg, global_step, optim_step = _run_train_epoch(
            epoch=epoch,
            model=model,
            train_examples=train_examples,
            task_cycle=task_cycle,
            cfg=cfg,
            args=args,
            device=device,
            use_amp=use_amp,
            boundaries=boundaries,
            struct_provider=struct_provider,
            strong_thresholds=strong_thresholds,
            mod_pu_loss=mod_pu_loss,
            bind_pu_loss=bind_pu_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            writer=writer,
            global_step=global_step,
            optim_step=optim_step,
            rank=rank,
            world_size=world_size,
        )
        train_elapsed = time.perf_counter() - train_t0

        # 每个 epoch 后做 reader 验证，作为 best 模型选择依据。
        if is_main:
            val_t0 = time.perf_counter()
            eval_model = model.module if isinstance(model, DDP) else model


            # val_metrics = evaluate_reader_binding(
            #     model=eval_model,
            #     examples=val_examples,
            #     struct_provider=struct_provider,
            #     strong_thresholds=strong_thresholds,
            #     batch_token_budget=int(cfg["batch_token_budget"]),
            #     boundaries=boundaries,
            #     device=device,
            # )
            # mod_val_metrics = evaluate_mod_task(
            #     model=eval_model,
            #     examples=val_examples,
            #     struct_provider=struct_provider,
            #     batch_token_budget=int(cfg["batch_token_budget"]),
            #     boundaries=boundaries,
            #     device=device,
            # )
            val_metrics = evaluate_bind_all_types(
                model=eval_model,
                examples=val_examples,
                batch_token_budget=int(cfg["batch_token_budget"]),
                device=device,
                neg_ratio=1.0,
            )
            mod_val_metrics = evaluate_mod_all_types(
                model=eval_model,
                examples=val_examples,
                batch_token_budget=int(cfg["batch_token_budget"]),
                boundaries=boundaries,
                device=device,
                neg_ratio=1.0,   # 正负 1:1
            )
            val_metrics.update(mod_val_metrics)
            val_elapsed = time.perf_counter() - val_t0

            # 记录 epoch 结果快照。
            epoch_record = {
                "epoch": epoch,
                "train": agg,
                "val": val_metrics,
                "optim_step": optim_step,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_record)
            # 按 epoch 写入 TensorBoard。
            if writer is not None:
                writer.add_scalar("train/loss_total_epoch", agg["loss_total"], epoch)
                writer.add_scalar("train/loss_bind_epoch", agg["loss_bind"], epoch)
                writer.add_scalar("train/loss_mod_epoch", agg["loss_mod"], epoch)
                writer.add_scalar("train/loss_struct_epoch", agg["loss_struct"], epoch)
                writer.add_scalar("train/loss_mlm_epoch", agg["loss_mlm"], epoch)
                writer.add_scalar("train/loss_unc_epoch", agg["loss_unc"], epoch)
                writer.add_scalar("train/loss_dir_epoch", agg["loss_dir"], epoch)
                for role in ROLE_NAMES:
                    for mt in ["m6A"]:
                        writer.add_scalar(f"val/bind_{mt}_{role}_auprc", val_metrics[f"bind_{mt}_{role}_auprc"], epoch)
                        writer.add_scalar(f"val/bind_{mt}_{role}_auroc", val_metrics[f"bind_{mt}_{role}_auroc"], epoch)
                        writer.add_scalar(f"val/bind_{mt}_{role}_f1", val_metrics[f"bind_{mt}_{role}_f1"], epoch)
                        writer.add_scalar(f"val/bind_{mt}_{role}_ece", val_metrics[f"bind_{mt}_{role}_ece"], epoch) 
                writer.add_scalar(f"bind_avg_auprc", val_metrics[f"bind_avg_auprc"], epoch)
                writer.add_scalar(f"bind_avg_auroc", val_metrics[f"bind_avg_auroc"], epoch) 
                for mt in MOD_TYPE_NAMES:
                    writer.add_scalar(f"val/mod_{mt}_auprc", val_metrics[f"mod_{mt}_auprc"], epoch)
                    writer.add_scalar(f"val/mod_{mt}_auroc", val_metrics[f"mod_{mt}_auroc"], epoch)
                    writer.add_scalar(f"val/mod_{mt}_f1", val_metrics[f"mod_{mt}_f1"], epoch)
                    writer.add_scalar(f"val/mod_{mt}_ece", val_metrics[f"mod_{mt}_ece"], epoch)
                writer.add_scalar(f"mod_avg_auprc", val_metrics[f"mod_avg_auprc"], epoch)
                writer.add_scalar(f"mod_avg_auroc", val_metrics[f"mod_avg_auroc"], epoch)
             
                writer.add_scalar("train/lr_epoch", optimizer.param_groups[0]["lr"], epoch)

            epoch_elapsed = time.perf_counter() - epoch_t0
            # 终端打印摘要，便于实时监控。
            # print(
            #     f"第{epoch:03d}轮 | "
            #     f"训练损失: 总={agg['loss_total']:.4f}, bind={agg['loss_bind']:.4f}, mod={agg['loss_mod']:.4f}, "
            #     f"struct={agg['loss_struct']:.4f}, mlm={agg['loss_mlm']:.4f}, 不确定度={agg['loss_unc']:.4f}, 方向={agg['loss_dir']:.4f} | "
            #     f"验证(bind): AUPRC={val_metrics['reader_auprc']:.4f}, AUROC={val_metrics['reader_auroc']:.4f}, "
            #     f"F1={val_metrics['reader_f1']:.4f}, ECE={val_metrics['reader_ece']:.4f} | "
            #     f"验证(mod): AUPRC={val_metrics['mod_avg_auprc']:.4f}, AUROC={val_metrics['mod_avg_auroc']:.4f}"
            # )
            # print(
            #     f"第{epoch:03d}轮详情 | "
            #     f"本轮batch数={int(agg['steps'])}, 优化步={optim_step}, 全局步={global_step}, 学习率={optimizer.param_groups[0]['lr']:.6e} | "
            #     f"耗时: 训练={train_elapsed:.1f}s, 验证={val_elapsed:.1f}s, 总计={epoch_elapsed:.1f}s | "
            #     f"reader不确定度: 强样本={val_metrics['reader_unc_strong']:.4f}, 弱样本={val_metrics['reader_unc_weak']:.4f}, 相对差值={val_metrics['reader_unc_gap']:.4f} | "
            #     f"当前最佳(reader AUPRC): 轮次={best['epoch']}, 数值={best['reader_auprc']:.4f}",
            #     flush=True,
            # )
            model_for_save = model.module if isinstance(model, DDP) else model
            # 保存每个 epoch 的完整 checkpoint（含优化器与调度器状态）。
            ckpt = {
                "epoch": epoch,
                "model_state": model_for_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config": run_config,
                "history": history,
            }
            torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pt")

            # 用验证集 avg_auprc 作为 early-best 选择标准。
            # score = val_metrics.get("bind_avg_auprc", float("nan"))
            # if np.isfinite(score) and score > best["bind_avg_auprc"]:
            #     best = {
            #         "epoch": epoch,
            #         "bind_avg_auprc": float(score),
            #         "bind_avg_auroc": float(val_metrics.get("bind_avg_auroc", float("nan"))),
            #     }
            #     torch.save(ckpt, out_dir / "best.pt")

            # 每个 epoch 刷新一次 metrics.json，避免中断时丢失历史。
            save_json(out_dir / "metrics.json", {"history": history, "best": best})
        _barrier()
    if is_main:
        model_for_save = model.module if isinstance(model, DDP) else model
    # 17) 训练结束：另存轻量 last.pt，并再次刷新 metrics。
        torch.save(
            {
                "model_state": model_for_save.state_dict(),
                "config": run_config,
                "history": history,
                # "best": best,
            },
            out_dir / "last.pt",
        )
        # save_json(out_dir / "metrics.json", {"history": history, "best": best})
        save_json(out_dir / "metrics.json", {"history": history})
        if writer is not None:
            writer.close()
    _cleanup_distributed()


if __name__ == "__main__":
    main()
