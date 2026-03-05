#!/usr/bin/env python3
"""Train ETD multi-task model for m6A multitask objectives."""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import random
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

from models.etd_multitask.constants import COND_BASE_IDS, ROLE_IDS, ROLE_NAMES, TASK_IDS
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


def parse_args() -> argparse.Namespace:
    """训练脚本参数入口（具体定义与配置覆盖逻辑在 experiment_config 模块）。"""
    return parse_train_args(REPO_ROOT)


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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
    base_lr: float,
):
    """构建 warmup + cosine 学习率调度器。

    训练步长定义：
    - step < warmup_steps: 线性升温
    - 其余阶段: 从 base_lr 余弦衰减到 min_lr

    返回：
    - `torch.optim.lr_scheduler.LambdaLR`
    """
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))
    min_ratio = float(min_lr / base_lr) if base_lr > 0 else 0.1

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

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

    with torch.no_grad():
        for batch_examples in batches:
            # 评估阶段固定随机种子，避免 collate 内随机采样导致指标抖动。
            rng = random.Random(0)
            batch = collate_batch(
                examples=batch_examples,
                task_name="bind",
                role_name="reader",
                cond_base="A",
                struct_provider=struct_provider,
                strong_binding_threshold=strong_thresholds["reader"],
                rng=rng,
                mod_unlabeled_ratio=1.0,
                mask_prob=0.15,
            )
            batch = to_device(batch, device)

            bsz = batch["tokens"].shape[0]
            cond_task = torch.full((bsz,), TASK_IDS["bind"], device=device, dtype=torch.long)
            cond_role = torch.full((bsz,), ROLE_IDS["reader"], device=device, dtype=torch.long)
            cond_base = torch.full((bsz,), COND_BASE_IDS["A"], device=device, dtype=torch.long)

            outputs = model(
                tokens=batch["tokens"],
                cond_task=cond_task,
                cond_role=cond_role,
                cond_base=cond_base,
                attn_mask=batch["attn_mask"],
                struct_feats=batch["struct_feats"],
                site_positions=batch["site_positions"],
                site_mask=batch["site_mask"],
            )

            # 只统计有效 site（site_mask=True），跳过 padding 槽位。
            mask = batch["site_mask"].detach().cpu().numpy().astype(bool)
            y_true = (batch["site_pu_labels"].detach().cpu().numpy() == 1).astype(np.int64)
            y_prob = torch.sigmoid(outputs["bind_logits"]).detach().cpu().numpy()
            unc = outputs["bind_uncertainty"].detach().cpu().numpy()
            strong = batch["strong_binding_mask"].detach().cpu().numpy().astype(bool)

            y_true_all.append(y_true[mask])
            y_prob_all.append(y_prob[mask])
            unc_all.append(unc[mask])
            strong_all.append(strong[mask])

    if not y_true_all:
        return {
            "reader_auroc": float("nan"),
            "reader_auprc": float("nan"),
            "reader_f1": float("nan"),
            "reader_ece": float("nan"),
            "reader_unc_strong": float("nan"),
            "reader_unc_weak": float("nan"),
            "reader_unc_gap": float("nan"),
        }

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    unc = np.concatenate(unc_all)
    strong = np.concatenate(strong_all)

    strong_mean = float(unc[strong].mean()) if np.any(strong) else float("nan")
    weak_mean = float(unc[~strong].mean()) if np.any(~strong) else float("nan")
    gap = float((weak_mean - strong_mean) / max(weak_mean, 1e-6)) if np.isfinite(strong_mean) and np.isfinite(weak_mean) else float("nan")

    return {
        "reader_auroc": binary_auroc(y_true, y_prob),
        "reader_auprc": binary_auprc(y_true, y_prob),
        "reader_f1": binary_f1(y_true, y_prob),
        "reader_ece": expected_calibration_error(y_true, y_prob),
        "reader_unc_strong": strong_mean,
        "reader_unc_weak": weak_mean,
        "reader_unc_gap": gap,
    }


def _prepare_data_and_priors(cfg: dict):
    """加载 train/val 样本，并计算训练依赖的先验统计。
    返回顺序：
    1) train_examples
    2) val_examples
    3) struct_provider
    4) mod_prior
    5) bind_priors
    6) strong_thresholds
    """
    train_examples = load_examples(
        sites_path=cfg["sites"],
        transcripts_path=cfg["transcripts"],
        splits_path=cfg["splits"],
        split_names=["train"],
        max_len=int(cfg["max_len"]),
        smoke_ratio=float(cfg["smoke_ratio"]),
        seed=int(cfg["seed"]),
    )
    val_examples = load_examples(
        sites_path=cfg["sites"],
        transcripts_path=cfg["transcripts"],
        splits_path=cfg["splits"],
        split_names=["val"],
        max_len=int(cfg["max_len"]),
        smoke_ratio=1.0,
        seed=int(cfg["seed"]),
    )

    if str(cfg.get("struct_source", "precomputed")) == "online":
        struct_provider = OnlineRNAfoldProvider(
            rnafold_bin=str(cfg.get("online_rnafold_bin", "RNAfold")),
            timeout_seconds=int(cfg.get("online_rnafold_timeout_seconds", 240)),
            cache_size=int(cfg.get("online_rnafold_cache_size", 2048)),
        )
    else:
        struct_provider = BPPCache(cfg["rnafold_cache"])
    mod_prior = estimate_mod_prior(train_examples)
    bind_priors = estimate_binding_priors(train_examples)
    strong_thresholds = estimate_strong_binding_thresholds(train_examples, q=0.75)
    return train_examples, val_examples, struct_provider, mod_prior, bind_priors, strong_thresholds


def _build_train_components(
    cfg: dict,
    device: torch.device,
    train_examples,
    boundaries: list[int],
    mod_prior: float,
    bind_priors: dict[str, float],
    use_amp: bool,
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
    print(f"Model initialized with {num_parameters:,} trainable parameters.")

    mod_pu_loss = NonNegativePULoss(prior=mod_prior)
    bind_pu_loss = {role: NonNegativePULoss(prior=bind_priors.get(role, 0.5)) for role in ROLE_NAMES}

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
    steps_per_epoch = max(1, math.ceil(len(dry_batches) / int(cfg["grad_accum"])))
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
) -> tuple[dict[str, float], int, int]:
    """执行单个 epoch 训练。

    返回：
    - agg: 本 epoch 平均损失统计
    - global_step: 更新后的全局 batch 计数
    - optim_step: 更新后的优化器步数（受 grad_accum 影响）
    """
    model.train()
    batches = build_length_bucketed_batches(
        examples=train_examples,
        batch_token_budget=int(cfg["batch_token_budget"]),
        boundaries=boundaries,
        shuffle=True,
        seed=int(cfg["seed"]) + epoch,
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

    for batch_idx, batch_examples in enumerate(batches, start=1):
        # 多任务轮转：按全局步在 task_cycle 中循环。
        task_name = task_cycle[global_step % len(task_cycle)]
        # 每个 batch 构造独立 RNG，保证同一 seed 可复现。
        rng = random.Random(int(cfg["seed"]) + epoch * 100000 + batch_idx)

        # 两段条件机制：先采样任务条件，再按概率做 mask。
        sampled_role, sampled_base = sample_task_condition(task_name, rng)
        cond_role, cond_base = apply_condition_mask(
            task_name=task_name,
            sampled_role=sampled_role,
            sampled_base=sampled_base,
            rng=rng,
            role_mask_prob=float(cfg["cond_mask_role_prob"]),
            base_mask_prob=float(cfg["cond_mask_base_prob"]),
        )

        # bind 的监督标签随采样 role 切换；其它任务只占位用 reader。
        role_for_labels = sampled_role if task_name == "bind" else "reader"
        # cond_base 做保护性兜底，防止非法值破坏映射。
        struct_base = cond_base if cond_base in COND_BASE_IDS else "mask"

        batch = collate_batch(
            examples=batch_examples,
            task_name=task_name,
            role_name=role_for_labels,
            cond_base=struct_base,
            struct_provider=struct_provider,
            strong_binding_threshold=strong_thresholds.get(role_for_labels, 1.0),
            rng=rng,
            mod_unlabeled_ratio=float(cfg["mod_unlabeled_ratio"]),
            mask_prob=float(cfg["mask_prob"]),
            aprime_enable=bool(cfg["aprime_enable"]),
            aprime_prob=float(cfg["aprime_prob"]),
            aprime_max_per_seq=int(cfg["aprime_max_per_seq"]),
        )
        batch = to_device(batch, device)

        # 组装条件 token（task/role/base）送入模型条件分支。
        bsz = batch["tokens"].shape[0]
        cond_task_tensor = torch.full((bsz,), TASK_IDS[task_name], dtype=torch.long, device=device)
        cond_role_tensor = torch.full((bsz,), ROLE_IDS.get(cond_role, ROLE_IDS["none"]), dtype=torch.long, device=device)
        cond_base_tensor = torch.full((bsz,), COND_BASE_IDS.get(cond_base, COND_BASE_IDS["mask"]), dtype=torch.long, device=device)

        if bool(cfg["ablate_no_condition"]):
            # 条件消融：三路条件全部置为无信息状态。
            cond_task_tensor = torch.zeros_like(cond_task_tensor)
            cond_role_tensor = torch.zeros_like(cond_role_tensor)
            cond_base_tensor = torch.full_like(cond_base_tensor, COND_BASE_IDS["mask"])

        # mask 任务使用 MLM 输入，其余任务使用原 token 输入。
        tokens_in = batch["mlm_input"] if task_name == "mask" else batch["tokens"]
        # legacy 策略下，base=A 视为监督更强的条件。
        supervised_a = cond_base == "A"

        with torch.amp.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            outputs = model(
                tokens=tokens_in,
                cond_task=cond_task_tensor,
                cond_role=cond_role_tensor,
                cond_base=cond_base_tensor,
                attn_mask=batch["attn_mask"],
                struct_feats=batch["struct_feats"],
                site_positions=batch["site_positions"],
                site_mask=batch["site_mask"],
            )

            losses = compute_multitask_losses(
                task_name=task_name,
                outputs=outputs,
                batch=batch,
                args=args,
                supervised_a=supervised_a,
                sampled_role=sampled_role,
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

        # 梯度累积：反传前先除以累积步数，保证等效梯度尺度稳定。
        loss_for_step = loss_total / int(cfg["grad_accum"])
        if not loss_for_step.requires_grad:
            loss_for_step = loss_for_step + 0.0 * outputs["mod_logits_acu"].sum()
        scaler.scale(loss_for_step).backward()

        if (batch_idx % int(cfg["grad_accum"]) == 0) or (batch_idx == len(batches)):
            # 优化前先反缩放并做梯度裁剪，抑制异常梯度。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            optim_step += 1

        agg["loss_total"] += float(loss_total.detach().cpu())
        agg["loss_bind"] += float(loss_bind.detach().cpu())
        agg["loss_mod"] += float(loss_mod.detach().cpu())
        agg["loss_struct"] += float(loss_struct.detach().cpu())
        agg["loss_mlm"] += float(loss_mlm.detach().cpu())
        agg["loss_unc"] += float(loss_unc.detach().cpu())
        agg["loss_dir"] += float(loss_dir.detach().cpu())
        agg["steps"] += 1

        if writer is not None and (global_step % max(1, int(cfg["tb_log_steps"])) == 0):
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

    for key in list(agg.keys()):
        if key.startswith("loss_"):
            agg[key] = agg[key] / max(1, agg["steps"])
    return agg, global_step, optim_step


def main() -> None:
    """训练脚本主入口。"""
    args = parse_args()
    # 统一配置入口：后续训练逻辑尽量只读 cfg，不直接散读 args。
    cfg = resolve_train_config(args)
    set_seed(int(cfg["seed"]))
    use_amp = bool(cfg["use_amp"])
    device = torch.device(cfg["device"])
    boundaries = list(cfg["bucket_boundaries_list"])

    train_examples, val_examples, struct_provider, mod_prior, bind_priors, strong_thresholds = _prepare_data_and_priors(cfg)
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
        mod_prior=mod_prior,
        bind_priors=bind_priors,
        use_amp=use_amp,
    )

    # 输出目录统一管理：checkpoint、metrics、run_config、TensorBoard。
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if bool(cfg["tensorboard"]):
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
        "mod_prior": mod_prior,
        "binding_priors": bind_priors,
        "strong_thresholds": strong_thresholds,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }
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
        )

        # 每个 epoch 后做 reader 验证，作为 best 模型选择依据。
        val_metrics = evaluate_reader_binding(
            model=model,
            examples=val_examples,
            struct_provider=struct_provider,
            strong_thresholds=strong_thresholds,
            batch_token_budget=int(cfg["batch_token_budget"]),
            boundaries=boundaries,
            device=device,
        )

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
            writer.add_scalar("val/reader_auprc", val_metrics["reader_auprc"], epoch)
            writer.add_scalar("val/reader_auroc", val_metrics["reader_auroc"], epoch)
            writer.add_scalar("val/reader_f1", val_metrics["reader_f1"], epoch)
            writer.add_scalar("val/reader_ece", val_metrics["reader_ece"], epoch)
            writer.add_scalar("val/reader_unc_strong", val_metrics["reader_unc_strong"], epoch)
            writer.add_scalar("val/reader_unc_weak", val_metrics["reader_unc_weak"], epoch)
            writer.add_scalar("val/reader_unc_gap", val_metrics["reader_unc_gap"], epoch)
            writer.add_scalar("train/lr_epoch", optimizer.param_groups[0]["lr"], epoch)

        # 终端打印一行摘要，便于实时监控。
        print(
            f"Epoch {epoch:03d} | "
            f"loss={agg['loss_total']:.4f} bind={agg['loss_bind']:.4f} mod={agg['loss_mod']:.4f} "
            f"struct={agg['loss_struct']:.4f} mlm={agg['loss_mlm']:.4f} unc={agg['loss_unc']:.4f} dir={agg['loss_dir']:.4f} "
            f"val_reader_auprc={val_metrics['reader_auprc']:.4f} val_reader_auroc={val_metrics['reader_auroc']:.4f}"
        )

        # 保存每个 epoch 的完整 checkpoint（含优化器与调度器状态）。
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": run_config,
            "history": history,
        }
        torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pt")

        # 用验证集 reader_auprc 作为 early-best 选择标准。
        score = val_metrics.get("reader_auprc", float("nan"))
        if np.isfinite(score) and score > best["reader_auprc"]:
            best = {
                "epoch": epoch,
                "reader_auprc": float(score),
                "reader_auroc": float(val_metrics.get("reader_auroc", float("nan"))),
            }
            torch.save(ckpt, out_dir / "best.pt")

        # 每个 epoch 刷新一次 metrics.json，避免中断时丢失历史。
        save_json(out_dir / "metrics.json", {"history": history, "best": best})

    # 17) 训练结束：另存轻量 last.pt，并再次刷新 metrics。
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": run_config,
            "history": history,
            "best": best,
        },
        out_dir / "last.pt",
    )
    save_json(out_dir / "metrics.json", {"history": history, "best": best})
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
