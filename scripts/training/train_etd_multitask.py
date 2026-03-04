#!/usr/bin/env python3
"""Train ETD multi-task model for m6A multitask objectives."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
from pathlib import Path

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
import torch.nn.functional as F

from models.etd_multitask.constants import COND_BASE_IDS, ROLE_IDS, ROLE_NAMES, TASK_IDS, VALID_BASES
from models.etd_multitask.data import (
    BPPCache,
    build_length_bucketed_batches,
    collate_batch,
    estimate_binding_priors,
    estimate_mod_prior,
    estimate_strong_binding_thresholds,
    load_examples,
)
from models.etd_multitask.losses import (
    NonNegativePULoss,
    evidential_positive_loss,
    structure_bce_dice_loss,
)
from models.etd_multitask.metrics import (
    binary_auprc,
    binary_auroc,
    binary_f1,
    expected_calibration_error,
)
from models.etd_multitask.model import ETDMultiTaskModel
from models.etd_multitask.utils import save_json, set_seed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ETD multitask model.")
    parser.add_argument(
        "--config",
        default="",
        help="Path to JSON config file. CLI args override config values.",
    )
    parser.add_argument("--sites", default=str(REPO_ROOT / "data/processed/m6a_multitask_sites.parquet"))
    parser.add_argument("--transcripts", default=str(REPO_ROOT / "data/processed/m6a_multitask_transcripts.parquet"))
    parser.add_argument("--splits", default=str(REPO_ROOT / "data/processed/m6a_multitask_splits.json"))
    parser.add_argument("--rnafold-cache", default=str(REPO_ROOT / "data/processed/rnafold_bpp"))
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--smoke-ratio", type=float, default=1.0)
    parser.add_argument("--batch-token-budget", type=int, default=24000)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/etd_multitask"))

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--mod-unlabeled-ratio", type=float, default=1.0)
    parser.add_argument("--cond-mask-role-prob", type=float, default=0.3)
    parser.add_argument("--cond-mask-base-prob", type=float, default=0.3)
    parser.add_argument("--struct-min-sep", type=int, default=4)
    parser.add_argument("--bucket-boundaries", default="1024,2048,4096,8192,12000")

    parser.add_argument("--ablate-no-condition", action="store_true")
    parser.add_argument("--ablate-no-dirichlet", action="store_true")
    parser.add_argument("--ablate-no-struct", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tb-dir", default="")
    parser.add_argument("--tb-log-steps", type=int, default=20)
    return parser


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="")
    known, _ = bootstrap.parse_known_args()

    parser = _build_arg_parser()

    if known.config:
        config_path = Path(known.config)
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError(f"Config file must contain a JSON object: {config_path}")

        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(k for k in config if k not in valid_keys)
        if unknown_keys:
            raise ValueError(f"Unknown config keys in {config_path}: {unknown_keys}")

        parser.set_defaults(**config)
        parser.set_defaults(config=str(config_path))

    return parser.parse_args()


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def compute_uncertainty_loss(
    uncertainty: torch.Tensor,
    probs: torch.Tensor,
    strong_mask: torch.Tensor,
    site_mask: torch.Tensor,
    supervised: bool,
) -> torch.Tensor:
    if supervised:
        target_mask = strong_mask & site_mask
        if target_mask.any():
            low_unc = torch.relu(uncertainty[target_mask] - 0.2)
            high_prob = torch.relu(0.8 - probs[target_mask])
            return (low_unc + high_prob).mean()
        return uncertainty.new_tensor(0.0)

    target_mask = site_mask
    if target_mask.any():
        return torch.relu(0.6 - uncertainty[target_mask]).mean()
    return uncertainty.new_tensor(0.0)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
    base_lr: float,
):
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
    bpp_cache: BPPCache,
    strong_thresholds: dict[str, float],
    batch_token_budget: int,
    boundaries: list[int],
    device: torch.device,
) -> dict[str, float]:
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
            rng = random.Random(0)
            batch = collate_batch(
                examples=batch_examples,
                task_name="bind",
                role_name="reader",
                cond_base="A",
                bpp_cache=bpp_cache,
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


def _sample_task_condition(task_name: str, rng: random.Random) -> tuple[str, str]:
    if task_name == "bind":
        role = rng.choice(list(ROLE_NAMES))
        base = rng.choice(list(VALID_BASES))
        return role, base
    if task_name == "mod":
        return "none", rng.choice(["A", "C", "U"])
    if task_name == "struct":
        return "none", rng.choice(list(VALID_BASES))
    return "none", "mask"


def _apply_condition_mask(
    task_name: str,
    sampled_role: str,
    sampled_base: str,
    rng: random.Random,
    role_mask_prob: float,
    base_mask_prob: float,
) -> tuple[str, str]:
    cond_role = sampled_role
    cond_base = sampled_base
    if task_name in ("bind", "mod", "struct"):
        if rng.random() < role_mask_prob:
            cond_role = "none"
        if rng.random() < base_mask_prob:
            cond_base = "mask"
    return cond_role, cond_base


def main() -> None:
    # 1) 解析参数与基础运行模式（是否开启 AMP）。
    args = parse_args()
    use_amp = bool(args.amp) and not bool(args.no_amp)

    # 2) 固定随机种子，构造设备对象（cpu/cuda）。
    set_seed(args.seed)
    device = torch.device(args.device)

    # 3) 解析长度分桶边界；为空时回退到默认分桶。
    # 分桶用于动态 batch（按 token budget 组 batch），减少 padding 浪费。
    boundaries = [int(x.strip()) for x in args.bucket_boundaries.split(",") if x.strip()]
    if not boundaries:
        boundaries = [1024, 2048, 4096, 8192, 12000]

    # 4) 加载训练/验证样本。训练集可受 smoke_ratio 下采样，验证集始终全量 split。
    train_examples = load_examples(
        sites_path=args.sites,
        transcripts_path=args.transcripts,
        splits_path=args.splits,
        split_names=["train"],
        max_len=args.max_len,
        smoke_ratio=args.smoke_ratio,
        seed=args.seed,
    )
    val_examples = load_examples(
        sites_path=args.sites,
        transcripts_path=args.transcripts,
        splits_path=args.splits,
        split_names=["val"],
        max_len=args.max_len,
        smoke_ratio=1.0,
        seed=args.seed,
    )

    # 没有训练样本时直接报错，避免空跑。
    if not train_examples:
        raise RuntimeError("No training examples loaded. Check inputs and splits.")

    # 5) 初始化 RNA 二级结构缓存读取器。
    bpp_cache = BPPCache(args.rnafold_cache)

    # 6) 从训练集估计先验：
    # - mod_prior: m6A 任务的 PU 先验
    # - bind_priors: 各 role 的 PU 先验
    # - strong_thresholds: 强结合阈值（用于不确定性监督）
    mod_prior = estimate_mod_prior(train_examples)
    bind_priors = estimate_binding_priors(train_examples)
    strong_thresholds = estimate_strong_binding_thresholds(train_examples, q=0.75)

    # 7) 初始化模型与损失函数。
    model = ETDMultiTaskModel().to(device)
    num_parameters = np.sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Model initialized with {num_parameters:,} trainable parameters.")
    mod_pu_loss = NonNegativePULoss(prior=mod_prior)
    bind_pu_loss = {
        role: NonNegativePULoss(prior=bind_priors.get(role, 0.5)) for role in ROLE_NAMES
    }

    # 8) 优化器设置。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    # 9) 先用不打乱的数据估算每个 epoch 的优化步数（考虑 grad_accum）。
    # 该步只用于学习率调度器参数估计，不参与训练。
    dry_batches = build_length_bucketed_batches(
        examples=train_examples,
        batch_token_budget=args.batch_token_budget,
        boundaries=boundaries,
        shuffle=False,
        seed=args.seed,
    )
    steps_per_epoch = max(1, math.ceil(len(dry_batches) / args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        base_lr=args.lr,
    )

    # 10) AMP scaler。优先使用新接口，兼容旧版 PyTorch。
    try:
        scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and device.type == "cuda")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    # 11) 输出目录与可选 TensorBoard writer。
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            raise RuntimeError(
                "TensorBoard logging requested, but tensorboard is not available. "
                "Install with `pip install tensorboard`."
            ) from exc
        tb_dir = Path(args.tb_dir) if args.tb_dir else (out_dir / "tb")
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    # 12) 保存一次运行配置，便于复现。
    run_config = {
        "args": vars(args),
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "mod_prior": mod_prior,
        "binding_priors": bind_priors,
        "strong_thresholds": strong_thresholds,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
    }
    save_json(out_dir / "config.json", run_config)

    # 13) 任务轮转顺序：按 batch 级别交替训练多任务。
    # 若 ablate_no_struct=True，则去掉结构任务。
    task_cycle = ["bind", "mod", "struct", "mask"]
    if args.ablate_no_struct:
        task_cycle = ["bind", "mod", "mask"]

    history = []
    best = {"epoch": 0, "reader_auprc": -1.0}
    global_step = 0
    optim_step = 0

    # 14) epoch 训练循环。
    for epoch in range(1, args.epochs + 1):
        model.train()
        # 每个 epoch 重新打乱并按长度分桶组 batch。
        batches = build_length_bucketed_batches(
            examples=train_examples,
            batch_token_budget=args.batch_token_budget,
            boundaries=boundaries,
            shuffle=True,
            seed=args.seed + epoch,
        )

        optimizer.zero_grad(set_to_none=True)

        # 记录本 epoch 的平均训练损失。
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

        # 15) batch 循环：采样任务与条件，前向计算，反向更新。
        for batch_idx, batch_examples in enumerate(batches, start=1):
            # 按 global_step 循环取任务，保证多任务均匀出现。
            task_name = task_cycle[global_step % len(task_cycle)]
            # 为当前 batch 构造可复现随机源（依赖 seed/epoch/batch_idx）。
            rng = random.Random(args.seed + epoch * 100000 + batch_idx)

            # 采样任务条件（role/base），再按概率做条件 mask（none/mask）。
            sampled_role, sampled_base = _sample_task_condition(task_name, rng)

            cond_role, cond_base = _apply_condition_mask(
                task_name=task_name,
                sampled_role=sampled_role,
                sampled_base=sampled_base,
                rng=rng,
                role_mask_prob=args.cond_mask_role_prob,
                base_mask_prob=args.cond_mask_base_prob,
            )

            # bind 任务的监督 role 使用 sampled_role；其他任务默认用 reader 相关字段占位。
            role_for_labels = sampled_role if task_name == "bind" else "reader"
            # 结构特征构造时 cond_base 需可映射，不合法则回退到 mask。
            struct_base = cond_base if cond_base in COND_BASE_IDS else "mask"

            # 组装 batch（含 tokens、mask、struct 特征、site 信息、各任务标签等）。
            batch = collate_batch(
                examples=batch_examples,
                task_name=task_name,
                role_name=role_for_labels,
                cond_base=struct_base,
                bpp_cache=bpp_cache,
                strong_binding_threshold=strong_thresholds.get(role_for_labels, 1.0),
                rng=rng,
                mod_unlabeled_ratio=args.mod_unlabeled_ratio,
                mask_prob=args.mask_prob,
            )
            batch = to_device(batch, device)

            # 构造模型条件 token（task/role/base）。
            bsz = batch["tokens"].shape[0]
            cond_task_tensor = torch.full((bsz,), TASK_IDS[task_name], dtype=torch.long, device=device)
            cond_role_tensor = torch.full((bsz,), ROLE_IDS.get(cond_role, ROLE_IDS["none"]), dtype=torch.long, device=device)
            cond_base_tensor = torch.full((bsz,), COND_BASE_IDS.get(cond_base, COND_BASE_IDS["mask"]), dtype=torch.long, device=device)

            # 条件消融：强制所有条件恒定，测试模型对条件分支的依赖程度。
            if args.ablate_no_condition:
                cond_task_tensor = torch.zeros_like(cond_task_tensor)
                cond_role_tensor = torch.zeros_like(cond_role_tensor)
                cond_base_tensor = torch.full_like(cond_base_tensor, COND_BASE_IDS["mask"])

            # mask 任务使用 MLM 输入，其它任务使用原 tokens。
            tokens_in = batch["mlm_input"] if task_name == "mask" else batch["tokens"]
            # 仅当 base 条件为 A 时，对 bind/mod 施加强监督损失。
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

                # 各任务损失先置零，再按 task_name 选择性计算。
                loss_mod = torch.tensor(0.0, device=device)
                loss_bind = torch.tensor(0.0, device=device)
                loss_struct = torch.tensor(0.0, device=device)
                loss_mlm = torch.tensor(0.0, device=device)
                loss_unc = torch.tensor(0.0, device=device)
                loss_dir = torch.tensor(0.0, device=device)

                if task_name == "bind":
                    bind_probs = torch.sigmoid(outputs["bind_logits"])
                    if supervised_a:
                        # 结合位点 PU 损失（按 sampled_role 选择先验）。
                        loss_bind = bind_pu_loss[sampled_role](
                            logits=outputs["bind_logits"],
                            labels=batch["site_pu_labels"],
                            mask=batch["site_mask"],
                        )
                        if not args.ablate_no_dirichlet:
                            # 仅正例位点上计算 evidential 正例损失。
                            positive_mask = (batch["site_pu_labels"] == 1) & batch["site_mask"]
                            loss_dir = evidential_positive_loss(outputs["bind_alpha"], positive_mask=positive_mask)
                    # 无论是否 supervised_a，都计算不确定性正则项。
                    loss_unc = compute_uncertainty_loss(
                        uncertainty=outputs["bind_uncertainty"],
                        probs=bind_probs,
                        strong_mask=batch["strong_binding_mask"],
                        site_mask=batch["site_mask"],
                        supervised=supervised_a,
                    )

                elif task_name == "mod":
                    if supervised_a:
                        # m6A 可修饰性 PU 损失，取 mod_logits_acu[...,0] 通道作为目标 logit。
                        loss_mod = mod_pu_loss(
                            logits=outputs["mod_logits_acu"][..., 0],
                            labels=batch["mod_pu_labels"],
                            mask=batch["mod_pu_mask"],
                        )

                elif task_name == "struct" and not args.ablate_no_struct:
                    # 结构任务损失：BCE + Dice 组合。
                    loss_struct = structure_bce_dice_loss(
                        logits=outputs["struct_logits"],
                        target=batch["struct_target"],
                        valid_lengths=batch["struct_lengths"],
                        min_sep=args.struct_min_sep,
                    )

                elif task_name == "mask":
                    # MLM 任务损失：仅在未被 ignore 的 token 上有效。
                    loss_mlm = F.cross_entropy(
                        outputs["mask_logits"].reshape(-1, 4),
                        batch["mlm_target"].reshape(-1),
                        ignore_index=-100,
                    )

                # 多任务总损失（固定加权系数）。
                loss_total = (
                    1.0 * loss_mod
                    + 1.2 * (loss_bind + 0.2 * loss_dir)
                    + 0.8 * loss_struct
                    + 0.2 * loss_mlm
                    + 0.2 * loss_unc
                )

            # 梯度累积：先按 grad_accum 缩放后反传。
            loss_for_step = loss_total / args.grad_accum
            # 极端情况下若 loss 图断开，用 0*sum(logits) 保持图可反传（数值不变）。
            if not loss_for_step.requires_grad:
                loss_for_step = loss_for_step + 0.0 * outputs["mod_logits_acu"].sum()
            scaler.scale(loss_for_step).backward()

            # 达到累积步数（或最后一批）时执行一次优化器更新。
            if (batch_idx % args.grad_accum == 0) or (batch_idx == len(batches)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optim_step += 1

            # 聚合 epoch 级训练统计。
            agg["loss_total"] += float(loss_total.detach().cpu())
            agg["loss_bind"] += float(loss_bind.detach().cpu())
            agg["loss_mod"] += float(loss_mod.detach().cpu())
            agg["loss_struct"] += float(loss_struct.detach().cpu())
            agg["loss_mlm"] += float(loss_mlm.detach().cpu())
            agg["loss_unc"] += float(loss_unc.detach().cpu())
            agg["loss_dir"] += float(loss_dir.detach().cpu())
            agg["steps"] += 1

            # 可选：按 step 写入 TensorBoard。
            if writer is not None and (global_step % max(1, args.tb_log_steps) == 0):
                writer.add_scalar("train/loss_total_step", float(loss_total.detach().cpu()), global_step)
                writer.add_scalar("train/loss_bind_step", float(loss_bind.detach().cpu()), global_step)
                writer.add_scalar("train/loss_mod_step", float(loss_mod.detach().cpu()), global_step)
                writer.add_scalar("train/loss_struct_step", float(loss_struct.detach().cpu()), global_step)
                writer.add_scalar("train/loss_mlm_step", float(loss_mlm.detach().cpu()), global_step)
                writer.add_scalar("train/loss_unc_step", float(loss_unc.detach().cpu()), global_step)
                writer.add_scalar("train/loss_dir_step", float(loss_dir.detach().cpu()), global_step)
                writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)

            global_step += 1

        # 计算本 epoch 平均损失。
        for key in list(agg.keys()):
            if key.startswith("loss_"):
                agg[key] = agg[key] / max(1, agg["steps"])

        # 16) 每个 epoch 后固定评估 reader 结合任务（验证集）。
        val_metrics = evaluate_reader_binding(
            model=model,
            examples=val_examples,
            bpp_cache=bpp_cache,
            strong_thresholds=strong_thresholds,
            batch_token_budget=args.batch_token_budget,
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
        # 可选：按 epoch 写入 TensorBoard。
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
    # 关闭 TensorBoard writer，确保事件文件落盘完整。
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
