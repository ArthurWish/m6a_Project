"""Task Loss Composer.

本模块负责“训练阶段的任务级损失编排”，而不是定义基础损失公式本身。

定位：
- `losses.py` 放基础损失函数（nnPU、evidential、structure 等）。
- 本文件放任务路由与组合逻辑（bind/mod/struct/mask 何时启用哪些项）。

这样做的目的：
1) 缩短 `train_etd_multitask.py` 主循环，提高可读性。
2) 把任务策略集中在一个文件，便于实验对照与后续扩展。
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .losses import evidential_positive_loss, grouped_binding_loss, structure_bce_dice_loss


def compute_uncertainty_loss(
    uncertainty: torch.Tensor,
    probs: torch.Tensor,
    strong_mask: torch.Tensor,
    site_mask: torch.Tensor,
    supervised: bool,
    supervised_unc_max: float = 0.2,
    supervised_prob_min: float = 0.8,
    unsupervised_unc_min: float = 0.6,
) -> torch.Tensor:
    """不确定性约束损失（legacy 路径）。

    参数：
    - uncertainty: 模型输出不确定度，形状与 site 维度对齐
    - probs: bind 概率（sigmoid(logits)）
    - strong_mask: 强结合位点掩码
    - site_mask: 有效位点掩码
    - supervised: 是否在“有监督 A 条件”下

    行为：
    - supervised=True:
      仅对 strong 位点施加“低不确定度 + 高概率”约束
    - supervised=False:
      对有效位点施加“不要过低不确定度”约束（避免过度自信）
    """
    if supervised:
        target_mask = strong_mask & site_mask
        if target_mask.any():
            low_unc = torch.relu(uncertainty[target_mask] - supervised_unc_max)
            high_prob = torch.relu(supervised_prob_min - probs[target_mask])
            return (low_unc + high_prob).mean()
        return uncertainty.new_tensor(0.0)

    target_mask = site_mask
    if target_mask.any():
        return torch.relu(unsupervised_unc_min - uncertainty[target_mask]).mean()
    return uncertainty.new_tensor(0.0)


def compute_multitask_losses(
    task_name: str,
    outputs: dict[str, torch.Tensor],
    batch: dict,
    args: argparse.Namespace,
    supervised_a: bool,
    sampled_role: str,
    mod_pu_loss,
    bind_pu_loss: dict,
) -> dict[str, torch.Tensor]:
    """按任务类型计算并组合损失，返回统一字典。

    参数（关键）：
    - task_name: 当前 batch 对应任务（bind/mod/struct/mask）
    - outputs: 模型前向输出字典
    - batch: collate 后批数据（含标签、mask、分组信息）
    - args: 训练参数命名空间（包含开关与阈值）
    - supervised_a: 当前条件是否为 A（legacy bind/mod 会用到）
    - sampled_role: 当前 bind 任务采样到的 role（writer/reader/eraser）
    - mod_pu_loss: mod 任务的 nnPU 损失对象
    - bind_pu_loss: bind 任务按 role 分开的 nnPU 损失对象

    返回：
    - `loss_total`: 最终用于反传的总损失
    - 其余 `loss_*`: 各子损失分量，便于日志统计与 TensorBoard 记录
    """

    device = outputs["mod_logits"].device
    loss_mod = torch.tensor(0.0, device=device)
    loss_bind = torch.tensor(0.0, device=device)
    loss_struct = torch.tensor(0.0, device=device)
    loss_mlm = torch.tensor(0.0, device=device)
    loss_unc = torch.tensor(0.0, device=device)
    loss_dir = torch.tensor(0.0, device=device)

    # ---- bind 任务 ----
    # 两条路径：
    # 1) grouped loss（基于 G1~G5 分组）
    # 2) legacy loss（原 PU + evidential + uncertainty 约束）
    if task_name == "bind":
        bind_probs = torch.sigmoid(outputs["bind_logits"])
        if args.bind_grouped_loss:
            # G1~G5 分组损失：更细粒度地控制 reveal/unlabeled/普通A 行为。
            grouped = grouped_binding_loss(
                logits=outputs["bind_logits"],
                alpha=outputs["bind_alpha"],
                uncertainty=outputs["bind_uncertainty"],
                g1_mask=batch["g1_mask"] & batch["site_mask"],
                g2_mask=batch["g2_mask"] & batch["site_mask"],
                g3_mask=batch["g3_mask"] & batch["site_mask"],
                g5_mask=batch["g5_mask"] & batch["site_mask"],
                g3_prob_max=args.bind_g3_prob_max,
                g5_prob_max=args.bind_g5_prob_max,
                g5_unc_min=args.bind_g5_unc_min,
                g1_unc_max=args.bind_g1_unc_max,
            )
            loss_bind = grouped["core"]
            if not args.ablate_no_dirichlet:
                loss_dir = grouped["dir"]
            loss_unc = grouped["unc"]
        else:
            # Legacy bind：
            # 仅在 supervised_a 条件下计算 PU 主损失与 evidential 正例损失。
            if supervised_a:
                loss_bind = bind_pu_loss[sampled_role](
                    logits=outputs["bind_logits"],
                    labels=batch["site_pu_labels"],
                    mask=batch["site_mask"],
                )
                if not args.ablate_no_dirichlet:
                    positive_mask = (batch["site_pu_labels"] == 1) & batch["site_mask"]
                    loss_dir = evidential_positive_loss(outputs["bind_alpha"], positive_mask=positive_mask)
            # 不确定度项可在有/无监督两种模式下计算。
            loss_unc = compute_uncertainty_loss(
                uncertainty=outputs["bind_uncertainty"],
                probs=bind_probs,
                strong_mask=batch["strong_binding_mask"],
                site_mask=batch["site_mask"],
                supervised=supervised_a,
                supervised_unc_max=args.bind_legacy_supervised_unc_max,
                supervised_prob_min=args.bind_legacy_supervised_prob_min,
                unsupervised_unc_min=args.bind_legacy_unsupervised_unc_min,
            )

    # ---- mod 任务 ----
    # 当前策略：仅在 supervised_a 条件下计算 mod 的 nnPU 主损失。
    elif task_name == "mod":
        if supervised_a:
            loss_mod = mod_pu_loss(
                logits=outputs["mod_logits_acu"][..., 0],
                labels=batch["mod_pu_labels"],
                mask=batch["mod_pu_mask"],
            )

    # ---- struct 任务 ----
    # 使用 BCE + Dice 组合损失；可由 ablation 开关关闭。
    elif task_name == "struct" and not args.ablate_no_struct:
        loss_struct = structure_bce_dice_loss(
            logits=outputs["struct_logits"],
            target=batch["struct_target"],
            valid_lengths=batch["struct_lengths"],
            min_sep=args.struct_min_sep,
        )

    # ---- mask 任务 ----
    # 经典 MLM 交叉熵；ignore_index 位置不参与损失。
    elif task_name == "mask":
        loss_mlm = F.cross_entropy(
            outputs["mask_logits"].reshape(-1, 4),
            batch["mlm_target"].reshape(-1),
            ignore_index=-100,
        )

    # ---- 总损失加权 ----
    # 保持与原训练脚本一致的系数，便于行为对齐与历史结果可比。
    loss_total = (
        float(args.loss_w_mod) * loss_mod
        + float(args.loss_w_bind) * (loss_bind + float(args.loss_w_dir_in_bind) * loss_dir)
        + float(args.loss_w_struct) * loss_struct
        + float(args.loss_w_mlm) * loss_mlm
        + float(args.loss_w_unc) * loss_unc
    )

    return {
        "loss_total": loss_total,
        "loss_bind": loss_bind,
        "loss_mod": loss_mod,
        "loss_struct": loss_struct,
        "loss_mlm": loss_mlm,
        "loss_unc": loss_unc,
        "loss_dir": loss_dir,
    }
