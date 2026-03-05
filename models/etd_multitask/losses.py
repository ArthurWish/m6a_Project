"""Loss functions for ETD multi-task model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class NonNegativePULoss(nn.Module):
    """非负 PU（Positive-Unlabeled）风险估计损失。

    适用场景：
    - 标签只有正类(1)和未标注(-1)，没有可靠负类。
    - 通过先验 prior 估计总体正类比例。

    形式（logistic surrogate）：
    - 正风险:   pi * E_p[softplus(-f(x))]
    - 负风险:   E_u[softplus(f(x))] - pi * E_p[softplus(f(x))]
    - 总风险:   正风险 + clamp(负风险, min=0)（nnPU）
      当负风险小于 -beta 时，使用修正项避免过度负化：
      正风险 - gamma * 负风险
    """

    def __init__(self, prior: float, beta: float = 0.0, gamma: float = 1.0):
        super().__init__()
        self.prior = float(prior)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # 若未提供 mask，默认全部样本有效。
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)

        # PU 训练中：
        # - labels == 1  : 已知正类
        # - labels == -1 : 未标注（不等同于负类）
        pos_mask = (labels == 1) & mask
        unl_mask = (labels == -1) & mask

        # 当前 batch 没有有效样本时返回 0，避免 NaN。
        if not pos_mask.any() and not unl_mask.any():
            return logits.new_tensor(0.0)

        # 取出正类与未标注对应 logits；为空时用空张量占位。
        pos_logits = logits[pos_mask] if pos_mask.any() else logits.new_zeros((0,))
        unl_logits = logits[unl_mask] if unl_mask.any() else logits.new_zeros((0,))

        # 正风险项：希望正样本 logit 更大（softplus(-logit) 越小越好）。
        positive_risk = logits.new_tensor(0.0)
        if pos_logits.numel() > 0:
            positive_risk = self.prior * F.softplus(-pos_logits).mean()

        # 负风险估计：
        # 先用未标注集合估计“负向项”，再减去先验加权的正类泄漏项。
        negative_risk = logits.new_tensor(0.0)
        if unl_logits.numel() > 0:
            negative_risk = F.softplus(unl_logits).mean()
        if pos_logits.numel() > 0:
            negative_risk = negative_risk - self.prior * F.softplus(pos_logits).mean()

        # nnPU 非负修正：
        # - 常规分支：负风险截断到 >=0
        # - 极端分支：负风险过小（<-beta）时使用反向修正，缓解训练不稳定
        if negative_risk < -self.beta:
            return positive_risk - self.gamma * negative_risk
        return positive_risk + torch.clamp(negative_risk, min=0.0)


def dirichlet_binary_nll(alpha: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """二分类 Dirichlet 负对数似然（基于期望对数概率）。

    参数：
    - alpha: (..., 2) 的 Dirichlet 参数（证据+1，需为正）
    - labels: 0/1 类别标签
    - mask: 有效样本掩码
    """
    # 默认所有位置都参与。
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)

    # 无有效样本时返回 0。
    if not mask.any():
        return alpha.new_tensor(0.0)

    # 仅保留有效位置。
    alpha = alpha[mask]
    labels = labels[mask].long()

    # E[-log p(y)] = digamma(sum(alpha)) - digamma(alpha_y)
    # 这里按标签索引 alpha_y 并取均值。
    alpha_sum = alpha.sum(dim=-1)
    log_prob = torch.digamma(alpha_sum) - torch.digamma(alpha.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1))
    return log_prob.mean()


def evidential_positive_loss(
    alpha: torch.Tensor,
    positive_mask: torch.Tensor,
    min_strength: float = 5.0,
) -> torch.Tensor:
    """正样本证据约束损失（evidential regularization）。

    目标：
    - 在已知正样本上，提高正类概率（希望 >= 0.8）
    - 同时提高总证据强度（alpha.sum() 至少达到 min_strength）
    """
    # 空张量直接返回 0。
    if alpha.numel() == 0:
        return alpha.new_tensor(0.0)

    # 容忍传入非 bool，统一转 bool。
    if positive_mask.dtype != torch.bool:
        positive_mask = positive_mask.bool()

    # 没有正样本监督点时返回 0。
    if not positive_mask.any():
        return alpha.new_tensor(0.0)

    # 仅取正样本位置。
    alpha_pos = alpha[positive_mask]
    # Dirichlet 期望概率（正类 index=1）。
    pos_prob = alpha_pos[..., 1] / alpha_pos.sum(dim=-1).clamp_min(1e-8)
    # 证据强度（alpha 总和）。
    strength = alpha_pos.sum(dim=-1)

    # 概率不足 0.8 才产生惩罚；超过则为 0。
    loss_prob = F.relu(0.8 - pos_prob).mean()
    # 强度不足 min_strength 才惩罚；超过则为 0。
    loss_strength = F.relu(min_strength - strength).mean()
    # 组合损失：概率约束为主，强度约束为辅（0.1 权重）。
    return loss_prob + 0.1 * loss_strength


def structure_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_lengths: torch.Tensor,
    min_sep: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RNA 结构矩阵损失：上三角有效区间内的 BCE + Dice。

    参数：
    - logits/target: (B, L, L)
    - valid_lengths: 每条序列的有效长度（padding 之外无效）
    - min_sep: 最小配对间距，抑制过近配对
    """
    bsz, length, _ = logits.shape
    device = logits.device

    # tri: 只保留严格上三角 (i < j)。
    idx = torch.arange(length, device=device)
    tri = (idx[None, :, None] < idx[None, None, :])
    # 加入最小间隔约束：i + min_sep < j。
    if min_sep > 0:
        tri = tri & ((idx[None, :, None] + min_sep) < idx[None, None, :])

    # 按每个样本 valid_lengths 构建 2D 有效区域 mask（去掉 padding）。
    valid = idx[None, :] < valid_lengths[:, None]
    valid2d = valid[:, :, None] & valid[:, None, :]
    mask = tri & valid2d

    # 没有有效结构对时返回 0。
    if not mask.any():
        return logits.new_tensor(0.0)

    # 点级监督：二分类 BCE（仅在有效 mask 上计算）。
    bce = F.binary_cross_entropy_with_logits(logits[mask], target[mask], reduction="mean")

    # 区域级监督：Dice，提升稀疏结构预测的重叠质量。
    probs = torch.sigmoid(logits)
    probs = probs * mask.float()
    target_masked = target * mask.float()

    intersection = (probs * target_masked).sum()
    union = probs.sum() + target_masked.sum()
    dice = 1.0 - (2.0 * intersection + eps) / (union + eps)

    # 组合损失。
    return bce + dice


def grouped_binding_loss(
    logits: torch.Tensor,
    alpha: torch.Tensor,
    uncertainty: torch.Tensor,
    g1_mask: torch.Tensor,
    g2_mask: torch.Tensor,
    g3_mask: torch.Tensor,
    g5_mask: torch.Tensor,
    g3_prob_max: float = 0.3,
    g5_prob_max: float = 0.2,
    g5_unc_min: float = 0.6,
    g1_unc_max: float = 0.2,
) -> dict[str, torch.Tensor]:
    """按 G1~G5 分组规则构建 bind 任务损失。

    分组语义（由 collate_batch 提供 mask）：
    - G1: m6A + A' + role=1   -> 高概率、低不确定度、正例证据增强
    - G2: m6A + A' + role=-1  -> PU 的 unlabeled 端
    - G3: m6A + A  + role=1   -> 负向约束（不应高概率）
    - G5: 非 m6A 的 A         -> 低概率 + 高不确定度锚定

    返回值字段：
    - core: 主监督项（G1/G2/G3/G5 的概率约束）
    - dir:  证据项（G1）
    - unc:  不确定度项（G1/G5）
    """

    def _zero() -> torch.Tensor:
        return logits.new_tensor(0.0)

    probs = torch.sigmoid(logits)

    # G1: 正样本核心监督（希望 logit 高）。
    loss_g1_pos = F.softplus(-logits[g1_mask]).mean() if g1_mask.any() else _zero()
    # G2: unlabeled 端（与 PU 的 U 风险方向一致）。
    loss_g2_unl = F.softplus(logits[g2_mask]).mean() if g2_mask.any() else _zero()
    # G3: 负向约束，抑制未 reveal 的正位点出现过高结合概率。
    loss_g3 = F.relu(probs[g3_mask] - g3_prob_max).mean() if g3_mask.any() else _zero()
    # G5: 普通 A 锚点，概率不应高。
    loss_g5_prob = F.relu(probs[g5_mask] - g5_prob_max).mean() if g5_mask.any() else _zero()

    core = loss_g1_pos + loss_g2_unl + loss_g3 + loss_g5_prob

    # G1 evidential 正例项：鼓励正类证据与强度。
    loss_dir = evidential_positive_loss(alpha, positive_mask=g1_mask)

    # 不确定度约束：
    # - G1 需要低不确定度
    # - G5 需要高不确定度
    loss_g1_unc = F.relu(uncertainty[g1_mask] - g1_unc_max).mean() if g1_mask.any() else _zero()
    loss_g5_unc = F.relu(g5_unc_min - uncertainty[g5_mask]).mean() if g5_mask.any() else _zero()
    loss_unc = loss_g1_unc + loss_g5_unc

    return {
        "core": core,
        "dir": loss_dir,
        "unc": loss_unc,
    }
