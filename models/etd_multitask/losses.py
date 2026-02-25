"""Loss functions for ETD multi-task model."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class NonNegativePULoss(nn.Module):
    """Non-negative PU risk estimator with logistic surrogate."""

    def __init__(self, prior: float, beta: float = 0.0, gamma: float = 1.0):
        super().__init__()
        self.prior = float(prior)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)

        pos_mask = (labels == 1) & mask
        unl_mask = (labels == -1) & mask

        if not pos_mask.any() and not unl_mask.any():
            return logits.new_tensor(0.0)

        pos_logits = logits[pos_mask] if pos_mask.any() else logits.new_zeros((0,))
        unl_logits = logits[unl_mask] if unl_mask.any() else logits.new_zeros((0,))

        positive_risk = logits.new_tensor(0.0)
        if pos_logits.numel() > 0:
            positive_risk = self.prior * F.softplus(-pos_logits).mean()

        negative_risk = logits.new_tensor(0.0)
        if unl_logits.numel() > 0:
            negative_risk = F.softplus(unl_logits).mean()
        if pos_logits.numel() > 0:
            negative_risk = negative_risk - self.prior * F.softplus(pos_logits).mean()

        if negative_risk < -self.beta:
            return positive_risk - self.gamma * negative_risk
        return positive_risk + torch.clamp(negative_risk, min=0.0)


def dirichlet_binary_nll(alpha: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)

    if not mask.any():
        return alpha.new_tensor(0.0)

    alpha = alpha[mask]
    labels = labels[mask].long()

    alpha_sum = alpha.sum(dim=-1)
    log_prob = torch.digamma(alpha_sum) - torch.digamma(alpha.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1))
    return log_prob.mean()


def evidential_positive_loss(
    alpha: torch.Tensor,
    positive_mask: torch.Tensor,
    min_strength: float = 5.0,
) -> torch.Tensor:
    """Encourage confident positive evidence on known positive samples."""
    if alpha.numel() == 0:
        return alpha.new_tensor(0.0)

    if positive_mask.dtype != torch.bool:
        positive_mask = positive_mask.bool()

    if not positive_mask.any():
        return alpha.new_tensor(0.0)

    alpha_pos = alpha[positive_mask]
    pos_prob = alpha_pos[..., 1] / alpha_pos.sum(dim=-1).clamp_min(1e-8)
    strength = alpha_pos.sum(dim=-1)

    loss_prob = F.relu(0.8 - pos_prob).mean()
    loss_strength = F.relu(min_strength - strength).mean()
    return loss_prob + 0.1 * loss_strength


def structure_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_lengths: torch.Tensor,
    min_sep: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    bsz, length, _ = logits.shape
    device = logits.device

    idx = torch.arange(length, device=device)
    tri = (idx[None, :, None] < idx[None, None, :])
    if min_sep > 0:
        tri = tri & ((idx[None, :, None] + min_sep) < idx[None, None, :])

    valid = idx[None, :] < valid_lengths[:, None]
    valid2d = valid[:, :, None] & valid[:, None, :]
    mask = tri & valid2d

    if not mask.any():
        return logits.new_tensor(0.0)

    bce = F.binary_cross_entropy_with_logits(logits[mask], target[mask], reduction="mean")

    probs = torch.sigmoid(logits)
    probs = probs * mask.float()
    target_masked = target * mask.float()

    intersection = (probs * target_masked).sum()
    union = probs.sum() + target_masked.sum()
    dice = 1.0 - (2.0 * intersection + eps) / (union + eps)

    return bce + dice
