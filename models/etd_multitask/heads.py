"""Prediction heads for ETD multi-task model."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class MaskHead(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_model, 4)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class ModHead(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_model, 3)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class BindDirichletHead(nn.Module):
    def __init__(self, d_model: int = 256, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.evidence = nn.Linear(hidden, 2)
        self.logit = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor, site_positions: torch.Tensor, site_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        _, seq_len, channels = h.shape

        clamped = site_positions.clamp(min=0, max=max(0, seq_len - 1))
        gather_idx = clamped.unsqueeze(-1).expand(-1, -1, channels)
        site_feat = torch.gather(h, dim=1, index=gather_idx)
        site_feat = self.mlp(site_feat)

        evidence = F.softplus(self.evidence(site_feat))
        alpha = evidence + 1.0
        probs = alpha[..., 1] / alpha.sum(dim=-1).clamp_min(1e-8)
        uncertainty = 2.0 / alpha.sum(dim=-1).clamp_min(1e-8)

        bind_logits = self.logit(site_feat).squeeze(-1)

        probs = probs * site_mask.float()
        uncertainty = uncertainty * site_mask.float()
        bind_logits = bind_logits * site_mask.float()

        return {
            "bind_alpha": alpha,
            "alpha": alpha,
            "bind_prob": probs,
            "bind_uncertainty": uncertainty,
            "bind_logits": bind_logits,
        }


class StructHead(nn.Module):
    def __init__(self, bottleneck_dim: int = 512, hidden: int = 64):
        super().__init__()
        self.q = nn.Linear(bottleneck_dim, hidden)
        self.k = nn.Linear(bottleneck_dim, hidden)
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, bottleneck: torch.Tensor, adjacency_logits: torch.Tensor) -> torch.Tensor:
        q = self.q(bottleneck)
        k = self.k(bottleneck)
        logits = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(q.shape[-1])
        logits = logits + adjacency_logits
        logits = logits.unsqueeze(1)
        logits = self.refine(logits).squeeze(1)
        logits = 0.5 * (logits + logits.transpose(-1, -2))
        return logits
