"""Condition encoder for task/role/base control."""

from __future__ import annotations

import torch
from torch import nn


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        task_vocab: int = 4,
        role_vocab: int = 4,
        base_vocab: int = 5,
        mod_type_vocab: int = 5, 
        embed_dim: int = 64,
        bottleneck_dim: int = 512,
        pair_dim: int = 128,
        decoder_dim: int = 256,
    ):
        super().__init__()
        self.task_embed = nn.Embedding(task_vocab, embed_dim)
        self.role_embed = nn.Embedding(role_vocab, embed_dim)
        self.base_embed = nn.Embedding(base_vocab, embed_dim)
        self.mod_type_embed = nn.Embedding(mod_type_vocab, embed_dim)

        hidden = embed_dim * 4
        out_dim = 2 * (bottleneck_dim + pair_dim + decoder_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        

        self.bottleneck_dim = bottleneck_dim
        self.pair_dim = pair_dim
        self.decoder_dim = decoder_dim

    def forward(self, task_ids: torch.Tensor, role_ids: torch.Tensor, base_ids: torch.Tensor, mod_type_ids: torch.Tensor ) -> dict[str, torch.Tensor]:
        task_vec = self.task_embed(task_ids) 
        role_vec = self.role_embed(role_ids)
        base_vec = self.base_embed(base_ids)
        mod_type_vec = self.mod_type_embed(mod_type_ids) 


        cond = torch.cat([task_vec, role_vec, base_vec, mod_type_vec], dim=-1)
        params = self.mlp(cond)

        split1 = self.bottleneck_dim
        split2 = split1 + self.bottleneck_dim
        split3 = split2 + self.pair_dim
        split4 = split3 + self.pair_dim
        split5 = split4 + self.decoder_dim

        gamma_b = params[:, :split1]
        beta_b = params[:, split1:split2]
        gamma_p = params[:, split2:split3]
        beta_p = params[:, split3:split4]
        gamma_d = params[:, split4:split5]
        beta_d = params[:, split5:]

        return {
            "gamma_b": gamma_b,
            "beta_b": beta_b,
            "gamma_p": gamma_p,
            "beta_p": beta_p,
            "gamma_d": gamma_d,
            "beta_d": beta_d,
        }
