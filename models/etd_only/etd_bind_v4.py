"""Multi-task model v4: shared backbone + m6A head + RBP bind head.

m6A head: 对每个 A 位置输出 P(m6A), 替代 presence_head
bind head: 对 m6A 位点输出 17 个 RBP 概率
推理: P(RBP_i) = P(m6A) × P(RBP_i | m6A)

所有位置组共享同一次 backbone forward。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from models.etd_multitask.constants import NUM_INDIVIDUAL_RBPS


@dataclass
class MultiTaskConfig:
    vocab_size: int = 7
    d_model: int = 256
    encoder_channels: tuple[int, ...] = (256, 384, 512)
    n_transformer_layers: int = 4
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    head_hidden: int = 256
    head_dropout: float = 0.2


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample):
        super().__init__()
        stride = 2 if downsample else 1
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = (nn.Conv1d(in_ch, out_ch, 1, stride=stride)
                     if in_ch != out_ch or stride != 1 else nn.Identity())

    def forward(self, x):
        xc = x.transpose(1, 2)
        res = self.skip(xc)
        out = self.conv1(F.gelu(self.norm1(xc)))
        out = self.conv2(F.gelu(self.norm2(out)))
        return F.gelu(out + res).transpose(1, 2)


class TransformerBottleneck(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, ff_mult, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        return self.norm(self.encoder(x, src_key_padding_mask=key_padding_mask))


class MultiTaskBindModel(nn.Module):
    def __init__(self, cfg: MultiTaskConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)

        ch = cfg.encoder_channels
        enc_in = [cfg.d_model, *ch[:-1]]
        self.encoder = nn.ModuleList([
            EncoderBlock(enc_in[i], ch[i], downsample=(i < len(ch) - 1))
            for i in range(len(ch))
        ])
        self.n_downsample = sum(1 for i in range(len(ch)) if i < len(ch) - 1)

        self.bottleneck = TransformerBottleneck(
            ch[-1], cfg.n_transformer_layers, cfg.n_heads, cfg.ff_mult, cfg.dropout)

        feat_dim = ch[-1] + cfg.d_model

        # m6A detection head (binary: is this A an m6A?)
        self.m6a_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden), 
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )
    

        # RBP binding head (17-class multi-label)
        self.bind_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, NUM_INDIVIDUAL_RBPS),
        )

    @staticmethod
    def _downsample_mask(mask, n):
        x = mask.float().unsqueeze(1)
        for _ in range(n):
            x = F.max_pool1d(x, 2, stride=2, ceil_mode=True)
        return x.squeeze(1) > 0.5

    def _gather_feat(self, bn, x0, positions):
        """Gather multi-scale features at given positions.
        positions: [B, N] (window-local coords)
        returns: [B, N, feat_dim]
        """
        B, N = positions.shape
        down = (positions // (2 ** self.n_downsample)).clamp(0, bn.shape[1] - 1)
        idx_bn = down.unsqueeze(-1).expand(B, N, bn.shape[-1])
        feat_bn = bn.gather(1, idx_bn)

        idx_orig = positions.clamp(0, x0.shape[1] - 1).unsqueeze(-1).expand(B, N, x0.shape[-1])
        feat_orig = x0.gather(1, idx_orig)
        return torch.cat([feat_bn, feat_orig], dim=-1)

    def forward(
        self,
        tokens,            # [B, W]
        attn_mask,         # [B, W]
        center_indices,    # [B]
        # m6A detection positions
        m6a_det_positions=None,  # [B, K]
        m6a_det_mask=None,       # [B, K]
        # RBP: extra positive m6A+reader
        extra_pos_positions=None,  # [B, P]
        extra_pos_mask=None,
        # RBP: m6A without reader
        m6a_neg_positions=None,    # [B, M]
        m6a_neg_mask=None,
        # RBP: clean A negatives
        clean_neg_positions=None,  # [B, N]
        clean_neg_mask=None,
    ):
        B = tokens.shape[0]

        # ---- Shared backbone (run once) ----
        x0 = self.embed_drop(self.token_embed(tokens))
        x = x0
        for block in self.encoder:
            x = block(x)
        if attn_mask is not None:
            kpm = ~self._downsample_mask(attn_mask, self.n_downsample)
        else:
            kpm = None
        bn = self.bottleneck(x, key_padding_mask=kpm)

        # ---- Center site (always present) ----
        center_feat = self._gather_feat(bn, x0, center_indices.unsqueeze(1)).squeeze(1)
        center_bind_logits = self.bind_head(center_feat)         # [B, 17]
        center_m6a_logit = self.m6a_head(center_feat).squeeze(-1)  # [B]

        result = {
            "center_bind_logits": center_bind_logits,
            "center_m6a_logit": center_m6a_logit,
        }

        # ---- m6A detection task ----
        if m6a_det_positions is not None and m6a_det_mask is not None and m6a_det_mask.any():
            feat = self._gather_feat(bn, x0, m6a_det_positions)
            result["m6a_det_logits"] = self.m6a_head(feat).squeeze(-1)  # [B, K]
            result["m6a_det_mask"] = m6a_det_mask

        # ---- Extra positives ----
        if extra_pos_positions is not None and extra_pos_mask is not None and extra_pos_mask.any():
            feat = self._gather_feat(bn, x0, extra_pos_positions)
            result["extra_pos_bind_logits"] = self.bind_head(feat)  # [B, P, 17]
            result["extra_pos_mask"] = extra_pos_mask

        # ---- m6A neg (uncertain) ----
        if m6a_neg_positions is not None and m6a_neg_mask is not None and m6a_neg_mask.any():
            feat = self._gather_feat(bn, x0, m6a_neg_positions)
            result["m6a_neg_bind_logits"] = self.bind_head(feat)   # [B, M, 17]
            result["m6a_neg_mask"] = m6a_neg_mask

        # ---- Clean A neg ----
        if clean_neg_positions is not None and clean_neg_mask is not None and clean_neg_mask.any():
            feat = self._gather_feat(bn, x0, clean_neg_positions)
            result["clean_neg_bind_logits"] = self.bind_head(feat)  # [B, N, 17]
            result["clean_neg_mask"] = clean_neg_mask

        return result