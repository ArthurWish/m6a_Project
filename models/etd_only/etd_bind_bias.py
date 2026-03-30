"""带 attention bias 的窗口多任务模型。

设计目标：
1. 保持与 `etd_bind_v4.py` 基本一致的输入/输出接口；
2. 仅把 bottleneck Transformer 换成“支持外部 attention bias”的版本；
3. 让 RNAfold 等结构先验可以直接加到 attention logits 上；
4. 不影响原始 v4 baseline 的可复现实验。

实现策略：
- CNN encoder 部分保持不变；
- 只在 bottleneck 自注意力里新增 `attn_bias`；
- task heads 仍然沿用中心位点 m6A 预测 + bind 多标签预测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from models.etd_multitask.constants import NUM_INDIVIDUAL_RBPS


@dataclass
class MultiTaskBiasConfig:
    vocab_size: int = 7
    d_model: int = 256
    encoder_channels: tuple[int, ...] = (256, 384, 512)
    n_transformer_layers: int = 4
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    head_hidden: int = 256
    head_dropout: float = 0.2
    attn_dropout: float = 0.1


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool):
        super().__init__()
        stride = 2 if downsample else 1
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1, stride=stride)
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x.transpose(1, 2)
        res = self.skip(xc)
        out = self.conv1(F.gelu(self.norm1(xc)))
        out = self.conv2(F.gelu(self.norm2(out)))
        return F.gelu(out + res).transpose(1, 2)


class MultiHeadSelfAttentionWithBias(nn.Module):
    """带加性 bias 的多头自注意力。

    这里的 `attn_bias` 语义是：
    - 在 softmax 前，直接加到 attention logits 上；
    - 值越大，表示模型越倾向关注对应 token 对；
    - 当前实验里，这个 bias 来自 RNAfold 配对概率矩阵的下采样结果。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 输入 x: [B, L, D]
        # 先投影成标准多头注意力的 q/k/v。
        bsz, length, _ = x.shape

        q = self.q_proj(x).view(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 外部结构先验在这里注入。
        # 支持两种形状：
        # - [B, L, L]：所有头共享同一张 bias
        # - [B, H, L, L]：每个头单独一张 bias
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                scores = scores + attn_bias.unsqueeze(1)
            elif attn_bias.dim() == 4:
                scores = scores + attn_bias
            else:
                raise ValueError(
                    f"attn_bias must be [B,L,L] or [B,H,L,L], got shape={tuple(attn_bias.shape)}"
                )

        # padding 位置仍然要屏蔽掉，避免被 bias 重新抬起来。
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, length, self.d_model)
        out = self.out_drop(self.out_proj(out))
        return out


class TransformerBlockWithBias(nn.Module):
    """最小 Transformer block。

    结构上仍然是：
    - pre-norm attention
    - pre-norm FFN
    - 两次残差连接

    与普通 block 的唯一差异是 attention 支持外部 bias。
    """
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttentionWithBias(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerBottleneckWithBias(nn.Module):
    """堆叠多个支持 bias 的 Transformer block。"""
    def __init__(self, d_model: int, n_layers: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlockWithBias(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        return self.norm(x)


class MultiTaskBindBiasModel(nn.Module):
    """带结构 bias 的 v4 风格模型。

    输入仍然是窗口 token，但 forward 额外支持：
    - `attn_bias`: 下采样后的结构先验矩阵

    输出仍然保持 v4 的字段命名，方便训练脚本和 loss 直接复用。
    """

    def __init__(self, cfg: MultiTaskBiasConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)

        ch = cfg.encoder_channels
        enc_in = [cfg.d_model, *ch[:-1]]
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(enc_in[i], ch[i], downsample=(i < len(ch) - 1))
                for i in range(len(ch))
            ]
        )
        self.n_downsample = sum(1 for i in range(len(ch)) if i < len(ch) - 1)

        self.bottleneck = TransformerBottleneckWithBias(
            d_model=ch[-1],
            n_layers=cfg.n_transformer_layers,
            n_heads=cfg.n_heads,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
        )

        feat_dim = ch[-1] + cfg.d_model
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
        self.bind_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, NUM_INDIVIDUAL_RBPS),
        )

    @staticmethod
    def _downsample_mask(mask: torch.Tensor, n: int) -> torch.Tensor:
        x = mask.float().unsqueeze(1)
        for _ in range(n):
            x = F.max_pool1d(x, 2, stride=2, ceil_mode=True)
        return x.squeeze(1) > 0.5

    def _gather_feat(self, bn: torch.Tensor, x0: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """在多个局部坐标上抽取多尺度表征。

        返回的特征由两部分拼接：
        - bottleneck 上的全局语义特征
        - 原始 embedding 空间的局部细节特征
        """
        bsz, n_pos = positions.shape
        down = (positions // (2 ** self.n_downsample)).clamp(0, bn.shape[1] - 1)
        idx_bn = down.unsqueeze(-1).expand(bsz, n_pos, bn.shape[-1])
        feat_bn = bn.gather(1, idx_bn)

        idx_orig = positions.clamp(0, x0.shape[1] - 1).unsqueeze(-1).expand(bsz, n_pos, x0.shape[-1])
        feat_orig = x0.gather(1, idx_orig)
        return torch.cat([feat_bn, feat_orig], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        attn_mask: torch.Tensor,
        center_indices: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        m6a_det_positions: torch.Tensor | None = None,
        m6a_det_mask: torch.Tensor | None = None,
        extra_pos_positions: torch.Tensor | None = None,
        extra_pos_mask: torch.Tensor | None = None,
        m6a_neg_positions: torch.Tensor | None = None,
        m6a_neg_mask: torch.Tensor | None = None,
        clean_neg_positions: torch.Tensor | None = None,
        clean_neg_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # 共享主干只跑一次，后续所有任务头都在同一份特征上 gather。
        x0 = self.embed_drop(self.token_embed(tokens))
        x = x0
        for block in self.encoder:
            x = block(x)
        kpm = ~self._downsample_mask(attn_mask, self.n_downsample) if attn_mask is not None else None

        # attn_bias 在 bottleneck 内部生效。
        # 它的空间尺度已经和下采样后的序列长度对齐。
        bn = self.bottleneck(x, key_padding_mask=kpm, attn_bias=attn_bias)

        center_feat = self._gather_feat(bn, x0, center_indices.unsqueeze(1)).squeeze(1)
        center_bind_logits = self.bind_head(center_feat)
        center_m6a_logit = self.m6a_head(center_feat).squeeze(-1)

        result: dict[str, torch.Tensor] = {
            "center_bind_logits": center_bind_logits,
            "center_m6a_logit": center_m6a_logit,
        }

        # m6A detection 头：在窗口内一组候选 A 上做密集判别。
        if m6a_det_positions is not None and m6a_det_mask is not None and m6a_det_mask.any():
            feat = self._gather_feat(bn, x0, m6a_det_positions)
            result["m6a_det_logits"] = self.m6a_head(feat).squeeze(-1)
            result["m6a_det_mask"] = m6a_det_mask

        # extra positives：窗口内其他 m6A+reader 位点也参与 bind 监督。
        if extra_pos_positions is not None and extra_pos_mask is not None and extra_pos_mask.any():
            feat = self._gather_feat(bn, x0, extra_pos_positions)
            result["extra_pos_bind_logits"] = self.bind_head(feat)
            result["extra_pos_mask"] = extra_pos_mask

        # uncertain negative：已知 m6A，但无 reader 注释。
        if m6a_neg_positions is not None and m6a_neg_mask is not None and m6a_neg_mask.any():
            feat = self._gather_feat(bn, x0, m6a_neg_positions)
            result["m6a_neg_bind_logits"] = self.bind_head(feat)
            result["m6a_neg_mask"] = m6a_neg_mask

        # clean negative：普通 A，作为更高置信度的负样本。
        if clean_neg_positions is not None and clean_neg_mask is not None and clean_neg_mask.any():
            feat = self._gather_feat(bn, x0, clean_neg_positions)
            result["clean_neg_bind_logits"] = self.bind_head(feat)
            result["clean_neg_mask"] = clean_neg_mask

        return result
