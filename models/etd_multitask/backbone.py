"""High-fidelity ETD backbone with sequence/pairwise co-updates."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool):
        super().__init__()
        stride = 2 if downsample else 1
        self.downsample = downsample

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch or downsample:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cf = x.transpose(1, 2)
        residual = self.skip(x_cf)

        out = self.conv1(F.gelu(self.norm1(x_cf)))
        out = self.conv2(F.gelu(self.norm2(out)))
        out = F.gelu(out + residual)
        return out.transpose(1, 2)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, upsample: bool):
        super().__init__()
        self.upsample = upsample
        self.residual_scale = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))

        self.pre_conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.skip_conv = nn.Conv1d(skip_ch, out_ch, kernel_size=1)

        merged = out_ch * 2
        self.merge_norm = nn.GroupNorm(8, merged)
        self.merge_conv1 = nn.Conv1d(merged, out_ch, kernel_size=3, padding=1)
        self.merge_conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_cf = x.transpose(1, 2)
        skip_cf = skip.transpose(1, 2)

        out = F.gelu(self.pre_conv(x_cf))
        if self.upsample:
            out = out.repeat_interleave(2, dim=-1)
            out = out[..., : skip_cf.shape[-1]]
            out = out * self.residual_scale

        skip_proj = F.gelu(self.skip_conv(skip_cf))
        merged = torch.cat([out, skip_proj], dim=1)

        residual = self.merge_conv1(F.gelu(self.merge_norm(merged)))
        out = self.merge_conv2(F.gelu(residual))
        out = F.gelu(out + residual)
        return out.transpose(1, 2)


def _apply_rope_q(x: torch.Tensor) -> torch.Tensor:
    """Apply RoPE on query tensor shaped [B, S, H, D]."""
    bsz, seq_len, n_heads, dim = x.shape
    half = dim // 2
    if half == 0:
        return x

    device = x.device
    dtype = x.dtype

    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freq = torch.arange(half, device=device, dtype=dtype)
    denom = torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), freq / max(1, half))
    theta = pos[:, None] / denom[None, :]

    sin = torch.sin(theta)[None, :, None, :]
    cos = torch.cos(theta)[None, :, None, :]

    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if dim % 2 == 1:
        x_rot = torch.cat([x_rot, x[..., -1:]], dim=-1)
    return x_rot


def _apply_rope_k(x: torch.Tensor) -> torch.Tensor:
    """Apply RoPE on key tensor shaped [B, S, D]."""
    bsz, seq_len, dim = x.shape
    half = dim // 2
    if half == 0:
        return x

    device = x.device
    dtype = x.dtype

    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freq = torch.arange(half, device=device, dtype=dtype)
    denom = torch.pow(torch.tensor(10000.0, device=device, dtype=dtype), freq / max(1, half))
    theta = pos[:, None] / denom[None, :]

    sin = torch.sin(theta)[None, :, :]
    cos = torch.cos(theta)[None, :, :]

    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if dim % 2 == 1:
        x_rot = torch.cat([x_rot, x[..., -1:]], dim=-1)
    return x_rot


class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, value_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by n_heads")

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.value_dim = value_dim if value_dim is not None else self.head_dim

        self.q_proj = nn.Linear(embed_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.value_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.value_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = _apply_rope_q(q)
        k = _apply_rope_k(k)

        logits = torch.einsum("bshd,btd->bhst", q, k) / math.sqrt(self.head_dim)
        logits = torch.tanh((logits + attention_bias) / 5.0) * 5.0

        if key_padding_mask is not None:
            invalid = ~key_padding_mask
            logits = logits.masked_fill(invalid[:, None, None, :], -1e4)
            logits = logits.masked_fill(invalid[:, None, :, None], -1e4)

        weights = torch.softmax(logits, dim=-1)
        weights = self.dropout(weights)

        out = torch.einsum("bhst,btd->bshd", weights, v)
        out = out.reshape(bsz, seq_len, self.n_heads * self.value_dim)
        out = self.out_proj(out)
        return out


class SequenceTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiQueryAttention(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        hidden = embed_dim * mlp_ratio
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), attention_bias=attention_bias, key_padding_mask=key_padding_mask))
        x = x + self.mlp(self.norm2(x))
        return x


class RowAttentionBlock(nn.Module):
    """Row-wise attention over pairwise tensor [B, P, P, F]."""

    def __init__(self, pair_dim: int = 128, attn_dim: int = 64, dropout: float = 0.1, row_chunk_size: int = 96):
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim)
        self.q_proj = nn.Linear(pair_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(pair_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(pair_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, pair_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(attn_dim)
        self.row_chunk_size = max(16, int(row_chunk_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, p_len, _, _ = x.shape
        y = self.norm(x)

        q = self.q_proj(y).reshape(bsz * p_len, p_len, -1)
        k = self.k_proj(y).reshape(bsz * p_len, p_len, -1)
        v = self.v_proj(y).reshape(bsz * p_len, p_len, -1)

        outs = []
        for start in range(0, q.shape[0], self.row_chunk_size):
            end = min(start + self.row_chunk_size, q.shape[0])
            q_chunk = q[start:end]
            k_chunk = k[start:end]
            v_chunk = v[start:end]

            attn_logits = torch.bmm(q_chunk, k_chunk.transpose(1, 2)) / self.scale
            attn = torch.softmax(attn_logits, dim=-1)
            attn = self.dropout(attn)
            out_chunk = torch.bmm(attn, v_chunk)
            outs.append(out_chunk)

        out = torch.cat(outs, dim=0).reshape(bsz, p_len, p_len, -1)
        out = self.out_proj(out)
        return self.dropout(out)


class PairMLPBlock(nn.Module):
    def __init__(self, pair_dim: int = 128, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = pair_dim * mlp_ratio
        self.norm = nn.LayerNorm(pair_dim)
        self.net = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pair_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class SequenceToPairBlock(nn.Module):
    def __init__(self, seq_dim: int = 512, pair_dim: int = 128):
        super().__init__()
        self.seq_norm = nn.LayerNorm(seq_dim)
        self.q_proj = nn.Linear(seq_dim, pair_dim, bias=False)
        self.k_proj = nn.Linear(seq_dim, pair_dim, bias=False)
        self.y_q = nn.Linear(seq_dim, pair_dim, bias=False)
        self.y_k = nn.Linear(seq_dim, pair_dim, bias=False)
        self.dist_proj = nn.Linear(3, pair_dim, bias=False)
        self.out_proj = nn.Linear(pair_dim, pair_dim)

    def forward(self, seq_input: torch.Tensor) -> torch.Tensor:
        # Downsample S -> P where P ~= ceil(S/2)
        x = self.seq_norm(seq_input)
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)
        x = x.transpose(1, 2)

        bsz, p_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)

        pair_qk = q[:, :, None, :] + k[:, None, :, :]

        idx = torch.arange(p_len, device=x.device, dtype=x.dtype)
        dist = idx[:, None] - idx[None, :]
        dist_abs = torch.abs(dist)
        dist_sign = torch.sign(dist)
        dist_scale = dist_abs / max(1.0, float(p_len - 1))
        dist_feats = torch.stack([dist_scale, dist_sign, torch.exp(-dist_scale)], dim=-1)
        dist_feats = dist_feats.unsqueeze(0).expand(bsz, -1, -1, -1)
        pair_dist = self.dist_proj(dist_feats)

        y_q = self.y_q(F.gelu(x))
        y_k = self.y_k(F.gelu(x))
        pair_outer = y_q[:, :, None, :] + y_k[:, None, :, :]

        pair = self.out_proj(pair_qk + pair_dist + pair_outer)
        return pair


class PairUpdateBlock(nn.Module):
    def __init__(self, seq_dim: int = 512, pair_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.seq_to_pair = SequenceToPairBlock(seq_dim=seq_dim, pair_dim=pair_dim)
        self.row_attn = RowAttentionBlock(pair_dim=pair_dim, attn_dim=64, dropout=dropout)
        self.pair_mlp = PairMLPBlock(pair_dim=pair_dim, mlp_ratio=2, dropout=dropout)

    def forward(self, sequence_input: torch.Tensor, pair_input: torch.Tensor | None) -> torch.Tensor:
        y = self.seq_to_pair(sequence_input)
        x = y if pair_input is None else pair_input + y
        x = x + self.row_attn(x)
        x = x + self.pair_mlp(x)
        return x


class ETDBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        channels: tuple[int, ...] = (320, 384, 448, 512, 512, 512, 512),
        n_layers: int = 9,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        pair_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        if len(channels) != 7:
            raise ValueError("channels must provide exactly 7 encoder blocks")

        enc_in = [d_model, *channels[:-1]]
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(in_ch=enc_in[idx], out_ch=channels[idx], downsample=(idx < 4))
                for idx in range(7)
            ]
        )

        self.seq_blocks = nn.ModuleList(
            [
                SequenceTransformerBlock(embed_dim=channels[-1], n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.seq_norm = nn.LayerNorm(channels[-1])

        self.pair_update = PairUpdateBlock(seq_dim=channels[-1], pair_dim=pair_dim, dropout=dropout)
        self.pair_input_proj = nn.Linear(4, pair_dim)
        self.pair_bias_proj = nn.Linear(pair_dim, n_heads, bias=False)
        
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(in_ch=channels[-1], skip_ch=channels[-1], out_ch=channels[-1], upsample=False),
                DecoderBlock(in_ch=channels[-1], skip_ch=channels[-2], out_ch=channels[-1], upsample=False),
                DecoderBlock(in_ch=channels[-1], skip_ch=channels[-3], out_ch=channels[-1], upsample=False),
                DecoderBlock(in_ch=channels[-1], skip_ch=channels[2], out_ch=channels[2], upsample=True),
                DecoderBlock(in_ch=channels[2], skip_ch=channels[1], out_ch=channels[1], upsample=True),
                DecoderBlock(in_ch=channels[1], skip_ch=channels[0], out_ch=channels[0], upsample=True),
                DecoderBlock(in_ch=channels[0], skip_ch=d_model, out_ch=d_model, upsample=True),
            ]
        )

        self.adj_q = nn.Linear(channels[-1], 64)
        self.adj_k = nn.Linear(channels[-1], 64)
        self.pair_to_adj = nn.Linear(pair_dim, 1)

    @staticmethod
    def _repeat_pair_to_seq(pair_bias: torch.Tensor, target_len: int) -> torch.Tensor:
        # pair_bias: [B, P, P, H] -> [B, H, S, S]
        expanded = pair_bias.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        expanded = expanded[:, :target_len, :target_len, :]
        return expanded.permute(0, 3, 1, 2)

    def forward(
        self,
        x: torch.Tensor,
        pair_feats: torch.Tensor,
        down_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        film_params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x0 = x
        enc_out = []
        cur = x
        for block in self.encoder:
            cur = block(cur)
            enc_out.append(cur)

        bottleneck = enc_out[-1]

        gamma_b = film_params["gamma_b"].unsqueeze(1)
        beta_b = film_params["beta_b"].unsqueeze(1)
        seq_x = bottleneck * (1.0 + gamma_b) + beta_b

        seq_len = seq_x.shape[1]
        pair_x = self.pair_input_proj(pair_feats)
        pair_valid = (pair_mask.unsqueeze(1) & pair_mask.unsqueeze(2)).unsqueeze(-1).float()
        pair_x = pair_x * pair_valid

        gamma_p = film_params["gamma_p"].unsqueeze(1).unsqueeze(1)
        beta_p = film_params["beta_p"].unsqueeze(1).unsqueeze(1)

        for idx, block in enumerate(self.seq_blocks):
            if idx % 2 == 0:
                pair_x = self.pair_update(seq_x, pair_x)
                pair_x = pair_x * (1.0 + gamma_p) + beta_p
                pair_x = pair_x * pair_valid

            pair_bias = self.pair_bias_proj(F.gelu(pair_x))
            attention_bias = self._repeat_pair_to_seq(pair_bias, target_len=seq_len)
            seq_x = block(seq_x, attention_bias=attention_bias, key_padding_mask=down_mask)

        seq_x = self.seq_norm(seq_x)

        y = seq_x
        y = self.decoder[0](y, enc_out[6])
        y = self.decoder[1](y, enc_out[5])
        y = self.decoder[2](y, enc_out[4])
        y = self.decoder[3](y, enc_out[2])
        y = self.decoder[4](y, enc_out[1])
        y = self.decoder[5](y, enc_out[0])
        y = self.decoder[6](y, x0)

        gamma_d = film_params["gamma_d"].unsqueeze(1)
        beta_d = film_params["beta_d"].unsqueeze(1)
        y = y * (1.0 + gamma_d) + beta_d

        if y.shape[1] != x0.shape[1]:
            y_cf = y.transpose(1, 2)
            y_cf = F.interpolate(y_cf, size=x0.shape[1], mode="nearest")
            y = y_cf.transpose(1, 2)

        q = self.adj_q(seq_x)
        k = self.adj_k(seq_x)
        seq_adj = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(q.shape[-1])

        pair_adj = self.pair_to_adj(pair_x).squeeze(-1)
        pair_adj = pair_adj.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        pair_adj = pair_adj[:, :seq_len, :seq_len]

        adj_logits = seq_adj + pair_adj
        adj_logits = 0.5 * (adj_logits + adj_logits.transpose(-1, -2))

        return {
            "decoded": y,
            "bottleneck": seq_x,
            "adjacency_logits": adj_logits,
            "pair_state": pair_x,
        }
