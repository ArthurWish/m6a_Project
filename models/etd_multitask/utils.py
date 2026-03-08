"""Utility functions for ETD multi-task pipeline."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .constants import BASE_TO_ID


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_sequence(seq: str) -> list[int]:
    seq = seq.upper().replace("T", "U")
    ids = []
    for base in seq:
        ids.append(BASE_TO_ID.get(base, BASE_TO_ID["N"]))
    return ids


def parse_list_field(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, tuple):
        return [int(x) for x in value]
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.tolist()]
    if isinstance(value, (int, float)):
        return [int(value)]

    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        return []

    chunks = [chunk.strip() for chunk in raw.replace(";", ",").split(",")]
    out = []
    for chunk in chunks:
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError:
            continue
    return out


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def downsample_mask(mask: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return mask
    bsz, length = mask.shape
    pad = (factor - (length % factor)) % factor
    if pad:
        mask = F.pad(mask, (0, pad), value=False)
    new_len = mask.shape[1] // factor
    return mask.view(bsz, new_len, factor).any(dim=-1)


def downsample_1d(features: torch.Tensor, factor: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    if factor <= 1:
        return features

    bsz, length, channels = features.shape
    pad = (factor - (length % factor)) % factor
    if pad:
        features = F.pad(features, (0, 0, 0, pad), value=0.0)
        if mask is not None:
            mask = F.pad(mask, (0, pad), value=False)

    new_len = features.shape[1] // factor
    reshaped = features.view(bsz, new_len, factor, channels)

    if mask is None:
        return reshaped.mean(dim=2)

    mask_float = mask.view(bsz, new_len, factor).float().unsqueeze(-1)
    denom = mask_float.sum(dim=2).clamp_min(1.0)
    return (reshaped * mask_float).sum(dim=2) / denom


def make_pair_features(struct_down: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Build pairwise features using distance only.

    Returns [B, L, L, 4].
    约定：当前仅使用距离先验，前 3 个通道置 0，第 4 通道为归一化 |i-j|。
    """
    bsz, length, _ = struct_down.shape
    device = struct_down.device

    idx = torch.arange(length, device=device, dtype=torch.float32)
    dist = (idx[None, :, None] - idx[None, None, :]).abs()
    dist = dist / max(1.0, float(length - 1))
    dist = dist.expand(bsz, -1, -1)

    valid2d = (valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)).float()

    zeros = torch.zeros_like(dist)
    pair_feats = torch.stack(
        [
            zeros,
            zeros,
            zeros,
            dist,
        ],
        dim=-1,
    )
    pair_feats = pair_feats * valid2d.unsqueeze(-1)
    return pair_feats


def batched(iterable: Iterable, n: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def ceil_div(a: int, b: int) -> int:
    return int(math.ceil(a / b))
