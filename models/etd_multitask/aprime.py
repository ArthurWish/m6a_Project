

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .constants import BASE_TO_ID, MOD_BASE_MAP, MOD_TOKEN_IDS
from .utils import encode_sequence


@dataclass(frozen=True)
class APrimeConfig:
    """修饰 token 替换配置。

    字段说明：
    - enabled: 是否启用该增强；False 时函数会直接返回原序列副本。
    - replace_prob: 对每个候选位点执行替换的伯努利概率。
    - orig_token_id: 原始碱基的 token id（如 A=0, C=1, U=3）。
    - mod_token_id: 修饰揭示 token id（如 MOD_TOKEN_m6A=7）。
    - max_replace_per_sequence:
      每条序列最多替换多少个位置。None 表示不限制，仅受 replace_prob 控制。
    """

    enabled: bool = False
    replace_prob: float = 0.1
    orig_token_id: int = 0
    mod_token_id: int = 7
    max_replace_per_sequence: int | None = None


def _sanitize_positions(positions: np.ndarray, length: int) -> np.ndarray:
    """清洗位点数组，确保唯一、升序、且位于 [0, length)。

    输入通常是 m6a_positions，可能包含：
    - 重复值
    - 越界值
    - 非整型 dtype
    """

    if positions.size == 0:
        return np.zeros((0,), dtype=np.int64)

    pos = np.asarray(positions, dtype=np.int64)
    pos = pos[(pos >= 0) & (pos < length)]
    if pos.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(pos)


def apply_mod_token_replacement(
    seq_ids: np.ndarray,
    mod_positions: np.ndarray,
    rng: random.Random,
    config: APrimeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """在修饰位点上按配置进行替换。

    参数：
    - seq_ids: 1D token 序列（长度 L）
    - mod_positions: 已知修饰位点（0-based）
    - rng: 外部随机源（保证可复现）
    - config: 替换配置

    返回：
    - seq_out: 替换后的 token 序列（不会原地修改输入）
    - replaced_positions: 实际被替换的位置数组（升序）
    """

    seq = np.asarray(seq_ids, dtype=np.int64).copy()
    length = int(seq.shape[0])

    if not config.enabled:
        return seq, np.zeros((0,), dtype=np.int64)
    if config.replace_prob <= 0.0:
        return seq, np.zeros((0,), dtype=np.int64)

    candidates = _sanitize_positions(mod_positions, length=length)
    if candidates.size == 0:
        return seq, np.zeros((0,), dtype=np.int64)


    is_orig = seq[candidates] == int(config.orig_token_id)
    candidates = candidates[is_orig]
    if candidates.size == 0:
        return seq, np.zeros((0,), dtype=np.int64)

    chosen: list[int] = []
    p = float(config.replace_prob)
    for pos in candidates.tolist():
        if rng.random() < p:
            chosen.append(int(pos))

    if not chosen:
        return seq, np.zeros((0,), dtype=np.int64)

    # 可选上限：防止单条样本被替换过多，导致输入分布偏移过大。
    max_k = config.max_replace_per_sequence
    if max_k is not None and max_k >= 0 and len(chosen) > max_k:
        chosen = rng.sample(chosen, k=max_k)

    replaced = np.asarray(sorted(chosen), dtype=np.int64)
    seq[replaced] = int(config.orig_token_id)
    return seq, replaced


def build_mod_aprime_view(
    seq_ids: np.ndarray,
    mod_positions: np.ndarray,
    rng: random.Random,
    enable: bool,
    replace_prob: float,
    orig_token_id: int = 0,
    mod_token_id: int = 7,
    max_replace_per_sequence: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """便捷包装函数：不需要外部手动构造 APrimeConfig。

    适合在 `collate_batch` 里直接调用，例如：
    `seq_mod, replaced = build_mod_aprime_view(...)`
    """

    cfg = APrimeConfig(
        enabled=enable,
        replace_prob=replace_prob,
        orig_token_id=int(orig_token_id),
        mod_token_id=int(mod_token_id),
        max_replace_per_sequence=max_replace_per_sequence,
    )

    return apply_mod_token_replacement(
        seq_ids=seq_ids,
        mod_positions=mod_positions,
        rng=rng,
        config=cfg,
    )


def encode_with_optional_aprime(
    sequence: str,
    mod_positions: np.ndarray,
    mod_type: str,
    aprime_enable: bool,
    aprime_prob: float,
    aprime_max_per_seq: int,
    rng_aprime: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """

    当前策略（单点替换）：
    - aprime_enable=False: 不替换。
    - aprime_enable=True:
      aprime_prob 作为"该样本是否执行修饰 token 增强"的门控概率；
      一旦命中门控，就只在候选位点里随机替换 1 个。
    - 因此，replaced_positions 的长度只会是 0 或 1。
    """
    seq_ids = np.asarray(encode_sequence(sequence), dtype=np.int64)
    replaced_positions = np.zeros((0,), dtype=np.int64)
    if aprime_enable and mod_type in MOD_TOKEN_IDS:

        target_base = MOD_BASE_MAP[mod_type]
        orig_token_id = BASE_TO_ID[target_base]
        mod_token_id = MOD_TOKEN_IDS[mod_type]


        if rng_aprime.random() < float(aprime_prob):
            seq_ids, replaced_positions = build_mod_aprime_view(
                seq_ids=seq_ids,
                mod_positions=mod_positions,
                rng=rng_aprime,
                enable=True,
                replace_prob=1.0,
                orig_token_id=orig_token_id,
                mod_token_id=mod_token_id,
                max_replace_per_sequence=1,
            )
    return seq_ids, replaced_positions
