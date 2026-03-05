"""A'（修饰 A）输入构造工具。

本模块用于你提出的实验思路：
- 在已知 m6A 位点中，按一定概率把输入 token 从 A 替换为 A'（新 token）。
- 仅改变“输入表示”，不改变监督标签（这些位置仍按原任务正常预测）。

设计目标：
1) 与现有 `collate_batch` 解耦，先作为独立工具模块落地。
2) 可控、可复现（通过外部传入 `random.Random`）。
3) 仅替换合法 A 位点，避免误改其他碱基 token。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .constants import APRIME_TOKEN_ID
from .utils import encode_sequence


@dataclass(frozen=True)
class APrimeConfig:
    """A' 替换配置。

    字段说明：
    - enabled: 是否启用该增强；False 时函数会直接返回原序列副本。
    - replace_prob: 对每个候选 m6A 位点执行 A->A' 的伯努利概率。
    - a_token_id: 原始 A 的 token id（当前工程通常是 0）。
    - aprime_token_id: A' 的 token id（需要你在 constants 里新增）。
    - max_replace_per_sequence:
      每条序列最多替换多少个位置。None 表示不限制，仅受 replace_prob 控制。
    """

    enabled: bool = False
    replace_prob: float = 0.1
    a_token_id: int = 0
    aprime_token_id: int = 7
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


def apply_aprime_on_m6a_positions(
    seq_ids: np.ndarray,
    m6a_positions: np.ndarray,
    rng: random.Random,
    config: APrimeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """在 m6A 位点上按配置进行 A->A' 替换。

    参数：
    - seq_ids: 1D token 序列（长度 L）
    - m6a_positions: 已知 m6A 位点（0-based）
    - rng: 外部随机源（保证可复现）
    - config: A' 替换配置

    返回：
    - seq_out: 替换后的 token 序列（不会原地修改输入）
    - replaced_positions: 实际被替换的位置数组（升序）

    逻辑细节：
    1) 若 disabled / prob<=0 / 无候选位点，直接返回原序列副本。
    2) 候选位点必须满足当前 token 正好是 A（a_token_id）。
       这样可避免错误替换到 C/G/U/N/PAD/MASK 等 token。
    3) 对候选位点按 replace_prob 独立采样。
    4) 若配置了 max_replace_per_sequence，则在被采样位置里随机下采样到上限。
    """

    seq = np.asarray(seq_ids, dtype=np.int64).copy()
    length = int(seq.shape[0])

    if not config.enabled:
        return seq, np.zeros((0,), dtype=np.int64)
    if config.replace_prob <= 0.0:
        return seq, np.zeros((0,), dtype=np.int64)

    candidates = _sanitize_positions(m6a_positions, length=length)
    if candidates.size == 0:
        return seq, np.zeros((0,), dtype=np.int64)

    # 只允许替换当前 token 为 A 的位置，避免语义污染。
    is_a = seq[candidates] == int(config.a_token_id)
    candidates = candidates[is_a]
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
    seq[replaced] = int(config.aprime_token_id)
    return seq, replaced


def build_mod_aprime_view(
    seq_ids: np.ndarray,
    m6a_positions: np.ndarray,
    rng: random.Random,
    enable: bool,
    replace_prob: float,
    a_token_id: int = 0,
    aprime_token_id: int = 7,
    max_replace_per_sequence: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """便捷包装函数：不需要外部手动构造 APrimeConfig。

    适合在 `collate_batch` 里直接调用，例如：
    `seq_mod, replaced = build_mod_aprime_view(...)`
    """

    cfg = APrimeConfig(
        enabled=enable,
        replace_prob=replace_prob,
        a_token_id=a_token_id,
        aprime_token_id=aprime_token_id,
        max_replace_per_sequence=max_replace_per_sequence,
    )
    return apply_aprime_on_m6a_positions(
        seq_ids=seq_ids,
        m6a_positions=m6a_positions,
        rng=rng,
        config=cfg,
    )


def encode_with_optional_aprime(
    sequence: str,
    m6a_positions: np.ndarray,
    aprime_enable: bool,
    aprime_prob: float,
    aprime_max_per_seq: int,
    rng_aprime: random.Random,
    a_token_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """编码序列并按配置可选执行 A' 替换。

    这是从 `data.collate_batch` 抽离出的工具函数，便于把 A' 逻辑集中维护。
    """
    seq_ids = np.asarray(encode_sequence(sequence), dtype=np.int64)
    replaced_positions = np.zeros((0,), dtype=np.int64)
    if aprime_enable:
        max_replace = None if aprime_max_per_seq < 0 else int(aprime_max_per_seq)
        seq_ids, replaced_positions = build_mod_aprime_view(
            seq_ids=seq_ids,
            m6a_positions=m6a_positions,
            rng=rng_aprime,
            enable=True,
            replace_prob=float(aprime_prob),
            a_token_id=int(a_token_id),
            aprime_token_id=int(APRIME_TOKEN_ID),
            max_replace_per_sequence=max_replace,
        )
    return seq_ids, replaced_positions
