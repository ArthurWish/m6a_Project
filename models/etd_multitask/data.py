"""Dataset loading and batch construction for ETD multi-task training."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .constants import BASE_TO_ID, MASK_TOKEN_ID, PAD_TOKEN_ID
from .utils import encode_sequence, parse_list_field


@dataclass
class TranscriptExample:
    """单条转录本训练样本的内存结构。

    字段说明：
    - transcript_id: 转录本 ID（如 ENST...）
    - sequence: RNA 序列（通常已标准化为 A/C/G/U）
    - seq_len: 序列长度
    - m6a_positions: 已知 m6A 位点位置（1D ndarray，0-based）
    - unlabeled_a_positions: 未标注的 A 位点（PU 学习中的 unlabeled 候选）
    - role_labels: 各 role 的 PU 标签数组（1=positive, -1=unlabeled）
    - role_support: 各 role 的支持证据计数（与 role_labels 按位对齐）
    """
    transcript_id: str
    sequence: str
    seq_len: int
    m6a_positions: np.ndarray
    unlabeled_a_positions: np.ndarray
    role_labels: dict[str, np.ndarray]
    role_support: dict[str, np.ndarray]


class BPPCache:
    """读取并缓存 RNAfold 生成的稀疏配对概率。

    缓存层次：
    - _raw_cache: 原始 npz 内容（ij/p_ref/p_modA/L）
    - _position_cache: 逐碱基统计（pair_prob, pair_count）
    - _downsample_cache: 下采样后的结构目标矩阵（L' x L'）
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self._raw_cache: dict[str, dict[str, Any]] = {}
        self._position_cache: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray]] = {}
        self._downsample_cache: dict[tuple[str, bool, int], np.ndarray] = {}

    def _load_raw(self, transcript_id: str) -> dict[str, Any] | None:
        """加载单条 transcript 的 npz 原始数据并做进程内缓存。"""
        if transcript_id in self._raw_cache:
            return self._raw_cache[transcript_id]

        path = self.cache_dir / f"{transcript_id}.npz"
        if not path.exists():
            self._raw_cache[transcript_id] = None
            return None

        with np.load(path, allow_pickle=False) as payload:
            data = {
                "ij": payload["ij"].astype(np.int32),
                "p_ref": payload["p_ref"].astype(np.float32),
                "p_modA": payload["p_modA"].astype(np.float32),
                "L": int(payload["L"]),
            }

        self._raw_cache[transcript_id] = data
        return data

    def get_positional_stats(self, transcript_id: str, use_mod_a: bool, seq_len_fallback: int) -> tuple[np.ndarray, np.ndarray]:
        """返回逐位点配对统计特征。

        输出：
        - pair_prob[i]: 位点 i 的配对概率和（由 ij 边累加）
        - pair_count[i]: 位点 i 参与配对的边数

        说明：
        - use_mod_a=True 时使用 p_modA，否则使用 p_ref。
        - 若缓存缺失，返回按 seq_len_fallback 构造的全零特征。
        """
        key = (transcript_id, bool(use_mod_a))
        if key in self._position_cache:
            return self._position_cache[key]

        data = self._load_raw(transcript_id)
        if data is None:
            length = int(seq_len_fallback)
            pair_prob = np.zeros(length, dtype=np.float32)
            pair_count = np.zeros(length, dtype=np.float32)
            self._position_cache[key] = (pair_prob, pair_count)
            return pair_prob, pair_count

        ij = data["ij"]
        p = data["p_modA"] if use_mod_a else data["p_ref"]
        length = data["L"]

        pair_prob = np.zeros(length, dtype=np.float32)
        pair_count = np.zeros(length, dtype=np.float32)

        if ij.size > 0:
            # 对每条边 (i,j,p) 同时给 i、j 两个端点累加统计。
            i = ij[:, 0]
            j = ij[:, 1]
            np.add.at(pair_prob, i, p)
            np.add.at(pair_prob, j, p)
            np.add.at(pair_count, i, 1.0)
            np.add.at(pair_count, j, 1.0)

        pair_prob = np.clip(pair_prob, 0.0, 1.0)
        self._position_cache[key] = (pair_prob, pair_count)
        return pair_prob, pair_count

    def get_downsampled_target(self, transcript_id: str, use_mod_a: bool, seq_len_fallback: int, factor: int = 16) -> np.ndarray:
        """构建结构任务用的下采样目标矩阵。

        逻辑：
        - 将原始配对边 (i,j) 映射到块索引 (i//factor, j//factor)
        - 同一块内取平均概率
        - 最后做对称化并清空对角线
        """
        key = (transcript_id, bool(use_mod_a), int(factor))
        if key in self._downsample_cache:
            return self._downsample_cache[key]

        data = self._load_raw(transcript_id)
        if data is None:
            l_prime = int(math.ceil(seq_len_fallback / factor))
            mat = np.zeros((l_prime, l_prime), dtype=np.float32)
            self._downsample_cache[key] = mat
            return mat

        ij = data["ij"]
        p = data["p_modA"] if use_mod_a else data["p_ref"]
        length = data["L"]
        l_prime = int(math.ceil(length / factor))

        mat = np.zeros((l_prime, l_prime), dtype=np.float32)
        counts = np.zeros((l_prime, l_prime), dtype=np.float32)

        if ij.size > 0:
            bi = ij[:, 0] // factor
            bj = ij[:, 1] // factor
            np.add.at(mat, (bi, bj), p)
            np.add.at(counts, (bi, bj), 1.0)

        nz = counts > 0
        mat[nz] = mat[nz] / counts[nz]
        mat = mat + mat.T
        np.fill_diagonal(mat, 0.0)

        self._downsample_cache[key] = mat
        return mat


def load_split_ids(splits_path: str | Path, split_names: list[str]) -> set[str]:
    """从 splits.json 读取指定 split 的 transcript_id 集合。"""
    with Path(splits_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    ids: set[str] = set()
    for split_name in split_names:
        entries = payload.get(split_name, [])
        ids.update(str(x) for x in entries)
    return ids


def _build_site_lookup(sites_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """把位点表转成 transcript_id -> 位点子表 的索引结构。

    细节：
    - 先计算 support_sum 作为重复位点的优先级
    - 同一 transcript_id + site_pos 若重复，保留 support_sum 更高的一条
    - 最终每个 transcript 子表按 site_pos 升序
    """
    sites_df = sites_df.copy()
    sites_df["support_sum"] = (
        sites_df["writer_support_count"].astype(float)
        + sites_df["reader_support_count"].astype(float)
        + sites_df["eraser_support_count"].astype(float)
    )
    sites_df = sites_df.sort_values(
        ["transcript_id", "site_pos", "support_sum"],
        ascending=[True, True, False],
    )
    sites_df = sites_df.drop_duplicates(subset=["transcript_id", "site_pos"], keep="first")

    out: dict[str, pd.DataFrame] = {}
    for transcript_id, group in sites_df.groupby("transcript_id"):
        out[str(transcript_id)] = group.sort_values("site_pos")
    return out


def _sample_examples(examples: list[TranscriptExample], smoke_ratio: float, seed: int) -> list[TranscriptExample]:
    """按 smoke_ratio 对样本做可复现子采样（用于 smoke 训练）。"""
    if smoke_ratio <= 0.0 or smoke_ratio >= 1.0:
        return examples
    rng = random.Random(seed)
    keep = max(1, int(len(examples) * smoke_ratio))
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    selected = sorted(indices[:keep])
    return [examples[idx] for idx in selected]


def load_examples(
    sites_path: str | Path,
    transcripts_path: str | Path,
    splits_path: str | Path,
    split_names: list[str],
    max_len: int = 12000,
    smoke_ratio: float = 1.0,
    seed: int = 42,
) -> list[TranscriptExample]:
    """从 parquet + splits 构建训练/验证样本列表。

    处理步骤：
    1) 读取位点表与转录本表
    2) 依据 split_names 过滤 transcript
    3) 构建 m6A 位点、unlabeled A 位点、role 标签与支持计数
    4) 组装为 TranscriptExample
    5) 可选 smoke 子采样
    """
    sites_df = pd.read_parquet(sites_path)
    transcripts_df = pd.read_parquet(transcripts_path)
    split_ids = load_split_ids(splits_path, split_names)

    site_lookup = _build_site_lookup(sites_df)

    examples: list[TranscriptExample] = []
    for row in transcripts_df.itertuples(index=False):
        transcript_id = str(row.transcript_id)
        # 只保留目标 split 内的 transcript。
        if transcript_id not in split_ids:
            continue

        seq = str(row.full_sequence).upper().replace("T", "U")
        seq_len = int(row.seq_len)
        # 安全过滤：再按 max_len 截一道，避免异常长序列进入训练。
        if seq_len > max_len:
            continue

        site_df = site_lookup.get(transcript_id)
        if site_df is None or site_df.empty:
            continue

        m6a_positions = site_df["site_pos"].astype(int).to_numpy(dtype=np.int64)
        m6a_positions = np.unique(m6a_positions)

        # unlabeled A = 全序列 A 位点 - 已知 m6A 位点。
        m6a_set = set(int(x) for x in m6a_positions.tolist())
        unlabeled_a = [idx for idx, base in enumerate(seq) if base == "A" and idx not in m6a_set]

        role_labels = {
            "writer": site_df["writer_pu_label"].astype(int).to_numpy(dtype=np.int64),
            "reader": site_df["reader_pu_label"].astype(int).to_numpy(dtype=np.int64),
            "eraser": site_df["eraser_pu_label"].astype(int).to_numpy(dtype=np.int64),
        }
        role_support = {
            "writer": site_df["writer_support_count"].astype(int).to_numpy(dtype=np.int64),
            "reader": site_df["reader_support_count"].astype(int).to_numpy(dtype=np.int64),
            "eraser": site_df["eraser_support_count"].astype(int).to_numpy(dtype=np.int64),
        }

        example = TranscriptExample(
            transcript_id=transcript_id,
            sequence=seq,
            seq_len=seq_len,
            m6a_positions=m6a_positions,
            unlabeled_a_positions=np.asarray(unlabeled_a, dtype=np.int64),
            role_labels=role_labels,
            role_support=role_support,
        )
        examples.append(example)

    # smoke_ratio<1 时做随机子集抽样，加速联调。
    examples = _sample_examples(examples, smoke_ratio=smoke_ratio, seed=seed)
    return examples


def split_bucket(length: int, boundaries: list[int]) -> int:
    """根据长度边界返回分桶编号。"""
    for idx, boundary in enumerate(boundaries):
        if length <= boundary:
            return idx
    return len(boundaries)


def build_length_bucketed_batches(
    examples: list[TranscriptExample],
    batch_token_budget: int,
    boundaries: list[int],
    shuffle: bool,
    seed: int,
) -> list[list[TranscriptExample]]:
    """按长度分桶并按 token 预算组 batch。

    组 batch 规则：
    - 先按长度桶分组，桶内按 seq_len 降序
    - 贪心累加当前 batch 的 token 数
    - 超过 batch_token_budget 就切 batch

    这不是固定 batch_size，而是“动态 batch size”策略。
    """
    by_bucket: dict[int, list[TranscriptExample]] = {}
    for item in examples:
        bucket = split_bucket(item.seq_len, boundaries)
        by_bucket.setdefault(bucket, []).append(item)

    rng = random.Random(seed)
    batches: list[list[TranscriptExample]] = []

    for bucket in sorted(by_bucket):
        bucket_items = by_bucket[bucket]
        if shuffle:
            rng.shuffle(bucket_items)
        bucket_items = sorted(bucket_items, key=lambda x: x.seq_len, reverse=True)

        current: list[TranscriptExample] = []
        current_tokens = 0
        for item in bucket_items:
            if current and current_tokens + item.seq_len > batch_token_budget:
                batches.append(current)
                current = []
                current_tokens = 0

            current.append(item)
            current_tokens += item.seq_len

        if current:
            batches.append(current)

    if shuffle:
        rng.shuffle(batches)
    return batches


def _build_struct_features(
    seq_ids: np.ndarray,
    m6a_positions: np.ndarray,
    pair_prob: np.ndarray,
    pair_count: np.ndarray,
) -> np.ndarray:
    """构造每个位点 8 维结构/序列特征。

    维度定义：
    0: 配对概率和（clip 后）
    1: 配对计数（按长度归一化后 clip）
    2: 是否为已知 m6A 位点
    3-6: A/C/G/U one-hot
    7: 相对位置（0~1）
    """
    length = int(seq_ids.shape[0])
    out = np.zeros((length, 8), dtype=np.float32)

    out[:, 0] = np.clip(pair_prob, 0.0, 1.0)
    out[:, 1] = np.clip(pair_count / max(1.0, length / 2.0), 0.0, 1.0)

    if m6a_positions.size > 0:
        out[m6a_positions, 2] = 1.0

    out[:, 3] = (seq_ids == BASE_TO_ID["A"]).astype(np.float32)
    out[:, 4] = (seq_ids == BASE_TO_ID["C"]).astype(np.float32)
    out[:, 5] = (seq_ids == BASE_TO_ID["G"]).astype(np.float32)
    out[:, 6] = (seq_ids == BASE_TO_ID["U"]).astype(np.float32)

    if length > 1:
        out[:, 7] = np.arange(length, dtype=np.float32) / float(length - 1)
    return out


def collate_batch(
    examples: list[TranscriptExample],
    task_name: str,
    role_name: str,
    cond_base: str,
    bpp_cache: BPPCache,
    strong_binding_threshold: float,
    rng: random.Random,
    mod_unlabeled_ratio: float = 1.0,
    mask_prob: float = 0.15,
) -> dict[str, torch.Tensor | list[str]]:
    """把一组 TranscriptExample 打包成模型输入张量。

    该函数同时服务多任务（mod/bind/struct/mask），因此会统一返回：
    - tokens / attn_mask / struct_feats
    - mod 任务标签：mod_pu_labels/mod_pu_mask
    - bind 任务标签：site_positions/site_pu_labels/site_mask/site_support/strong_binding_mask
    - mask 任务标签：mlm_input/mlm_target
    - struct 任务标签：struct_target/struct_lengths
    """
    # 动态 padding 形状：按本 batch 最大序列长度与最大位点数分配。
    batch_size = len(examples)
    max_len = max(item.seq_len for item in examples)
    max_sites = max(max(1, item.m6a_positions.shape[0]) for item in examples)

    tokens = np.full((batch_size, max_len), PAD_TOKEN_ID, dtype=np.int64)
    attn_mask = np.zeros((batch_size, max_len), dtype=bool)
    struct_feats = np.zeros((batch_size, max_len, 8), dtype=np.float32)

    mod_pu_labels = np.zeros((batch_size, max_len), dtype=np.int64)
    mod_pu_mask = np.zeros((batch_size, max_len), dtype=bool)

    site_positions = np.full((batch_size, max_sites), -1, dtype=np.int64)
    site_pu_labels = np.zeros((batch_size, max_sites), dtype=np.int64)
    site_mask = np.zeros((batch_size, max_sites), dtype=bool)
    site_support = np.zeros((batch_size, max_sites), dtype=np.float32)
    strong_mask = np.zeros((batch_size, max_sites), dtype=bool)

    mlm_input = np.full((batch_size, max_len), PAD_TOKEN_ID, dtype=np.int64)
    mlm_target = np.full((batch_size, max_len), -100, dtype=np.int64)

    struct_mats: list[np.ndarray] = []
    struct_lengths = np.zeros((batch_size,), dtype=np.int64)

    transcript_ids: list[str] = []

    # cond_base=A 时使用 p_modA，否则使用 p_ref。
    use_mod_a = cond_base == "A"
    for row_idx, item in enumerate(examples):
        transcript_ids.append(item.transcript_id)

        seq_ids = np.asarray(encode_sequence(item.sequence), dtype=np.int64)
        length = seq_ids.shape[0]

        tokens[row_idx, :length] = seq_ids
        attn_mask[row_idx, :length] = True

        # 结构特征由 RNAfold 统计 + 序列特征拼接而成。
        pair_prob, pair_count = bpp_cache.get_positional_stats(
            transcript_id=item.transcript_id,
            use_mod_a=use_mod_a,
            seq_len_fallback=length,
        )
        struct_feat = _build_struct_features(seq_ids, item.m6a_positions, pair_prob[:length], pair_count[:length])
        struct_feats[row_idx, :length] = struct_feat

        # mod 任务：正样本用 m6A 位点；未标注样本从 unlabeled A 中按比例抽样。
        if task_name == "mod":
            positives = item.m6a_positions
            if positives.size > 0:
                mod_pu_labels[row_idx, positives] = 1
                mod_pu_mask[row_idx, positives] = True

            unlabeled = item.unlabeled_a_positions
            if positives.size > 0:
                target_unlabeled = max(1, int(round(float(positives.size) * mod_unlabeled_ratio)))
            else:
                target_unlabeled = min(64, int(unlabeled.size))

            if unlabeled.size > 0 and target_unlabeled > 0:
                if unlabeled.size <= target_unlabeled:
                    sampled_u = unlabeled
                else:
                    sampled_u = np.asarray(rng.sample(unlabeled.tolist(), target_unlabeled), dtype=np.int64)
                mod_pu_labels[row_idx, sampled_u] = -1
                mod_pu_mask[row_idx, sampled_u] = True

        # bind 任务公共位点信息（也可供其他任务作为占位输入）。
        current_sites = item.m6a_positions
        n_sites = int(current_sites.shape[0])
        if n_sites > 0:
            site_positions[row_idx, :n_sites] = current_sites
            site_mask[row_idx, :n_sites] = True

            labels = item.role_labels.get(role_name)
            support = item.role_support.get(role_name)
            if labels is None:
                labels = np.full((n_sites,), -1, dtype=np.int64)
            if support is None:
                support = np.zeros((n_sites,), dtype=np.int64)

            site_pu_labels[row_idx, :n_sites] = labels[:n_sites]
            site_support[row_idx, :n_sites] = support[:n_sites].astype(np.float32)

            # 强结合定义：正例且 support>=阈值。
            positives = labels[:n_sites] == 1
            strong_here = positives & (support[:n_sites] >= strong_binding_threshold)
            strong_mask[row_idx, :n_sites] = strong_here

        # mask 任务：构造 MLM 输入和目标（被选中位置替换为 [MASK]）。
        if task_name == "mask":
            mlm_input[row_idx, :length] = seq_ids
            valid = seq_ids < 4
            if valid.any():
                random_draw = np.random.rand(length)
                chosen = (random_draw < mask_prob) & valid
                if chosen.any():
                    chosen_idx = np.where(chosen)[0]
                    mlm_target[row_idx, chosen_idx] = seq_ids[chosen_idx]
                    mlm_input[row_idx, chosen_idx] = MASK_TOKEN_ID
        else:
            mlm_input[row_idx, :length] = seq_ids

        # 结构目标按 16 倍下采样后的长度记录。
        l_prime = int(math.ceil(length / 16))
        struct_lengths[row_idx] = l_prime
        if task_name == "struct":
            mat = bpp_cache.get_downsampled_target(
                transcript_id=item.transcript_id,
                use_mod_a=use_mod_a,
                seq_len_fallback=length,
                factor=16,
            )
        else:
            mat = np.zeros((l_prime, l_prime), dtype=np.float32)
        struct_mats.append(mat)

    # 将每条样本的结构目标矩阵 pad 到 batch 内统一大小。
    max_l_prime = max(int(mat.shape[0]) for mat in struct_mats)
    struct_target = np.zeros((batch_size, max_l_prime, max_l_prime), dtype=np.float32)
    for idx, mat in enumerate(struct_mats):
        l_prime = mat.shape[0]
        struct_target[idx, :l_prime, :l_prime] = mat

    return {
        "tokens": torch.tensor(tokens, dtype=torch.long),
        "attn_mask": torch.tensor(attn_mask, dtype=torch.bool),
        "struct_feats": torch.tensor(struct_feats, dtype=torch.float32),
        "mod_pu_labels": torch.tensor(mod_pu_labels, dtype=torch.long),
        "mod_pu_mask": torch.tensor(mod_pu_mask, dtype=torch.bool),
        "site_positions": torch.tensor(site_positions, dtype=torch.long),
        "site_pu_labels": torch.tensor(site_pu_labels, dtype=torch.long),
        "site_mask": torch.tensor(site_mask, dtype=torch.bool),
        "site_support": torch.tensor(site_support, dtype=torch.float32),
        "strong_binding_mask": torch.tensor(strong_mask, dtype=torch.bool),
        "mlm_input": torch.tensor(mlm_input, dtype=torch.long),
        "mlm_target": torch.tensor(mlm_target, dtype=torch.long),
        "struct_target": torch.tensor(struct_target, dtype=torch.float32),
        "struct_lengths": torch.tensor(struct_lengths, dtype=torch.long),
        "transcript_ids": transcript_ids,
    }


def estimate_mod_prior(examples: list[TranscriptExample]) -> float:
    """估计 mod 任务 PU 先验：pos / (pos + unlabeled)。"""
    total_pos = sum(int(item.m6a_positions.size) for item in examples)
    total_unlabeled = sum(int(item.unlabeled_a_positions.size) for item in examples)
    denom = total_pos + total_unlabeled
    if denom <= 0:
        return 0.5
    return float(total_pos / denom)


def estimate_binding_priors(examples: list[TranscriptExample]) -> dict[str, float]:
    """估计各 role 绑定任务先验（正例比例）。"""
    out: dict[str, float] = {}
    for role in ("writer", "reader", "eraser"):
        positives = 0
        total = 0
        for item in examples:
            labels = item.role_labels.get(role)
            if labels is None:
                continue
            positives += int((labels == 1).sum())
            total += int(labels.shape[0])
        if total == 0:
            out[role] = 0.5
        else:
            out[role] = float(positives / total)
    return out


def estimate_strong_binding_thresholds(examples: list[TranscriptExample], q: float = 0.75) -> dict[str, float]:
    """估计各 role 的强结合阈值（正例 support 的 q 分位数）。"""
    out: dict[str, float] = {}
    for role in ("writer", "reader", "eraser"):
        supports: list[int] = []
        for item in examples:
            labels = item.role_labels.get(role)
            support = item.role_support.get(role)
            if labels is None or support is None:
                continue
            positive_support = support[labels == 1]
            if positive_support.size > 0:
                supports.extend(int(x) for x in positive_support.tolist())
        if supports:
            out[role] = float(np.quantile(np.asarray(supports, dtype=np.float32), q=q))
        else:
            out[role] = 1.0
    return out


def tensor_to_numpy_flat(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> np.ndarray:
    """张量转 numpy 并展平；可选按 mask 过滤有效位置。"""
    data = tensor.detach().cpu().numpy()
    if mask is None:
        return data.reshape(-1)
    mask_np = mask.detach().cpu().numpy().astype(bool)
    return data[mask_np]


def read_parquet_list_column(df: pd.DataFrame, column: str) -> pd.Series:
    """把 parquet 中的列表列解析为 Python list/ndarray 形式。"""
    return df[column].apply(parse_list_field)
