"""Dataset loading and batch construction for ETD multi-task training."""

from __future__ import annotations
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from .constants import BASE_TO_ID, MASK_TOKEN_ID, MOD_BASE_MAP, PAD_TOKEN_ID, ROLE_NAMES, VALID_BASES, MOD_TYPE_NAMES, TASK_PROBS
from .aprime import encode_with_optional_aprime
from .utils import parse_list_field


@dataclass
class TranscriptExample:
    """单条转录本训练样本的内存结构（多修饰泛化版）。

    字段说明：
    - transcript_id: 转录本 ID（如 ENST...）
    - sequence: RNA 序列（已标准化为 A/C/G/U）
    - seq_len: 序列长度

    - mod_positions: {mod_type -> positions ndarray}
      每种修饰类型的已知位点（0-based），如 {"m6A": [100,200], "m5C": [400]}

    - unlabeled_positions: {mod_type -> positions ndarray}
      每种修饰对应碱基中未标注的候选位点（PU unlabeled 集）
      如 {"m6A": [非m6A的A位点], "m5C": [非m5C的C位点], ...}

    - role_labels: {mod_type -> {role -> labels ndarray}}
      按修饰类型和 role 分组的 PU 标签，与 mod_positions[mod_type] 对齐

    - role_support: {mod_type -> {role -> support ndarray}}
      同上结构，证据支持计数

    - all_mod_positions: 所有修饰位点的并集
    """
    transcript_id: str
    sequence: str
    seq_len: int
    mod_positions: dict[str, np.ndarray] = field(default_factory=dict)
    unlabeled_positions: dict[str, np.ndarray] = field(default_factory=dict)
    role_labels: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    role_support: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    all_mod_positions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

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
    - 同一 transcript_id + site_pos + mod_type 若重复，保留 support_sum 更高的一条
    - 最终每个 transcript 子表按 site_pos 升序
    """
    sites_df = sites_df.copy()
    sites_df["support_sum"] = (
        sites_df["writer_support_count"].astype(float)
        + sites_df["reader_support_count"].astype(float)
        + sites_df["eraser_support_count"].astype(float)
    )
    sites_df = sites_df.sort_values(
        ["transcript_id", "site_pos", "mod_type", "support_sum"],
        ascending=[True, True, True, False],
    )
    sites_df = sites_df.drop_duplicates(subset=["transcript_id", "site_pos", "mod_type"], keep="first")

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
    3) 按 mod_type 分组，构建修饰位点、unlabeled 候选、role 标签与支持计数
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
        
        mod_positions: dict[str, np.ndarray] = {}
        role_labels: dict[str, dict[str, np.ndarray]] = {}
        role_support: dict[str, dict[str, np.ndarray]] = {}

        for mod_type, mod_group in site_df.groupby("mod_type"):
            mod_type = str(mod_type)
            if mod_type not in MOD_BASE_MAP:
                continue
            positions = np.unique(mod_group["site_pos"].astype(int).to_numpy(dtype=np.int64))
            mod_positions[mod_type] = positions

            role_labels[mod_type] = {
                "writer": mod_group["writer_pu_label"].astype(int).to_numpy(dtype=np.int64),
                "reader": mod_group["reader_pu_label"].astype(int).to_numpy(dtype=np.int64),
                "eraser": mod_group["eraser_pu_label"].astype(int).to_numpy(dtype=np.int64),
            }
            role_support[mod_type] = {
                "writer": mod_group["writer_support_count"].astype(int).to_numpy(dtype=np.int64),
                "reader": mod_group["reader_support_count"].astype(int).to_numpy(dtype=np.int64),
                "eraser": mod_group["eraser_support_count"].astype(int).to_numpy(dtype=np.int64),
            }
        if not mod_positions:
            continue
        
        base_all_positions: dict[str, list[int]] = {}
        for idx, base in enumerate(seq):
            if base in ("A", "C", "G", "U"):
                base_all_positions.setdefault(base, []).append(idx)
        unlabeled_positions: dict[str, np.ndarray] = {}
        for mt, positions in mod_positions.items():
            target_base = MOD_BASE_MAP[mt]
            all_pos_of_base = base_all_positions.get(target_base, [])
            own_positive_set = set(int(x) for x in positions.tolist())
            unl = [p for p in all_pos_of_base if p not in own_positive_set]
            unlabeled_positions[mt] = np.asarray(unl, dtype=np.int64)
        
        all_pos_list = []
        for positions in mod_positions.values():
            all_pos_list.append(positions)
        all_mod_positions = np.unique(np.concatenate(all_pos_list)) if all_pos_list else np.zeros(0, dtype=np.int64)

        example = TranscriptExample(
            transcript_id=transcript_id,
            sequence=seq,
            seq_len=seq_len,
            mod_positions=mod_positions,
            unlabeled_positions=unlabeled_positions,
            role_labels=role_labels,
            role_support=role_support,
            all_mod_positions=all_mod_positions,
        )
        examples.append(example)

    #     m6a_positions = site_df["site_pos"].astype(int).to_numpy(dtype=np.int64)
    #     m6a_positions = np.unique(m6a_positions)

    #     # unlabeled A = 全序列 A 位点 - 已知 m6A 位点。
    #     m6a_set = set(int(x) for x in m6a_positions.tolist())
    #     unlabeled_a = [idx for idx, base in enumerate(seq) if base == "A" and idx not in m6a_set]

    #     role_labels = {
    #         "writer": site_df["writer_pu_label"].astype(int).to_numpy(dtype=np.int64),
    #         "reader": site_df["reader_pu_label"].astype(int).to_numpy(dtype=np.int64),
    #         "eraser": site_df["eraser_pu_label"].astype(int).to_numpy(dtype=np.int64),
    #     }
    #     role_support = {
    #         "writer": site_df["writer_support_count"].astype(int).to_numpy(dtype=np.int64),
    #         "reader": site_df["reader_support_count"].astype(int).to_numpy(dtype=np.int64),
    #         "eraser": site_df["eraser_support_count"].astype(int).to_numpy(dtype=np.int64),
    #     }

    #     example = TranscriptExample(
    #         transcript_id=transcript_id,
    #         sequence=seq,
    #         seq_len=seq_len,
    #         m6a_positions=m6a_positions,
    #         unlabeled_a_positions=np.asarray(unlabeled_a, dtype=np.int64),
    #         role_labels=role_labels,
    #         role_support=role_support,
    #     )
    #     examples.append(example)

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



def sample_task_name(rng: random.Random, task_probs: dict[str, float] | None = None) -> str:
    """按给定概率采样任务名。"""
    if task_probs is None:
        task_probs = TASK_PROBS

    tasks = list(task_probs.keys())
    probs = list(task_probs.values())

    s = sum(probs)
    if s <= 0:
        raise ValueError("task_probs 的概率和必须大于 0")

    probs = [p / s for p in probs]
    return rng.choices(tasks, weights=probs, k=1)[0]

def sample_task_condition(task_name: str, rng: random.Random) -> tuple[str, str]:
    """
    按任务采样一组条件 token（role / base / mod_type）
    采样规则：
    - bind: mod_type 从 4 种均匀采样 → base 由 mod_type 决定；role 均匀采样
    - mod:  mod_type 从 4 种均匀采样 → base 由 mod_type 决定；role="none"
    - struct: mod_type="none"；role="none"；base 从 ACGU 采样
    - mask:   mod_type="none"；role="none"；base="mask"
    """
    if task_name == "bind":
        role = rng.choice(list(ROLE_NAMES))
        # role = "reader"
        mod_type = rng.choice(list(MOD_TYPE_NAMES))
        # mod_type = "m6A"
        base = MOD_BASE_MAP[mod_type]
        return role, base, mod_type
    if task_name == "mod":
        mod_type = rng.choice(list(MOD_TYPE_NAMES))
        base = MOD_BASE_MAP[mod_type]
        return "none", base, mod_type
    if task_name == "struct":
        return "none", rng.choice(list(VALID_BASES)), "none"
    return "none", "mask", "none"

def apply_condition_mask(
    task_name: str,
    sampled_role: str,
    sampled_base: str,
    sampled_mod_type: str,
    rng: random.Random,
    role_mask_prob: float,
    base_mask_prob: float,
    mod_type_mask_prob: float,
) -> tuple[str, str]:
    """对采样条件做随机遮蔽，得到真正送入模型的 cond_role/cond_base/cond_mod_type。

    语义：
    - role 被遮蔽时置为 "none"
    - base 被遮蔽时置为 "mask"
    - mod_type 被遮蔽时置为 "none"

    边界约束：
    - 仅对 bind/mod/struct 三类任务生效。
    - mask 任务保持固定条件,不再二次随机化。
    """
    cond_role = sampled_role
    cond_base = sampled_base
    cond_mod_type = sampled_mod_type

    if task_name in ("bind", "mod", "struct"):
        if rng.random() < role_mask_prob:
            cond_role = "none"
        if rng.random() < base_mask_prob:
            cond_base = "mask"
        if rng.random() < mod_type_mask_prob:
            cond_mod_type = "none"
    return cond_role, cond_base, cond_mod_type



def tensor_to_numpy_flat(tensor: torch.Tensor, mask: torch.Tensor | None = None) -> np.ndarray:
    """张量转 numpy 并展平；可选按 mask 过滤有效位置。"""
    data = tensor.detach().cpu().numpy()
    if mask is None:
        return data.reshape(-1)
    mask_np = mask.detach().cpu().numpy().astype(bool)
    return data[mask_np]

def _spawn_row_rngs(rng: random.Random) -> tuple[random.Random, random.Random]:
    """为单条样本拆分独立随机源。
    """
    row_seed_aprime = rng.randint(0, 2**31 - 1)
    row_seed_unlabeled = rng.randint(0, 2**31 - 1)
    return random.Random(row_seed_aprime), random.Random(row_seed_unlabeled)

def _sample_indices_or_all(
    candidates: np.ndarray,
    target_count: int,
    rng: random.Random,
) -> np.ndarray:
    """从候选位点中采样固定数量；若候选不足则全取。

    参数：
    - candidates: 候选位点下标数组（1D，int64）
    - target_count: 目标采样数
    - rng: 随机源（用于可复现采样）

    返回：
    - 若 `candidates` 为空或 `target_count<=0`：返回空数组
    - 若候选数 <= 目标数：返回全部候选（不打乱）
    - 否则：无放回随机采样 `target_count` 个位点

    备注：
    - 该函数是一个通用工具，当前用于 mod 的 unlabeled 采样和 bind 的 G5 采样。
    """
    if candidates.size == 0 or target_count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if candidates.size <= target_count:
        return candidates
    return np.asarray(rng.sample(candidates.tolist(), target_count), dtype=np.int64)

def read_parquet_list_column(df: pd.DataFrame, column: str) -> pd.Series:
    """把 parquet 中的列表列解析为 Python list/ndarray 形式。"""
    return df[column].apply(parse_list_field)

################ 先验比例相关
def estimate_mod_prior(examples: list[TranscriptExample]) -> dict[str, float]:
    """估计 mod 任务 PU 先验：pos / (pos + unlabeled)。"""
    priors: dict[str, float] = {}
    for mt in MOD_TYPE_NAMES:
        total_pos = sum(
            item.mod_positions.get(mt, np.zeros(0)).size for item in examples
        )
        total_unl = sum(
            item.unlabeled_positions.get(mt, np.zeros(0)).size for item in examples
        )
        denom = total_pos + total_unl
        priors[mt] = float(total_pos / denom) if denom > 0 else 0.5
    return priors
    # total_pos = sum(int(item.m6a_positions.size) for item in examples)
    # total_unlabeled = sum(int(item.unlabeled_a_positions.size) for item in examples)
    # denom = total_pos + total_unlabeled
    # if denom <= 0:
    #     return 0.5
    # return float(total_pos / denom)


def estimate_binding_priors(examples: list[TranscriptExample]) -> dict[str, dict[str, float]]:
    """估计各 role 绑定任务先验（正例比例）。"""
    out: dict[str, dict[str, float]] = {}
    for mt in MOD_TYPE_NAMES:
        role_priors: dict[str, float] = {}
        for role in ROLE_NAMES:
            positives = 0
            total = 0
            for item in examples:
                labels = item.role_labels.get(mt, {}).get(role)
                if labels is None:
                    continue
                positives += int((labels == 1).sum())
                total += int(labels.shape[0])
            role_priors[role] = float(positives / total) if total > 0 else 0.5
        out[mt] = role_priors
    return out
    # for role in ("writer", "reader", "eraser"):
    #     positives = 0
    #     total = 0
    #     for item in examples:
    #         labels = item.role_labels.get(role)
    #         if labels is None:
    #             continue
    #         positives += int((labels == 1).sum())
    #         total += int(labels.shape[0])
    #     if total == 0:
    #         out[role] = 0.5
    #     else:
    #         out[role] = float(positives / total)
    # return out

################### bind 相关
def estimate_strong_binding_thresholds(examples: list[TranscriptExample], q: float = 0.75) ->  dict[str, dict[str, float]]:
    """估计各 (mod_type, role) 的强结合阈值（正例 support 的 q 分位数）。"""
    out: dict[str, dict[str, float]] = {}
    for mt in MOD_TYPE_NAMES:
        role_thresholds: dict[str, float] = {}
        for role in ROLE_NAMES:
            supports: list[int] = []
            for item in examples:
                labels = item.role_labels.get(mt, {}).get(role)
                support = item.role_support.get(mt, {}).get(role)
                if labels is None or support is None:
                    continue
                positive_support = support[labels == 1]
                if positive_support.size > 0:
                    supports.extend(int(x) for x in positive_support.tolist())
            if supports:
                role_thresholds[role] = float(
                    np.quantile(np.asarray(supports, dtype=np.float32), q=q)
                )
            else:
                role_thresholds[role] = 1.0
        out[mt] = role_thresholds
    return out
    # for role in ("writer", "reader", "eraser"):
    #     supports: list[int] = []
    #     for item in examples:
    #         labels = item.role_labels.get(role)
    #         support = item.role_support.get(role)
    #         if labels is None or support is None:
    #             continue
    #         positive_support = support[labels == 1]
    #         if positive_support.size > 0:
    #             supports.extend(int(x) for x in positive_support.tolist())
    #     if supports:
    #         out[role] = float(np.quantile(np.asarray(supports, dtype=np.float32), q=q))
    #     else:
    #         out[role] = 1.0
    # return out



####### structure feature / target 构建相关函数（涉及 RNAfold 统计） #######

def _build_fold_sequence_from_replacements(raw_seq: str, replaced_positions: np.ndarray) -> str:
    """基于原始 RNA 序列构造“用于在线 RNAfold”的序列字符串。

    规则：
    - 普通 A 保持 A
    - 仅在 replaced_positions 里的 A 变为 6
    """
    seq = str(raw_seq).upper().replace("T", "U")
    if replaced_positions.size == 0:
        return seq
    chars = list(seq)
    for pos in replaced_positions.tolist():
        i = int(pos)
        if 0 <= i < len(chars) and chars[i] == "A":
            chars[i] = "6"
    return "".join(chars)


def _get_struct_pair_stats(
    struct_provider: Any, # 结构数据提供器（离线缓存 BPPCache/在线 RNAfold 接口）
    item: TranscriptExample, # 当前转录本样本序列（含 sequence、transcript_id）
    replaced_positions: np.ndarray, # 这条样本里被替换成 A' 的位置
    length: int, #当前序列长度
    use_rnafold_struct_feats: bool, # 是否把 RNAfold 统计作为输入特征
) -> tuple[np.ndarray, np.ndarray, bool]:
    use_mod_a_row = bool(replaced_positions.size > 0)
    if not use_rnafold_struct_feats:
        return (
            np.zeros((length,), dtype=np.float32),
            np.zeros((length,), dtype=np.float32),
            use_mod_a_row,
        )

    if hasattr(struct_provider, "get_positional_stats_from_sequence"):
        fold_seq = _build_fold_sequence_from_replacements(item.sequence, replaced_positions) # A'变为6
        pair_prob, pair_count = struct_provider.get_positional_stats_from_sequence(fold_seq)
    else: # 否则走离线缓存接口（兼容旧版 BPPCache）
        pair_prob, pair_count = struct_provider.get_positional_stats(
            transcript_id=item.transcript_id,
            use_mod_a=use_mod_a_row,
            seq_len_fallback=length,
        )
    return pair_prob, pair_count, use_mod_a_row


def _get_struct_target_matrix(
    struct_provider: Any,
    item: TranscriptExample,
    task_name: str,
    replaced_positions: np.ndarray,
    use_mod_a_row: bool,
    length: int,
    factor: int = 16,
) -> np.ndarray:
    """获取结构任务监督目标矩阵（下采样后）。"""
    if task_name != "struct":
        l_prime = int(math.ceil(length / factor))
        return np.zeros((l_prime, l_prime), dtype=np.float32)

    if hasattr(struct_provider, "get_downsampled_target_from_sequence"):
        fold_seq = _build_fold_sequence_from_replacements(item.sequence, replaced_positions)
        return struct_provider.get_downsampled_target_from_sequence(fold_seq, factor=factor)

    return struct_provider.get_downsampled_target(
        transcript_id=item.transcript_id,
        use_mod_a=use_mod_a_row,
        seq_len_fallback=length,
        factor=factor,
    )


def _estimate_max_sites(examples: list[TranscriptExample], task_name: str, sampled_mod_type: str) -> int:
    """估算当前 batch 在 site 维度上需要的最大容量。"""
    if task_name == "bind":
        site_caps = []
        for item in examples:
            n_mod = int(item.mod_positions.get(sampled_mod_type, np.zeros(0)).shape[0])
            n_unl = int(item.unlabeled_positions.get(sampled_mod_type, np.zeros(0)).shape[0])
            # n_m6a = int(item.m6a_positions.shape[0])
            # n_unl = int(item.unlabeled_a_positions.shape[0])
            g5_cap = min(n_unl, n_mod if n_mod > 0 else 16)
            site_caps.append(max(1, n_mod + g5_cap))
            # g5_cap = min(n_unl, n_m6a if n_m6a > 0 else 16)
            # site_caps.append(max(1, n_m6a + g5_cap))
        return max(site_caps) if site_caps else 1

    return max(
        max(1, item.mod_positions.get(sampled_mod_type, np.zeros(0)).shape[0])
        for item in examples
    )


def collate_batch(
    examples: list[TranscriptExample],
    task_name: str,
    role_name: str,
    cond_base: str,
    sampled_mod_type: str,  
    struct_provider: BPPCache,
    strong_binding_threshold: float,
    rng: random.Random,
    mod_unlabeled_ratio: float = 1.0,
    mask_prob: float = 0.15,
    aprime_enable: bool = True,
    aprime_prob: float = 0.1,
    aprime_max_per_seq: int = -1,
) -> dict[str, torch.Tensor | list[str]]:
    
    """把一组 TranscriptExample 打包成模型输入张量。

    该函数同时服务多任务（mod/bind/struct/mask），统一返回以下字段。

    输入字段：
    - tokens: [B,L] 主序列 token（随机替换）
    - attn_mask: [B,L] 有效 token 掩码（True=非 padding）

    mod 任务监督：
    - mod_pu_labels: [B,L]，1=positive，-1=unlabeled，0=不参与监督
    - mod_pu_mask: [B,L]，True 的位置才参与 mod loss

    bind 任务监督（site 级）：
    - site_positions: [B,S] site 坐标（原序列下标）
    - site_pu_labels: [B,S] site 级 PU 标签（1/-1）
    - site_mask: [B,S] 有效 site 掩码（过滤 padding site）
    - site_support: [B,S] 证据支持计数
    - strong_binding_mask: [B,S] 强结合位点掩码
    - g1_mask~g5_mask: [B,S] grouped bind loss 的分组掩码（用来监督loss）

    mask 任务监督：
    - mlm_input: [B,L] 带 [MASK] 的输入
    - mlm_target: [B,L] 被 mask 位点真值；其余位置为 -100（ignore）

    struct 任务监督：
    - struct_target: [B,L',L'] RNAfold 下采样结构目标（默认 16 倍）（也就是struct_mats）
    - struct_lengths: [B] 每条样本有效 L'
    """
    # 动态 padding 形状：按本 batch 最大序列长度与最大位点数分配。
    batch_size = len(examples) # 当前 batch 里有多少条转录本样本（B）
    max_len = max(item.seq_len for item in examples) # 都按这个长度做 padding
    max_sites = _estimate_max_sites(examples, task_name=task_name, sampled_mod_type=sampled_mod_type)

    # 初始化输出张量
    tokens = np.full((batch_size, max_len), PAD_TOKEN_ID, dtype=np.int64)
    attn_mask = np.zeros((batch_size, max_len), dtype=bool)
    mod_pu_labels = np.zeros((batch_size, max_len), dtype=np.int64)
    mod_pu_mask = np.zeros((batch_size, max_len), dtype=bool)
    site_positions = np.full((batch_size, max_sites), -1, dtype=np.int64)
    site_pu_labels = np.zeros((batch_size, max_sites), dtype=np.int64)
    site_mask = np.zeros((batch_size, max_sites), dtype=bool)
    site_support = np.zeros((batch_size, max_sites), dtype=np.float32)
    strong_mask = np.zeros((batch_size, max_sites), dtype=bool)
    g1_mask = np.zeros((batch_size, max_sites), dtype=bool)
    g2_mask = np.zeros((batch_size, max_sites), dtype=bool)
    g3_mask = np.zeros((batch_size, max_sites), dtype=bool)
    g4_mask = np.zeros((batch_size, max_sites), dtype=bool)
    g5_mask = np.zeros((batch_size, max_sites), dtype=bool)
    mlm_input = np.full((batch_size, max_len), PAD_TOKEN_ID, dtype=np.int64)
    mlm_target = np.full((batch_size, max_len), -100, dtype=np.int64)
    struct_mats: list[np.ndarray] = []
    struct_lengths = np.zeros((batch_size,), dtype=np.int64)
    transcript_ids: list[str] = []



    for row_idx, item in enumerate(examples): # 遍历 batch 里的每条转录本
        transcript_ids.append(item.transcript_id)
        rng_aprime, rng_unlabeled = _spawn_row_rngs(rng) # 随机源拆分：一个用于 A' 替换，一个用于未标注位点采样
    
        current_mod_positions = item.mod_positions.get(sampled_mod_type, np.zeros(0, dtype=np.int64))

        # ── 序列编码 + 修饰 token 替换 ──
        seq_ids, replaced_positions = encode_with_optional_aprime(
        # seq_ids是替换后的 token 序列，replaced_positions 是被替换成 A' 的位置（0-based）
            sequence=item.sequence,
            mod_positions=current_mod_positions,
            mod_type=sampled_mod_type,
            aprime_enable=aprime_enable,
            aprime_prob=aprime_prob,
            aprime_max_per_seq=aprime_max_per_seq,
            rng_aprime=rng_aprime,
        )
        # 写进 token
        length = seq_ids.shape[0] 
        tokens[row_idx, :length] = seq_ids
        attn_mask[row_idx, :length] = True

        # ############################ mod 任务标签构造：
        """
        已知:
            m6A 位点 m6a_positions
            其他 A 位点 unlabeled_a_positions
        标签构造：
            m6a_positions -> label=1, mask=True
            从 unlabeled_a_positions 里采样一部分 -> label=-1, mask=True
            其余位置 -> label=0, mask=False（不参与监督）
        损失：
            - 只在 mask=True 的位置计算 mod loss
            - 目标是让 label=1 的位点得分高于 label=-1 的位点
        """
        if task_name == "mod":
            positives = current_mod_positions
            if positives.size > 0:
                mod_pu_labels[row_idx, positives] = 1 #（标正类）
                mod_pu_mask[row_idx, positives] = True # 这些标注位置参与 mod loss

            unlabeled = item.unlabeled_positions.get(sampled_mod_type, np.zeros(0, dtype=np.int64))

            if positives.size > 0: # 有正例时：target_unlabeled ≈ 正例数 * ratio （保持正/未标注的大致平衡）
                target_unlabeled = max(1, int(round(float(positives.size) * mod_unlabeled_ratio)))
            else: # 没正例时：最多 64
                target_unlabeled = min(64, int(unlabeled.size))

            if unlabeled.size > 0 and target_unlabeled > 0: 
                sampled_u = _sample_indices_or_all(unlabeled, target_unlabeled, rng_unlabeled) # 从未标注中采样固定数量（或全取）
                mod_pu_labels[row_idx, sampled_u] = -1 # PU里的未标注 A 记为 -1
                mod_pu_mask[row_idx, sampled_u] = True # 这些未标注位置要参与mod损失

        ############################### bind 任务监督：site 级标签 + 强结合掩码 + G1~G5 分组。
        if task_name == "bind":
            current_sites = current_mod_positions
            n_sites = int(current_sites.shape[0])
            if n_sites > 0:
                site_positions[row_idx, :n_sites] = current_sites
                site_mask[row_idx, :n_sites] = True

                labels = item.role_labels.get(sampled_mod_type, {}).get(role_name)
                support_arr = item.role_support.get(sampled_mod_type, {}).get(role_name)
                if labels is None:
                    labels = np.full((n_sites,), -1, dtype=np.int64)
                if support_arr is None:
                    support_arr = np.zeros((n_sites,), dtype=np.int64)

                site_pu_labels[row_idx, :n_sites] = labels[:n_sites]# 写入 site 级 PU 标签
                site_support[row_idx, :n_sites] = support_arr[:n_sites].astype(np.float32) # 写入 site 级证据强度

                # 强结合定义：正例且 support>=阈值。
                positives = labels[:n_sites] == 1
                strong_here = positives & (support_arr[:n_sites] >= strong_binding_threshold) # 正例且证据达到阈值 -> 强结合位点
                strong_mask[row_idx, :n_sites] = strong_here # 写入强结合掩码

                # G1~G4 分组赋值：
                # - 依据两个维度划分：是否被替换、role_label(1/-1)
                replaced_set = set(int(x) for x in replaced_positions.tolist())
                # 对每个 m6A 位点判断是否在 A' 替换集合中
                is_revealed = np.asarray(
                    [int(pos) in replaced_set for pos in current_sites[:n_sites]], dtype=bool,
                )
                is_pos = labels[:n_sites] == 1 # 位点中 role 正例掩码
                is_unl = labels[:n_sites] == -1 # 位点中 role 未标注掩码

                g1_mask[row_idx, :n_sites] = is_revealed & is_pos
                g2_mask[row_idx, :n_sites] = is_revealed & is_unl
                g3_mask[row_idx, :n_sites] = (~is_revealed) & is_pos
                g4_mask[row_idx, :n_sites] = (~is_revealed) & is_unl

            # G5 分组（bind 专用）：
            # 从非修饰位点里采样一批锚点，追加到 site_positions 后部。
            # 这些位置没有 role 标签，统一按 unlabeled 处理（site_pu_labels=-1）。
            unlabeled_for_g5 = item.unlabeled_positions.get(
                sampled_mod_type, np.zeros(0, dtype=np.int64),
            )
            if unlabeled_for_g5.size > 0:
                # 采样数量策略与 max_sites 预留一致。
                if n_sites > 0:
                    target_g5 = min(int(unlabeled_for_g5.size), n_sites) 
                else:
                    target_g5 = min(int(unlabeled_for_g5.size), 16) 

                if target_g5 > 0: # # 从普通候选中随机采样（不够则全取）
                    sampled_g5 = _sample_indices_or_all(unlabeled_for_g5, target_g5, rng_unlabeled) 

                    # G5 从 site 槽位的第 n_sites 个位置开始写（接在 m6A 后面）
                    start = n_sites
                    end = min(max_sites, start + int(sampled_g5.shape[0])) # 截断到 batch 允许的最大 site 容量
                    k = max(0, end - start)
                    if k > 0:
                        site_positions[row_idx, start:end] = sampled_g5[:k] # 写入 G5 坐标
                        site_mask[row_idx, start:end] = True # 这些 G5 槽位也是有效 site
                        site_pu_labels[row_idx, start:end] = -1 # G5 没有 role 正例标签，按未标注处理
                        site_support[row_idx, start:end] = 0.0 # G5 没有支持证据
                        strong_mask[row_idx, start:end] = False # G5 不属于强结合
                        g5_mask[row_idx, start:end] = True # 标记这些位点属于 G5 组

        # ########################### mask 任务：构造 MLM 输入和目标（被选中位置替换为 [MASK]）。
        if task_name == "mask":
            mlm_input[row_idx, :length] = seq_ids
            # 仅对标准 RNA 碱基 A/C/G/U 做 MLM 采样；
            # 不 mask 随机替换（以及 N/[PAD]/[MASK] 等），让修饰作为上下文保留。
            valid = (
                (seq_ids == BASE_TO_ID["A"])
                | (seq_ids == BASE_TO_ID["C"])
                | (seq_ids == BASE_TO_ID["G"])
                | (seq_ids == BASE_TO_ID["U"])
            )
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
        use_mod_a_row = bool(replaced_positions.size > 0)
        l_prime = int(math.ceil(length / 16))
        struct_lengths[row_idx] = l_prime
        mat = _get_struct_target_matrix(
            struct_provider=struct_provider,
            item=item,
            task_name=task_name,
            replaced_positions=replaced_positions,
            use_mod_a_row=use_mod_a_row,
            length=length,
            factor=16,
        )
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
        "mod_pu_labels": torch.tensor(mod_pu_labels, dtype=torch.long),
        "mod_pu_mask": torch.tensor(mod_pu_mask, dtype=torch.bool),
        "site_positions": torch.tensor(site_positions, dtype=torch.long),
        "site_pu_labels": torch.tensor(site_pu_labels, dtype=torch.long),
        "site_mask": torch.tensor(site_mask, dtype=torch.bool),
        "site_support": torch.tensor(site_support, dtype=torch.float32),
        "strong_binding_mask": torch.tensor(strong_mask, dtype=torch.bool),
        "g1_mask": torch.tensor(g1_mask, dtype=torch.bool),
        "g2_mask": torch.tensor(g2_mask, dtype=torch.bool),
        "g3_mask": torch.tensor(g3_mask, dtype=torch.bool),
        "g4_mask": torch.tensor(g4_mask, dtype=torch.bool),
        "g5_mask": torch.tensor(g5_mask, dtype=torch.bool),
        "mlm_input": torch.tensor(mlm_input, dtype=torch.long),
        "mlm_target": torch.tensor(mlm_target, dtype=torch.long),
        "struct_target": torch.tensor(struct_target, dtype=torch.float32),
        "struct_lengths": torch.tensor(struct_lengths, dtype=torch.long),
        "transcript_ids": transcript_ids,
    }
