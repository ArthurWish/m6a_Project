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
    transcript_id: str
    sequence: str
    seq_len: int
    m6a_positions: np.ndarray
    unlabeled_a_positions: np.ndarray
    role_labels: dict[str, np.ndarray]
    role_support: dict[str, np.ndarray]


class BPPCache:
    """Read and cache sparse RNAfold base pairing probabilities."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self._raw_cache: dict[str, dict[str, Any]] = {}
        self._position_cache: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray]] = {}
        self._downsample_cache: dict[tuple[str, bool, int], np.ndarray] = {}

    def _load_raw(self, transcript_id: str) -> dict[str, Any] | None:
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
    with Path(splits_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    ids: set[str] = set()
    for split_name in split_names:
        entries = payload.get(split_name, [])
        ids.update(str(x) for x in entries)
    return ids


def _build_site_lookup(sites_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
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
    sites_df = pd.read_parquet(sites_path)
    transcripts_df = pd.read_parquet(transcripts_path)
    split_ids = load_split_ids(splits_path, split_names)

    site_lookup = _build_site_lookup(sites_df)

    examples: list[TranscriptExample] = []
    for row in transcripts_df.itertuples(index=False):
        transcript_id = str(row.transcript_id)
        if transcript_id not in split_ids:
            continue

        seq = str(row.full_sequence).upper().replace("T", "U")
        seq_len = int(row.seq_len)
        if seq_len > max_len:
            continue

        site_df = site_lookup.get(transcript_id)
        if site_df is None or site_df.empty:
            continue

        m6a_positions = site_df["site_pos"].astype(int).to_numpy(dtype=np.int64)
        m6a_positions = np.unique(m6a_positions)

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

    examples = _sample_examples(examples, smoke_ratio=smoke_ratio, seed=seed)
    return examples


def split_bucket(length: int, boundaries: list[int]) -> int:
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

    use_mod_a = cond_base == "A"
    for row_idx, item in enumerate(examples):
        transcript_ids.append(item.transcript_id)

        seq_ids = np.asarray(encode_sequence(item.sequence), dtype=np.int64)
        length = seq_ids.shape[0]

        tokens[row_idx, :length] = seq_ids
        attn_mask[row_idx, :length] = True

        pair_prob, pair_count = bpp_cache.get_positional_stats(
            transcript_id=item.transcript_id,
            use_mod_a=use_mod_a,
            seq_len_fallback=length,
        )
        struct_feat = _build_struct_features(seq_ids, item.m6a_positions, pair_prob[:length], pair_count[:length])
        struct_feats[row_idx, :length] = struct_feat

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

            positives = labels[:n_sites] == 1
            strong_here = positives & (support[:n_sites] >= strong_binding_threshold)
            strong_mask[row_idx, :n_sites] = strong_here

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
    total_pos = sum(int(item.m6a_positions.size) for item in examples)
    total_unlabeled = sum(int(item.unlabeled_a_positions.size) for item in examples)
    denom = total_pos + total_unlabeled
    if denom <= 0:
        return 0.5
    return float(total_pos / denom)


def estimate_binding_priors(examples: list[TranscriptExample]) -> dict[str, float]:
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
    data = tensor.detach().cpu().numpy()
    if mask is None:
        return data.reshape(-1)
    mask_np = mask.detach().cpu().numpy().astype(bool)
    return data[mask_np]


def read_parquet_list_column(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].apply(parse_list_field)
