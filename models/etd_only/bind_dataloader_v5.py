
from __future__ import annotations

from dataclasses import dataclass
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.etd_multitask.constants import (
    BASE_TO_ID,
    PAD_TOKEN_ID,
    NUM_INDIVIDUAL_RBPS,
    INDIVIDUAL_RBPS,
    RBP_TO_IDX,
)
from models.etd_only.struct_bias_v5 import (
    tokens_to_sequence,
    pair_map_to_dense,
    downsample_pair_bias,
    compute_single_bias,
    OfflineStructBiasCache,
)

A_TOKEN_ID = BASE_TO_ID["A"]  # 0

_DRACH_VALID = {
    -2: {0, 2, 3},   # D
    -1: {0, 2},      # R
     0: {0},         # A (center)
    +1: {1},         # C
    +2: {0, 1, 3},   # H
}


def _is_drach(window: np.ndarray, pos: int) -> bool:
    for offset, valid in _DRACH_VALID.items():
        idx = pos + offset
        if idx < 0 or idx >= len(window):
            return False
        if int(window[idx]) not in valid:
            return False
    return True


@dataclass
class SiteMeta:
    transcript_id: str
    seq_idx: int
    site_pos: int
    rbp_targets: np.ndarray
    is_positive: bool


class SequenceStore:
    def __init__(self):
        self.sequences: list[np.ndarray] = []
        self.seq_lens: list[int] = []
        self._tid_to_idx: dict[str, int] = {}
        self.known_m6a: list[set[int]] = []
        self.site_targets: dict[tuple[int, int], np.ndarray] = {}

    def add(self, transcript_id: str, token_ids: np.ndarray) -> int:
        if transcript_id in self._tid_to_idx:
            return self._tid_to_idx[transcript_id]
        idx = len(self.sequences)
        self.sequences.append(token_ids)
        self.seq_lens.append(token_ids.shape[0])
        self._tid_to_idx[transcript_id] = idx
        self.known_m6a.append(set())
        return idx

    def add_m6a_site(self, seq_idx: int, pos: int, rbp_targets: np.ndarray) -> None:
        self.known_m6a[seq_idx].add(pos)
        self.site_targets[(seq_idx, pos)] = rbp_targets

    def get(self, idx: int) -> np.ndarray:
        return self.sequences[idx]

    def get_m6a_set(self, idx: int) -> set[int]:
        return self.known_m6a[idx]

    def get_site_target(self, seq_idx: int, pos: int) -> np.ndarray | None:
        return self.site_targets.get((seq_idx, pos))

    def __len__(self) -> int:
        return len(self.sequences)


@dataclass
class StructBiasConfig:
    enabled: bool = False
    mode: str = "online"          # "online" | "offline"
    scale: float = 1.0
    n_downsample: int = 2
    window_size: int = 513
    # online 参数
    rnafold_bin: str = "RNAfold"
    rnafold_timeout: int = 240
    # offline 参数
    cache_dir: str = ""
    offline_cache_max_transcripts: int = 256


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        metas: list[SiteMeta],
        seq_store: SequenceStore,
        half_window: int = 512,
        max_jitter: int = 128,
        training: bool = True,
        n_extra_pos: int = 5,
        n_m6a_neg: int = 3,
        n_clean_neg: int = 8,
        n_m6a_eval_a: int = 64,
        m6a_neg_smooth: float = 0.002,
        struct_bias_cfg: StructBiasConfig | None = None,
    ):
        self.metas = metas
        self.seq_store = seq_store
        self.half_window = half_window
        self.max_jitter = max_jitter
        self.training = training
        self.window_size = 2 * half_window + 1
        self.n_extra_pos = n_extra_pos
        self.n_m6a_neg = n_m6a_neg
        self.n_clean_neg = n_clean_neg
        self.n_m6a_eval_a = n_m6a_eval_a
        self.m6a_neg_smooth = m6a_neg_smooth

        # struct bias config
        self.sb_cfg = struct_bias_cfg or StructBiasConfig()
        self._offline_cache = None  

    def __len__(self) -> int:
        return len(self.metas)

    def __getitem__(self, idx: int) -> dict:
        meta = self.metas[idx]
        full_tokens = self.seq_store.get(meta.seq_idx)
        seq_len = full_tokens.shape[0]

        if self.training and self.max_jitter > 0:
            jitter = random.randint(-self.max_jitter, self.max_jitter)
        else:
            jitter = 0

        center_in_window = self.half_window + jitter
        win_start = meta.site_pos - center_in_window
        win_end = win_start + self.window_size

        window = np.full(self.window_size, PAD_TOKEN_ID, dtype=np.int64)
        src_start = max(0, win_start)
        src_end = min(seq_len, win_end)
        dst_start = src_start - win_start
        dst_end = dst_start + (src_end - src_start)
        window[dst_start:dst_end] = full_tokens[src_start:src_end]

        known_set = self.seq_store.get_m6a_set(meta.seq_idx)

        all_a_locs = np.where(window == A_TOKEN_ID)[0]
        cat_a_pos, cat_a_targets = [], []
        cat_b_pos, cat_c_pos = [], []

        for loc in all_a_locs:
            if loc == center_in_window:
                continue
            global_pos = loc + win_start
            if global_pos in known_set:
                tgt = self.seq_store.get_site_target(meta.seq_idx, global_pos)
                if tgt is not None and tgt.sum() > 0:
                    cat_a_pos.append(loc)
                    cat_a_targets.append(tgt)
                else:
                    cat_b_pos.append(loc)
            else:
                cat_c_pos.append(loc)

        extra_pos = np.zeros(self.n_extra_pos, dtype=np.int64)
        extra_targets = np.zeros((self.n_extra_pos, NUM_INDIVIDUAL_RBPS), dtype=np.float32)
        n_extra = 0
        if cat_a_pos:
            chosen = np.random.choice(len(cat_a_pos), min(len(cat_a_pos), self.n_extra_pos), replace=False)
            n_extra = len(chosen)
            for i, ci in enumerate(chosen):
                extra_pos[i] = cat_a_pos[ci]
                extra_targets[i] = cat_a_targets[ci]

        m6a_neg = np.zeros(self.n_m6a_neg, dtype=np.int64)
        n_m6a_neg = 0
        if cat_b_pos:
            chosen = np.random.choice(cat_b_pos, min(len(cat_b_pos), self.n_m6a_neg), replace=False)
            n_m6a_neg = len(chosen)
            m6a_neg[:n_m6a_neg] = chosen

        clean_neg = np.zeros(self.n_clean_neg, dtype=np.int64)
        n_clean_neg = 0
        if cat_c_pos:
            chosen = np.random.choice(cat_c_pos, min(len(cat_c_pos), self.n_clean_neg), replace=False)
            n_clean_neg = len(chosen)
            clean_neg[:n_clean_neg] = chosen

        m6a_positions = np.zeros(self.n_m6a_eval_a, dtype=np.int64)
        m6a_targets = np.full(self.n_m6a_eval_a, 0.0, dtype=np.float32)
        m6a_global_positions = np.zeros(self.n_m6a_eval_a, dtype=np.int64)
        n_m6a_sites = 0

        candidate_locs, candidate_labels = [], []
        for loc in all_a_locs:
            global_pos = loc + win_start
            is_m6a = global_pos in known_set
            if self.training:
                is_drach = _is_drach(window, loc)
                if not is_drach and not is_m6a:
                    if random.random() > 0.2:
                        continue
            label = 1.0 if is_m6a else self.m6a_neg_smooth
            candidate_locs.append(loc)
            candidate_labels.append(label)

        if candidate_locs:
            if len(candidate_locs) > self.n_m6a_eval_a:
                pos_idx = [i for i, l in enumerate(candidate_labels) if l > 0.5]
                neg_idx = [i for i, l in enumerate(candidate_labels) if l <= 0.5]
                if len(pos_idx) > self.n_m6a_eval_a:
                    pos_idx = list(np.random.choice(pos_idx, self.n_m6a_eval_a, replace=False))
                n_remaining = self.n_m6a_eval_a - len(pos_idx)
                chosen_neg = (
                    list(np.random.choice(neg_idx, min(n_remaining, len(neg_idx)), replace=False))
                    if n_remaining > 0 and neg_idx
                    else []
                )
                chosen = pos_idx + chosen_neg
            else:
                chosen = list(range(len(candidate_locs)))

            n_m6a_sites = len(chosen)
            for i, ci in enumerate(chosen):
                m6a_positions[i] = candidate_locs[ci]
                m6a_targets[i] = candidate_labels[ci]
                m6a_global_positions[i] = candidate_locs[ci] + win_start

       
        attn_bias = None
        if self.sb_cfg.enabled:
            factor = 2 ** self.sb_cfg.n_downsample
            valid_mask = window != PAD_TOKEN_ID
            valid_indices = np.where(valid_mask)[0]

            if self.sb_cfg.mode == "offline":
                if self._offline_cache is None:
                    self._offline_cache = OfflineStructBiasCache(
                        self.sb_cfg.cache_dir,
                        max_transcripts=self.sb_cfg.offline_cache_max_transcripts,
                    )
                try:
                    attn_bias = self._offline_cache.get_downsampled_bias(
                        transcript_id=meta.transcript_id,
                        site_pos=meta.site_pos,
                        window_len=self.window_size,
                        factor=factor,
                        scale=self.sb_cfg.scale,
                    )
                except Exception:
                    out_len = (self.window_size + factor - 1) // factor
                    attn_bias = np.zeros((out_len, out_len), dtype=np.float32)
            else:
                
                seq = tokens_to_sequence(window, valid_mask)
                attn_bias = compute_single_bias(
                    seq=seq,
                    valid_indices=valid_indices,
                    window_size=self.window_size,
                    factor=factor,
                    scale=self.sb_cfg.scale,
                    rnafold_bin=self.sb_cfg.rnafold_bin,
                    rnafold_timeout=self.sb_cfg.rnafold_timeout,
                )

        result = {
            "transcript_id": meta.transcript_id,
            "site_pos": np.int64(meta.site_pos),
            "token_ids": window,
            "center_idx": np.int64(center_in_window),
            "rbp_targets": meta.rbp_targets,
            "is_positive": meta.is_positive,
            "extra_pos_positions": extra_pos,
            "extra_pos_targets": extra_targets,
            "extra_pos_count": np.int64(n_extra),
            "m6a_neg_positions": m6a_neg,
            "m6a_neg_count": np.int64(n_m6a_neg),
            "clean_neg_positions": clean_neg,
            "clean_neg_count": np.int64(n_clean_neg),
            "m6a_det_positions": m6a_positions,
            "m6a_det_targets": m6a_targets,
            "m6a_det_count": np.int64(n_m6a_sites),
            "m6a_det_global_pos": m6a_global_positions,
            "m6a_det_seq_idx": np.int64(meta.seq_idx),
        }

        if attn_bias is not None:
            result["_attn_bias"] = attn_bias  # np.ndarray [out_len, out_len]

        return result



class ETDCollate:
    """轻量 collate：只 stack tensor，不做结构 bias 计算。"""

    def __init__(self, use_struct_bias: bool = False, **_kwargs):
        """保留旧接口签名以兼容，多余参数忽略。"""
        self.use_struct_bias = use_struct_bias

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        tokens = np.stack([b["token_ids"] for b in batch])
        attn_mask = (tokens != PAD_TOKEN_ID)

        def _stack_and_mask(key_pos, key_count, key_targets=None):
            pos = np.stack([b[key_pos] for b in batch])
            counts = np.array([b[key_count] for b in batch], dtype=np.int64)
            mask = np.zeros_like(pos, dtype=bool)
            for i, c in enumerate(counts):
                mask[i, :c] = True
            res = {
                "positions": torch.tensor(pos, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.bool),
            }
            if key_targets:
                res["targets"] = torch.tensor(
                    np.stack([b[key_targets] for b in batch]), dtype=torch.float32
                )
            return res

        extra_pos = _stack_and_mask("extra_pos_positions", "extra_pos_count", "extra_pos_targets")
        m6a_neg = _stack_and_mask("m6a_neg_positions", "m6a_neg_count")
        clean_neg = _stack_and_mask("clean_neg_positions", "clean_neg_count")
        m6a_det = _stack_and_mask("m6a_det_positions", "m6a_det_count", "m6a_det_targets")
        m6a_det_global = np.stack([b["m6a_det_global_pos"] for b in batch])
        m6a_det_seq_idx = np.array([b["m6a_det_seq_idx"] for b in batch], dtype=np.int64)

        result_dict = {
            "transcript_ids": [b["transcript_id"] for b in batch],
            "site_positions": torch.tensor(
                np.array([b["site_pos"] for b in batch], dtype=np.int64), dtype=torch.long
            ),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.bool),
            "center_indices": torch.tensor(
                np.array([b["center_idx"] for b in batch], dtype=np.int64), dtype=torch.long
            ),
            "rbp_targets": torch.tensor(
                np.stack([b["rbp_targets"] for b in batch]), dtype=torch.float32
            ),
            "is_positive": torch.tensor(
                np.array([b["is_positive"] for b in batch], dtype=bool), dtype=torch.bool
            ),
            "extra_pos_positions": extra_pos["positions"],
            "extra_pos_targets": extra_pos["targets"],
            "extra_pos_mask": extra_pos["mask"],
            "m6a_neg_positions": m6a_neg["positions"],
            "m6a_neg_mask": m6a_neg["mask"],
            "clean_neg_positions": clean_neg["positions"],
            "clean_neg_mask": clean_neg["mask"],
            "m6a_det_positions": m6a_det["positions"],
            "m6a_det_targets": m6a_det["targets"],
            "m6a_det_mask": m6a_det["mask"],
            "m6a_det_global_pos": torch.tensor(m6a_det_global, dtype=torch.long),
            "m6a_det_seq_idx": torch.tensor(m6a_det_seq_idx, dtype=torch.long),
        }

        if self.use_struct_bias and "_attn_bias" in batch[0]:
            attn_biases = np.stack([b["_attn_bias"] for b in batch])
            result_dict["attn_bias"] = torch.tensor(attn_biases, dtype=torch.float32)

        return result_dict



def _encode_sequence(seq: str) -> np.ndarray:
    seq = str(seq).upper().replace("T", "U")
    return np.array([BASE_TO_ID.get(ch, BASE_TO_ID["N"]) for ch in seq], dtype=np.int64)


_RBP_ALIASES = {"HNRNPG": "RBMX", "FMRP": "FMR1"}


def _rbp_names_to_multihot(names: object) -> np.ndarray:
    out = np.zeros(NUM_INDIVIDUAL_RBPS, dtype=np.float32)
    if names is None:
        return out
    if isinstance(names, np.ndarray):
        names = names.tolist()
    if not isinstance(names, (list, tuple)):
        return out
    for name in names:
        canonical = _RBP_ALIASES.get(str(name).upper(), str(name).upper())
        idx = RBP_TO_IDX.get(canonical)
        if idx is not None:
            out[idx] = 1.0
    return out


def load_split_ids(splits_path: str, split_names: list[str]) -> set[str]:
    with open(splits_path, "r") as f:
        payload = json.load(f)
    out = set()
    for s in split_names:
        for tid in payload.get(s, []):
            out.add(str(tid))
    return out


import ast


def _parse_rbp_field(x):
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return {}
    return x if isinstance(x, dict) else {}


def _merge_rbp_dicts(series: pd.Series) -> dict:
    merged = {"reader": []}
    reader_set = set()
    for item in series:
        d = _parse_rbp_field(item)
        readers = d.get("reader", [])
        if isinstance(readers, np.ndarray):
            readers = readers.tolist()
        if isinstance(readers, (list, tuple)):
            for r in readers:
                if r is not None:
                    reader_set.add(str(r).upper())
    merged["reader"] = sorted(reader_set)
    return merged


def load_site_metas_and_seqs(
    sites_path: str,
    transcripts_path: str,
    splits_path: str | None = None,
    split_names: list[str] | None = None,
    mod_type: str = "m6A",
    role_name: str = "reader",
    max_len: int = 12000,
    neg_ratio: float = 0.3,
    smoke_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[list[SiteMeta], SequenceStore]:
    sites_df = pd.read_parquet(
        sites_path, columns=["transcript_id", "site_pos", "mod_type", "rbp_name"]
    )
    transcripts_df = pd.read_parquet(
        transcripts_path, columns=["transcript_id", "full_sequence", "seq_len"]
    )

    split_ids = load_split_ids(splits_path, split_names) if splits_path and split_names else None
    sites_df = sites_df[sites_df["mod_type"].astype(str) == mod_type].copy()
    if sites_df.empty:
        return [], SequenceStore()

    sites_df = sites_df.sort_values(["transcript_id", "site_pos"]).copy()
    sites_df["rbp_name"] = sites_df["rbp_name"].apply(_parse_rbp_field)
    sites_df = sites_df.groupby(
        ["transcript_id", "site_pos", "mod_type"], as_index=False
    ).agg({"rbp_name": _merge_rbp_dicts})

    seq_store = SequenceStore()
    tid_to_idx: dict[str, int] = {}
    for row in transcripts_df.itertuples(index=False):
        tid = str(row.transcript_id)
        if split_ids is not None and tid not in split_ids:
            continue
        if int(row.seq_len) > max_len:
            continue
        sid = seq_store.add(tid, _encode_sequence(row.full_sequence))
        tid_to_idx[tid] = sid

    rng = random.Random(seed)
    pos_metas, neg_metas = [], []

    for tid, group in sites_df.groupby("transcript_id"):
        tid = str(tid)
        if tid not in tid_to_idx:
            continue
        seq_idx = tid_to_idx[tid]

        for srow in group.itertuples(index=False):
            site_pos = int(srow.site_pos)
            rbp_field = getattr(srow, "rbp_name", {})
            if not isinstance(rbp_field, dict):
                rbp_field = {}
            target = _rbp_names_to_multihot(rbp_field.get("reader"))
            is_positive = target.sum() > 0

            seq_store.add_m6a_site(seq_idx, site_pos, target)
            meta = SiteMeta(
                transcript_id=tid,
                seq_idx=seq_idx,
                site_pos=site_pos,
                rbp_targets=target,
                is_positive=is_positive,
            )
            if is_positive:
                pos_metas.append(meta)
            else:
                neg_metas.append(meta)

    n_neg = int(len(pos_metas) * neg_ratio)
    if len(neg_metas) > n_neg:
        rng.shuffle(neg_metas)
        neg_metas = neg_metas[:n_neg]

    all_metas = pos_metas + neg_metas
    if smoke_ratio < 1.0:
        k = max(1, int(len(all_metas) * smoke_ratio))
        rng.shuffle(all_metas)
        all_metas = all_metas[:k]

    return all_metas, seq_store


def build_balanced_sampler(metas: list[SiteMeta]) -> WeightedRandomSampler:
    counts = np.zeros(NUM_INDIVIDUAL_RBPS, dtype=np.float64)
    for m in metas:
        counts += m.rbp_targets.astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))

    pos_ws = []
    for m in metas:
        if m.is_positive:
            active = m.rbp_targets > 0.5
            pos_ws.append(float(w[active].max()) if active.any() else 1.0)
    avg = float(np.mean(pos_ws)) if pos_ws else 1.0

    sample_w = []
    for m in metas:
        if m.is_positive:
            active = m.rbp_targets > 0.5
            sample_w.append(float(w[active].max()) if active.any() else 1.0)
        else:
            sample_w.append(avg)

    return WeightedRandomSampler(weights=sample_w, num_samples=len(metas), replacement=True)
