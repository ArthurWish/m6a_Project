#!/usr/bin/env python3
"""为 ETD together v4 attention-bias 生成离线结构 cache。

设计目标：
- 输入固定窗口（默认 half_window=256，总长度 513）；
- 对每个中心位点窗口跑普通 RNAfold；
- 不存稠密矩阵，而是存稀疏配对边；
- 训练时再快速组装成下采样后的 attention bias。

缓存格式：
- 每个 transcript 一个 `.npz`
- 字段：
  - site_positions: [N]
  - edge_offsets:   [N+1]
  - edge_i:         [E]
  - edge_j:         [E]
  - edge_p:         [E]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import random
import sys
from urllib.parse import quote_plus

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.etd_multitask.constants import BASE_TO_ID, PAD_TOKEN_ID
from models.etd_multitask.rnafold_online import OnlineRNAfoldProvider
from models.etd_only.bind_dataloader_v4 import load_site_metas_and_seqs


ID_TO_BASE = {
    int(BASE_TO_ID["A"]): "A",
    int(BASE_TO_ID["C"]): "C",
    int(BASE_TO_ID["G"]): "G",
    int(BASE_TO_ID["U"]): "U",
    int(BASE_TO_ID["N"]): "N",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline sparse struct-bias cache for ETD together",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sites-path", default=str(REPO_ROOT / "data/processed/all_multitask_sites.parquet"))
    parser.add_argument("--transcripts-path", default=str(REPO_ROOT / "data/processed/all_multitask_transcripts.parquet"))
    parser.add_argument("--splits-path", default=str(REPO_ROOT / "data/processed/all_multitask_splits.json"))
    parser.add_argument("--split-names", default="train,val", help="逗号分隔")
    parser.add_argument("--mod-type", default="m6A")
    parser.add_argument("--role-name", default="reader")
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--neg-ratio", type=float, default=0.3)
    parser.add_argument("--half-window", type=int, default=256, help="half window；总长度 = 2*half_window+1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rnafold-bin", default="/root/miniconda3/envs/m6a/bin/RNAfold")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--cache-size", type=int, default=4096)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data/processed/etd_together_struct_bias_hw256"))
    parser.add_argument("--num-shards", type=int, default=1, help="按 transcript 切 shard，便于多机/多进程并行")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-transcripts", type=int, default=0, help="调试用，0 表示不限制")
    return parser.parse_args()


def transcript_filename(transcript_id: str) -> str:
    return quote_plus(str(transcript_id), safe="") + ".npz"


def shuffled_transcript_ids(transcript_ids, seed: int) -> list[str]:
    """按固定 seed 打乱 transcript_id，和训练时的 transcript 子集规则保持一致。"""
    tids = sorted({str(tid) for tid in transcript_ids})
    rng = random.Random(int(seed))
    rng.shuffle(tids)
    return tids


def extract_window_sequence(full_tokens: np.ndarray, site_pos: int, half_window: int) -> str:
    window_size = 2 * half_window + 1
    seq_len = int(full_tokens.shape[0])
    center_in_window = half_window
    win_start = int(site_pos) - center_in_window
    win_end = win_start + window_size

    window = np.full(window_size, PAD_TOKEN_ID, dtype=np.int64)
    src_start = max(0, win_start)
    src_end = min(seq_len, win_end)
    dst_start = src_start - win_start
    dst_end = dst_start + (src_end - src_start)
    window[dst_start:dst_end] = full_tokens[src_start:src_end]

    chars = [ID_TO_BASE.get(int(tok), "N") for tok in window.tolist() if int(tok) != PAD_TOKEN_ID]
    return "".join(chars)


def main() -> None:
    args = parse_args()
    split_names = [x.strip() for x in str(args.split_names).split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] splits={split_names} half_window={args.half_window} output_dir={output_dir}", flush=True)
    metas, seq_store = load_site_metas_and_seqs(
        sites_path=args.sites_path,
        transcripts_path=args.transcripts_path,
        splits_path=args.splits_path,
        split_names=split_names,
        mod_type=args.mod_type,
        role_name=args.role_name,
        max_len=args.max_len,
        neg_ratio=args.neg_ratio,
        smoke_ratio=1.0,
        seed=args.seed,
    )
    print(f"[data] metas={len(metas)} transcripts={len(seq_store)}", flush=True)

    grouped: dict[str, list] = defaultdict(list)
    for meta in metas:
        grouped[meta.transcript_id].append(meta)

    transcript_ids = shuffled_transcript_ids(grouped.keys(), seed=int(args.seed))
    if args.num_shards > 1:
        transcript_ids = [
            tid for i, tid in enumerate(transcript_ids)
            if i % int(args.num_shards) == int(args.shard_index)
        ]
    if args.limit_transcripts > 0:
        transcript_ids = transcript_ids[: int(args.limit_transcripts)]

    provider = OnlineRNAfoldProvider(
        rnafold_bin=args.rnafold_bin,
        timeout_seconds=int(args.timeout_seconds),
        cache_size=int(args.cache_size),
    )

    manifest = {
        "split_names": split_names,
        "half_window": int(args.half_window),
        "window_len": int(2 * args.half_window + 1),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "n_transcripts": int(len(transcript_ids)),
    }
    (output_dir / f"manifest.shard{int(args.shard_index):03d}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for t_idx, transcript_id in enumerate(transcript_ids, start=1):
        metas_t = sorted(grouped[transcript_id], key=lambda m: int(m.site_pos))
        seq_idx = metas_t[0].seq_idx
        full_tokens = seq_store.get(seq_idx)
        path = output_dir / transcript_filename(transcript_id)

        if path.exists():
            if t_idx == 1 or t_idx % 100 == 0 or t_idx == len(transcript_ids):
                print(
                    f"[skip] transcript={t_idx}/{len(transcript_ids)} "
                    f"cached={path.name}",
                    flush=True,
                )
            continue

        site_positions: list[int] = []
        edge_offsets = [0]
        edge_i_all: list[np.ndarray] = []
        edge_j_all: list[np.ndarray] = []
        edge_p_all: list[np.ndarray] = []

        for meta in metas_t:
            seq = extract_window_sequence(full_tokens, int(meta.site_pos), int(args.half_window))
            pair_map = provider._get_pair_map(seq)
            items = sorted(pair_map.items())
            if items:
                edge_i = np.array([int(k[0]) for k, _ in items], dtype=np.uint16)
                edge_j = np.array([int(k[1]) for k, _ in items], dtype=np.uint16)
                edge_p = np.array([float(v) for _, v in items], dtype=np.float16)
            else:
                edge_i = np.zeros((0,), dtype=np.uint16)
                edge_j = np.zeros((0,), dtype=np.uint16)
                edge_p = np.zeros((0,), dtype=np.float16)

            site_positions.append(int(meta.site_pos))
            edge_i_all.append(edge_i)
            edge_j_all.append(edge_j)
            edge_p_all.append(edge_p)
            edge_offsets.append(edge_offsets[-1] + int(edge_i.shape[0]))

        np.savez_compressed(
            path,
            transcript_id=np.array([transcript_id], dtype=object),
            site_positions=np.asarray(site_positions, dtype=np.int32),
            edge_offsets=np.asarray(edge_offsets, dtype=np.int64),
            edge_i=np.concatenate(edge_i_all) if edge_i_all else np.zeros((0,), dtype=np.uint16),
            edge_j=np.concatenate(edge_j_all) if edge_j_all else np.zeros((0,), dtype=np.uint16),
            edge_p=np.concatenate(edge_p_all) if edge_p_all else np.zeros((0,), dtype=np.float16),
        )

        if t_idx == 1 or t_idx % 100 == 0 or t_idx == len(transcript_ids):
            print(
                f"[build] transcript={t_idx}/{len(transcript_ids)} "
                f"sites={len(metas_t)} cached={path.name}",
                flush=True,
            )


if __name__ == "__main__":
    main()
