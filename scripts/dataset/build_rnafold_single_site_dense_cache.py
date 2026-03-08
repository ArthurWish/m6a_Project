#!/usr/bin/env python3
"""Build offline RNAfold cache for single-site m6A replacement.

目标：
- 对每条 transcript 的每个 m6A 位点，单独构造一个“只替换该位点 A->6”的序列。
- 运行 RNAfold，得到该替换条件下的配对概率。
- 按当前 struct 目标的方式（块平均 + 对称 + 清对角）构建稠密矩阵，并存盘。

输出格式（每个 transcript 一个 npz）：
- site_positions: [N] int32，N 为该 transcript 可替换 m6A 位点数
- mats: [N, L, L] float16，L 为原序列长度（不下采样）
- seq_len: int32
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.etd_multitask.rnafold import parse_dot_ps_ubox
from models.etd_multitask.utils import parse_list_field


def _sanitize_sequence(seq: str) -> str:
    seq = str(seq).upper().replace("T", "U")
    allowed = {"A", "C", "G", "U", "N", "6"}
    return "".join(ch if ch in allowed else "N" for ch in seq)


def _replace_single_site(seq: str, pos: int) -> str:
    chars = list(seq)
    if 0 <= pos < len(chars) and chars[pos] == "A":
        chars[pos] = "6"
    return "".join(chars)


def _run_rnafold_pairs(
    seq: str,
    rnafold_bin: str,
    timeout_seconds: int,
    tmp_dir: str | None = None,
) -> dict[tuple[int, int], float]:
    seq = _sanitize_sequence(seq)
    with tempfile.TemporaryDirectory(prefix="rnafold_single_site_", dir=tmp_dir) as work_dir:
        tmp_path = Path(work_dir)
        cmd = [rnafold_bin, "-p", "--noLP", "-d2", "--modifications"]
        proc = subprocess.run(
            cmd,
            input=f">seq\n{seq}\n",
            text=True,
            cwd=tmp_path,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"RNAfold failed: {err}")

        dot_ps = tmp_path / "seq_dp.ps"
        if not dot_ps.exists():
            raise RuntimeError("RNAfold did not generate seq_dp.ps")
        return parse_dot_ps_ubox(dot_ps)


def _pair_map_to_dense(pair_map: dict[tuple[int, int], float], seq_len: int) -> np.ndarray:
    mat = np.zeros((seq_len, seq_len), dtype=np.float32)

    for (i, j), p in pair_map.items():
        if i < 0 or j < 0 or i >= seq_len or j >= seq_len:
            continue
        # 上三角概率直接写入，再对称化到下三角。
        mat[i, j] = float(p)
    mat = mat + mat.T
    np.fill_diagonal(mat, 0.0)
    return mat


def _process_one(
    transcript_id: str,
    seq: str,
    m6a_positions: list[int],
    out_path: Path,
    *,
    overwrite: bool,
    retries: int,
    rnafold_bin: str,
    timeout_seconds: int,
    tmp_dir: str | None,
) -> tuple[str, str]:
    if out_path.exists() and not overwrite:
        return transcript_id, "skipped"

    seq = _sanitize_sequence(seq)
    valid_positions = [int(p) for p in sorted(set(m6a_positions)) if 0 <= int(p) < len(seq) and seq[int(p)] == "A"]
    if not valid_positions:
        return transcript_id, "no_valid_sites"

    mats: list[np.ndarray] = []
    kept_positions: list[int] = []

    for pos in valid_positions:
        seq_one = _replace_single_site(seq, pos)
        last_error = None
        ok = False
        for _ in range(max(1, int(retries) + 1)):
            try:
                pair_map = _run_rnafold_pairs(
                    seq_one,
                    rnafold_bin=rnafold_bin,
                    timeout_seconds=timeout_seconds,
                    tmp_dir=tmp_dir,
                )
                mat = _pair_map_to_dense(pair_map, seq_len=len(seq))
                mats.append(mat.astype(np.float16))
                kept_positions.append(int(pos))
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        if not ok:
            return transcript_id, f"error:site={pos}:{last_error}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        site_positions=np.asarray(kept_positions, dtype=np.int32),
        mats=np.asarray(mats, dtype=np.float16),
        seq_len=np.asarray(len(seq), dtype=np.int32),
    )
    return transcript_id, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-site replaced RNAfold dense cache.")
    parser.add_argument("--transcripts", default=str(REPO_ROOT / "data/processed/m6a_multitask_transcripts.parquet"))
    parser.add_argument("--splits", default=str(REPO_ROOT / "data/processed/m6a_multitask_splits.json"))
    parser.add_argument("--include-splits", default="train,val,test")
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data/processed/rnafold_single_site_dense"))
    parser.add_argument("--manifest-out", default=str(REPO_ROOT / "data/processed/rnafold_single_site_dense_manifest.json"))
    parser.add_argument("--rnafold-bin", default="RNAfold")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() // 2 if os.cpu_count() else 1))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards for multi-process distributed run.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index in [0, num_shards).")
    parser.add_argument(
        "--tmp-dir",
        default="/dev/shm",
        help="Temporary directory for RNAfold working files. /dev/shm is fastest if memory allows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transcripts_df = pd.read_parquet(args.transcripts)
    with Path(args.splits).open("r", encoding="utf-8") as handle:
        split_payload = json.load(handle)

    include_splits = [x.strip() for x in str(args.include_splits).split(",") if x.strip()]
    selected_ids: set[str] = set()
    for split_name in include_splits:
        selected_ids.update(str(x) for x in split_payload.get(split_name, []))

    tasks = []
    for row in transcripts_df.itertuples(index=False):
        tid = str(row.transcript_id)
        if tid not in selected_ids:
            continue
        seq = str(row.full_sequence).upper().replace("T", "U")
        if len(seq) > int(args.max_len):
            continue
        positions = parse_list_field(row.m6a_positions)
        if not positions:
            continue
        tasks.append((tid, seq, positions, Path(args.output_dir) / f"{tid}.npz"))

    tasks = sorted(tasks, key=lambda x: len(x[1]))

    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    if num_shards > 1:
        tasks = [task for idx, task in enumerate(tasks) if (idx % num_shards) == shard_index]

    if int(args.limit) > 0:
        tasks = tasks[: int(args.limit)]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"Start single-site RNAfold cache build: tasks={len(tasks)} jobs={int(args.jobs)} "
        f"shard={shard_index}/{num_shards} tmp_dir={args.tmp_dir}",
        flush=True,
    )

    results = {"ok": 0, "skipped": 0, "no_valid_sites": 0, "errors": {}}
    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
        future_map = {
            executor.submit(
                _process_one,
                transcript_id=tid,
                seq=seq,
                m6a_positions=positions,
                out_path=out_path,
                overwrite=bool(args.overwrite),
                retries=int(args.retries),
                rnafold_bin=str(args.rnafold_bin),
                timeout_seconds=int(args.timeout_seconds),
                tmp_dir=str(args.tmp_dir) if args.tmp_dir else None,
            ): (tid, len(seq), len(positions))
            for tid, seq, positions, out_path in tasks
        }

        done = 0
        total = len(future_map)
        for future in as_completed(future_map):
            tid, seq_len, n_sites = future_map[future]
            done += 1
            try:
                _, status = future.result()
            except Exception as exc:  # noqa: BLE001
                status = f"error:{exc}"

            if status in ("ok", "skipped", "no_valid_sites"):
                results[status] += 1
            else:
                results["errors"][tid] = status

            if done % 20 == 0 or done == total:
                print(
                    f"[{done}/{total}] tid={tid} len={seq_len} m6a={n_sites} status={status} "
                    f"ok={results['ok']} skip={results['skipped']} novalid={results['no_valid_sites']} "
                    f"err={len(results['errors'])}",
                    flush=True,
                )

    manifest = {
        "transcripts": str(args.transcripts),
        "splits": str(args.splits),
        "include_splits": include_splits,
        "max_len": int(args.max_len),
        "output_dir": str(args.output_dir),
        "rnafold_bin": str(args.rnafold_bin),
        "timeout_seconds": int(args.timeout_seconds),
        "retries": int(args.retries),
        "jobs": int(args.jobs),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "tmp_dir": str(args.tmp_dir),
        "limit": int(args.limit),
        "overwrite": bool(args.overwrite),
        "n_tasks": len(tasks),
        "results": results,
    }
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
