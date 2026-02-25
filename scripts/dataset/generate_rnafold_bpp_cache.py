#!/usr/bin/env python3
"""Generate transcript-level RNAfold sparse BPP cache for ETD multitask training."""

from __future__ import annotations

import argparse
import json
import math
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


def _sanitize_sequence(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    allowed = {"A", "C", "G", "U", "N", "6"}
    return "".join(ch if ch in allowed else "N" for ch in seq)


def _apply_modification(seq: str, m6a_positions: list[int]) -> str:
    chars = list(seq)
    for pos in m6a_positions:
        if pos < 0 or pos >= len(chars):
            continue
        if chars[pos] == "A":
            chars[pos] = "6"
    return "".join(chars)


def _run_rnafold_single(seq: str, use_modifications: bool) -> dict[tuple[int, int], float]:
    seq = _sanitize_sequence(seq)
    with tempfile.TemporaryDirectory(prefix="rnafold_bpp_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_text = f">seq\n{seq}\n"
        cmd = ["RNAfold", "-p", "--noLP", "-d2"]
        if use_modifications:
            cmd.append("--modifications")

        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            cwd=tmp_path,
            capture_output=True,
            timeout=_run_rnafold_single.timeout_seconds,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"RNAfold failed: {proc.stderr.strip()}")

        dot_ps = tmp_path / "seq_dp.ps"
        if not dot_ps.exists():
            raise RuntimeError("RNAfold did not generate seq_dp.ps")

        pairs = parse_dot_ps_ubox(dot_ps)
        return pairs


_run_rnafold_single.timeout_seconds = 0


def _build_sparse_arrays(
    pair_ref: dict[tuple[int, int], float],
    pair_mod: dict[tuple[int, int], float],
    length: int,
) -> dict[str, np.ndarray]:
    keys = sorted(set(pair_ref.keys()) | set(pair_mod.keys()))
    if not keys:
        ij = np.zeros((0, 2), dtype=np.int32)
        p_ref = np.zeros((0,), dtype=np.float16)
        p_mod = np.zeros((0,), dtype=np.float16)
    else:
        ij = np.asarray(keys, dtype=np.int32)
        p_ref = np.asarray([pair_ref.get(k, 0.0) for k in keys], dtype=np.float16)
        p_mod = np.asarray([pair_mod.get(k, 0.0) for k in keys], dtype=np.float16)

    return {
        "ij": ij,
        "p_ref": p_ref,
        "p_modA": p_mod,
        "L": np.asarray(length, dtype=np.int32),
    }


def _process_one(
    transcript_id: str,
    sequence: str,
    m6a_positions: list[int],
    out_path: Path,
    overwrite: bool,
    retries: int,
    timeout_seconds: int,
) -> tuple[str, str]:
    if out_path.exists() and not overwrite:
        return transcript_id, "skipped"

    last_error = None
    for _ in range(max(1, retries + 1)):
        try:
            _run_rnafold_single.timeout_seconds = timeout_seconds
            seq_ref = _sanitize_sequence(sequence)
            seq_mod = _apply_modification(seq_ref, m6a_positions)

            pair_ref = _run_rnafold_single(seq_ref, use_modifications=True)
            pair_mod = _run_rnafold_single(seq_mod, use_modifications=True)

            arrays = _build_sparse_arrays(pair_ref, pair_mod, length=len(seq_ref))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, **arrays)
            return transcript_id, "ok"
        except subprocess.TimeoutExpired:
            last_error = f"timeout>{timeout_seconds}s"
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

    return transcript_id, f"error:{last_error}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate transcript-level RNAfold sparse BPP cache.")
    parser.add_argument(
        "--transcripts",
        default=str(REPO_ROOT / "data/processed/m6a_multitask_transcripts.parquet"),
    )
    parser.add_argument(
        "--splits",
        default=str(REPO_ROOT / "data/processed/m6a_multitask_splits.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data/processed/rnafold_bpp"),
    )
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument(
        "--include-splits",
        default="train,val,test",
        help="Comma-separated split names to process.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() // 2 if os.cpu_count() else 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument(
        "--manifest-out",
        default=str(REPO_ROOT / "data/processed/rnafold_bpp_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transcripts_df = pd.read_parquet(args.transcripts)
    with Path(args.splits).open("r", encoding="utf-8") as handle:
        split_payload = json.load(handle)

    split_names = [name.strip() for name in args.include_splits.split(",") if name.strip()]
    selected_ids = set()
    for split_name in split_names:
        selected_ids.update(str(x) for x in split_payload.get(split_name, []))

    tasks = []
    for row in transcripts_df.itertuples(index=False):
        transcript_id = str(row.transcript_id)
        if transcript_id not in selected_ids:
            continue

        seq = str(row.full_sequence).upper().replace("T", "U")
        if len(seq) > args.max_len:
            continue

        m6a_positions = parse_list_field(row.m6a_positions)
        out_path = Path(args.output_dir) / f"{transcript_id}.npz"
        tasks.append((transcript_id, seq, m6a_positions, out_path))

    tasks = sorted(tasks, key=lambda x: len(x[1]))
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "ok": 0,
        "skipped": 0,
        "errors": {},
    }

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(
                _process_one,
                transcript_id=tid,
                sequence=seq,
                m6a_positions=positions,
                out_path=out_path,
                overwrite=args.overwrite,
                retries=args.retries,
                timeout_seconds=args.timeout_seconds,
            ): (tid, len(seq))
            for tid, seq, positions, out_path in tasks
        }

        for future in as_completed(futures):
            tid, seq_len = futures[future]
            try:
                _, status = future.result()
            except Exception as exc:  # noqa: BLE001
                status = f"error:{exc}"

            if status == "ok":
                results["ok"] += 1
            elif status == "skipped":
                results["skipped"] += 1
            else:
                results["errors"][tid] = {
                    "status": status,
                    "seq_len": seq_len,
                }

    manifest = {
        "transcripts": args.transcripts,
        "splits": args.splits,
        "output_dir": str(out_dir),
        "max_len": args.max_len,
        "include_splits": split_names,
        "jobs": args.jobs,
        "overwrite": bool(args.overwrite),
        "retries": int(args.retries),
        "timeout_seconds": int(args.timeout_seconds),
        "submitted": len(tasks),
        "ok": results["ok"],
        "skipped": results["skipped"],
        "error_count": len(results["errors"]),
        "errors": results["errors"],
    }

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
