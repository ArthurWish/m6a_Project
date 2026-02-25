#!/usr/bin/env python3
"""Build m6A multitask dataset tables for ETD training."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def normalize_transcript_id(raw: str) -> str:
    if not raw:
        return ""
    raw = str(raw).strip()
    if not raw or raw.lower() == "na":
        return ""
    return raw.split(".")[0]


def parse_fasta(path: str) -> tuple[dict[str, str], dict[str, int]]:
    seqs: dict[str, str] = {}
    stats = {
        "records": 0,
        "duplicates": 0,
        "duplicate_exact": 0,
        "duplicate_replaced": 0,
        "duplicate_conflict": 0,
    }

    with open_text(path) as handle:
        seq_id = None
        chunks = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    seq = "".join(chunks).upper().replace("T", "U")
                    stats["records"] += 1
                    if seq_id in seqs:
                        stats["duplicates"] += 1
                        existing = seqs[seq_id]
                        if seq == existing:
                            stats["duplicate_exact"] += 1
                        elif len(seq) > len(existing):
                            seqs[seq_id] = seq
                            stats["duplicate_replaced"] += 1
                        elif len(seq) == len(existing):
                            stats["duplicate_conflict"] += 1
                    else:
                        seqs[seq_id] = seq
                header = line[1:].split()[0]
                seq_id = normalize_transcript_id(header)
                chunks = []
            else:
                chunks.append(line)

        if seq_id is not None:
            seq = "".join(chunks).upper().replace("T", "U")
            stats["records"] += 1
            if seq_id in seqs:
                stats["duplicates"] += 1
                existing = seqs[seq_id]
                if seq == existing:
                    stats["duplicate_exact"] += 1
                elif len(seq) > len(existing):
                    seqs[seq_id] = seq
                    stats["duplicate_replaced"] += 1
                elif len(seq) == len(existing):
                    stats["duplicate_conflict"] += 1
            else:
                seqs[seq_id] = seq

    return seqs, stats


def locate_site(full_seq: str, short_seq: str) -> int | None:
    pattern = short_seq.upper().replace("T", "U")
    idx = full_seq.find(pattern)
    if idx < 0:
        return None
    if full_seq.count(pattern) != 1:
        return None
    center = idx + len(pattern) // 2
    return center


def parse_m6a_sites(m6a_path: str, transcript_seqs: dict[str, str], protein_coding_only: bool = True):
    kept = {}
    stats = Counter()

    with open_text(m6a_path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            stats["total_rows"] += 1

            if len(row) < 19:
                stats["invalid_rows"] += 1
                continue

            gene_type = row[16].strip()
            if protein_coding_only and gene_type != "protein_coding":
                stats["non_protein_coding"] += 1
                continue

            mod_id = row[3].strip()
            transcript_id = normalize_transcript_id(row[14])
            short_seq = row[18].strip()

            if not mod_id or not transcript_id or not short_seq:
                stats["missing_fields"] += 1
                continue

            full_seq = transcript_seqs.get(transcript_id)
            if not full_seq:
                stats["missing_transcript"] += 1
                continue

            site_pos = locate_site(full_seq, short_seq)
            if site_pos is None:
                stats["unresolved_position"] += 1
                continue

            key = (transcript_id, site_pos)
            if key in kept:
                stats["duplicate_site"] += 1
                # Prefer first mod_id to keep deterministic.
                continue

            site_base = full_seq[site_pos]
            if site_base != "A":
                stats["non_a_center"] += 1
                continue

            kept[key] = {
                "mod_id": mod_id,
                "transcript_id": transcript_id,
                "site_pos": int(site_pos),
                "site_base": site_base,
            }
            stats["kept"] += 1

    return kept, stats


def parse_role_support(path: str) -> Counter:
    counts: Counter = Counter()
    if not os.path.exists(path):
        return counts

    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or len(row) < 8:
                continue
            mod_id = row[7].strip()
            if mod_id:
                counts[mod_id] += 1
    return counts


def deterministic_split(transcript_id: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    key = f"{transcript_id}-{seed}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()[:8]
    value = int(digest, 16) / 0xFFFFFFFF

    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def write_parquet(df: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def build_tables(args: argparse.Namespace) -> dict:
    transcript_seqs, fasta_stats = parse_fasta(args.fasta)

    sites_by_key, site_stats = parse_m6a_sites(
        m6a_path=args.m6a_file,
        transcript_seqs=transcript_seqs,
        protein_coding_only=not args.all_genes,
    )

    writer_support = parse_role_support(args.writer_file)
    reader_support = parse_role_support(args.reader_file)
    eraser_support = parse_role_support(args.eraser_file)

    sites_rows = []
    transcript_to_sites: dict[str, list[int]] = defaultdict(list)

    for key in sorted(sites_by_key):
        site = sites_by_key[key]
        mod_id = site["mod_id"]

        w = int(writer_support.get(mod_id, 0))
        r = int(reader_support.get(mod_id, 0))
        e = int(eraser_support.get(mod_id, 0))

        row = {
            "mod_id": mod_id,
            "transcript_id": site["transcript_id"],
            "site_pos": int(site["site_pos"]),
            "site_base": site["site_base"],
            "m6a_pu_label": 1,
            "writer_pu_label": 1 if w > 0 else -1,
            "reader_pu_label": 1 if r > 0 else -1,
            "eraser_pu_label": 1 if e > 0 else -1,
            "writer_support_count": w,
            "reader_support_count": r,
            "eraser_support_count": e,
        }
        sites_rows.append(row)
        transcript_to_sites[row["transcript_id"]].append(int(row["site_pos"]))

    transcripts_rows = []
    for transcript_id, positions in transcript_to_sites.items():
        seq = transcript_seqs.get(transcript_id, "")
        if not seq:
            continue
        unique_pos = sorted(set(int(x) for x in positions))
        row = {
            "transcript_id": transcript_id,
            "full_sequence": seq,
            "seq_len": int(len(seq)),
            "m6a_positions": unique_pos,
        }
        transcripts_rows.append(row)

    transcripts_df = pd.DataFrame(transcripts_rows)
    sites_df = pd.DataFrame(sites_rows)

    split_payload = {
        "train": [],
        "val": [],
        "test": [],
        "holdout_long": [],
    }

    for row in transcripts_rows:
        transcript_id = row["transcript_id"]
        seq_len = int(row["seq_len"])
        if seq_len > args.max_len:
            split_payload["holdout_long"].append(transcript_id)
            continue
        split_name = deterministic_split(
            transcript_id=transcript_id,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        split_payload[split_name].append(transcript_id)

    for key in split_payload:
        split_payload[key] = sorted(split_payload[key])

    split_payload["meta"] = { # type: ignore
        "max_len": args.max_len,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "counts": {k: len(v) for k, v in split_payload.items() if isinstance(v, list)},
    }

    write_parquet(sites_df, args.sites_out)
    write_parquet(transcripts_df, args.transcripts_out)

    splits_path = Path(args.splits_out)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2)

    return {
        "fasta_stats": dict(fasta_stats),
        "site_stats": dict(site_stats),
        "n_sites": int(len(sites_df)),
        "n_transcripts": int(len(transcripts_df)),
        "split_counts": split_payload["meta"]["counts"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build m6A multitask dataset tables.")
    parser.add_argument(
        "--m6a-file",
        default=str(REPO_ROOT / "data/raw/human.hg38.m6A.result.col29.bed"),
    )
    parser.add_argument(
        "--writer-file",
        default=str(REPO_ROOT / "data/raw/human.hg38.modrbp.m6A.writer.bed"),
    )
    parser.add_argument(
        "--reader-file",
        default=str(REPO_ROOT / "data/raw/human.hg38.modrbp.m6A.reader.bed"),
    )
    parser.add_argument(
        "--eraser-file",
        default=str(REPO_ROOT / "data/raw/human.hg38.modrbp.m6A.eraser.bed"),
    )
    parser.add_argument(
        "--fasta",
        default=str(REPO_ROOT / "data/raw/Homo_sapiens.GRCh38.cdna.all.fa.gz"),
    )
    parser.add_argument(
        "--sites-out",
        default=str(REPO_ROOT / "data/processed/m6a_multitask_sites.parquet"),
    )
    parser.add_argument(
        "--transcripts-out",
        default=str(REPO_ROOT / "data/processed/m6a_multitask_transcripts.parquet"),
    )
    parser.add_argument(
        "--splits-out",
        default=str(REPO_ROOT / "data/processed/m6a_multitask_splits.json"),
    )
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-genes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1 and both non-negative.")

    summary = build_tables(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
