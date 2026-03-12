from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

from RMBase import (
    parse_sites_RMBase,
    build_sites_rows,
    # build_transcript_rows,
)
from directRMDB import (
    parse_sites_directRMDB,
    build_sites_rows_directRMDB,
    # build_transcript_rows_directRMDB,
)
from RMPore import (
    parse_sites_RMPore,
    build_sites_rows_RMPore,
    # build_transcript_rows_RMPore,
)
from Atlas import (
    parse_sites_Atlas,
    build_sites_rows_Atlas,
    # build_transcript_rows_Atlas,
)
from merge_and_split import (
    merge_all_sites,
    rebuild_transcripts_from_sites,
    stratified_split,
    print_merge_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# FASTA parsing
# ─────────────────────────────────────────────────────────────────────────────

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

    def _store(sid, seq):
        stats["records"] += 1
        if sid in seqs:
            stats["duplicates"] += 1
            existing = seqs[sid]
            if seq == existing:
                stats["duplicate_exact"] += 1
            elif len(seq) > len(existing):
                seqs[sid] = seq
                stats["duplicate_replaced"] += 1
            elif len(seq) == len(existing):
                stats["duplicate_conflict"] += 1
        else:
            seqs[sid] = seq

    with open_text(path) as handle:
        seq_id = None
        chunks = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    _store(seq_id, "".join(chunks).upper().replace("T", "U"))
                header = line[1:].split()[0]
                seq_id = normalize_transcript_id(header)
                chunks = []
            else:
                chunks.append(line)
        if seq_id is not None:
            _store(seq_id, "".join(chunks).upper().replace("T", "U"))

    return seqs, stats


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_counter(title: str, counter_obj) -> None:
    print(f"\n[{title}]")
    if not counter_obj:
        print("  (empty)")
        return
    for k in sorted(counter_obj):
        print(f"  {k}: {counter_obj[k]}")


def write_parquet(df: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def print_sites_summary(sites_rows: list[dict]) -> None:
    print("\n[Sites summary]")
    print(f"  total_site_rows: {len(sites_rows)}")

    mod_type_counter = Counter()
    site_base_counter = Counter()
    writer_label_counter = Counter()
    reader_label_counter = Counter()
    eraser_label_counter = Counter()
    writer_supported = 0
    reader_supported = 0
    eraser_supported = 0

    for row in sites_rows:
        mod_type_counter[row["mod_type"]] += 1
        site_base_counter[row["site_base"]] += 1
        writer_label_counter[row["writer_pu_label"]] += 1
        reader_label_counter[row["reader_pu_label"]] += 1
        eraser_label_counter[row["eraser_pu_label"]] += 1
        if row["writer_support_count"] > 0:
            writer_supported += 1
        if row["reader_support_count"] > 0:
            reader_supported += 1
        if row["eraser_support_count"] > 0:
            eraser_supported += 1

    print("  by mod_type:")
    for k in sorted(mod_type_counter):
        print(f"    {k}: {mod_type_counter[k]}")
    print("  by site_base:")
    for k in sorted(site_base_counter):
        print(f"    {k}: {site_base_counter[k]}")
    print("  writer_pu_label:")
    for k in sorted(writer_label_counter):
        print(f"    {k}: {writer_label_counter[k]}")
    print("  reader_pu_label:")
    for k in sorted(reader_label_counter):
        print(f"    {k}: {reader_label_counter[k]}")
    print("  eraser_pu_label:")
    for k in sorted(eraser_label_counter):
        print(f"    {k}: {eraser_label_counter[k]}")
    print(f"  writer_supported_rows: {writer_supported}")
    print(f"  reader_supported_rows: {reader_supported}")
    print(f"  eraser_supported_rows: {eraser_supported}")


def print_rbp_summary(sites_rows: list[dict]) -> None:
    print("\n[RBP summary by mod_type]")
    summary = defaultdict(lambda: {"writer": set(), "reader": set(), "eraser": set()})
    for row in sites_rows:
        mod_type = row["mod_type"]
        rbp_name = row.get("rbp_name", {})
        for role in ["writer", "reader", "eraser"]:
            names = rbp_name.get(role)
            if names:
                summary[mod_type][role].update(names)
    for mod_type in sorted(summary):
        print(f"  {mod_type}:")
        for role in ["writer", "reader", "eraser"]:
            names = sorted(summary[mod_type][role])
            print(f"    {role}_rbps ({len(names)}): {', '.join(names) if names else '-'}")


def print_transcripts_summary(transcripts_rows: list[dict]) -> None:
    print("\n[Transcripts summary]")
    print(f"  total_transcripts: {len(transcripts_rows)}")
    if not transcripts_rows:
        return
    seq_lens = [row["seq_len"] for row in transcripts_rows]
    n_sites_per_tx = [len(row["mod_positions"]) for row in transcripts_rows]
    print(f"  seq_len_min: {min(seq_lens)}")
    print(f"  seq_len_max: {max(seq_lens)}")
    print(f"  seq_len_mean: {sum(seq_lens) / len(seq_lens):.2f}")
    print(f"  seq_len_median: {float(np.median(seq_lens)):.2f}")
    print(f"  mod_sites_per_tx_min: {min(n_sites_per_tx)}")
    print(f"  mod_sites_per_tx_max: {max(n_sites_per_tx)}")
    print(f"  mod_sites_per_tx_mean: {sum(n_sites_per_tx) / len(n_sites_per_tx):.2f}")
    print(f"  mod_sites_per_tx_median: {float(np.median(n_sites_per_tx)):.2f}")
    multi_type_positions = 0
    total_positions = 0
    mod_type_counter = Counter()
    for row in transcripts_rows:
        for types_here in row["mod_types"]:
            total_positions += 1
            if len(types_here) > 1:
                multi_type_positions += 1
            for t in types_here:
                mod_type_counter[t] += 1
    print(f"  total_unique_positions: {total_positions}")
    print(f"  multi_type_positions: {multi_type_positions}")
    print("  flattened mod_type counts:")
    for k in sorted(mod_type_counter):
        print(f"    {k}: {mod_type_counter[k]}")


def print_position_type_multiplicity(transcripts_rows: list[dict]) -> None:
    print("\n[Position type multiplicity]")
    multiplicity_counter = Counter()
    for row in transcripts_rows:
        for types_here in row["mod_types"]:
            multiplicity_counter[len(types_here)] += 1
    for k in sorted(multiplicity_counter):
        print(f"  positions_with_{k}_type(s): {multiplicity_counter[k]}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_rmbase(transcript_seqs, args, verbose):
    sites_by_key, site_stats = parse_sites_RMBase(
        mod_types=["m6A", "m1A", "m5C", "pseu"],
        transcript_seqs=transcript_seqs,
        protein_coding_only=not args.all_genes,
    )
    if verbose:
        print_counter("RMBase site parsing stats", site_stats)
        print(f"\n[RMBase summary]  kept: {len(sites_by_key)}")
    sites_rows, _ = build_sites_rows(sites_by_key)
    return sites_rows


def _build_directrmdb(transcript_seqs, args, verbose):
    sites_by_key, site_stats = parse_sites_directRMDB(
        transcript_seqs=transcript_seqs,
        mod_types=["m6A", "m1A", "m5C", "Psi"],
        protein_coding_only=not args.all_genes,
        verbose=verbose,
    )
    if verbose:
        print_counter("directRMDB site parsing stats", site_stats)
        print(f"\n[directRMDB summary]  kept: {len(sites_by_key)}")
    sites_rows, _ = build_sites_rows_directRMDB(sites_by_key)
    return sites_rows


def _build_rmpore(transcript_seqs, args, verbose):
    sites_by_key, site_stats = parse_sites_RMPore(
        transcript_seqs=transcript_seqs,
        mod_types=["m6A", "m1A", "m5C", "psU"],
        protein_coding_only=not args.all_genes,
        verbose=verbose,
    )
    if verbose:
        print_counter("RMPore site parsing stats", site_stats)
        print(f"\n[RMPore summary]  kept: {len(sites_by_key)}")
    sites_rows, _ = build_sites_rows_RMPore(sites_by_key)
    return sites_rows


def _build_atlas(transcript_seqs, args, verbose):
    sites_by_key, site_stats = parse_sites_Atlas(
        transcript_seqs=transcript_seqs,
        protein_coding_only=not args.all_genes,
        verbose=verbose,
    )
    if verbose:
        print_counter("Atlas site parsing stats", site_stats)
        print(f"\n[Atlas summary]  kept: {len(sites_by_key)}")
    sites_rows, _ = build_sites_rows_Atlas(sites_by_key)
    return sites_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_tables(args: argparse.Namespace, verbose: bool = True) -> dict:
    transcript_seqs, fasta_stats = parse_fasta(args.fasta)
    if verbose:
        print_counter("FASTA stats", fasta_stats)
        print(f"\n[FASTA summary]\n  unique_transcripts: {len(transcript_seqs)}")

    # ── single dataset mode ──
    if args.dataset != "all":
        if args.dataset == "RMBase":
            sites_rows = _build_rmbase(transcript_seqs, args, verbose)
        elif args.dataset == "directRMDB":
            sites_rows = _build_directrmdb(transcript_seqs, args, verbose)
        elif args.dataset == "RMPore":
            sites_rows = _build_rmpore(transcript_seqs, args, verbose)
        elif args.dataset == "Atlas":
            sites_rows = _build_atlas(transcript_seqs, args, verbose)

        transcripts_rows = rebuild_transcripts_from_sites(
            sites_rows, transcript_seqs, verbose,
        )

        if verbose:
            print_sites_summary(sites_rows)
            if args.dataset != "Atlas":
                print_rbp_summary(sites_rows)
            print_transcripts_summary(transcripts_rows)
            print_position_type_multiplicity(transcripts_rows)

        return {}

    # ── merge all mode ──
    print("\n" + "=" * 70)
    print("  MERGE ALL DATASETS")
    print("=" * 70)

    source_rows = {}

    print("\n--- [1/4] RMBase ---")
    source_rows["RMBase"] = _build_rmbase(transcript_seqs, args, verbose)

    print("\n--- [2/4] directRMDB ---")
    source_rows["directRMDB"] = _build_directrmdb(transcript_seqs, args, verbose)

    print("\n--- [3/4] RMPore ---")
    source_rows["RMPore"] = _build_rmpore(transcript_seqs, args, verbose)

    print("\n--- [4/4] Atlas ---")
    source_rows["Atlas"] = _build_atlas(transcript_seqs, args, verbose)

    # ── merge ──
    merged_sites = merge_all_sites(
        source_rows["RMBase"],
        source_rows["directRMDB"],
        source_rows["RMPore"],
        source_rows["Atlas"],
        verbose=verbose,
    )

    transcripts_rows = rebuild_transcripts_from_sites(
        merged_sites, transcript_seqs, verbose,
    )

    if verbose:
        print_merge_summary(
            merged_sites, transcripts_rows,
            {k: len(v) for k, v in source_rows.items()},
        )
        print_sites_summary(merged_sites)
        print_rbp_summary(merged_sites)
        print_transcripts_summary(transcripts_rows)
        print_position_type_multiplicity(transcripts_rows)

    # ── save sites & transcripts ──
    sites_df = pd.DataFrame(merged_sites)
    transcripts_df = pd.DataFrame(transcripts_rows)
    write_parquet(sites_df, args.sites_out)
    write_parquet(transcripts_df, args.transcripts_out)
    if verbose:
        print(f"\n[Saved] sites → {args.sites_out}  ({len(sites_df)} rows)")
        print(f"[Saved] transcripts → {args.transcripts_out}  ({len(transcripts_df)} rows)")

    # ── stratified split ──
    split_payload = stratified_split(
        transcripts_rows=transcripts_rows,
        merged_sites=merged_sites,
        max_len=args.max_len,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        verbose=verbose,
    )

    splits_path = Path(args.splits_out)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("w", encoding="utf-8") as f:
        json.dump(split_payload, f, indent=2)
    if verbose:
        print(f"[Saved] splits → {args.splits_out}")

    return split_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multitask dataset tables.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["RMBase", "directRMDB", "RMPore", "Atlas", "all"],
        default="all",
    )
    parser.add_argument(
        "--fasta",
        default=str(REPO_ROOT / "data/raw/Homo_sapiens.GRCh38.cdna.all.fa.gz"),
    )
    parser.add_argument(
        "--gtf",
        default=str(REPO_ROOT / "data/raw/Homo_sapiens.GRCh38.115.gtf.gz"),
    )
    parser.add_argument(
        "--atlas",
        default=str(REPO_ROOT / "data/raw/Atlas/m1A_Human_Basic_Site_Information.txt"),
    )
    parser.add_argument(
        "--sites-out",
        default=str(REPO_ROOT / "data/processed/all_multitask_sites.parquet"),
    )
    parser.add_argument(
        "--transcripts-out",
        default=str(REPO_ROOT / "data/processed/all_multitask_transcripts.parquet"),
    )
    parser.add_argument(
        "--splits-out",
        default=str(REPO_ROOT / "data/processed/all_multitask_splits.json"),
    )
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--all-genes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1 and both non-negative.")

    summary = build_tables(args)
    if summary:
        print(json.dumps(summary.get("meta", {}), indent=2))


if __name__ == "__main__":
    main()