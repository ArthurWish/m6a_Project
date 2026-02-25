#!/usr/bin/env python3
import argparse
import csv
import gzip
import os
import sys
from pathlib import Path


YTH_READERS = {"YTHDF1", "YTHDF2", "YTHDF3", "YTHDC1", "YTHDC2"}
YTH_FOLDER_MAP = {
    "YTHDF1": "DF1",
    "YTHDF2": "DF2",
    "YTHDF3": "DF3",
    "YTHDC1": "DC1",
    "YTHDC2": "DC2",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def normalize_transcript_id(raw_id):
    if not raw_id:
        return ""
    if raw_id.lower() == "na":
        return ""
    return raw_id.split(".")[0]


def parse_transcript_ids(field):
    if not field:
        return []
    ids = []
    for item in field.split(","):
        item = item.strip()
        if not item or item.lower() == "na":
            continue
        ids.append(normalize_transcript_id(item))
    return [item for item in ids if item]


def parse_fasta(path):
    seqs = {}
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
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    seq = "".join(chunks).strip().upper().replace("T", "U")
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
                header = line[1:].strip().split()[0]
                seq_id = normalize_transcript_id(header)
                chunks = []
            else:
                chunks.append(line.strip())
        if seq_id is not None:
            seq = "".join(chunks).strip().upper().replace("T", "U")
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


def write_clean_fasta(path, seqs):
    with open(path, "w") as out_handle:
        for seq_id in sorted(seqs):
            out_handle.write(f">{seq_id}\n{seqs[seq_id]}\n")


def load_positive_ids(reader_path, require_protein_coding=True):
    positive_ids = set()
    total = 0
    kept = 0
    missing_transcript = 0
    with open_text(reader_path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            total += 1
            if len(row) < 12:
                continue
            rbp = row[1]
            if rbp not in YTH_READERS:
                continue
            transcript_ids = parse_transcript_ids(row[11])
            if not transcript_ids:
                missing_transcript += 1
                continue
            if require_protein_coding and len(row) > 13:
                gene_types = row[13].split(",")
                if "protein_coding" not in gene_types:
                    continue
            m6a_id = row[7]
            positive_ids.add(m6a_id)
            kept += 1
    return positive_ids, total, kept, missing_transcript


def split_rbp_by_yth(reader_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    writers = {}
    handles = {}
    counts = {rbp: 0 for rbp in YTH_READERS}
    stats = {
        "total_rows": 0,
        "invalid_rows": 0,
        "non_yth_rows": 0,
        "missing_transcript": 0,
    }
    for rbp in YTH_READERS:
        folder = os.path.join(output_dir, YTH_FOLDER_MAP[rbp])
        os.makedirs(folder, exist_ok=True)
        out_path = os.path.join(folder, "rbp.tsv")
        handle = open(out_path, "w", newline="")
        handles[rbp] = handle
        writers[rbp] = csv.writer(handle, delimiter="\t", lineterminator="\n")

    try:
        with open_text(reader_path) as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                stats["total_rows"] += 1
                if len(row) < 12:
                    stats["invalid_rows"] += 1
                    continue
                rbp = row[1]
                if rbp not in YTH_READERS:
                    stats["non_yth_rows"] += 1
                    continue
                transcript_ids = parse_transcript_ids(row[11])
                if not transcript_ids:
                    stats["missing_transcript"] += 1
                    continue
                writers[rbp].writerow(row)
                counts[rbp] += 1
    finally:
        for handle in handles.values():
            handle.close()

    return stats, counts


def build_dataset(
    m6a_path,
    positive_ids,
    transcript_seqs,
    output_path,
    window_size=None,
):
    total = 0
    kept = 0
    missing_transcript = 0
    missing_pattern = 0
    non_unique_pattern = 0
    non_protein = 0
    invalid_rows = 0

    with open(output_path, "w", newline="") as out_handle:
        writer = csv.writer(out_handle)
        header = [
            "modId",
            "transcriptId",
            "full_sequence",
            "m6A_position_index",
            "label",
        ]
        if window_size:
            header.append("window_sequence")
        writer.writerow(header)

        with open_text(m6a_path) as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                total += 1
                if len(row) < 19:
                    invalid_rows += 1
                    continue
                gene_type = row[16]
                if gene_type != "protein_coding":
                    non_protein += 1
                    continue
                mod_id = row[3]
                transcript_id = normalize_transcript_id(row[14])
                short_seq = row[18]

                label = 1 if mod_id in positive_ids else 0

                full_seq = transcript_seqs.get(transcript_id)
                if not full_seq:
                    missing_transcript += 1
                    continue

                pattern = short_seq.upper().replace("T", "U")
                site_index = full_seq.find(pattern)
                if site_index == -1:
                    missing_pattern += 1
                    continue
                if full_seq.count(pattern) != 1:
                    non_unique_pattern += 1
                    continue

                center_offset = len(pattern) // 2
                absolute_pos = site_index + center_offset

                row_out = [
                    mod_id,
                    transcript_id,
                    full_seq,
                    absolute_pos,
                    label,
                ]

                if window_size:
                    half = window_size // 2
                    start = absolute_pos - half
                    end = absolute_pos + half + (1 if window_size % 2 else 0)
                    left_pad = max(0, -start)
                    right_pad = max(0, end - len(full_seq))
                    start = max(0, start)
                    end = min(len(full_seq), end)
                    window = ("N" * left_pad) + full_seq[start:end] + ("N" * right_pad)
                    row_out.append(window)

                writer.writerow(row_out)
                kept += 1

    stats = {
        "total_m6a_rows": total,
        "kept": kept,
        "invalid_rows": invalid_rows,
        "non_protein_coding": non_protein,
        "missing_transcript": missing_transcript,
        "missing_pattern": missing_pattern,
        "non_unique_pattern": non_unique_pattern,
    }
    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build m6A-YTH dataset from RMBase and Ensembl cDNA."
    )
    parser.add_argument(
        "--m6a-file",
        default=str(REPO_ROOT / "data/raw/human.hg38.m6A.result.col29.bed"),
        help="RMBase m6A sites file (tab-delimited).",
    )
    parser.add_argument(
        "--reader-file",
        default=str(REPO_ROOT / "data/raw/human.hg38.modrbp.m6A.reader.bed"),
        help="RMBase reader binding file (tab-delimited).",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Ensembl cDNA FASTA file (.fa or .fa.gz).",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data/processed/m6A_YTH_dataset.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--rbp-out-dir",
        default=str(REPO_ROOT / "yth_rbp"),
        help="Output directory for YTH RBP subtype folders.",
    )
    parser.add_argument(
        "--clean-fasta-out",
        default="",
        help="Optional path to write de-wrapped FASTA (one-line sequences).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="Optional window size around the m6A site (0 to disable).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    window_size = args.window_size if args.window_size > 0 else None

    print("Filtering RBP table and splitting by YTH subtype...", file=sys.stderr)
    rbp_stats, rbp_counts = split_rbp_by_yth(args.reader_file, args.rbp_out_dir)
    for rbp in sorted(rbp_counts):
        print(
            f"RBP {rbp} rows kept: {rbp_counts[rbp]}",
            file=sys.stderr,
        )
    for key in sorted(rbp_stats):
        print(f"rbp_{key}: {rbp_stats[key]}", file=sys.stderr)

    print("Loading YTH positive site IDs...", file=sys.stderr)
    positive_ids, total_readers, kept_readers, missing_transcript = load_positive_ids(
        args.reader_file
    )
    print(
        f"Reader rows: {total_readers}, YTH protein_coding: {kept_readers}, "
        f"unique m6A IDs: {len(positive_ids)}",
        file=sys.stderr,
    )
    print(
        f"Reader rows missing transcript IDs: {missing_transcript}",
        file=sys.stderr,
    )

    print("Loading cDNA sequences...", file=sys.stderr)
    transcript_seqs, fasta_stats = parse_fasta(args.fasta)
    if args.clean_fasta_out:
        write_clean_fasta(args.clean_fasta_out, transcript_seqs)
    print(
        "Transcripts loaded: "
        f"{len(transcript_seqs)}, duplicates: {fasta_stats['duplicates']}, "
        f"replaced: {fasta_stats['duplicate_replaced']}, "
        f"conflicts: {fasta_stats['duplicate_conflict']}",
        file=sys.stderr,
    )

    print("Building dataset...", file=sys.stderr)
    stats = build_dataset(
        args.m6a_file,
        positive_ids,
        transcript_seqs,
        args.output,
        window_size=window_size,
    )
    print("Done.", file=sys.stderr)
    for key in sorted(stats):
        print(f"{key}: {stats[key]}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
