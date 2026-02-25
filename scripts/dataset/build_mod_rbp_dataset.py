#!/usr/bin/env python3
"""
Build a combined modification + RBP binding dataset for m6A/m5C/pseudo.

Inputs:
  - Mod-site files (RMBase 29-column .bed): human.hg38.<mod>.result.col29.bed
  - ModRBP binding files (RMBase 17-column .bed): human.hg38.modrbp.<mod>.<role>.bed
  - Ensembl cDNA FASTA (fa/fa.gz)

Outputs:
  - data/processed/mod_rbp_dataset.csv
  - outputs/reports/mod_rbp_site_table.csv
  - outputs/reports/mod_rbp_stats_summary.csv
  - outputs/reports/mod_rbp_stats_summary.md
"""
import argparse
import csv
import gzip
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

MODS = ("m6A", "m5C", "pseudo")
ROLE_ORDER = ("writer", "reader", "eraser")
MOD_ROLE_MAP = {
    "m6A": ("writer", "reader", "eraser"),
    "m5C": ("writer", "reader"),
    "pseudo": ("writer",),
}


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


def normalize_mod_type(raw):
    if not raw:
        return ""
    key = raw.strip().lower()
    if key == "m6a":
        return "m6A"
    if key == "m5c":
        return "m5C"
    if key in ("pseudo", "pseudouridine", "psi", "y"):
        return "pseudo"
    return raw


def normalize_role(raw):
    if not raw:
        return ""
    return raw.strip().lower()


def default_mod_site_path(mod_name, mod_site_dir):
    return os.path.join(mod_site_dir, f"human.hg38.{mod_name}.result.col29.bed")


def default_rbp_path(mod_name, role, rbp_dir):
    return os.path.join(rbp_dir, f"human.hg38.modrbp.{mod_name}.{role}.bed")


def load_rbp_bindings(mods, rbp_dir):
    bindings = {
        mod: defaultdict(lambda: {role: set() for role in ROLE_ORDER}) for mod in mods
    }
    stats = {
        "files_found": 0,
        "rows_total": 0,
        "rows_kept": 0,
        "rows_invalid": 0,
        "rows_unknown_mod": 0,
        "rows_role_mismatch": 0,
    }
    for mod in mods:
        roles = MOD_ROLE_MAP.get(mod, ROLE_ORDER)
        for role in roles:
            path = default_rbp_path(mod, role, rbp_dir)
            if not os.path.exists(path):
                print(f"[warn] missing RBP file: {path}", file=sys.stderr)
                continue
            stats["files_found"] += 1
            with open_text(path) as handle:
                reader = csv.reader(handle, delimiter="\t")
                for row in reader:
                    if not row:
                        continue
                    stats["rows_total"] += 1
                    if len(row) < 17:
                        stats["rows_invalid"] += 1
                        continue
                    rbp_name = row[1].strip()
                    mod_id = row[7].strip()
                    row_mod = normalize_mod_type(row[9])
                    row_role = normalize_role(row[16]) if len(row) > 16 else ""
                    if row_mod and row_mod != mod:
                        stats["rows_unknown_mod"] += 1
                        continue
                    if row_role and row_role in ROLE_ORDER and row_role != role:
                        stats["rows_role_mismatch"] += 1
                    if not rbp_name or not mod_id:
                        stats["rows_invalid"] += 1
                        continue
                    bindings[mod][mod_id][role].add(rbp_name)
                    stats["rows_kept"] += 1
    return bindings, stats


def build_dataset(
    mod_site_path,
    mod_type,
    transcript_seqs,
    bindings,
    protein_coding_only=True,
):
    rows = []
    stats = {
        "total_mod_rows": 0,
        "kept": 0,
        "invalid_rows": 0,
        "non_protein_coding": 0,
        "missing_transcript": 0,
        "missing_pattern": 0,
        "non_unique_pattern": 0,
    }
    with open_text(mod_site_path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            stats["total_mod_rows"] += 1
            if len(row) < 19:
                stats["invalid_rows"] += 1
                continue
            gene_type = row[16]
            if protein_coding_only and gene_type != "protein_coding":
                stats["non_protein_coding"] += 1
                continue
            mod_id = row[3].strip()
            transcript_id = normalize_transcript_id(row[14])
            short_seq = row[18].strip()
            region = row[17].strip() if len(row) > 17 else ""

            full_seq = transcript_seqs.get(transcript_id)
            if not full_seq:
                stats["missing_transcript"] += 1
                continue

            pattern = short_seq.upper().replace("T", "U")
            site_index = full_seq.find(pattern)
            if site_index == -1:
                stats["missing_pattern"] += 1
                continue
            if full_seq.count(pattern) != 1:
                stats["non_unique_pattern"] += 1
                continue

            center_offset = len(pattern) // 2
            absolute_pos = site_index + center_offset

            role_map = bindings.get(mod_type, {}).get(mod_id, {})
            writer_names = sorted(role_map.get("writer", set()))
            reader_names = sorted(role_map.get("reader", set()))
            eraser_names = sorted(role_map.get("eraser", set()))
            all_names = sorted(set(writer_names) | set(reader_names) | set(eraser_names))

            role_flags = [role for role in ROLE_ORDER if role_map.get(role)]
            role_combo = "+".join(role_flags) if role_flags else ""

            row_out = {
                "modId": mod_id,
                "modType": mod_type,
                "transcriptId": transcript_id,
                "full_sequence": full_seq,
                "mod_position_index": absolute_pos,
                "short_sequence": pattern,
                "gene_type": gene_type,
                "region": region,
                "rbp_names": ";".join(all_names),
                "rbp_writer_names": ";".join(writer_names),
                "rbp_reader_names": ";".join(reader_names),
                "rbp_eraser_names": ";".join(eraser_names),
                "rbp_name_count": len(all_names),
                "rbp_role_count": len(role_flags),
                "writer_count": len(writer_names),
                "reader_count": len(reader_names),
                "eraser_count": len(eraser_names),
                "rbp_role_combo": role_combo,
                "has_multi_rbp": 1 if len(all_names) >= 2 else 0,
                "has_multi_role": 1 if len(role_flags) >= 2 else 0,
            }
            rows.append(row_out)
            stats["kept"] += 1
    return rows, stats


def summarize_sites(rows, mod_type):
    total = len(rows)
    with_any = sum(1 for r in rows if r["rbp_name_count"] > 0)
    summary = {
        "modType": mod_type,
        "total_sites": total,
        "sites_with_any_rbp": with_any,
    }
    return summary


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_markdown(path, summaries, rbp_rows, top_n=20):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("# Mod RBP stats summary\n\n")
        for mod in summaries:
            handle.write(f"## {mod}\n\n")
            summary = summaries[mod]
            handle.write("Summary:\n\n")
            handle.write(f"- total_sites: {summary['total_sites']}\n")
            handle.write(f"- sites_with_any_rbp: {summary['sites_with_any_rbp']}\n")
            handle.write("\nTop RBP by site count (see CSV for full list):\n\n")
            rows = [r for r in rbp_rows if r["modType"] == mod]
            rows = sorted(rows, key=lambda r: (-int(r["site_count"]), r["rbp_name"]))
            for row in rows[:top_n]:
                handle.write(
                    f"- {row['rbp_name']} ({row['role']}): {row['site_count']}\n"
                )
            handle.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a mod + RBP binding dataset for m6A/m5C/pseudo."
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Ensembl cDNA FASTA file (.fa or .fa.gz).",
    )
    parser.add_argument(
        "--mod-site-dir",
        default=str(REPO_ROOT / "data/raw"),
        help="Directory containing mod-site .bed files.",
    )
    parser.add_argument(
        "--rbp-dir",
        default=str(REPO_ROOT / "data/raw"),
        help="Directory containing modRBP .bed files.",
    )
    parser.add_argument(
        "--mods",
        default=",".join(MODS),
        help="Comma-separated list of mods to process (m6A,m5C,pseudo).",
    )
    parser.add_argument(
        "--all-genes",
        action="store_true",
        help="Include all gene types (default is protein-coding only).",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data/processed/mod_rbp_dataset.csv"),
        help="Output dataset CSV path.",
    )
    parser.add_argument(
        "--site-stats-out",
        default=str(REPO_ROOT / "outputs/reports/mod_rbp_site_table.csv"),
        help="Output per-site stats table.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(REPO_ROOT / "outputs/reports/mod_rbp_stats_summary.csv"),
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--summary-md-out",
        default=str(REPO_ROOT / "outputs/reports/mod_rbp_stats_summary.md"),
        help="Output summary Markdown path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mods = [m.strip() for m in args.mods.split(",") if m.strip()]
    valid_mods = [m for m in mods if m in MODS]
    if not valid_mods:
        raise SystemExit("No valid mods provided. Use m6A,m5C,pseudo.")

    protein_coding_only = not bool(args.all_genes)

    print("Loading RBP binding files...", file=sys.stderr)
    bindings, binding_stats = load_rbp_bindings(valid_mods, args.rbp_dir)
    for key in sorted(binding_stats):
        print(f"rbp_{key}: {binding_stats[key]}", file=sys.stderr)

    print("Loading cDNA sequences...", file=sys.stderr)
    transcript_seqs, fasta_stats = parse_fasta(args.fasta)
    print(
        "Transcripts loaded: "
        f"{len(transcript_seqs)}, duplicates: {fasta_stats['duplicates']}, "
        f"replaced: {fasta_stats['duplicate_replaced']}, "
        f"conflicts: {fasta_stats['duplicate_conflict']}",
        file=sys.stderr,
    )

    dataset_rows = []
    site_table_rows = []
    summaries = {}
    rbp_site_counts = Counter()

    for mod in valid_mods:
        mod_site_path = default_mod_site_path(mod, args.mod_site_dir)
        if not os.path.exists(mod_site_path):
            print(f"[warn] missing mod-site file: {mod_site_path}", file=sys.stderr)
            continue
        print(f"Building dataset for {mod}...", file=sys.stderr)
        rows, stats = build_dataset(
            mod_site_path,
            mod,
            transcript_seqs,
            bindings,
            protein_coding_only=protein_coding_only,
        )
        for key in sorted(stats):
            print(f"{mod}_{key}: {stats[key]}", file=sys.stderr)
        dataset_rows.extend(rows)
        for row in rows:
            site_table_rows.append(
                {
                    "modId": row["modId"],
                    "modType": row["modType"],
                    "transcriptId": row["transcriptId"],
                    "mod_position_index": row["mod_position_index"],
                    "gene_type": row["gene_type"],
                    "region": row["region"],
                    "rbp_names": row["rbp_names"],
                    "rbp_name_count": row["rbp_name_count"],
                    "writer_count": row["writer_count"],
                    "reader_count": row["reader_count"],
                    "eraser_count": row["eraser_count"],
                    "rbp_role_combo": row["rbp_role_combo"],
                    "has_multi_rbp": row["has_multi_rbp"],
                    "has_multi_role": row["has_multi_role"],
                }
            )
            for name in [n for n in row["rbp_writer_names"].split(";") if n]:
                rbp_site_counts[(mod, "writer", name)] += 1
            for name in [n for n in row["rbp_reader_names"].split(";") if n]:
                rbp_site_counts[(mod, "reader", name)] += 1
            for name in [n for n in row["rbp_eraser_names"].split(";") if n]:
                rbp_site_counts[(mod, "eraser", name)] += 1
        summary = summarize_sites(rows, mod)
        summaries[mod] = summary

    if not dataset_rows:
        print("No dataset rows generated. Check input files.", file=sys.stderr)
        return

    dataset_fields = [
        "modId",
        "modType",
        "transcriptId",
        "full_sequence",
        "mod_position_index",
        "short_sequence",
        "gene_type",
        "region",
        "rbp_names",
        "rbp_writer_names",
        "rbp_reader_names",
        "rbp_eraser_names",
        "rbp_name_count",
        "rbp_role_count",
        "writer_count",
        "reader_count",
        "eraser_count",
        "rbp_role_combo",
        "has_multi_rbp",
        "has_multi_role",
    ]
    print(f"Writing dataset to {args.output}...", file=sys.stderr)
    write_csv(args.output, dataset_fields, dataset_rows)

    site_fields = [
        "modId",
        "modType",
        "transcriptId",
        "mod_position_index",
        "gene_type",
        "region",
        "rbp_names",
        "rbp_name_count",
        "writer_count",
        "reader_count",
        "eraser_count",
        "rbp_role_combo",
        "has_multi_rbp",
        "has_multi_role",
    ]
    print(f"Writing site table to {args.site_stats_out}...", file=sys.stderr)
    write_csv(args.site_stats_out, site_fields, site_table_rows)

    summary_rows = []
    for mod in sorted(summaries):
        summary_rows.append(summaries[mod])
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    print(f"Writing summary CSV to {args.summary_out}...", file=sys.stderr)
    write_csv(args.summary_out, summary_fields, summary_rows)

    rbp_rows = []
    for (mod, role, name), count in sorted(
        rbp_site_counts.items(), key=lambda item: (item[0][0], item[0][1], -item[1], item[0][2])
    ):
        rbp_rows.append(
            {
                "modType": mod,
                "role": role,
                "rbp_name": name,
                "site_count": count,
            }
        )
    rbp_counts_out = args.summary_out.replace(
        "mod_rbp_stats_summary.csv", "mod_rbp_rbp_counts.csv"
    )
    print(f"Writing RBP counts CSV to {rbp_counts_out}...", file=sys.stderr)
    write_csv(rbp_counts_out, ["modType", "role", "rbp_name", "site_count"], rbp_rows)

    print(f"Writing summary markdown to {args.summary_md_out}...", file=sys.stderr)
    write_summary_markdown(args.summary_md_out, summaries, rbp_rows)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
