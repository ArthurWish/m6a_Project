
from __future__ import annotations

import csv
import gzip
import re
from collections import Counter, defaultdict
from typing import Optional

GENOME_PATH = "/media/scw-workspace/m6a_dataset/data/raw/directRMDB/HomoSapiens_genome.txt"
RBP_PATH    = "/media/scw-workspace/m6a_dataset/data/raw/directRMDB/HomoSapiens_RBP.txt"
GTF_PATH    = "/media/scw-workspace/m6a_dataset/data/raw/Homo_sapiens.GRCh38.115.gtf.gz"


MOD_TYPE_MAP = {
    "m6A": "m6A",
    "m1A": "m1A",
    "m5C": "m5C",
    "Psi": "pseu",
}

MOD_LETTER = {"m6A": "A", "m1A": "A", "m5C": "C", "pseu": "U"}


RBP_ROLE_BY_MOD: dict[str, dict[str, str]] = {
    "m6A": {
        # writers (METTL3-METTL14 complex + cofactors)
        "METTL3": "writer", "METTL14": "writer", "METTL16": "writer",
        "WTAP": "writer", "VIRMA": "writer", "KIAA1429": "writer",
        "RBM15": "writer", "RBM15B": "writer", "ZC3H13": "writer",
        "CBLL1": "writer", "HAKAI": "writer",
        # readers
        "YTHDF1": "reader", "YTHDF2": "reader", "YTHDF3": "reader",
        "YTHDC1": "reader", "YTHDC2": "reader",
        "IGF2BP1": "reader", "IGF2BP2": "reader", "IGF2BP3": "reader",
        "HNRNPA2B1": "reader", "HNRNPC": "reader",
        "HNRNPG": "reader", "RBMX": "reader",
        "FMR1": "reader", "FMRP": "reader", "PRRC2A": "reader",
        "EIF3A": "reader", "EIF3B": "reader", "LRPPRC": "reader",
        # erasers
        "FTO": "eraser", "ALKBH5": "eraser",
    },
    "m1A": {
        # writers
        "TRMT6": "writer", "TRMT61A": "writer",
        "TRMT61B": "writer", "TRMT10C": "writer",
        # readers
        "YTHDF1": "reader", "YTHDF2": "reader", "YTHDF3": "reader",
        # erasers
        "ALKBH1": "eraser", "ALKBH3": "eraser",
    },
    "m5C": {
        # writers (NSUN family + DNMT2/TRDMT1)
        "NSUN1": "writer", "NSUN2": "writer", "NSUN3": "writer",
        "NSUN4": "writer", "NSUN5": "writer", "NSUN6": "writer",
        "NSUN7": "writer", "DNMT2": "writer", "TRDMT1": "writer",
        # readers
        "ALYREF": "reader", "YBX1": "reader", "YBX2": "reader",
    },
    "pseu": {
        # writers (pseudouridine synthases)
        "PUS1": "writer", "PUS3": "writer", "PUS7": "writer",
        "PUS7L": "writer", "PUS10": "writer",
        "TRUB1": "writer", "TRUB2": "writer", "DKC1": "writer",
        "RPUSD1": "writer", "RPUSD2": "writer",
        "RPUSD3": "writer", "RPUSD4": "writer",
    },
}



def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def normalize_id(raw: str) -> str:
    """Strip version suffix: ENSG00000004455.16 → ENSG00000004455."""
    if not raw:
        return ""
    raw = str(raw).strip().strip('"')
    if not raw or raw.lower() == "na":
        return ""
    return raw.split(".")[0]

def _strip_chr(s: str) -> str:
    return s[3:] if s.startswith("chr") else s


def _gtf_attr(attr_str: str, key: str) -> Optional[str]:
    """Extract value from GTF attributes column: gene_id "ENSG…"; …"""
    m = re.search(key + r'\s+"([^"]+)"', attr_str)
    return m.group(1) if m else None


def parse_gtf_exons(
    gtf_path: str,
    verbose: bool = True,
) -> tuple[dict[str, list[tuple]], dict[str, list[str]]]:
    """
    Returns
    -------
    transcript_exons : {transcript_id: [(chrom, start, end, strand), ...]}
        Sorted by genomic start.  Coordinates are 1-based inclusive (GTF).
    gene_to_transcripts : {gene_id: [transcript_id, ...]}
    """
    transcript_exons: dict[str, list[tuple]] = defaultdict(list)
    gene_to_transcripts: dict[str, set[str]] = defaultdict(set)
    n = 0

    with open_text(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "exon":
                continue
            attrs = cols[8]
            gene_id = normalize_id(_gtf_attr(attrs, "gene_id"))
            tid     = normalize_id(_gtf_attr(attrs, "transcript_id"))
            if not gene_id or not tid:
                continue
            transcript_exons[tid].append(
                (cols[0], int(cols[3]), int(cols[4]), cols[6])
            )
            gene_to_transcripts[gene_id].add(tid)
            n += 1

    for tid in transcript_exons:
        transcript_exons[tid].sort(key=lambda x: x[1])

    if verbose:
        print(f"\n[GTF parsing]")
        print(f"  exon_lines:   {n}")
        print(f"  transcripts:  {len(transcript_exons)}")
        print(f"  genes:        {len(gene_to_transcripts)}")

    return (
        dict(transcript_exons),
        {k: sorted(v) for k, v in gene_to_transcripts.items()},
    )


def genomic_to_transcript_pos(
    chrom: str,
    gpos: int,
    exons: list[tuple[str, int, int, str]],
) -> Optional[int]:
    """
    Convert a 1-based genomic position to a 0-based cDNA position.

    Walks through exons in transcription order (5'→3'), accumulating
    an offset.  Returns None if the position does not fall in any exon.
    """
    if not exons:
        return None

    strand  = exons[0][3]
    chrom_n = _strip_chr(chrom)

    relevant = [(s, e) for (c, s, e, _) in exons if _strip_chr(c) == chrom_n]
    if not relevant:
        return None

    if strand == "+":
        offset = 0
        for s, e in relevant:
            if s <= gpos <= e:
                return offset + (gpos - s)
            offset += (e - s + 1)
    else:
        offset = 0
        for s, e in reversed(relevant):
            if s <= gpos <= e:
                return offset + (e - gpos)
            offset += (e - s + 1)

    return None


def classify_rbp_role(name: str, mod_type: str) -> str:
    role_map = RBP_ROLE_BY_MOD.get(mod_type, {})
    return role_map.get(name.upper(), "unknown")
    
def _split_rbp_by_role(names: list[str], mod_type: str) -> dict[str, list[str] | None]:

    buckets: dict[str, list[str]] = {
        "writer": [], "reader": [], "eraser": [],
    }
    for n in names:
        role = classify_rbp_role(n, mod_type)
        if role in buckets:
            buckets[role].append(n)
    return {k: (sorted(v) if v else None) for k, v in buckets.items()}


def parse_sites_directRMDB(
    transcript_seqs: dict[str, str],
    genome_path: str = GENOME_PATH,
    rbp_path: str    = RBP_PATH,
    gtf_path: str    = GTF_PATH,
    mod_types: list[str] | None = None,
    protein_coding_only: bool = True,
    verbose: bool = True,
) -> tuple[dict, Counter]:

    if mod_types is None:
        mod_types = ["m6A", "m1A", "m5C", "Psi"]
    allowed = set(mod_types)

    transcript_exons, gene_to_transcripts = parse_gtf_exons(gtf_path, verbose)
    rbp_by_id = parse_rbp_file(rbp_path, verbose)

    kept: dict[tuple, dict] = {}
    stats: Counter = Counter()

    if verbose:
        print(f"\n[directRMDB site parsing]")
    
    with open_text(genome_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)                        # skip header

        for row in reader:
            if not row:
                continue
            stats["total_rows"] += 1

            chrom        = row[0].strip()
            gpos         = int(row[1])            # 1-based
            mod_id       = row[9].strip()
            modification = row[10].strip()
            gene_id      = normalize_id(row[13])
            gene_biotype = row[18].strip()

            if modification not in allowed:
                stats["filtered_mod_type"] += 1
                continue
            if protein_coding_only and gene_biotype != "protein_coding":
                stats["non_protein_coding"] += 1
                continue
            mod_type      = MOD_TYPE_MAP[modification]
            expected_base = MOD_LETTER[mod_type]

            candidates = gene_to_transcripts.get(gene_id)
            if not candidates:
                stats["no_transcripts_for_gene"] += 1
                continue

            best = None
            best_len = -1

            for tid in candidates:
                seq = transcript_seqs.get(tid)
                if not seq:
                    continue
                exons = transcript_exons.get(tid)
                if not exons:
                    continue
                tpos = genomic_to_transcript_pos(chrom, gpos, exons)
                if tpos is None or tpos < 0 or tpos >= len(seq):
                    continue
                if seq[tpos] != expected_base:
                    continue
                if len(seq) > best_len:
                    best_len = len(seq)
                    best = {"tid": tid, "tpos": tpos, "base": seq[tpos]}
            if best is None:
                stats["unmapped_position"] += 1
                continue

            key = (best["tid"], best["tpos"], mod_type)
            if key in kept:
                stats["duplicate_site"] += 1
                continue

            rbp_names = rbp_by_id.get(mod_id, [])
            roles = _split_rbp_by_role(rbp_names, mod_type)

            kept[key] = {
                "mod_id":               mod_id,
                "transcript_id":        best["tid"],
                "mod_type":             mod_type,
                "site_pos":             int(best["tpos"]),
                "site_base":            best["base"],
                "writer_support_count": len(roles["writer"] or []),
                "reader_support_count": len(roles["reader"] or []),
                "eraser_support_count": len(roles["eraser"] or []),
                "rbp_names_by_role":    roles,
            }
            stats["kept"] += 1

    if verbose:
        kept_counts = Counter(v["mod_type"] for v in kept.values())
        print("  Kept site counts by mod_type:")
        for mt in ["m6A", "m1A", "m5C", "pseu"]:
            print(f"    {mt}: {kept_counts.get(mt, 0)}")

    return kept, stats

def parse_rbp_file(
    rbp_path: str,
    verbose: bool = True,
) -> dict[str, list[str]]:
    """
    Parse HomoSapiens_RBP.txt.

    Columns (0-indexed, tab-separated, with header):
        5  ID    (= mod_id, links to genome file)
        6  Name  (= RBP name)

    Returns {mod_id: [unique rbp names sorted]}.
    """
    rbp_by_id: dict[str, set[str]] = defaultdict(set)
    n = 0

    with open_text(rbp_path) as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)                        # skip header
        for row in reader:
            if not row or len(row) < 7:
                continue
            mod_id   = row[5].strip()
            rbp_name = row[6].strip()
            if mod_id and rbp_name:
                rbp_by_id[mod_id].add(rbp_name)
                n += 1

    if verbose:
        print(f"\n[RBP parsing]")
        print(f"  binding_rows:  {n}")
        print(f"  unique_mod_ids_with_rbp: {len(rbp_by_id)}")

    return {k: sorted(v) for k, v in rbp_by_id.items()}



def build_sites_rows_directRMDB(
    sites_by_key: dict,
) -> tuple[list[dict], dict[str, dict[int, set[str]]]]:
    """sites_by_key → (sites_rows, transcript_to_sites)."""
    sites_rows: list[dict] = []
    transcript_to_sites: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for key in sorted(sites_by_key):
        site  = sites_by_key[key]
        roles = site["rbp_names_by_role"]
        w = site["writer_support_count"]
        r = site["reader_support_count"]
        e = site["eraser_support_count"]
    
        row = {
            "mod_id":               site["mod_id"],
            "transcript_id":        site["transcript_id"],
            "site_pos":             int(site["site_pos"]),
            "site_base":            site["site_base"],
            "mod_type":             site["mod_type"],
            "pu_label":             1,
            "writer_pu_label":      1 if w > 0 else -1,
            "reader_pu_label":      1 if r > 0 else -1,
            "eraser_pu_label":      1 if e > 0 else -1,
            "writer_support_count": w,
            "reader_support_count": r,
            "eraser_support_count": e,
            "rbp_name": {
                "reader":  roles.get("reader"),
                "writer":  roles.get("writer"),
                "eraser":  roles.get("eraser"),
            },
        }
        sites_rows.append(row)

        transcript_to_sites[row["transcript_id"]][int(row["site_pos"])].add(row["mod_type"])


    return sites_rows, transcript_to_sites


def build_transcript_rows_directRMDB(
    transcript_to_sites: dict[str, dict[int, set[str]]],
    transcript_seqs: dict[str, str],
) -> list[dict]:
    """Per-transcript rows.  Schema identical to build_transcript_rows()."""
    rows: list[dict] = []
    for tid, pos_to_types in transcript_to_sites.items():
        seq = transcript_seqs.get(tid, "")
        if not seq:
            continue
        mod_positions = sorted(pos_to_types.keys())
        mod_types     = [sorted(pos_to_types[p]) for p in mod_positions]
        rows.append({
            "transcript_id": tid,
            "full_sequence":  seq,
            "seq_len":        len(seq),
            "mod_positions":  mod_positions,
            "mod_types":      mod_types,
        })
    return rows
