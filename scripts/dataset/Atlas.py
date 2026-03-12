
from __future__ import annotations

import gzip
from collections import Counter, defaultdict
from typing import Optional

# Reuse GTF parser from directRMDB (avoid code duplication)
from directRMDB import (
    parse_gtf_exons,
    normalize_id,
    open_text,
)


ATLAS_PATH = "/media/scw-workspace/m6a_dataset/data/raw/Altas/m1A_Human_Basic_Site_Information.txt"



def locate_site(full_seq: str, short_seq: str) -> Optional[int]:
    """
    Find the unique centre position of short_seq within full_seq.
    Returns 0-based index of the centre base, or None if not unique / not found.
    """
    pattern = short_seq.upper().replace("T", "U")
    idx = full_seq.find(pattern)
    if idx < 0:
        return None
    if full_seq.count(pattern) != 1:
        return None
    return idx + len(pattern) // 2


def parse_sites_Atlas(
    transcript_seqs: dict[str, str],
    atlas_path: str = ATLAS_PATH,
    gtf_path: str   = None,
    protein_coding_only: bool = True,
    verbose: bool = True,
) -> tuple[dict, Counter]:

    from directRMDB import GTF_PATH as DEFAULT_GTF
    if gtf_path is None:
        gtf_path = DEFAULT_GTF

    # ── GTF ──
    _, gene_to_transcripts = parse_gtf_exons(gtf_path, verbose)

    kept: dict[tuple, dict] = {}
    stats: Counter = Counter()

    with open_text(atlas_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["total_rows"] += 1

            cols = line.split()
            if len(cols) < 11:
                stats["invalid_rows"] += 1
                continue

            mod_id    = cols[0]
            gene_type = cols[9]
            short_seq = cols[10]

            # ── filter: protein_coding ──
            if protein_coding_only and gene_type != "protein_coding":
                stats["non_protein_coding"] += 1
                continue

            # ── ENSG IDs (may be semicolon-separated) ──
            raw_ensembl = cols[7]
            gene_ids = [normalize_id(g) for g in raw_ensembl.split(";") if g.strip()]
            gene_ids = [g for g in gene_ids if g]
            if not gene_ids:
                stats["missing_gene_id"] += 1
                continue

            # ── collect all candidate transcripts ──
            candidate_tids: list[str] = []
            for gid in gene_ids:
                tids = gene_to_transcripts.get(gid, [])
                candidate_tids.extend(tids)
            candidate_tids = list(set(candidate_tids))

            if not candidate_tids:
                stats["no_transcripts_for_gene"] += 1
                continue

            # ── locate via sub-seq in each candidate, pick longest ──
            best = None
            best_len = -1

            for tid in candidate_tids:
                seq = transcript_seqs.get(tid)
                if not seq:
                    continue
                site_pos = locate_site(seq, short_seq)
                if site_pos is None:
                    continue
                if site_pos < 0 or site_pos >= len(seq):
                    continue
                # m1A → expect A
                if seq[site_pos] != "A":
                    continue
                if len(seq) > best_len:
                    best_len = len(seq)
                    best = {"tid": tid, "tpos": site_pos}

            if best is None:
                stats["unresolved_position"] += 1
                continue

            key = (best["tid"], best["tpos"], "m1A")
            if key in kept:
                stats["duplicate_site"] += 1
                continue

            kept[key] = {
                "mod_id":               mod_id,
                "transcript_id":        best["tid"],
                "mod_type":             "m1A",
                "site_pos":             int(best["tpos"]),
                "site_base":            "A",
                "writer_support_count": 0,
                "reader_support_count": 0,
                "eraser_support_count": 0,
                "rbp_names_by_role":    {"writer": None, "reader": None, "eraser": None},
            }
            stats["kept"] += 1

    if verbose:
        print(f"\n[Atlas m1A site parsing]")
        for k in sorted(stats):
            print(f"  {k}: {stats[k]}")
        print(f"  kept_unique_sites: {len(kept)}")

    return kept, stats


def build_sites_rows_Atlas(
    sites_by_key: dict,
) -> tuple[list[dict], dict[str, dict[int, set[str]]]]:
    """sites_by_key → (sites_rows, transcript_to_sites)."""
    sites_rows: list[dict] = []
    transcript_to_sites: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for key in sorted(sites_by_key):
        site = sites_by_key[key]

        row = {
            "mod_id":               site["mod_id"],
            "transcript_id":        site["transcript_id"],
            "site_pos":             int(site["site_pos"]),
            "site_base":            site["site_base"],
            "mod_type":             site["mod_type"],
            "pu_label":             1,
            "writer_pu_label":      -1,
            "reader_pu_label":      -1,
            "eraser_pu_label":      -1,
            "writer_support_count": 0,
            "reader_support_count": 0,
            "eraser_support_count": 0,
            "rbp_name": {
                "reader": None,
                "writer": None,
                "eraser": None,
            },
        }
        sites_rows.append(row)
        transcript_to_sites[row["transcript_id"]][row["site_pos"]].add("m1A")

    return sites_rows, transcript_to_sites


def build_transcript_rows_Atlas(
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