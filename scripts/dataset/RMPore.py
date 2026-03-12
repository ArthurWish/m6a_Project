from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from typing import Optional



BASIC_INFO_PATH = "/media/scw-workspace/m6a_dataset/data/raw/RMPore/RMPore_Combine_basic_info.Homo_sapiens.csv.gz"
CALC_PATH       = "/media/scw-workspace/m6a_dataset/data/raw/RMPore/RMPore_Calculation_results.Homo_sapiens.csv.gz"
RBP_INFO_PATH   = "/media/scw-workspace/m6a_dataset/data/raw/RMPore/RMPore_RBP_info.csv.gz"



MOD_TYPE_MAP = {
    "m6A": "m6A",
    "m1A": "m1A",
    "m5C": "m5C",
    "psU": "pseu",
}

MOD_LETTER = {"m6A": "A", "m1A": "A", "m5C": "C", "pseu": "U"}

RBP_ROLE_BY_MOD: dict[str, dict[str, str]] = {
    "m6A": {
        "METTL3": "writer", "METTL14": "writer", "METTL16": "writer",
        "WTAP": "writer", "VIRMA": "writer", "KIAA1429": "writer",
        "RBM15": "writer", "RBM15B": "writer", "ZC3H13": "writer",
        "CBLL1": "writer", "HAKAI": "writer",
        "YTHDF1": "reader", "YTHDF2": "reader", "YTHDF3": "reader",
        "YTHDC1": "reader", "YTHDC2": "reader",
        "IGF2BP1": "reader", "IGF2BP2": "reader", "IGF2BP3": "reader",
        "HNRNPA2B1": "reader", "HNRNPC": "reader",
        "HNRNPG": "reader", "RBMX": "reader",
        "FMR1": "reader", "FMRP": "reader", "PRRC2A": "reader",
        "EIF3A": "reader", "EIF3B": "reader", "LRPPRC": "reader",
        "FTO": "eraser", "ALKBH5": "eraser",
    },
    "m1A": {
        "TRMT6": "writer", "TRMT61A": "writer",
        "TRMT61B": "writer", "TRMT10C": "writer",
        "YTHDF1": "reader", "YTHDF2": "reader", "YTHDF3": "reader",
        "ALKBH1": "eraser", "ALKBH3": "eraser",
    },
    "m5C": {
        "NSUN1": "writer", "NSUN2": "writer", "NSUN3": "writer",
        "NSUN4": "writer", "NSUN5": "writer", "NSUN6": "writer",
        "NSUN7": "writer", "DNMT2": "writer", "TRDMT1": "writer",
        "ALYREF": "reader", "YBX1": "reader", "YBX2": "reader",
    },
    "pseu": {
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
    """Strip version suffix: ENST00000638356.1 → ENST00000638356."""
    if not raw:
        return ""
    raw = str(raw).strip().strip('"')
    if not raw or raw.lower() == "na":
        return ""
    return raw.split(".")[0]


def classify_rbp_role(name: str, mod_type: str) -> str:
    """Return writer/reader/eraser only if the RBP is known for *this* mod_type."""
    role_map = RBP_ROLE_BY_MOD.get(mod_type, {})
    return role_map.get(name.upper(), "unknown")


def _split_rbp_by_role(names: list[str], mod_type: str) -> dict[str, list[str] | None]:
    """Classify RBP names into writer / reader / eraser only.
    RBPs unrelated to the current mod_type are silently dropped."""
    buckets: dict[str, list[str]] = {
        "writer": [], "reader": [], "eraser": [],
    }
    for n in names:
        role = classify_rbp_role(n, mod_type)
        if role in buckets:
            buckets[role].append(n)
    return {k: (sorted(v) if v else None) for k, v in buckets.items()}


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



def parse_basic_info(
    basic_info_path: str = BASIC_INFO_PATH,
    mod_types: list[str] | None = None,
    protein_coding_only: bool = True,
    verbose: bool = True,
) -> tuple[dict[str, str], Counter]:
  
    if mod_types is None:
        mod_types = ["m6A", "m1A", "m5C", "psU"]
    allowed = set(mod_types)

    rm_id_to_mod: dict[str, str] = {}
    stats: Counter = Counter()

    with open_text(basic_info_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["total_rows"] += 1

            mod = row.get("mod", "").strip()
            if mod not in allowed:
                stats["filtered_mod_type"] += 1
                continue

            gene_type = row.get("gene_type", "").strip()
           
            if protein_coding_only and gene_type not in ("protein-coding", "protein_coding"):
                stats["non_protein_coding"] += 1
                continue

            level1 = row.get("level1", "0").strip()
            level2 = row.get("level2", "0").strip()
            if level1 != "1" and level2 != "1":
                stats["filtered_level"] += 1
                continue

            rm_id = row.get("rm_id", "").strip()
            if not rm_id:
                stats["missing_rm_id"] += 1
                continue

            rm_id_to_mod[rm_id] = MOD_TYPE_MAP[mod]
            stats["kept"] += 1

    if verbose:
        print(f"\n[RMPore basic info parsing]")
        for k in sorted(stats):
            print(f"  {k}: {stats[k]}")
        kept_counts = Counter(rm_id_to_mod.values())
        print("  Kept rm_ids by mod_type:")
        for mt in ["m6A", "m1A", "m5C", "pseu"]:
            print(f"    {mt}: {kept_counts.get(mt, 0)}")

    return rm_id_to_mod, stats



def parse_rbp_info(
    rbp_info_path: str = RBP_INFO_PATH,
    valid_rm_ids: set[str] | None = None,
    verbose: bool = True,
) -> dict[str, list[str]]:
    
    rbp_by_id: dict[str, set[str]] = defaultdict(set)
    n = 0

    with open_text(rbp_info_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rm_id = row.get("rm_id", "").strip()
            rbp_name = row.get("RBP", "").strip()
            if not rm_id or not rbp_name:
                continue
            if valid_rm_ids is not None and rm_id not in valid_rm_ids:
                continue
            rbp_by_id[rm_id].add(rbp_name.upper())
            n += 1

    if verbose:
        print(f"\n[RMPore RBP parsing]")
        print(f"  binding_rows_kept: {n}")
        print(f"  unique_rm_ids_with_rbp: {len(rbp_by_id)}")

    return {k: sorted(v) for k, v in rbp_by_id.items()}


def parse_sites_RMPore(
    transcript_seqs: dict[str, str],
    basic_info_path: str = BASIC_INFO_PATH,
    calc_path: str       = CALC_PATH,
    rbp_info_path: str   = RBP_INFO_PATH,
    mod_types: list[str] | None = None,
    protein_coding_only: bool = True,
    verbose: bool = True,
) -> tuple[dict, Counter]:

    if mod_types is None:
        mod_types = ["m6A", "m1A", "m5C", "psU"]
    

    rm_id_to_mod, basic_stats = parse_basic_info(
        basic_info_path, mod_types, protein_coding_only, verbose,
    )

    valid_rm_ids = set(rm_id_to_mod.keys())

    rbp_by_id = parse_rbp_info(rbp_info_path, valid_rm_ids, verbose)

    kept: dict[tuple, dict] = {}
    stats: Counter = Counter()

    resolved_rm_ids: set[str] = set()

    if verbose:
        print(f"\n[RMPore site parsing from calculation results]")

    with open_text(calc_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["total_calc_rows"] += 1

            rm_id = row.get("rm_id", "").strip()
            if rm_id not in valid_rm_ids:
                continue
            stats["matching_rm_id"] += 1


            if rm_id in resolved_rm_ids:
                stats["skipped_already_resolved"] += 1
                continue

            mod_type = rm_id_to_mod[rm_id]
            expected_base = MOD_LETTER[mod_type]

            tc_raw = row.get("transcript_coordinate", "").strip()
            if not tc_raw or "|" not in tc_raw:
                stats["missing_transcript_coord"] += 1
                continue

            parts = tc_raw.split("|")
            transcript_id = normalize_id(parts[0])
            try:
                tc_pos = int(parts[1])
            except (ValueError, IndexError):
                stats["invalid_transcript_coord"] += 1
                continue

            if not transcript_id:
                stats["missing_transcript_id"] += 1
                continue

            full_seq = transcript_seqs.get(transcript_id)
            if not full_seq:
                stats["missing_transcript_in_fasta"] += 1
                continue

            fivemer = row.get("5mer", "").strip().upper().replace("T", "U")


            site_pos = None

            # 策略1: tc_pos 是 1-based → 转 0-based
            if _validate_pos(full_seq, tc_pos - 1, expected_base, fivemer):
                site_pos = tc_pos - 1
                stats["resolved_by_coord_1based"] += 1

            # 策略2: tc_pos 本身就是 0-based
            if site_pos is None and _validate_pos(full_seq, tc_pos, expected_base, fivemer):
                site_pos = tc_pos
                stats["resolved_by_coord_0based"] += 1

            # 策略3: 用 5mer 在全序列中唯一定位
            if site_pos is None and fivemer and len(fivemer) == 5:
                located = locate_site(full_seq, fivemer)
                if located is not None and full_seq[located] == expected_base:
                    site_pos = located
                    stats["resolved_by_5mer"] += 1

            if site_pos is None:
                stats["unresolved_position"] += 1
                continue

            # ── dedup by (transcript, pos, mod_type) ──
            key = (transcript_id, site_pos, mod_type)
            if key in kept:
                stats["duplicate_site"] += 1
                resolved_rm_ids.add(rm_id)
                continue
            
            rbp_names = rbp_by_id.get(rm_id, [])
            roles = _split_rbp_by_role(rbp_names, mod_type)

            kept[key] = {
                "mod_id":               rm_id,
                "transcript_id":        transcript_id,
                "mod_type":             mod_type,
                "site_pos":             int(site_pos),
                "site_base":            full_seq[site_pos],
                "writer_support_count": len(roles["writer"] or []),
                "reader_support_count": len(roles["reader"] or []),
                "eraser_support_count": len(roles["eraser"] or []),
                "rbp_names_by_role":    roles,
            }
            stats["kept"] += 1
            resolved_rm_ids.add(rm_id)

    stats["rm_ids_not_in_calc"] = len(valid_rm_ids - resolved_rm_ids)

    if verbose:
        kept_counts = Counter(v["mod_type"] for v in kept.values())
        print("  Kept site counts by mod_type:")
        for mt in ["m6A", "m1A", "m5C", "pseu"]:
            print(f"    {mt}: {kept_counts.get(mt, 0)}")

    return kept, stats


def build_sites_rows_RMPore(
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
                "reader": roles.get("reader"),
                "writer": roles.get("writer"),
                "eraser": roles.get("eraser"),
            },
        }
        sites_rows.append(row)
        transcript_to_sites[row["transcript_id"]][row["site_pos"]].add(
            row["mod_type"]
        )

    return sites_rows, transcript_to_sites


def build_transcript_rows_RMPore(
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


def _validate_pos(full_seq: str, pos: int, expected_base: str, fivemer: str) -> bool:
    """Validate a candidate position against base and optional 5mer context."""
    if pos < 0 or pos >= len(full_seq):
        return False
    if full_seq[pos] != expected_base:
        return False
    if fivemer and len(fivemer) == 5:
        ctx_start = pos - 2
        ctx_end = pos + 3
        if 0 <= ctx_start and ctx_end <= len(full_seq):
            return full_seq[ctx_start:ctx_end] == fivemer
        # 边界位点，碱基已验证通过即可
    return True
