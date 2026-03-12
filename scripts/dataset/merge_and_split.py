"""
Merge multiple dataset outputs into unified sites_rows / transcripts_rows,
then perform stratified transcript-level splitting.

Usage:
    from merge_and_split import merge_all_sites, merge_all_transcripts, stratified_split
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Merge sites_rows from multiple datasets
# ─────────────────────────────────────────────────────────────────────────────

def merge_all_sites(
    *sites_rows_list: list[dict],
    verbose: bool = True,
) -> list[dict]:
    """
    Merge multiple sites_rows lists, deduplicating by
    (transcript_id, site_pos, mod_type, site_base).

    When duplicates occur, the row with more RBP support is kept
    (sum of writer + reader + eraser support counts).

    Returns the merged, deduplicated sites_rows.
    """
    seen: dict[tuple, dict] = {}
    stats = Counter()

    for i, rows in enumerate(sites_rows_list):
        stats[f"input_{i}_rows"] = len(rows)
        for row in rows:
            key = (
                row["transcript_id"],
                int(row["site_pos"]),
                row["mod_type"],
                row["site_base"],
            )
            if key in seen:
                stats["duplicates"] += 1
                # 保留 RBP 信息更丰富的那条
                existing = seen[key]
                existing_support = (
                    existing["writer_support_count"]
                    + existing["reader_support_count"]
                    + existing["eraser_support_count"]
                )
                new_support = (
                    row["writer_support_count"]
                    + row["reader_support_count"]
                    + row["eraser_support_count"]
                )
                if new_support > existing_support:
                    seen[key] = row
                    stats["duplicates_replaced"] += 1
            else:
                seen[key] = row

    merged = list(seen.values())

    if verbose:
        print(f"\n[Merge sites]")
        for k in sorted(stats):
            print(f"  {k}: {stats[k]}")
        print(f"  merged_total: {len(merged)}")

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rebuild transcripts_rows from merged sites
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_transcripts_from_sites(
    merged_sites: list[dict],
    transcript_seqs: dict[str, str],
    verbose: bool = True,
) -> list[dict]:
    """
    Rebuild transcripts_rows from the merged sites_rows.

    This is more reliable than merging transcripts_rows directly,
    because it guarantees consistency between sites and transcripts.
    """
    # transcript_id → { pos → set(mod_type) }
    tx_map: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

    for row in merged_sites:
        tx_map[row["transcript_id"]][int(row["site_pos"])].add(row["mod_type"])

    rows: list[dict] = []
    for tid, pos_to_types in tx_map.items():
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

    if verbose:
        print(f"\n[Rebuild transcripts from merged sites]")
        print(f"  total_transcripts: {len(rows)}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build per-transcript feature profile (for stratification)
# ─────────────────────────────────────────────────────────────────────────────

def _build_transcript_profiles(
    merged_sites: list[dict],
) -> dict[str, dict[str, Any]]:
    """
    For each transcript, compute a feature profile from its sites:
        - which mod_types are present
        - whether it has writer / reader / eraser supported sites
        - which specific RBP names appear

    Returns { transcript_id: profile_dict }
    """
    profiles: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "mod_types": set(),
        "has_writer": False,
        "has_reader": False,
        "has_eraser": False,
        "rbp_names": set(),
    })

    for row in merged_sites:
        tid = row["transcript_id"]
        p = profiles[tid]
        p["mod_types"].add(row["mod_type"])
        if row["writer_support_count"] > 0:
            p["has_writer"] = True
        if row["reader_support_count"] > 0:
            p["has_reader"] = True
        if row["eraser_support_count"] > 0:
            p["has_eraser"] = True

        rbp_name = row.get("rbp_name", {})
        for role in ["writer", "reader", "eraser"]:
            names = rbp_name.get(role)
            if names:
                p["rbp_names"].update(names)

    return dict(profiles)


def _make_stratum_key(profile: dict) -> str:
    """
    Convert a transcript's profile into a stratum string for grouping.
    e.g. "m1A+m6A|W1R1E0"
    """
    mods = "+".join(sorted(profile["mod_types"]))
    w = "1" if profile["has_writer"] else "0"
    r = "1" if profile["has_reader"] else "0"
    e = "1" if profile["has_eraser"] else "0"
    return f"{mods}|W{w}R{r}E{e}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stratified splitting
# ─────────────────────────────────────────────────────────────────────────────

def _deterministic_hash(transcript_id: str, seed: int) -> float:
    """Map transcript_id to a deterministic float in [0, 1)."""
    key = f"{transcript_id}-{seed}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def stratified_split(
    transcripts_rows: list[dict],
    merged_sites: list[dict],
    max_len: int = 12000,
    seed: int = 2026,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    Stratified transcript-level split that ensures balanced representation
    of mod_types, writer/reader/eraser, and RBP names across splits.

    Strategy:
        1. Group transcripts into strata by (mod_types_present, has_w/r/e).
        2. Within each stratum, use deterministic hashing to split.
        3. This ensures every stratum (and thus every feature) is proportionally
           represented in train/val/test.
        4. Transcripts longer than max_len go to holdout_long regardless.

    After splitting, verify that no mod_type or RBP name is completely
    missing from any split; if so, steal one transcript from train to fix it.
    """
    profiles = _build_transcript_profiles(merged_sites)

    # ── group by stratum ──
    strata: dict[str, list[dict]] = defaultdict(list)
    holdout_long: list[str] = []

    for row in transcripts_rows:
        tid = row["transcript_id"]
        if int(row["seq_len"]) > max_len:
            holdout_long.append(tid)
            continue
        profile = profiles.get(tid)
        if profile is None:
            # 没有 site 的 transcript（理论上不应该出现）
            continue
        stratum = _make_stratum_key(profile)
        strata[stratum].append(row)

    # ── split within each stratum ──
    split_tids: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for stratum_key, rows in sorted(strata.items()):
        for row in rows:
            tid = row["transcript_id"]
            h = _deterministic_hash(tid, seed)
            if h < test_ratio:
                split_tids["test"].append(tid)
            elif h < test_ratio + val_ratio:
                split_tids["val"].append(tid)
            else:
                split_tids["train"].append(tid)

    # ── post-hoc coverage check & repair ──
    # 确保 val 和 test 里每种 mod_type 和每个 RBP name 至少出现一次
    # 如果某个 feature 在 val/test 里缺失，从 train 中偷一条含该 feature 的 transcript

    train_set = set(split_tids["train"])
    val_set   = set(split_tids["val"])
    test_set  = set(split_tids["test"])

    all_mod_types = set()
    all_rbp_names = set()
    for p in profiles.values():
        all_mod_types.update(p["mod_types"])
        all_rbp_names.update(p["rbp_names"])

    def _features_in_split(tid_set):
        mods = set()
        rbps = set()
        for tid in tid_set:
            p = profiles.get(tid)
            if p:
                mods.update(p["mod_types"])
                rbps.update(p["rbp_names"])
        return mods, rbps

    repairs = 0
    for target_name, target_set in [("val", val_set), ("test", test_set)]:
        target_mods, target_rbps = _features_in_split(target_set)

        # 缺失的 mod_types
        for mt in all_mod_types - target_mods:
            # 从 train 里找一条含该 mod_type 的 transcript
            for tid in list(train_set):
                p = profiles.get(tid)
                if p and mt in p["mod_types"]:
                    train_set.discard(tid)
                    target_set.add(tid)
                    repairs += 1
                    break

        # 缺失的 RBP names
        target_mods, target_rbps = _features_in_split(target_set)
        for rn in all_rbp_names - target_rbps:
            for tid in list(train_set):
                p = profiles.get(tid)
                if p and rn in p["rbp_names"]:
                    train_set.discard(tid)
                    target_set.add(tid)
                    repairs += 1
                    break

    # 重建 sorted lists
    split_payload = {
        "train":        sorted(train_set),
        "val":          sorted(val_set),
        "test":         sorted(test_set),
        "holdout_long": sorted(holdout_long),
    }

    # ── verbose output ──
    if verbose:
        print(f"\n[Stratified split]")
        print(f"  strata_count: {len(strata)}")
        for s_key in sorted(strata):
            print(f"    {s_key}: {len(strata[s_key])} transcripts")
        print(f"  post-hoc repairs (train→val/test): {repairs}")
        print(f"  split sizes:")
        for k in ["train", "val", "test", "holdout_long"]:
            print(f"    {k}: {len(split_payload[k])}")

        # 验证各 split 的 mod_type 和 RBP 覆盖
        for split_name in ["train", "val", "test"]:
            mods, rbps = _features_in_split(set(split_payload[split_name]))
            print(f"  {split_name} coverage:")
            print(f"    mod_types ({len(mods)}): {', '.join(sorted(mods))}")
            print(f"    rbp_names ({len(rbps)}): {', '.join(sorted(rbps)) if rbps else '-'}")

    # ── meta ──
    split_payload["meta"] = {
        "max_len":    max_len,
        "seed":       seed,
        "val_ratio":  val_ratio,
        "test_ratio": test_ratio,
        "strata_count": len(strata),
        "repairs":    repairs,
        "counts": {
            k: len(v) for k, v in split_payload.items() if isinstance(v, list)
        },
    }

    return split_payload


# ─────────────────────────────────────────────────────────────────────────────
# 5. Print merged summary (convenience)
# ─────────────────────────────────────────────────────────────────────────────

def print_merge_summary(
    merged_sites: list[dict],
    transcripts_rows: list[dict],
    source_counts: dict[str, int],
) -> None:
    """Print a summary comparing source contributions."""
    print(f"\n[Merge summary]")
    print(f"  sources:")
    for name, count in sorted(source_counts.items()):
        print(f"    {name}: {count} sites")
    print(f"  merged_unique_sites: {len(merged_sites)}")
    print(f"  merged_transcripts:  {len(transcripts_rows)}")

    # mod_type breakdown
    mod_counter = Counter(row["mod_type"] for row in merged_sites)
    print(f"  by mod_type:")
    for k in sorted(mod_counter):
        print(f"    {k}: {mod_counter[k]}")

    # RBP coverage
    w = sum(1 for r in merged_sites if r["writer_support_count"] > 0)
    r = sum(1 for r in merged_sites if r["reader_support_count"] > 0)
    e = sum(1 for r in merged_sites if r["eraser_support_count"] > 0)
    print(f"  sites_with_writer: {w}")
    print(f"  sites_with_reader: {r}")
    print(f"  sites_with_eraser: {e}")