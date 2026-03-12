

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
import numpy as np

MOD_PATHS = {
    

            "m6A": "/media/scw-workspace/m6a_dataset/data/raw/RMBase/mod/human.hg38.m6A.result.col29.bed",
            "m1A": "/media/scw-workspace/m6a_dataset/data/raw/RMBase/mod/human.hg38.m1A.result.col29.bed",
            "m5C": "/media/scw-workspace/m6a_dataset/data/raw/RMBase/mod/human.hg38.m5C.result.col29.bed",
            "pseu": "/media/scw-workspace/m6a_dataset/data/raw/RMBase/mod/human.hg38.Pseudo.result.col29.bed",
    
}

BIND_PATHS = {
    'reader':["/media/scw-workspace/m6a_dataset/data/raw/RMBase/bind/m5C/human.hg38.modrbp.m5C.reader.bed", "/media/scw-workspace/m6a_dataset/data/raw/RMBase/bind/m6A/human.hg38.modrbp.m6A.reader.bed",],
    'writer':["/media/scw-workspace/m6a_dataset/data/raw/RMBase/bind/m5C/human.hg38.modrbp.m5C.writer.bed","/media/scw-workspace/m6a_dataset/data/raw/RMBase/bind/m6A/human.hg38.modrbp.m6A.writer.bed","/media/scw-workspace/m6a_dataset/data/raw/RMBase/bind/pseudo/human.hg38.modrbp.Y.writer.bed"],
    'eraser':["/media/scw-workspace/m6a_dataset/data/raw/RMBase/bind/m6A/human.hg38.modrbp.m6A.eraser.bed"],
}

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



def locate_site(full_seq: str, short_seq: str) -> int | None:
    pattern = short_seq.upper().replace("T", "U")
    idx = full_seq.find(pattern)
    if idx < 0:
        return None
    if full_seq.count(pattern) != 1:
        return None
    center = idx + len(pattern) // 2
    return center

def parse_sites_RMBase(mod_types: list[str], transcript_seqs: dict[str, str], protein_coding_only: bool = True):
    
    # kept 以 (transcript_id, site_pos, mod_type) 为唯一键，避免同位点重复入表。
    letter_dict = {'m6A': 'A', 'm1A': 'A', 'm5C': 'C', 'pseu': 'U'}
    kept = {}
    # stats 记录过滤路径，便于分析数据损失来源。
    stats = Counter()
    for mod_type in mod_types:
        mod_path = MOD_PATHS[mod_type]
        with open_text(mod_path) as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                # 空行直接跳过，不计统计。
                if not row:
                    continue

                # 至少需要访问到 row[18]；不足说明格式异常。
                if len(row) < 19:
                    stats["invalid_rows"] += 1
                    continue
                # 部分行把 gene_type / transcript_id 写成逗号分隔多值。
                # 这里按索引一一对应展开处理。
                gene_types = [x.strip() for x in row[16].split(",") if x.strip()]
                transcript_ids = [x.strip() for x in row[14].split(",") if x.strip()]
                mod_id = row[3].strip()
                short_seq = row[18].strip()

                # total_rows 统计“展开后样本数”，而不是原始文件行数。
                stats["total_rows"] += len(gene_types)
                for i in range(len(gene_types)):
                    gene_type = gene_types[i]

                    # 仅保留 protein_coding（可由参数关闭该过滤）。
                    if protein_coding_only and gene_type != "protein_coding":
                        stats["non_protein_coding"] += 1
                        continue
                    transcript_id = normalize_transcript_id(transcript_ids[i])

                    # 关键字段缺失时跳过。
                    if not mod_id or not transcript_id or not short_seq:
                        stats["missing_fields"] += 1
                        continue

                    # 当前 transcript 在 FASTA 字典中不存在。
                    full_seq = transcript_seqs.get(transcript_id)
                    if not full_seq:
                        stats["missing_transcript"] += 1
                        continue

                    # short_seq 需要在 full_seq 中唯一匹配，才能确定中心位点。
                    site_pos = locate_site(full_seq, short_seq)
                    if site_pos is None:
                        stats["unresolved_position"] += 1
                        continue

                    key = (transcript_id, site_pos, mod_type)
                    # 同一 transcript 同一位点重复出现时，只保留第一条，保证确定性。
                    if key in kept:
                        stats["duplicate_site"] += 1
                        # Prefer first mod_id to keep deterministic.
                        continue

                    # 位点中心必须一致；否则视为不一致样本。
                    site_base = full_seq[site_pos]
                    mod_letter = letter_dict[mod_type]
                    if site_base != mod_letter:
                        stats["non_" + mod_type.lower() + "_center"] += 1
                        continue
                    
                    kept[key] = {
                        "mod_id": mod_id,
                        "transcript_id": transcript_id,
                        "mod_type": mod_type,
                        "site_pos": int(site_pos),
                        "site_base": site_base,
                    }
                    # 最终进入输出表的样本数。
                    stats["kept"] += 1
    kept_mod_counts = Counter(item["mod_type"] for item in kept.values())
    print("Kept site counts by mod_type in RMBase:")
    for mod_type in mod_types:
        print(f"  {mod_type}: {kept_mod_counts.get(mod_type, 0)}")

    return kept, stats


def parse_role_support(bind_type: str) -> tuple[Counter, dict[str, list[str]]]:
    counts: Counter = Counter()
    names_by_mod: dict[str, set[str]] = defaultdict(set)

    paths = BIND_PATHS[bind_type]
    for path in paths:
        with open_text(path) as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if not row or len(row) < 8:
                    continue
                rbp_name = row[1].strip()
                mod_id = row[7].strip()
                if not mod_id:
                    continue
                counts[mod_id] += 1
                if rbp_name:
                    names_by_mod[mod_id].add(rbp_name)
    names_by_mod = {k: sorted(v) for k, v in names_by_mod.items()}
    return counts, names_by_mod

def build_sites_rows(sites_by_key):

    writer_support, writer_rbp_names = parse_role_support("writer")
    reader_support, reader_rbp_names = parse_role_support("reader")
    eraser_support, eraser_rbp_names = parse_role_support("eraser")

    sites_rows = []
    transcript_to_sites: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

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
            "mod_type": site["mod_type"],
            "pu_label": 1,
            "writer_pu_label": 1 if w > 0 else -1,
            "reader_pu_label": 1 if r > 0 else -1,
            "eraser_pu_label": 1 if e > 0 else -1,
            "writer_support_count": w,
            "reader_support_count": r,
            "eraser_support_count": e,
            "rbp_name": {
                "reader": reader_rbp_names.get(mod_id) or None,
                "writer": writer_rbp_names.get(mod_id) or None,
                "eraser": eraser_rbp_names.get(mod_id) or None,
            },
        }
        sites_rows.append(row)

        transcript_to_sites[row["transcript_id"]][int(row["site_pos"])].add(row["mod_type"])


    return sites_rows, transcript_to_sites

def build_transcript_rows(transcript_to_sites: dict[str, dict[int, set[str]]], transcript_seqs: dict[str, str]):
    transcripts_rows = []
    for transcript_id, pos_to_types in transcript_to_sites.items():
        seq = transcript_seqs.get(transcript_id, "")
        if not seq:
            continue

        mod_positions = sorted(pos_to_types.keys())
        mod_types = [sorted(list(pos_to_types[pos])) for pos in mod_positions]

        row = {
            "transcript_id": transcript_id,
            "full_sequence": seq,
            "seq_len": int(len(seq)),
            "mod_positions": mod_positions,
            "mod_types": mod_types,
        }
        transcripts_rows.append(row)

    return transcripts_rows


