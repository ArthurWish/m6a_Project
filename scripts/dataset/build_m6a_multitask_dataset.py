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
import numpy as np
import matplotlib.pyplot as plt


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
    """解析 m6A 主 BED 文件并定位位点到转录本坐标。

    返回：
    - kept: dict[(transcript_id, site_pos)] -> site payload
      只保留可定位、中心碱基为 A、且去重后的位点。
    - stats: Counter
      记录总处理量、各类过滤原因和最终保留数，便于数据质控。

    说明：
    - 文件中第 14/16 列可能是逗号分隔的多值字段，代码按“同索引配对”展开。
    - transcript_id 会做 normalize（去版本号等）。
    """
    # kept 以 (transcript_id, site_pos) 为唯一键，避免同位点重复入表。
    kept = {}
    # stats 记录过滤路径，便于分析数据损失来源。
    stats = Counter()

    with open_text(m6a_path) as handle:
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

                key = (transcript_id, site_pos)
                # 同一 transcript 同一位点重复出现时，只保留第一条，保证确定性。
                if key in kept:
                    stats["duplicate_site"] += 1
                    # Prefer first mod_id to keep deterministic.
                    continue

                # m6A 位点中心必须是 A；否则视为不一致样本。
                site_base = full_seq[site_pos]
                if site_base != "A":
                    stats["non_a_center"] += 1
                    continue

                # 保留规范化后的位点记录。
                kept[key] = {
                    "mod_id": mod_id,
                    "transcript_id": transcript_id,
                    "site_pos": int(site_pos),
                    "site_base": site_base,
                }
                # 最终进入输出表的样本数。
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
    """构建位点表/转录本表与可复现的数据划分文件。

    处理流程：
    1) 从 FASTA 读取转录本序列；
    2) 解析并定位 m6A 位点到转录本坐标；
    3) 合并 writer/reader/eraser 对应的支持计数；
    4) 生成两张输出表：
       - 位点级表（每个唯一 transcript-site 一行）
       - 转录本级表（每个 transcript 一行，含 m6A 位置列表）
    5) 按 transcript 做 train/val/test（以及 holdout_long）划分；
    6) 落盘 parquet/json，并返回汇总统计。
    """
    # 阶段1：FASTA -> 转录本序列字典与解析统计。
    # transcript_seqs 结构为 transcript_id -> 完整 RNA 序列（T 已转为 U）。
    transcript_seqs, fasta_stats = parse_fasta(args.fasta)

    # 阶段2：m6A BED -> 在转录本上唯一定位后的位点集合。
    # sites_by_key 的 key 为 (transcript_id, site_pos)，value 为位点信息字典。
    # site_stats 记录过滤原因统计（如缺失 transcript、定位失败等）。
    sites_by_key, site_stats = parse_m6a_sites(
        m6a_path=args.m6a_file,
        transcript_seqs=transcript_seqs,
        protein_coding_only=not args.all_genes,
    )

    # 阶段3：统计每个 mod_id 在不同 role 下的支持次数。
    # 后续用于生成 PU 标签：support > 0 记为正类(1)，否则未标注(-1)。
    writer_support = parse_role_support(args.writer_file)
    reader_support = parse_role_support(args.reader_file)
    eraser_support = parse_role_support(args.eraser_file)

    # 中间容器：
    # - sites_rows：位点级 parquet 的行记录
    # - transcript_to_sites：按 transcript 聚合位点位置，供生成转录本级表
    sites_rows = []
    transcript_to_sites: dict[str, list[int]] = defaultdict(list)

    # 阶段4a：构建位点级表。
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
        # 记录每条 transcript 的位点位置，后续生成 transcript 级表时使用。
        transcript_to_sites[row["transcript_id"]].append(int(row["site_pos"]))

    # 阶段4b：构建转录本级表。
    # 每行包含完整序列和去重排序后的 m6A 位置列表。
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

    # 汇总统计项：所有 transcript 的 m6A 位置总数。
    total_m6a_positions = sum(len(r["m6a_positions"]) for r in transcripts_rows)

    # 序列长度分布统计（n/均值/方差/分位数），用于数据质量检查。
    seq_lens = np.array([int(r["seq_len"]) for r in transcripts_rows], dtype=np.int64)
    if seq_lens.size == 0:
        len_stats = {
            "n": 0,
            "mean": 0.0,
            "var": 0.0,
            "std": 0.0,
            "min": 0,
            "p25": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    else:
        len_stats = {
            "n": int(seq_lens.size),
            "mean": float(seq_lens.mean()),
            "var": float(seq_lens.var()),   # population variance (ddof=0)
            "std": float(seq_lens.std()),
            "min": int(seq_lens.min()),
            "p25": int(np.percentile(seq_lens, 25)),
            "p50": int(np.percentile(seq_lens, 50)),
            "p75": int(np.percentile(seq_lens, 75)),
            "p90": int(np.percentile(seq_lens, 90)),
            "p95": int(np.percentile(seq_lens, 95)),
            "p99": int(np.percentile(seq_lens, 99)),
            "max": int(seq_lens.max()),
        }

    # 额外产物：输出 seq_len 直方图，便于快速观察长度分布。
    plot_path = Path(args.len_plot_out)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(seq_lens, bins=100)
    plt.xlabel("Transcript length (nt)")
    plt.ylabel("Count")
    plt.title("Transcript seq_len distribution")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    transcripts_df = pd.DataFrame(transcripts_rows)
    sites_df = pd.DataFrame(sites_rows)

    # 阶段5：按 transcript 做可复现划分。
    # 注意：按 transcript_id 划分可避免同一 transcript 的位点泄漏到不同集合。
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

    # 保存划分元信息与数量统计，便于复现与排查。
    split_payload["meta"] = {  # type: ignore
        "max_len": args.max_len,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "counts": {k: len(v) for k, v in split_payload.items() if isinstance(v, list)},
    }

    # 阶段6：落盘输出。
    write_parquet(sites_df, args.sites_out)
    write_parquet(transcripts_df, args.transcripts_out)

    splits_path = Path(args.splits_out)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with splits_path.open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2)

    # 返回精简汇总，供 CLI 打印和流水线日志记录。
    return {
        "fasta_stats": dict(fasta_stats),
        "site_stats": dict(site_stats),
        "n_sites": int(len(sites_df)),
        "n_transcripts": int(len(transcripts_df)),
        "split_counts": split_payload["meta"]["counts"],
        "total_m6a_positions": total_m6a_positions,
        "seq_len_stats": len_stats,
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
    parser.add_argument(
        "--len-plot-out",
        default=str(REPO_ROOT / "data/processed/seq_len_hist.png"),
        help="Output path for transcript length histogram PNG.",
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
