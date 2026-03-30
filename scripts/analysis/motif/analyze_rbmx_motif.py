#!/usr/bin/env python3
"""RBMX-focused motif analysis on single-label sites.

目标：
1. 从 together/v4 的同一份数据口径里提取 single-label 正样本；
2. 聚焦 RBMX，并和最常压过它的 HNRNPC / IGF2BP2 做局部序列对比；
3. 输出一个易读的 Markdown，总结：
   - 单标签样本数
   - 中心 5-mer 分布
   - 局部位置碱基偏好
   - RBMX 相比对照组更富集的中心 5-mer
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.etd_multitask.constants import ID_TO_BASE, PAD_TOKEN_ID, RBP_TO_IDX
from models.etd_only.bind_dataloader_v4 import MultiTaskDataset, load_site_metas_and_seqs


TARGET_MEMBERS = ("RBMX", "HNRNPC", "IGF2BP2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze single-label local motifs for RBMX and frequent confusers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-config",
        default=str(REPO_ROOT / "outputs/etd_bind_v4/v4_multitask_m6a_rbp_326_2300/run_config.json"),
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--flank", type=int, default=10, help="统计中心两侧多少个位置的碱基频率")
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "outputs/analysis/rbmx_motif_analysis.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "outputs/analysis/rbmx_motif_analysis.md"),
    )
    return parser.parse_args()


def load_run_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokens_to_window_seq(token_ids: np.ndarray) -> str:
    chars = []
    for tok in token_ids.tolist():
        if int(tok) == PAD_TOKEN_ID:
            chars.append("N")
        else:
            chars.append(ID_TO_BASE.get(int(tok), "N"))
    return "".join(chars)


def center_5mer(seq: str, center: int) -> str | None:
    if center - 2 < 0 or center + 2 >= len(seq):
        return None
    kmer = seq[center - 2:center + 3]
    if "N" in kmer:
        return None
    return kmer


def init_pos_counter(flank: int) -> dict[int, Counter[str]]:
    return {rel: Counter() for rel in range(-flank, flank + 1)}


def top_counter(counter: Counter[str], limit: int = 10) -> list[dict[str, object]]:
    return [{"item": k, "count": int(v)} for k, v in counter.most_common(limit)]


def summarize_group(
    member: str,
    sample_count: int,
    center5_counter: Counter[str],
    pos_counter: dict[int, Counter[str]],
    flank: int,
) -> dict[str, object]:
    pos_summary = []
    for rel in range(-flank, flank + 1):
        cnt = pos_counter[rel]
        total = sum(cnt.values())
        if total == 0:
            pos_summary.append({"rel_pos": rel, "top_base": None, "top_freq": 0.0, "counts": {}})
            continue
        top_base, top_count = cnt.most_common(1)[0]
        pos_summary.append(
            {
                "rel_pos": rel,
                "top_base": top_base,
                "top_freq": float(top_count / total),
                "counts": {base: int(cnt.get(base, 0)) for base in ("A", "C", "G", "U")},
            }
        )
    return {
        "member": member,
        "n_single_label": int(sample_count),
        "top_center_5mers": top_counter(center5_counter, limit=12),
        "position_summary": pos_summary,
    }


def rel_enrichment(
    target_counter: Counter[str],
    ref_counter: Counter[str],
    pseudocount: float = 1.0,
) -> list[dict[str, object]]:
    kmers = sorted(set(target_counter.keys()) | set(ref_counter.keys()))
    if not kmers:
        return []
    tgt_total = float(sum(target_counter.values()))
    ref_total = float(sum(ref_counter.values()))
    out = []
    for kmer in kmers:
        tgt = float(target_counter.get(kmer, 0))
        ref = float(ref_counter.get(kmer, 0))
        if tgt <= 0:
            continue
        tgt_rate = (tgt + pseudocount) / (tgt_total + pseudocount * len(kmers))
        ref_rate = (ref + pseudocount) / (ref_total + pseudocount * len(kmers))
        out.append(
            {
                "kmer": kmer,
                "target_count": int(tgt),
                "ref_count": int(ref),
                "enrichment": float(tgt_rate / ref_rate),
            }
        )
    out.sort(key=lambda x: (x["enrichment"], x["target_count"], -x["ref_count"]), reverse=True)
    return out[:12]


def build_markdown(
    result: dict[str, object],
    flank: int,
) -> str:
    lines: list[str] = []
    lines.append("# RBMX Motif Analysis")
    lines.append("")
    lines.append("口径：只看 `single-label` 正样本位点，比较 `RBMX`、`HNRNPC`、`IGF2BP2` 的局部序列。")
    lines.append("")
    lines.append("## Sample Counts")
    lines.append("")
    for row in result["groups"]:
        lines.append(f"- `{row['member']}`: {row['n_single_label']} single-label sites")
    lines.append("")

    for row in result["groups"]:
        lines.append(f"## {row['member']}")
        lines.append("")
        lines.append("Top center 5-mers:")
        for item in row["top_center_5mers"][:10]:
            lines.append(f"- `{item['item']}`: {item['count']}")
        lines.append("")
        lines.append(f"Position-wise dominant bases (`-{flank}`..`+{flank}`):")
        for item in row["position_summary"]:
            if item["top_base"] is None:
                continue
            lines.append(
                f"- `{item['rel_pos']:+d}`: {item['top_base']} ({item['top_freq']:.3f})"
            )
        lines.append("")

    lines.append("## RBMX-Enriched Center 5-mers")
    lines.append("")
    for comp_name, rows in result["rbmx_vs"].items():
        lines.append(f"Compared with `{comp_name}`:")
        for item in rows[:10]:
            lines.append(
                f"- `{item['kmer']}`: enrich={item['enrichment']:.2f}, "
                f"RBMX={item['target_count']}, {comp_name}={item['ref_count']}"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    run_cfg = load_run_config(Path(args.run_config))

    metas, seq_store = load_site_metas_and_seqs(
        sites_path=str(REPO_ROOT / run_cfg["sites_path"]),
        transcripts_path=str(REPO_ROOT / run_cfg["transcripts_path"]),
        splits_path=str(REPO_ROOT / run_cfg["splits_path"]),
        split_names=[args.split],
        mod_type=run_cfg["mod_type"],
        role_name=run_cfg["role_name"],
        max_len=int(run_cfg["max_len"]),
        neg_ratio=float(run_cfg["neg_ratio"]),
        smoke_ratio=1.0,
        seed=int(run_cfg["seed"]),
    )
    dataset = MultiTaskDataset(
        metas=metas,
        seq_store=seq_store,
        half_window=int(run_cfg["half_window"]),
        max_jitter=0,
        training=False,
        n_extra_pos=int(run_cfg["n_extra_pos"]),
        n_m6a_neg=int(run_cfg["n_m6a_neg"]),
        n_clean_neg=int(run_cfg["n_clean_neg"]),
        n_m6a_eval_a=int(run_cfg["n_m6a_eval_a"]),
        m6a_neg_smooth=float(run_cfg["m6a_neg_smooth"]),
    )

    member_indices = {name: int(RBP_TO_IDX[name]) for name in TARGET_MEMBERS}
    flank = int(args.flank)

    sample_counts = defaultdict(int)
    center5_counters = {name: Counter() for name in TARGET_MEMBERS}
    pos_counters = {name: init_pos_counter(flank) for name in TARGET_MEMBERS}

    for idx, meta in enumerate(metas):
        active = np.where(meta.rbp_targets > 0.5)[0]
        if active.size != 1:
            continue
        member_idx = int(active[0])
        member_name = next((name for name, idx_ in member_indices.items() if idx_ == member_idx), None)
        if member_name is None:
            continue

        sample = dataset[idx]
        seq = tokens_to_window_seq(sample["token_ids"])
        center = int(sample["center_idx"])
        kmer = center_5mer(seq, center)

        sample_counts[member_name] += 1
        if kmer is not None:
            center5_counters[member_name][kmer] += 1

        for rel in range(-flank, flank + 1):
            pos = center + rel
            if pos < 0 or pos >= len(seq):
                continue
            base = seq[pos]
            if base in ("A", "C", "G", "U"):
                pos_counters[member_name][rel][base] += 1

    groups = [
        summarize_group(
            member=name,
            sample_count=sample_counts[name],
            center5_counter=center5_counters[name],
            pos_counter=pos_counters[name],
            flank=flank,
        )
        for name in TARGET_MEMBERS
    ]

    result = {
        "split": args.split,
        "flank": flank,
        "groups": groups,
        "rbmx_vs": {
            "HNRNPC": rel_enrichment(center5_counters["RBMX"], center5_counters["HNRNPC"]),
            "IGF2BP2": rel_enrichment(center5_counters["RBMX"], center5_counters["IGF2BP2"]),
        },
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_markdown(result, flank=flank), encoding="utf-8")

    print(f"[ok] wrote {output_json}", flush=True)
    print(f"[ok] wrote {output_md}", flush=True)


if __name__ == "__main__":
    main()
