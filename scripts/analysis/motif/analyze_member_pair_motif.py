#!/usr/bin/env python3
"""Pairwise motif analysis for two or more members on single-label sites."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze single-label local motifs for a pair of members",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-config",
        default=str(REPO_ROOT / "outputs/etd_bind_v4/v4_multitask_m6a_rbp_326_2300/run_config.json"),
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--flank", type=int, default=10)
    parser.add_argument("--members", required=True, help="逗号分隔，例如 EIF3B,EIF3D")
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "outputs/analysis/member_pair_motif_analysis.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "outputs/analysis/member_pair_motif_analysis.md"),
    )
    return parser.parse_args()


def load_run_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokens_to_window_seq(token_ids: np.ndarray) -> str:
    chars = []
    for tok in token_ids.tolist():
        chars.append("N" if int(tok) == PAD_TOKEN_ID else ID_TO_BASE.get(int(tok), "N"))
    return "".join(chars)


def center_5mer(seq: str, center: int) -> str | None:
    if center - 2 < 0 or center + 2 >= len(seq):
        return None
    kmer = seq[center - 2:center + 3]
    return None if "N" in kmer else kmer


def init_pos_counter(flank: int) -> dict[int, Counter[str]]:
    return {rel: Counter() for rel in range(-flank, flank + 1)}


def top_counter(counter: Counter[str], limit: int = 12) -> list[dict[str, object]]:
    return [{"item": k, "count": int(v)} for k, v in counter.most_common(limit)]


def rel_enrichment(target_counter: Counter[str], ref_counter: Counter[str], pseudocount: float = 1.0):
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


def summarize_group(member: str, n: int, center5: Counter[str], pos_counter: dict[int, Counter[str]], flank: int):
    pos_summary = []
    for rel in range(-flank, flank + 1):
        cnt = pos_counter[rel]
        total = sum(cnt.values())
        if total == 0:
            continue
        top_base, top_count = cnt.most_common(1)[0]
        pos_summary.append({"rel_pos": rel, "top_base": top_base, "top_freq": float(top_count / total)})
    return {
        "member": member,
        "n_single_label": int(n),
        "top_center_5mers": top_counter(center5),
        "position_summary": pos_summary,
    }


def build_markdown(result: dict[str, object], flank: int) -> str:
    members = [row["member"] for row in result["groups"]]
    lines = [f"# {' vs '.join(members)} Motif Analysis", ""]
    lines.append("口径：只看 `single-label` 正样本位点，比较指定 member 的局部序列。")
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
            lines.append(f"- `{item['rel_pos']:+d}`: {item['top_base']} ({item['top_freq']:.3f})")
        lines.append("")
    lines.append("## Pairwise Enriched Center 5-mers")
    lines.append("")
    for name, rows in result["pairwise"].items():
        lines.append(f"{name}:")
        for item in rows[:10]:
            lhs, rhs = name.split(" vs ")
            lines.append(
                f"- `{item['kmer']}`: enrich={item['enrichment']:.2f}, "
                f"{lhs}={item['target_count']}, {rhs}={item['ref_count']}"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    members = [x.strip().upper() for x in args.members.split(",") if x.strip()]
    if len(members) < 2:
        raise ValueError("--members 至少需要两个 member")

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

    member_indices = {name: int(RBP_TO_IDX[name]) for name in members}
    flank = int(args.flank)
    sample_counts = defaultdict(int)
    center5_counters = {name: Counter() for name in members}
    pos_counters = {name: init_pos_counter(flank) for name in members}

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
            if 0 <= pos < len(seq):
                base = seq[pos]
                if base in ("A", "C", "G", "U"):
                    pos_counters[member_name][rel][base] += 1

    groups = [
        summarize_group(name, sample_counts[name], center5_counters[name], pos_counters[name], flank)
        for name in members
    ]
    pairwise = {}
    for i, lhs in enumerate(members):
        for rhs in members[i + 1:]:
            pairwise[f"{lhs} vs {rhs}"] = rel_enrichment(center5_counters[lhs], center5_counters[rhs])
            pairwise[f"{rhs} vs {lhs}"] = rel_enrichment(center5_counters[rhs], center5_counters[lhs])

    result = {"split": args.split, "flank": flank, "members": members, "groups": groups, "pairwise": pairwise}
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(result, flank), encoding="utf-8")
    print(f"[ok] wrote {out_json}", flush=True)
    print(f"[ok] wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
