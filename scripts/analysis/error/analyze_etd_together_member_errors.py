#!/usr/bin/env python3
"""离线分析 v4 together 模型的 member-level top-k 预测去向。

目标：
1. 复跑给定 checkpoint 在指定 split 上的中心位点预测。
2. 统计每个 member 的基础指标：n_pos / AUPRC / AUROC / F1opt / top1-hit / top3-hit / 平均 rank。
3. 对低 F1 的 member，分析：
   - 真值为该 member 时，top-1 最常预测成谁；
   - 当该 member 没进 top-3 时，最常有哪些 member 排在前面；
   - 哪些 member 最常“压过”它（概率高于该 member）。

输出：
- 一个 JSON，便于后续再加工
- 一个 Markdown，便于直接阅读
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.etd_multitask.constants import INDIVIDUAL_RBPS, RBP_TO_IDX
from models.etd_only.bind_dataloader_v4 import (
    MultiTaskDataset,
    collate_fn,
    load_site_metas_and_seqs,
)
from models.etd_only.bind_loss_v4 import full_evaluate, _optimize_thresholds
from models.etd_only.etd_bind_v4 import MultiTaskBindModel, MultiTaskConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze member-level top-k confusions for ETD together v4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-config",
        default=str(REPO_ROOT / "outputs/etd_bind_v4/v4_multitask_m6a_rbp_326_2300/run_config.json"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "outputs/etd_bind_v4/v4_multitask_m6a_rbp_326_2300/checkpoints/best.pt"),
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--focus-members",
        default="",
        help="逗号分隔；为空时自动取 F1 最低的若干 member",
    )
    parser.add_argument("--bottom-k", type=int, default=6, help="未指定 focus 时，自动分析 F1 最低的多少个 member")
    parser.add_argument("--top-k", type=int, default=3, help="分析 top-k 预测去向，默认 3")
    parser.add_argument("--max-examples", type=int, default=10, help="每个 member 最多保留多少条代表性误例")
    parser.add_argument("--print-every", type=int, default=200, help="每隔多少个 batch 打印一次进度")
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "outputs/analysis/etd_together_member_error_analysis.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "outputs/analysis/etd_together_member_error_analysis.md"),
    )
    return parser.parse_args()


def load_run_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model(checkpoint_path: Path, run_cfg: dict, device: torch.device) -> MultiTaskBindModel:
    cfg = MultiTaskConfig(
        d_model=int(run_cfg["d_model"]),
        encoder_channels=tuple(int(x) for x in str(run_cfg["encoder_channels"]).split(",")),
        n_transformer_layers=int(run_cfg["n_transformer_layers"]),
        n_heads=int(run_cfg["n_heads"]),
        ff_mult=int(run_cfg["ff_mult"]),
        dropout=float(run_cfg["dropout"]),
        head_dropout=float(run_cfg["head_dropout"]),
    )
    model = MultiTaskBindModel(cfg).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def topk_indices(probs: np.ndarray, k: int) -> np.ndarray:
    k = min(k, probs.shape[1])
    idx = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    row_probs = np.take_along_axis(probs, idx, axis=1)
    order = np.argsort(-row_probs, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def row_rank_of_member(probs: np.ndarray, member_idx: int) -> np.ndarray:
    member_prob = probs[:, member_idx][:, None]
    return (probs > member_prob).sum(axis=1) + 1


def format_counter(counter: Counter[str], limit: int = 8) -> list[dict[str, object]]:
    return [{"member": name, "count": int(cnt)} for name, cnt in counter.most_common(limit)]


def main() -> None:
    args = parse_args()
    run_cfg = load_run_config(Path(args.run_config))
    device = torch.device(args.device)

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
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    model = build_model(Path(args.checkpoint), run_cfg, device)

    print(f"[config] checkpoint={args.checkpoint}", flush=True)
    print(f"[config] split={args.split} batch_size={args.batch_size} top_k={args.top_k}", flush=True)
    print(f"[data] metas={len(metas)} dataset={len(dataset)} loader_batches={len(loader)}", flush=True)

    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    records: list[dict[str, object]] = []

    cursor = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            tokens = batch["tokens"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            center_indices = batch["center_indices"].to(device)
            out = model(tokens=tokens, attn_mask=attn_mask, center_indices=center_indices)
            bind_probs = torch.sigmoid(out["center_bind_logits"])
            m6a_gate = torch.sigmoid(out["center_m6a_logit"]).unsqueeze(-1)
            gated_probs = (bind_probs * m6a_gate).cpu().numpy().astype(np.float32)
            targets = batch["rbp_targets"].cpu().numpy().astype(np.float32)

            batch_size = gated_probs.shape[0]
            batch_metas = metas[cursor: cursor + batch_size]
            cursor += batch_size

            all_probs.append(gated_probs)
            all_targets.append(targets)

            for i, meta in enumerate(batch_metas):
                records.append(
                    {
                        "transcript_id": meta.transcript_id,
                        "site_pos": int(meta.site_pos),
                        "is_positive": bool(meta.is_positive),
                        "prob": gated_probs[i],
                        "target": targets[i],
                    }
                )

            if batch_idx == 1 or batch_idx % int(args.print_every) == 0 or batch_idx == len(loader):
                print(
                    f"[forward] batch={batch_idx}/{len(loader)} cursor={cursor}",
                    flush=True,
                )

    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    is_pos = np.array([bool(rec["is_positive"]) for rec in records], dtype=np.bool_)
    top1 = probs.argmax(axis=1)
    topk = topk_indices(probs, args.top_k)
    eval_result = full_evaluate(gated_probs=probs, targets=targets, is_pos=is_pos)
    opt_thresholds = _optimize_thresholds(probs, targets)

    summary_rows: list[dict[str, object]] = []
    for member_idx, member_name in enumerate(INDIVIDUAL_RBPS):
        y_true = (targets[:, member_idx] > 0.5).astype(np.int64)
        n_pos = int(y_true.sum())
        if n_pos == 0:
            summary_rows.append(
                {
                    "member": member_name,
                    "n_pos": 0,
                    "auprc": float("nan"),
                    "auroc": float("nan"),
                    "f1opt": float("nan"),
                    "best_threshold": 0.5,
                    "top1_hit": float("nan"),
                    "top3_hit": float("nan"),
                    "mean_rank": float("nan"),
                    "median_rank": float("nan"),
                }
            )
            continue

        pos_mask = y_true.astype(bool)
        ranks = row_rank_of_member(probs[pos_mask], member_idx)
        summary_rows.append(
            {
                "member": member_name,
                "n_pos": n_pos,
                "auprc": float(eval_result.get(f"{member_name}_auprc", float("nan"))),
                "auroc": float(eval_result.get(f"{member_name}_auroc", float("nan"))),
                "f1opt": float(eval_result.get(f"{member_name}_f1_opt", float("nan"))),
                "best_threshold": float(opt_thresholds[member_idx]),
                "top1_hit": float((top1[pos_mask] == member_idx).mean()),
                "top3_hit": float((topk[pos_mask] == member_idx).any(axis=1).mean()),
                "mean_rank": float(np.mean(ranks)),
                "median_rank": float(np.median(ranks)),
            }
        )

    valid_rows = [row for row in summary_rows if not np.isnan(row["f1opt"])]
    valid_rows.sort(key=lambda x: (x["f1opt"], x["auprc"]))

    if args.focus_members.strip():
        focus_members = [x.strip().upper() for x in args.focus_members.split(",") if x.strip()]
    else:
        focus_members = [row["member"] for row in valid_rows[: args.bottom_k]]

    focus_analysis: dict[str, dict[str, object]] = {}
    for member_name in focus_members:
        member_idx = RBP_TO_IDX[member_name]
        pos_sites = np.where(targets[:, member_idx] > 0.5)[0]
        if pos_sites.size == 0:
            continue

        top1_wrong = pos_sites[top1[pos_sites] != member_idx]
        topk_miss = pos_sites[~(topk[pos_sites] == member_idx).any(axis=1)]
        outranker_counter: Counter[str] = Counter()
        top1_counter: Counter[str] = Counter()
        topk_counter: Counter[str] = Counter()
        missed_examples: list[dict[str, object]] = []

        for row_idx in top1_wrong:
            top1_counter[INDIVIDUAL_RBPS[int(top1[row_idx])]] += 1

        for row_idx in topk_miss:
            for pred_idx in topk[row_idx]:
                topk_counter[INDIVIDUAL_RBPS[int(pred_idx)]] += 1

        for row_idx in pos_sites:
            member_prob = float(probs[row_idx, member_idx])
            higher = np.where(probs[row_idx] > member_prob)[0]
            for other_idx in higher.tolist():
                if other_idx != member_idx:
                    outranker_counter[INDIVIDUAL_RBPS[int(other_idx)]] += 1

        if topk_miss.size > 0:
            ranked_miss = sorted(
                topk_miss.tolist(),
                key=lambda i: float(probs[i, member_idx]),
                reverse=True,
            )
            for row_idx in ranked_miss[: args.max_examples]:
                pred_members = [
                    {"member": INDIVIDUAL_RBPS[int(j)], "prob": float(probs[row_idx, int(j)])}
                    for j in topk[row_idx]
                ]
                true_members = [
                    INDIVIDUAL_RBPS[j]
                    for j in np.where(targets[row_idx] > 0.5)[0].tolist()
                ]
                rec = records[row_idx]
                missed_examples.append(
                    {
                        "transcript_id": rec["transcript_id"],
                        "site_pos": int(rec["site_pos"]),
                        "true_members": true_members,
                        "this_member_prob": float(probs[row_idx, member_idx]),
                        "this_member_rank": int(row_rank_of_member(probs[row_idx : row_idx + 1], member_idx)[0]),
                        "topk_pred": pred_members,
                    }
                )

        focus_analysis[member_name] = {
            "member": member_name,
            "n_pos": int(pos_sites.size),
            "top1_hit_rate": float((top1[pos_sites] == member_idx).mean()),
            "topk_hit_rate": float((topk[pos_sites] == member_idx).any(axis=1).mean()),
            "top1_wrong_most_common": format_counter(top1_counter),
            "topk_miss_front_runners": format_counter(topk_counter),
            "most_common_outrankers": format_counter(outranker_counter),
            "miss_examples": missed_examples,
        }

    payload = {
        "run_config": str(args.run_config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "top_k": int(args.top_k),
        "summary_rows": summary_rows,
        "focus_members": focus_members,
        "focus_analysis": focus_analysis,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines: list[str] = []
    lines.append("# ETD Together Member Error Analysis")
    lines.append("")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- split: `{args.split}`")
    lines.append(f"- top-k: `{args.top_k}`")
    lines.append("")
    lines.append("## Lowest-F1 Members")
    lines.append("")
    lines.append("| member | n_pos | AUPRC | AUROC | F1opt | top1_hit | top3_hit | mean_rank |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in valid_rows[: max(args.bottom_k, len(focus_members))]:
        lines.append(
            f"| {row['member']} | {row['n_pos']} | {row['auprc']:.4f} | {row['auroc']:.4f} | "
            f"{row['f1opt']:.4f} | {row['top1_hit']:.4f} | {row['top3_hit']:.4f} | {row['mean_rank']:.2f} |"
        )

    for member_name in focus_members:
        item = focus_analysis.get(member_name)
        if item is None:
            continue
        lines.append("")
        lines.append(f"## {member_name}")
        lines.append("")
        lines.append(
            f"- n_pos: `{item['n_pos']}`; top1_hit: `{item['top1_hit_rate']:.4f}`; "
            f"top{args.top_k}_hit: `{item['topk_hit_rate']:.4f}`"
        )
        lines.append("- top1 最常错成：")
        for row in item["top1_wrong_most_common"]:
            lines.append(f"  - `{row['member']}`: {row['count']}")
        lines.append(f"- top{args.top_k} miss 时，前排最常出现：")
        for row in item["topk_miss_front_runners"]:
            lines.append(f"  - `{row['member']}`: {row['count']}")
        lines.append("- 最常压过它的 member：")
        for row in item["most_common_outrankers"]:
            lines.append(f"  - `{row['member']}`: {row['count']}")
        lines.append("- 代表性 miss 样本：")
        for row in item["miss_examples"]:
            topk_str = ", ".join(f"{x['member']}({x['prob']:.3f})" for x in row["topk_pred"])
            true_str = ", ".join(row["true_members"])
            lines.append(
                f"  - `{row['transcript_id']}:{row['site_pos']}` "
                f"true=[{true_str}] this_prob={row['this_member_prob']:.4f} "
                f"rank={row['this_member_rank']} topk=[{topk_str}]"
            )

    with output_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[ok] wrote {output_json}")
    print(f"[ok] wrote {output_md}")


if __name__ == "__main__":
    main()
