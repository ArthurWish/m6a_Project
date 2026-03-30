#!/usr/bin/env python3
"""离线分析 ETD-only bind 的错分模式。

功能：
1. 对给定 checkpoint 和 split，按 family 统计 TP/FP/FN/TN、precision/recall/F1。
2. 导出每个 family 的 hardest false positives / false negatives。
3. 每条错误样本附带：
   - transcript_id
   - site_pos
   - 以位点为中心的窗口序列
   - 是否 DRACH
   - 真值 family multi-hot
   - 预测概率 top-k families

默认分析口径：
- split: val
- 位点集合 = annotated reader m6A 位点 + sampled negative m6A 位点
- 阈值 = 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.etd_only.bind_dataloader import load_bind_examples
from models.etd_only.etd import ETDBindModel, ETDOnlyConfig
from models.etd_multitask.constants import ID_TO_BASE, NUM_RBP_FAMILIES, RBP_FAMILY_NAMES
from models.etd_multitask.metrics import binary_f1
from scripts.training.configs.etd_bind_config import parse_etd_bind_args


D_SET = {"A", "G", "U"}
R_SET = {"A", "G"}
H_SET = {"A", "C", "U"}


def is_drach_center(seq: str, pos: int) -> bool:
    if pos < 2 or pos + 2 >= len(seq):
        return False
    if seq[pos] != "A":
        return False
    return (
        seq[pos - 2] in D_SET
        and seq[pos - 1] in R_SET
        and seq[pos + 1] == "C"
        and seq[pos + 2] in H_SET
    )


def decode_token_ids(token_ids: np.ndarray) -> str:
    chars = []
    for x in token_ids.tolist():
        chars.append(ID_TO_BASE.get(int(x), "N"))
    return "".join(chars)


def seq_window(seq: str, pos: int, radius: int) -> str:
    left = max(0, pos - radius)
    right = min(len(seq), pos + radius + 1)
    return seq[left:right]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ETD-only bind errors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "outputs/etd_only/etd_bind_m6a_reader_neg05_0319/checkpoints/best.pt"),
        help="checkpoint 文件路径",
    )
    parser.add_argument("--sites-path", default=str(REPO_ROOT / "data/processed/all_multitask_sites.parquet"))
    parser.add_argument("--transcripts-path", default=str(REPO_ROOT / "data/processed/all_multitask_transcripts.parquet"))
    parser.add_argument("--splits", default=str(REPO_ROOT / "data/processed/all_multitask_splits.json"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--mod-type", default="m6A")
    parser.add_argument("--role-name", default="reader")
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--smoke-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--thresholds",
        default="0.1,0.2,0.3,0.4,0.5",
        help="额外输出哪些阈值下的 family summary，逗号分隔；会自动包含主 threshold 并去重",
    )
    parser.add_argument("--neg-ratio", type=float, default=0.5, help="分析时加入多少无注释 m6A 负位点，相对 annotated 位点数")
    parser.add_argument("--top-k", type=int, default=20, help="每个 family 保留多少条 hardest FP/FN")
    parser.add_argument("--window-radius", type=int, default=5, help="输出位点窗口半径")
    parser.add_argument("--output-json", default=str(REPO_ROOT / "outputs/analysis/etd_bind_error_analysis.json"))
    parser.add_argument("--output-md", default=str(REPO_ROOT / "outputs/analysis/etd_bind_error_analysis.md"))
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> ETDBindModel:
    cfg = ETDOnlyConfig()
    model = ETDBindModel(cfg).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def format_top_probs(prob_vec: np.ndarray, k: int = 3) -> list[dict[str, float]]:
    order = np.argsort(-prob_vec)[:k]
    return [{"family": RBP_FAMILY_NAMES[int(i)], "prob": float(prob_vec[int(i)])} for i in order]


def parse_threshold_list(primary: float, raw: str) -> list[float]:
    values = [float(primary)]
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return [float(v) for v in sorted({round(v, 6) for v in values})]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    threshold = float(args.threshold)
    thresholds = parse_threshold_list(threshold, args.thresholds)

    examples = load_bind_examples(
        sites_path=args.sites_path,
        transcripts_path=args.transcripts_path,
        splits_path=args.splits,
        split_names=[args.split],
        mod_type=args.mod_type,
        role_name=args.role_name,
        max_len=args.max_len,
        smoke_ratio=args.smoke_ratio,
        seed=args.seed,
    )
    model = build_model_from_checkpoint(checkpoint_path, device)

    rng = np.random.default_rng(args.seed)
    all_records: list[dict] = []

    print(f"[checkpoint] {checkpoint_path}")
    print(f"[split] {args.split} n_examples={len(examples)}")

    with torch.no_grad():
        for ex_idx, item in enumerate(examples, start=1):
            if ex_idx == 1 or ex_idx % 1000 == 0 or ex_idx == len(examples):
                print(f"[analyze] example {ex_idx}/{len(examples)}", flush=True)

            seq = decode_token_ids(item.token_ids)
            token_ids = torch.tensor(item.token_ids, dtype=torch.long, device=device).unsqueeze(0)
            attn_mask = torch.ones_like(token_ids, dtype=torch.bool, device=device)

            # annotated 位点
            if item.site_positions.size > 0:
                site_positions = torch.tensor(item.site_positions.reshape(1, -1), dtype=torch.long, device=device)
                out = model(token_ids=token_ids, attn_mask=attn_mask, site_positions=site_positions)
                probs = torch.sigmoid(out["site_logits"][0]).detach().cpu().numpy()
                for i, pos in enumerate(item.site_positions.tolist()):
                    all_records.append(
                        {
                            "transcript_id": item.transcript_id,
                            "site_pos": int(pos),
                            "site_kind": "annotated",
                            "sequence_window": seq_window(seq, int(pos), args.window_radius),
                            "is_drach": bool(is_drach_center(seq, int(pos))),
                            "target": item.rbp_family_targets[i].astype(np.float32),
                            "prob": probs[i].astype(np.float32),
                        }
                    )

            # sampled negative 位点
            if item.neg_site_positions.size > 0 and args.neg_ratio > 0:
                n_target = max(1, int(round(item.site_positions.shape[0] * args.neg_ratio)))
                n_sample = min(n_target, int(item.neg_site_positions.shape[0]))
                chosen = rng.choice(item.neg_site_positions.shape[0], size=n_sample, replace=False)
                neg_positions = item.neg_site_positions[chosen]
                site_positions = torch.tensor(neg_positions.reshape(1, -1), dtype=torch.long, device=device)
                out = model(token_ids=token_ids, attn_mask=attn_mask, site_positions=site_positions)
                probs = torch.sigmoid(out["site_logits"][0]).detach().cpu().numpy()
                zero_target = np.zeros((neg_positions.shape[0], NUM_RBP_FAMILIES), dtype=np.float32)
                for i, pos in enumerate(neg_positions.tolist()):
                    all_records.append(
                        {
                            "transcript_id": item.transcript_id,
                            "site_pos": int(pos),
                            "site_kind": "neg_m6a",
                            "sequence_window": seq_window(seq, int(pos), args.window_radius),
                            "is_drach": bool(is_drach_center(seq, int(pos))),
                            "target": zero_target[i],
                            "prob": probs[i].astype(np.float32),
                        }
                    )

    family_arrays: dict[str, dict[str, np.ndarray]] = {}
    for fam_idx, fam_name in enumerate(RBP_FAMILY_NAMES):
        y_true = np.asarray([int(rec["target"][fam_idx] > 0.5) for rec in all_records], dtype=np.int64)
        y_prob = np.asarray([float(rec["prob"][fam_idx]) for rec in all_records], dtype=np.float64)
        family_arrays[fam_name] = {"y_true": y_true, "y_prob": y_prob}

    threshold_summaries: dict[str, dict[str, dict]] = {}
    for current_threshold in thresholds:
        current_summary: dict[str, dict] = {}
        for fam_name in RBP_FAMILY_NAMES:
            y_true = family_arrays[fam_name]["y_true"]
            y_prob = family_arrays[fam_name]["y_prob"]
            y_pred = (y_prob >= current_threshold).astype(np.int64)

            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            precision = float(tp / max(tp + fp, 1))
            recall = float(tp / max(tp + fn, 1))
            f1 = float((2 * tp) / max((2 * tp + fp + fn), 1))
            current_summary[fam_name] = {
                "n_total": int(y_true.shape[0]),
                "n_pos": int(y_true.sum()),
                "n_neg": int((y_true == 0).sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        threshold_summaries[f"{current_threshold:.4f}"] = current_summary

    summary: dict[str, dict] = {}

    for fam_idx, fam_name in enumerate(RBP_FAMILY_NAMES):
        y_true = family_arrays[fam_name]["y_true"]
        y_prob = family_arrays[fam_name]["y_prob"]
        y_pred = (y_prob >= threshold).astype(np.int64)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(binary_f1(y_true, y_prob))

        fp_cases = []
        fn_cases = []
        for rec, yt, yp in zip(all_records, y_true.tolist(), y_prob.tolist()):
            item = {
                "transcript_id": rec["transcript_id"],
                "site_pos": int(rec["site_pos"]),
                "site_kind": rec["site_kind"],
                "sequence_window": rec["sequence_window"],
                "is_drach": bool(rec["is_drach"]),
                "true_families": [RBP_FAMILY_NAMES[i] for i, v in enumerate(rec["target"].tolist()) if v > 0.5],
                "this_family_prob": float(yp),
                "top_predicted_families": format_top_probs(rec["prob"], k=3),
            }
            pred_pos = yp >= threshold
            if yt == 0 and pred_pos:
                fp_cases.append(item)
            elif yt == 1 and not pred_pos:
                fn_cases.append(item)

        fp_cases = sorted(fp_cases, key=lambda x: -x["this_family_prob"])[: args.top_k]
        fn_cases = sorted(fn_cases, key=lambda x: -x["this_family_prob"])[: args.top_k]

        summary[fam_name] = {
            "n_total": int(y_true.shape[0]),
            "n_pos": int(y_true.sum()),
            "n_neg": int((y_true == 0).sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_topk": fp_cases,
            "fn_topk": fn_cases,
        }

    result = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "threshold": threshold,
        "thresholds": thresholds,
        "neg_ratio": args.neg_ratio,
        "n_records": len(all_records),
        "threshold_summaries": threshold_summaries,
        "families": summary,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    lines = [
        "# ETD Bind 错误分析",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- split: `{args.split}`",
        f"- threshold: `{threshold}`",
        f"- thresholds: `{', '.join(f'{v:.4f}' for v in thresholds)}`",
        f"- neg_ratio: `{args.neg_ratio}`",
        f"- n_records: `{len(all_records)}`",
        "",
        "## Family Summary",
        "",
        "| family | n_pos | tp | fp | fn | precision | recall | f1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fam_name in RBP_FAMILY_NAMES:
        item = summary[fam_name]
        lines.append(
            f"| {fam_name} | {item['n_pos']} | {item['tp']} | {item['fp']} | {item['fn']} | "
            f"{item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
        )
    lines.append("")

    if len(thresholds) > 1:
        lines.extend(
            [
                "## Family Summary By Threshold",
                "",
            ]
        )
        for current_threshold in thresholds:
            current_summary = threshold_summaries[f"{current_threshold:.4f}"]
            lines.append(f"### threshold = {current_threshold:.4f}")
            lines.append("")
            lines.append("| family | n_pos | tp | fp | fn | precision | recall | f1 |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for fam_name in RBP_FAMILY_NAMES:
                item = current_summary[fam_name]
                lines.append(
                    f"| {fam_name} | {item['n_pos']} | {item['tp']} | {item['fp']} | {item['fn']} | "
                    f"{item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |"
                )
            lines.append("")

    for fam_name in RBP_FAMILY_NAMES:
        item = summary[fam_name]
        lines.append(f"## {fam_name}")
        lines.append("")
        lines.append("### False Positives Top-K")
        lines.append("")
        if not item["fp_topk"]:
            lines.append("- none")
        else:
            for row in item["fp_topk"]:
                lines.append(
                    f"- {row['transcript_id']}:{row['site_pos']} kind={row['site_kind']} "
                    f"drach={row['is_drach']} prob={row['this_family_prob']:.4f} "
                    f"window=`{row['sequence_window']}` true={row['true_families']} "
                    f"top={row['top_predicted_families']}"
                )
        lines.append("")
        lines.append("### False Negatives Top-K")
        lines.append("")
        if not item["fn_topk"]:
            lines.append("- none")
        else:
            for row in item["fn_topk"]:
                lines.append(
                    f"- {row['transcript_id']}:{row['site_pos']} kind={row['site_kind']} "
                    f"drach={row['is_drach']} prob={row['this_family_prob']:.4f} "
                    f"window=`{row['sequence_window']}` true={row['true_families']} "
                    f"top={row['top_predicted_families']}"
                )
        lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"[done] json -> {out_json}")
    print(f"[done] md   -> {out_md}")


if __name__ == "__main__":
    main()
