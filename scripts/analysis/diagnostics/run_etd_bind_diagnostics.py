#!/usr/bin/env python3
"""离线运行单任务 ETD bind 的 family 诊断。

这个脚本复用单任务 bind 当前验证口径：
- 正位点: annotated `m6A + reader` sites
- 负位点: sampled unannotated `m6A` sites (`neg_m6a`)
- 输出: `targets [N, K]` / `preds [N, K]`

然后把这两个数组传给：
`models.etd_multitask.bind_diagnostics.run_full_diagnostics`
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.etd_multitask.bind_diagnostics import run_full_diagnostics
from models.etd_multitask.constants import NUM_RBP_FAMILIES, RBP_FAMILY_NAMES
from models.etd_only.bind_dataloader import load_bind_examples
from models.etd_only.etd import ETDBindModel, ETDOnlyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline diagnostics for ETD-only bind",
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
    parser.add_argument("--neg-ratio", type=float, default=0.5, help="sampled neg_m6a 数量相对 annotated 位点数的比例")
    parser.add_argument("--progress-interval", type=int, default=1000, help="每处理多少条 transcript 打一次进度；0 表示不打印")
    parser.add_argument(
        "--output-txt",
        default=str(REPO_ROOT / "outputs/analysis/etd_bind_diagnostics.txt"),
        help="诊断文本输出路径",
    )
    parser.add_argument(
        "--output-npz",
        default=str(REPO_ROOT / "outputs/analysis/etd_bind_diagnostics_arrays.npz"),
        help="保存 targets/preds 数组，便于复查",
    )
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> ETDBindModel:
    cfg = ETDOnlyConfig()
    model = ETDBindModel(cfg).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def collect_targets_preds(
    *,
    model: ETDBindModel,
    examples: list[object],
    device: torch.device,
    neg_ratio: float,
    seed: int,
    progress_interval: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    preds_annotated = []
    targets_annotated = []
    preds_neg = []
    n_annotated = 0
    n_neg_sampled = 0

    with torch.no_grad():
        for ex_idx, item in enumerate(examples, start=1):
            if progress_interval > 0 and (ex_idx == 1 or ex_idx % progress_interval == 0 or ex_idx == len(examples)):
                print(f"[collect] example {ex_idx}/{len(examples)}", flush=True)

            token_ids = torch.tensor(item.token_ids, dtype=torch.long, device=device).unsqueeze(0)
            attn_mask = torch.ones_like(token_ids, dtype=torch.bool, device=device)

            if item.site_positions.size > 0:
                site_positions = torch.tensor(item.site_positions.reshape(1, -1), dtype=torch.long, device=device)
                out = model(token_ids=token_ids, attn_mask=attn_mask, site_positions=site_positions)
                probs = torch.sigmoid(out["site_logits"][0]).detach().cpu().numpy().astype(np.float32)
                preds_annotated.append(probs)
                targets_annotated.append(item.rbp_family_targets.astype(np.float32))
                n_annotated += int(item.site_positions.shape[0])

            if item.neg_site_positions.size > 0 and neg_ratio > 0 and item.site_positions.size > 0:
                n_target = max(1, int(round(item.site_positions.shape[0] * neg_ratio)))
                n_sample = min(n_target, int(item.neg_site_positions.shape[0]))
                chosen = rng.choice(item.neg_site_positions.shape[0], size=n_sample, replace=False)
                neg_positions = item.neg_site_positions[chosen]
                site_positions = torch.tensor(neg_positions.reshape(1, -1), dtype=torch.long, device=device)
                out = model(token_ids=token_ids, attn_mask=attn_mask, site_positions=site_positions)
                probs = torch.sigmoid(out["site_logits"][0]).detach().cpu().numpy().astype(np.float32)
                preds_neg.append(probs)
                n_neg_sampled += int(neg_positions.shape[0])

    preds_annotated_all = np.concatenate(preds_annotated, axis=0) if preds_annotated else np.zeros((0, NUM_RBP_FAMILIES), dtype=np.float32)
    targets_annotated_all = np.concatenate(targets_annotated, axis=0) if targets_annotated else np.zeros((0, NUM_RBP_FAMILIES), dtype=np.float32)

    if preds_neg:
        preds_neg_all = np.concatenate(preds_neg, axis=0)
        targets_neg_all = np.zeros_like(preds_neg_all, dtype=np.float32)
        preds_final = np.concatenate([preds_annotated_all, preds_neg_all], axis=0)
        targets_final = np.concatenate([targets_annotated_all, targets_neg_all], axis=0)
    else:
        preds_final = preds_annotated_all
        targets_final = targets_annotated_all

    meta = {
        "n_examples": int(len(examples)),
        "n_annotated": int(n_annotated),
        "n_neg_sampled": int(n_neg_sampled),
        "n_total": int(preds_final.shape[0]),
    }
    return targets_final, preds_final, meta


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    device = torch.device(args.device)

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

    print(f"[checkpoint] {checkpoint_path}")
    print(f"[split] {args.split} n_examples={len(examples)}")
    print(f"[neg_ratio] {float(args.neg_ratio):.4f}")

    targets_final, preds_final, meta = collect_targets_preds(
        model=model,
        examples=examples,
        device=device,
        neg_ratio=float(args.neg_ratio),
        seed=int(args.seed),
        progress_interval=int(args.progress_interval),
    )

    print(
        f"[collected] annotated={meta['n_annotated']} neg_sampled={meta['n_neg_sampled']} "
        f"total={meta['n_total']}"
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"[checkpoint] {checkpoint_path}")
        print(f"[split] {args.split}")
        print(f"[neg_ratio] {float(args.neg_ratio):.4f}")
        print(
            f"[arrays] n_examples={meta['n_examples']} annotated={meta['n_annotated']} "
            f"neg_sampled={meta['n_neg_sampled']} total={meta['n_total']}"
        )
        run_full_diagnostics(targets_final.astype(np.int64), preds_final.astype(np.float64), list(RBP_FAMILY_NAMES))
    report = buf.getvalue()
    print(report, end="")

    out_txt = Path(args.output_txt)
    out_npz = Path(args.output_npz)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(report, encoding="utf-8")
    np.savez_compressed(
        out_npz,
        targets=targets_final.astype(np.int64),
        preds=preds_final.astype(np.float32),
        family_names=np.asarray(RBP_FAMILY_NAMES, dtype=object),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    print(f"[saved] txt -> {out_txt}")
    print(f"[saved] npz -> {out_npz}")


if __name__ == "__main__":
    main()
