#!/usr/bin/env python3
"""Dump one-example collate_batch result to JSON for debugging.

用途：
- 从 processed 数据集中取 1 条转录本样本。
- 走一遍当前 `collate_batch` 逻辑（含 A' 替换、mod/bind/mask/struct 标签构造）。
- 把关键结果写到输出 JSON，便于人工核对。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.etd_multitask.constants import APRIME_TOKEN_ID, BASE_TO_ID, MASK_TOKEN_ID, PAD_TOKEN_ID
from models.etd_multitask.data import BPPCache, collate_batch, load_examples
from models.etd_multitask.rnafold_online import OnlineRNAfoldProvider


TOKEN_ID_TO_SYMBOL = {
    BASE_TO_ID["A"]: "A",
    BASE_TO_ID["C"]: "C",
    BASE_TO_ID["G"]: "G",
    BASE_TO_ID["U"]: "U",
    BASE_TO_ID["N"]: "N",
    PAD_TOKEN_ID: "[PAD]",
    MASK_TOKEN_ID: "[MASK]",
    APRIME_TOKEN_ID: "A'",
}


class StageProgress:
    """Lightweight stage progress bar without external dependencies."""

    def __init__(self, total: int) -> None:
        self.total = max(1, int(total))
        self.current = 0
        self.width = 30
        self.is_tty = sys.stdout.isatty()

    def update(self, desc: str) -> None:
        self.current = min(self.current + 1, self.total)
        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        if self.is_tty:
            sys.stdout.write(f"\r[{bar}] {self.current}/{self.total} {desc}")
            sys.stdout.flush()
            if self.current == self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()
        else:
            print(f"[{bar}] {self.current}/{self.total} {desc}", flush=True)


def log(msg: str) -> None:
    print(f"[debug] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump one collate_batch example to JSON.")

    parser.add_argument("--sites", default=str(REPO_ROOT / "data/processed/m6a_multitask_sites.parquet"))
    parser.add_argument("--transcripts", default=str(REPO_ROOT / "data/processed/m6a_multitask_transcripts.parquet"))
    parser.add_argument("--splits", default=str(REPO_ROOT / "data/processed/m6a_multitask_splits.json"))
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])

    parser.add_argument("--task", default="bind", choices=["bind", "mod", "struct", "mask"])
    parser.add_argument("--role", default="reader", choices=["writer", "reader", "eraser", "none"])
    parser.add_argument("--cond-base", default="A", choices=["A", "C", "G", "U", "mask"])

    parser.add_argument("--index", type=int, default=0, help="index in selected split examples")
    parser.add_argument("--transcript-id", default="", help="optional exact transcript id to select")

    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--strong-binding-threshold", type=float, default=1.0)
    parser.add_argument("--mod-unlabeled-ratio", type=float, default=1.0)
    parser.add_argument("--mask-prob", type=float, default=0.15)

    parser.add_argument("--aprime-enable", action="store_true", default=True)
    parser.add_argument("--no-aprime-enable", action="store_false", dest="aprime_enable")
    parser.add_argument("--aprime-prob", type=float, default=0.1)
    parser.add_argument("--aprime-max-per-seq", type=int, default=-1)

    parser.add_argument("--struct-source", default="online", choices=["precomputed", "online"])
    parser.add_argument("--rnafold-cache", default=str(REPO_ROOT / "data/processed/rnafold_bpp"))
    parser.add_argument("--online-rnafold-bin", default="RNAfold")
    parser.add_argument("--online-rnafold-timeout-seconds", type=int, default=240)
    parser.add_argument("--online-rnafold-cache-size", type=int, default=128)
    parser.add_argument("--use-rnafold-struct-feats", action="store_true", default=True)
    parser.add_argument("--no-use-rnafold-struct-feats", action="store_false", dest="use_rnafold_struct_feats")

    parser.add_argument("--struct-feat-preview-len", type=int, default=64)
    parser.add_argument("--struct-target-preview-size", type=int, default=16)
    parser.add_argument("--output", default=str(REPO_ROOT / "debug/output/single_collate_debug.json"))
    return parser.parse_args()


def build_struct_provider(args: argparse.Namespace):
    if args.struct_source == "online":
        return OnlineRNAfoldProvider(
            rnafold_bin=args.online_rnafold_bin,
            timeout_seconds=args.online_rnafold_timeout_seconds,
            cache_size=args.online_rnafold_cache_size,
        )
    return BPPCache(args.rnafold_cache)


def select_example(examples, args: argparse.Namespace):
    if args.transcript_id:
        for ex in examples:
            if ex.transcript_id == args.transcript_id:
                return ex
        raise ValueError(f"transcript_id not found in split={args.split}: {args.transcript_id}")

    idx = int(args.index)
    if idx < 0 or idx >= len(examples):
        raise IndexError(f"index out of range: {idx}, total={len(examples)}")
    return examples[idx]


def ids_to_symbols(ids: np.ndarray) -> list[str]:
    return [TOKEN_ID_TO_SYMBOL.get(int(x), f"<{int(x)}>") for x in ids.tolist()]


def to_bool_list(arr: np.ndarray) -> list[bool]:
    return [bool(x) for x in arr.tolist()]


def main() -> None:
    progress = StageProgress(total=5)
    t0 = time.perf_counter()
    args = parse_args()
    log(
        f"start task={args.task} role={args.role} split={args.split} "
        f"struct_source={args.struct_source}"
    )

    examples = load_examples(
        sites_path=args.sites,
        transcripts_path=args.transcripts,
        splits_path=args.splits,
        split_names=[args.split],
        max_len=args.max_len,
        smoke_ratio=1.0,
        seed=args.seed,
    )
    progress.update("loaded examples")
    log(f"loaded {len(examples)} examples")
    if not examples:
        raise RuntimeError("No examples loaded. Check --sites/--transcripts/--splits and --split.")

    item = select_example(examples, args)
    progress.update("selected sample")
    log(f"selected transcript_id={item.transcript_id} seq_len={item.seq_len}")
    struct_provider = build_struct_provider(args)
    progress.update("built struct provider")
    log("built struct provider; running collate_batch (this may take time for online RNAfold)")

    rng = random.Random(args.seed)
    t_collate = time.perf_counter()
    batch = collate_batch(
        examples=[item],
        task_name=args.task,
        role_name=args.role,
        cond_base=args.cond_base,
        struct_provider=struct_provider,
        strong_binding_threshold=args.strong_binding_threshold,
        rng=rng,
        mod_unlabeled_ratio=args.mod_unlabeled_ratio,
        mask_prob=args.mask_prob,
        aprime_enable=args.aprime_enable,
        aprime_prob=args.aprime_prob,
        aprime_max_per_seq=args.aprime_max_per_seq,
        use_rnafold_struct_feats=args.use_rnafold_struct_feats,
    )
    progress.update("collated one sample")
    log(f"collate finished in {time.perf_counter() - t_collate:.2f}s")

    tokens = batch["tokens"][0].cpu().numpy()
    attn_mask = batch["attn_mask"][0].cpu().numpy().astype(bool)
    valid_len = int(attn_mask.sum())

    tokens_valid = tokens[:valid_len]
    token_symbols = ids_to_symbols(tokens_valid)
    aprime_positions = np.where(tokens_valid == APRIME_TOKEN_ID)[0].astype(int).tolist()

    mod_labels = batch["mod_pu_labels"][0].cpu().numpy()[:valid_len]
    mod_mask = batch["mod_pu_mask"][0].cpu().numpy().astype(bool)[:valid_len]

    mod_positive_positions = np.where(mod_mask & (mod_labels == 1))[0].astype(int).tolist()
    mod_unlabeled_positions = np.where(mod_mask & (mod_labels == -1))[0].astype(int).tolist()

    site_positions = batch["site_positions"][0].cpu().numpy()
    site_labels = batch["site_pu_labels"][0].cpu().numpy()
    site_mask = batch["site_mask"][0].cpu().numpy().astype(bool)
    site_support = batch["site_support"][0].cpu().numpy()
    strong = batch["strong_binding_mask"][0].cpu().numpy().astype(bool)
    g1 = batch["g1_mask"][0].cpu().numpy().astype(bool)
    g2 = batch["g2_mask"][0].cpu().numpy().astype(bool)
    g3 = batch["g3_mask"][0].cpu().numpy().astype(bool)
    g4 = batch["g4_mask"][0].cpu().numpy().astype(bool)
    g5 = batch["g5_mask"][0].cpu().numpy().astype(bool)

    bind_sites = []
    for i in np.where(site_mask)[0].astype(int).tolist():
        bind_sites.append(
            {
                "slot": int(i),
                "position": int(site_positions[i]),
                "label": int(site_labels[i]),
                "support": float(site_support[i]),
                "strong": bool(strong[i]),
                "g1": bool(g1[i]),
                "g2": bool(g2[i]),
                "g3": bool(g3[i]),
                "g4": bool(g4[i]),
                "g5": bool(g5[i]),
            }
        )

    mlm_input = batch["mlm_input"][0].cpu().numpy()[:valid_len]
    mlm_target = batch["mlm_target"][0].cpu().numpy()[:valid_len]
    mlm_masked_positions = np.where(mlm_target != -100)[0].astype(int).tolist()

    struct_feats = batch["struct_feats"][0].cpu().numpy()
    struct_preview_len = min(int(args.struct_feat_preview_len), valid_len)
    struct_target = batch["struct_target"][0].cpu().numpy()
    struct_target_n = min(int(args.struct_target_preview_size), int(struct_target.shape[0]))

    payload = {
        "config": {
            "split": args.split,
            "task": args.task,
            "role": args.role,
            "cond_base": args.cond_base,
            "seed": args.seed,
            "aprime_enable": bool(args.aprime_enable),
            "aprime_prob": float(args.aprime_prob),
            "aprime_max_per_seq": int(args.aprime_max_per_seq),
            "struct_source": args.struct_source,
            "use_rnafold_struct_feats": bool(args.use_rnafold_struct_feats),
        },
        "example": {
            "transcript_id": item.transcript_id,
            "seq_len": int(item.seq_len),
            "m6a_positions": [int(x) for x in item.m6a_positions.tolist()],
            "unlabeled_a_count": int(item.unlabeled_a_positions.shape[0]),
            "role_label_counts": {
                role: {
                    "pos": int((labels == 1).sum()),
                    "unlabeled": int((labels == -1).sum()),
                }
                for role, labels in item.role_labels.items()
            },
        },
        "tokens": {
            "valid_len": valid_len,
            "token_ids": [int(x) for x in tokens_valid.tolist()],
            "token_symbols": token_symbols,
            "aprime_positions": aprime_positions,
            "attn_mask": to_bool_list(attn_mask[:valid_len]),
        },
        "mod_supervision": {
            "positive_positions": mod_positive_positions,
            "unlabeled_positions": mod_unlabeled_positions,
            "n_positive": len(mod_positive_positions),
            "n_unlabeled": len(mod_unlabeled_positions),
        },
        "bind_supervision": {
            "n_valid_sites": int(site_mask.sum()),
            "sites": bind_sites,
        },
        "mask_supervision": {
            "masked_positions": mlm_masked_positions,
            "n_masked": len(mlm_masked_positions),
            "mlm_input_symbols": ids_to_symbols(mlm_input),
            "mlm_target_ids": [int(x) for x in mlm_target.tolist()],
        },
        "structure": {
            "struct_feats_shape": [int(x) for x in batch["struct_feats"][0].shape],
            "struct_feats_preview": struct_feats[:struct_preview_len].round(6).tolist(),
            "struct_target_shape": [int(x) for x in struct_target.shape],
            "struct_lengths": int(batch["struct_lengths"][0].item()),
            "struct_target_preview": struct_target[:struct_target_n, :struct_target_n].round(6).tolist(),
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    progress.update("wrote output")
    log(f"wrote debug payload to: {out}")
    log(f"done in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
