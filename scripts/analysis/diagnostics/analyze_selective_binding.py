#!/usr/bin/env python3
"""Analyze selective binding shifts across condition base choices."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _bootstrap_cuda_runtime_paths() -> None:
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidate = (
        Path(sys.prefix)
        / "lib"
        / f"python{pyver}"
        / "site-packages"
        / "nvidia"
        / "cusparselt"
        / "lib"
    )
    if candidate.exists():
        curr = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [str(candidate)]
        if curr:
            parts.append(curr)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        lib_path = candidate / "libcusparseLt.so.0"
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


_bootstrap_cuda_runtime_paths()

import torch

from models.etd_multitask.constants import COND_BASE_IDS, ROLE_IDS, TASK_IDS
from models.etd_multitask.data import BPPCache, build_length_bucketed_batches, collate_batch, load_examples
from models.etd_multitask.model import ETDMultiTaskModel
from models.etd_multitask.utils import save_json


BASES = ("A", "C", "G", "U")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze selective binding by condition base.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sites", default=str(REPO_ROOT / "data/processed/m6a_multitask_sites.parquet"))
    parser.add_argument("--transcripts", default=str(REPO_ROOT / "data/processed/m6a_multitask_transcripts.parquet"))
    parser.add_argument("--splits", default=str(REPO_ROOT / "data/processed/m6a_multitask_splits.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--role", default="reader", choices=("writer", "reader", "eraser"))
    parser.add_argument("--rnafold-cache", default=str(REPO_ROOT / "data/processed/rnafold_bpp"))
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--smoke-ratio", type=float, default=1.0)
    parser.add_argument("--batch-token-budget", type=int, default=24000)
    parser.add_argument("--bucket-boundaries", default="1024,2048,4096,8192,12000")
    parser.add_argument("--output-prefix", default=str(REPO_ROOT / "outputs/analysis/selective_binding_report"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _summarize_selective(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n_sites": 0,
            "mean_prob": {base: float("nan") for base in BASES},
            "mean_unc": {base: float("nan") for base in BASES},
            "delta_prob_vs_A": {base: float("nan") for base in ("C", "G", "U")},
            "delta_unc_vs_A": {base: float("nan") for base in ("C", "G", "U")},
        }

    out = {
        "n_sites": int(df.shape[0]),
        "mean_prob": {},
        "mean_unc": {},
        "delta_prob_vs_A": {},
        "delta_unc_vs_A": {},
    }

    for base in BASES:
        out["mean_prob"][base] = float(df[f"p_{base}"].mean())
        out["mean_unc"][base] = float(df[f"u_{base}"].mean())

    for base in ("C", "G", "U"):
        out["delta_prob_vs_A"][base] = float((df[f"p_{base}"] - df["p_A"]).mean())
        out["delta_unc_vs_A"][base] = float((df[f"u_{base}"] - df["u_A"]).mean())

    return out


def _plot_summary(summary: dict, png_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    bases = ["C", "G", "U"]
    delta_prob = [summary["delta_prob_vs_A"].get(base, float("nan")) for base in bases]
    delta_unc = [summary["delta_unc_vs_A"].get(base, float("nan")) for base in bases]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(bases, delta_prob)
    axes[0].set_title("Delta Prob vs A")
    axes[0].set_ylabel("mean(p_base - p_A)")

    axes[1].bar(bases, delta_unc)
    axes[1].set_title("Delta Unc vs A")
    axes[1].set_ylabel("mean(u_base - u_A)")

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    boundaries = [int(x.strip()) for x in args.bucket_boundaries.split(",") if x.strip()]
    if not boundaries:
        boundaries = [1024, 2048, 4096, 8192, 12000]

    examples = load_examples(
        sites_path=args.sites,
        transcripts_path=args.transcripts,
        splits_path=args.splits,
        split_names=[args.split],
        max_len=args.max_len,
        smoke_ratio=args.smoke_ratio,
        seed=42,
    )

    model = ETDMultiTaskModel().to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    model.eval()

    struct_provider = BPPCache(args.rnafold_cache)

    batches = build_length_bucketed_batches(
        examples=examples,
        batch_token_budget=args.batch_token_budget,
        boundaries=boundaries,
        shuffle=False,
        seed=0,
    )

    rows = []
    with torch.no_grad():
        for batch_examples in batches:
            base_outputs = {}
            common = {}

            for base in BASES:
                batch = collate_batch(
                    examples=batch_examples,
                    task_name="bind",
                    role_name=args.role,
                    cond_base=base,
                    struct_provider=struct_provider,
                    strong_binding_threshold=1.0,
                    rng=random.Random(0),
                    mod_unlabeled_ratio=1.0,
                    mask_prob=0.15,
                )
                if base == "A":
                    common = {
                        "site_mask": batch["site_mask"].numpy().astype(bool),
                        "site_pos": batch["site_positions"].numpy(),
                        "site_label": batch["site_pu_labels"].numpy(),
                        "transcript_ids": batch["transcript_ids"],
                    }

                batch_t = to_device(batch, device)
                bsz = batch_t["tokens"].shape[0]
                cond_task = torch.full((bsz,), TASK_IDS["bind"], device=device, dtype=torch.long)
                cond_role = torch.full((bsz,), ROLE_IDS[args.role], device=device, dtype=torch.long)
                cond_base = torch.full((bsz,), COND_BASE_IDS[base], device=device, dtype=torch.long)

                outputs = model(
                    tokens=batch_t["tokens"],
                    cond_task=cond_task,
                    cond_role=cond_role,
                    cond_base=cond_base,
                    attn_mask=batch_t["attn_mask"],
                    struct_feats=batch_t["struct_feats"],
                    site_positions=batch_t["site_positions"],
                    site_mask=batch_t["site_mask"],
                )

                base_outputs[base] = {
                    "prob": torch.sigmoid(outputs["bind_logits"]).cpu().numpy(),
                    "unc": outputs["bind_uncertainty"].cpu().numpy(),
                }

            mask = common["site_mask"]
            for row_idx in range(mask.shape[0]):
                txid = common["transcript_ids"][row_idx]
                for site_idx in range(mask.shape[1]):
                    if not mask[row_idx, site_idx]:
                        continue
                    record = {
                        "transcript_id": txid,
                        "site_pos": int(common["site_pos"][row_idx, site_idx]),
                        "role": args.role,
                        "label": int(common["site_label"][row_idx, site_idx]),
                    }
                    for base in BASES:
                        record[f"p_{base}"] = float(base_outputs[base]["prob"][row_idx, site_idx])
                        record[f"u_{base}"] = float(base_outputs[base]["unc"][row_idx, site_idx])
                    rows.append(record)

    df = pd.DataFrame(rows)
    summary = _summarize_selective(df)

    prefix = Path(args.output_prefix)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    png_path = prefix.with_suffix(".png")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    summary["plot_written"] = _plot_summary(summary, png_path)
    save_json(json_path, summary)

    payload_out = {
        "summary": summary,
        "csv": str(csv_path),
        "json": str(json_path),
        "png": str(png_path),
    }
    print(json.dumps(payload_out, indent=2))


if __name__ == "__main__":
    main()
