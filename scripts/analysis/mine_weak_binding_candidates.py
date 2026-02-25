#!/usr/bin/env python3
"""Mine weak-binding candidates from unlabeled sites with high uncertainty."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine weak-binding candidates.")
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

    parser.add_argument("--prob-min", type=float, default=0.2)
    parser.add_argument("--prob-max", type=float, default=0.6)
    parser.add_argument("--unc-min", type=float, default=0.6)

    parser.add_argument("--output-prefix", default=str(REPO_ROOT / "outputs/analysis/weak_binding_candidates"))
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


def is_weak_candidate(
    label: int,
    prob: float,
    uncertainty: float,
    prob_min: float,
    prob_max: float,
    unc_min: float,
) -> bool:
    if label != -1:
        return False
    if prob < prob_min or prob > prob_max:
        return False
    if uncertainty < unc_min:
        return False
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

    bpp_cache = BPPCache(args.rnafold_cache)

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
            batch = collate_batch(
                examples=batch_examples,
                task_name="bind",
                role_name=args.role,
                cond_base="A",
                bpp_cache=bpp_cache,
                strong_binding_threshold=1.0,
                rng=random.Random(0),
                mod_unlabeled_ratio=1.0,
                mask_prob=0.15,
            )

            batch_t = to_device(batch, device)
            bsz = batch_t["tokens"].shape[0]
            cond_task = torch.full((bsz,), TASK_IDS["bind"], device=device, dtype=torch.long)
            cond_role = torch.full((bsz,), ROLE_IDS[args.role], device=device, dtype=torch.long)
            cond_base = torch.full((bsz,), COND_BASE_IDS["A"], device=device, dtype=torch.long)

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

            probs = torch.sigmoid(outputs["bind_logits"]).cpu().numpy()
            unc = outputs["bind_uncertainty"].cpu().numpy()

            site_mask = batch["site_mask"].numpy().astype(bool)
            site_pos = batch["site_positions"].numpy()
            labels = batch["site_pu_labels"].numpy()
            support = batch["site_support"].numpy()
            txids = batch["transcript_ids"]

            for row_idx in range(site_mask.shape[0]):
                txid = txids[row_idx]
                for site_idx in range(site_mask.shape[1]):
                    if not site_mask[row_idx, site_idx]:
                        continue

                    label = int(labels[row_idx, site_idx])
                    prob = float(probs[row_idx, site_idx])
                    unc_val = float(unc[row_idx, site_idx])
                    if not is_weak_candidate(
                        label=label,
                        prob=prob,
                        uncertainty=unc_val,
                        prob_min=args.prob_min,
                        prob_max=args.prob_max,
                        unc_min=args.unc_min,
                    ):
                        continue

                    rows.append(
                        {
                            "transcript_id": txid,
                            "site_pos": int(site_pos[row_idx, site_idx]),
                            "role": args.role,
                            "site_pu_label": label,
                            "support_count": float(support[row_idx, site_idx]),
                            "bind_prob": prob,
                            "bind_uncertainty": unc_val,
                        }
                    )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["bind_uncertainty", "bind_prob"], ascending=[False, True]).reset_index(drop=True)

    prefix = Path(args.output_prefix)
    csv_path = prefix.with_suffix(".csv")
    parquet_path = prefix.with_suffix(".parquet")
    summary_path = prefix.with_suffix(".json")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    summary = {
        "n_candidates": int(df.shape[0]),
        "split": args.split,
        "role": args.role,
        "prob_min": args.prob_min,
        "prob_max": args.prob_max,
        "unc_min": args.unc_min,
        "csv": str(csv_path),
        "parquet": str(parquet_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
