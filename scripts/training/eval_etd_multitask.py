#!/usr/bin/env python3
"""Evaluate ETD multi-task model on m6A multitask splits."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
from pathlib import Path

import numpy as np

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
import torch.nn.functional as F

from models.etd_multitask.constants import COND_BASE_IDS, ROLE_IDS, ROLE_NAMES, TASK_IDS
from models.etd_multitask.data import BPPCache, build_length_bucketed_batches, collate_batch, load_examples
from models.etd_multitask.losses import dirichlet_binary_nll
from models.etd_multitask.metrics import (
    binary_accuracy,
    binary_auprc,
    binary_auroc,
    binary_f1,
    binary_nll,
    expected_calibration_error,
    structure_f1,
)
from models.etd_multitask.model import ETDMultiTaskModel
from models.etd_multitask.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ETD multitask checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sites", default=str(REPO_ROOT / "data/processed/m6a_multitask_sites.parquet"))
    parser.add_argument("--transcripts", default=str(REPO_ROOT / "data/processed/m6a_multitask_transcripts.parquet"))
    parser.add_argument("--splits", default=str(REPO_ROOT / "data/processed/m6a_multitask_splits.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--rnafold-cache", default=str(REPO_ROOT / "data/processed/rnafold_bpp"))
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--batch-token-budget", type=int, default=24000)
    parser.add_argument("--bucket-boundaries", default="1024,2048,4096,8192,12000")
    parser.add_argument("--max-struct-len", type=int, default=4000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def evaluate_binding(
    model: ETDMultiTaskModel,
    examples,
    role: str,
    struct_provider: BPPCache,
    boundaries: list[int],
    batch_token_budget: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    batches = build_length_bucketed_batches(
        examples=examples,
        batch_token_budget=batch_token_budget,
        boundaries=boundaries,
        shuffle=False,
        seed=0,
    )

    y_true_all = []
    y_prob_all = []
    alpha_all = []
    strong_unc = []
    weak_unc = []

    with torch.no_grad():
        for batch_examples in batches:
            batch = collate_batch(
                examples=batch_examples,
                task_name="bind",
                role_name=role,
                cond_base="A",
                struct_provider=struct_provider,
                strong_binding_threshold=1.0,
                rng=random.Random(0),
                mod_unlabeled_ratio=1.0,
                mask_prob=0.15,
            )
            batch = to_device(batch, device)

            bsz = batch["tokens"].shape[0]
            cond_task = torch.full((bsz,), TASK_IDS["bind"], device=device, dtype=torch.long)
            cond_role = torch.full((bsz,), ROLE_IDS[role], device=device, dtype=torch.long)
            cond_base = torch.full((bsz,), COND_BASE_IDS["A"], device=device, dtype=torch.long)

            outputs = model(
                tokens=batch["tokens"],
                cond_task=cond_task,
                cond_role=cond_role,
                cond_base=cond_base,
                attn_mask=batch["attn_mask"],
                site_positions=batch["site_positions"],
                site_mask=batch["site_mask"],
            )

            mask = batch["site_mask"].detach().cpu().numpy().astype(bool)
            y_true = (batch["site_pu_labels"].detach().cpu().numpy() == 1).astype(np.int64)
            y_prob = torch.sigmoid(outputs["bind_logits"]).detach().cpu().numpy()
            alpha = outputs["bind_alpha"].detach().cpu().numpy()
            unc = outputs["bind_uncertainty"].detach().cpu().numpy()
            support = batch["site_support"].detach().cpu().numpy()

            y_true_all.append(y_true[mask])
            y_prob_all.append(y_prob[mask])
            alpha_all.append(alpha[mask])

            strong = (support >= np.quantile(support[mask], 0.75) if np.any(mask) else support) & mask & (y_true == 1)
            weak = mask & ~strong
            if np.any(strong):
                strong_unc.append(unc[strong])
            if np.any(weak):
                weak_unc.append(unc[weak])

    if not y_true_all:
        return {}

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    alpha = np.concatenate(alpha_all, axis=0)

    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    y_true_t = torch.tensor(y_true, dtype=torch.long)
    dir_nll = float(dirichlet_binary_nll(alpha_t, y_true_t).item())

    strong_mean = float(np.concatenate(strong_unc).mean()) if strong_unc else float("nan")
    weak_mean = float(np.concatenate(weak_unc).mean()) if weak_unc else float("nan")

    return {
        "auroc": binary_auroc(y_true, y_prob),
        "auprc": binary_auprc(y_true, y_prob),
        "f1": binary_f1(y_true, y_prob),
        "acc": binary_accuracy(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "nll": binary_nll(y_true, y_prob),
        "dirichlet_nll": dir_nll,
        "unc_strong_mean": strong_mean,
        "unc_weak_mean": weak_mean,
        "unc_gap": float((weak_mean - strong_mean) / max(weak_mean, 1e-6)) if np.isfinite(weak_mean) and np.isfinite(strong_mean) else float("nan"),
        "n_samples": int(y_true.shape[0]),
    }


def evaluate_mod(
    model: ETDMultiTaskModel,
    examples,
    struct_provider: BPPCache,
    boundaries: list[int],
    batch_token_budget: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    batches = build_length_bucketed_batches(
        examples=examples,
        batch_token_budget=batch_token_budget,
        boundaries=boundaries,
        shuffle=False,
        seed=0,
    )

    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for batch_examples in batches:
            batch = collate_batch(
                examples=batch_examples,
                task_name="mod",
                role_name="reader",
                cond_base="A",
                struct_provider=struct_provider,
                strong_binding_threshold=1.0,
                rng=random.Random(0),
                mod_unlabeled_ratio=1.0,
                mask_prob=0.15,
            )
            batch = to_device(batch, device)

            bsz = batch["tokens"].shape[0]
            cond_task = torch.full((bsz,), TASK_IDS["mod"], device=device, dtype=torch.long)
            cond_role = torch.full((bsz,), ROLE_IDS["none"], device=device, dtype=torch.long)
            cond_base = torch.full((bsz,), COND_BASE_IDS["A"], device=device, dtype=torch.long)

            outputs = model(
                tokens=batch["tokens"],
                cond_task=cond_task,
                cond_role=cond_role,
                cond_base=cond_base,
                attn_mask=batch["attn_mask"],
                site_positions=batch["site_positions"],
                site_mask=batch["site_mask"],
            )

            mask = batch["mod_pu_mask"].detach().cpu().numpy().astype(bool)
            y_true = (batch["mod_pu_labels"].detach().cpu().numpy() == 1).astype(np.int64)
            y_prob = torch.sigmoid(outputs["mod_logits_acu"][..., 0]).detach().cpu().numpy()

            y_true_all.append(y_true[mask])
            y_prob_all.append(y_prob[mask])

    if not y_true_all:
        return {}

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    return {
        "auroc": binary_auroc(y_true, y_prob),
        "auprc": binary_auprc(y_true, y_prob),
        "f1": binary_f1(y_true, y_prob),
        "acc": binary_accuracy(y_true, y_prob),
        "n_samples": int(y_true.shape[0]),
    }


def evaluate_structure(
    model: ETDMultiTaskModel,
    examples,
    struct_provider: BPPCache,
    boundaries: list[int],
    batch_token_budget: int,
    max_struct_len: int,
    device: torch.device,
) -> dict[str, float]:
    subset = [item for item in examples if item.seq_len <= max_struct_len]
    if not subset:
        return {}

    model.eval()
    batches = build_length_bucketed_batches(
        examples=subset,
        batch_token_budget=batch_token_budget,
        boundaries=boundaries,
        shuffle=False,
        seed=0,
    )

    scores = []

    with torch.no_grad():
        for batch_examples in batches:
            batch = collate_batch(
                examples=batch_examples,
                task_name="struct",
                role_name="reader",
                cond_base="A",
                struct_provider=struct_provider,
                strong_binding_threshold=1.0,
                rng=random.Random(0),
                mod_unlabeled_ratio=1.0,
                mask_prob=0.15,
            )
            batch = to_device(batch, device)

            bsz = batch["tokens"].shape[0]
            cond_task = torch.full((bsz,), TASK_IDS["struct"], device=device, dtype=torch.long)
            cond_role = torch.full((bsz,), ROLE_IDS["none"], device=device, dtype=torch.long)
            cond_base = torch.full((bsz,), COND_BASE_IDS["A"], device=device, dtype=torch.long)

            outputs = model(
                tokens=batch["tokens"],
                cond_task=cond_task,
                cond_role=cond_role,
                cond_base=cond_base,
                attn_mask=batch["attn_mask"],
                site_positions=batch["site_positions"],
                site_mask=batch["site_mask"],
            )

            pred = torch.sigmoid(outputs["struct_logits"]).detach().cpu().numpy()
            target = batch["struct_target"].detach().cpu().numpy()
            lengths = batch["struct_lengths"].detach().cpu().numpy()

            for idx in range(pred.shape[0]):
                f1 = structure_f1(pred[idx], target[idx], int(lengths[idx]), threshold=0.5, min_sep=4)
                if np.isfinite(f1):
                    scores.append(float(f1))

    if not scores:
        return {}
    return {
        "pair_f1": float(np.mean(scores)),
        "n_sequences": len(scores),
    }


def evaluate_mask(
    model: ETDMultiTaskModel,
    examples,
    struct_provider: BPPCache,
    boundaries: list[int],
    batch_token_budget: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    batches = build_length_bucketed_batches(
        examples=examples,
        batch_token_budget=batch_token_budget,
        boundaries=boundaries,
        shuffle=False,
        seed=0,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_examples in batches:
            batch = collate_batch(
                examples=batch_examples,
                task_name="mask",
                role_name="reader",
                cond_base="mask",
                struct_provider=struct_provider,
                strong_binding_threshold=1.0,
                rng=random.Random(0),
                mod_unlabeled_ratio=1.0,
                mask_prob=0.15,
            )
            batch = to_device(batch, device)

            bsz = batch["tokens"].shape[0]
            cond_task = torch.full((bsz,), TASK_IDS["mask"], device=device, dtype=torch.long)
            cond_role = torch.full((bsz,), ROLE_IDS["none"], device=device, dtype=torch.long)
            cond_base = torch.full((bsz,), COND_BASE_IDS["mask"], device=device, dtype=torch.long)

            outputs = model(
                tokens=batch["mlm_input"],
                cond_task=cond_task,
                cond_role=cond_role,
                cond_base=cond_base,
                attn_mask=batch["attn_mask"],
                site_positions=batch["site_positions"],
                site_mask=batch["site_mask"],
            )

            target = batch["mlm_target"]
            logits = outputs["mask_logits"]
            pred = logits.argmax(dim=-1)

            mask = target >= 0
            if mask.any():
                correct += int((pred[mask] == target[mask]).sum().item())
                total += int(mask.sum().item())

    if total == 0:
        return {}
    return {
        "masked_token_acc": float(correct / total),
        "n_masked_tokens": int(total),
    }


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
        smoke_ratio=1.0,
        seed=42,
    )

    model = ETDMultiTaskModel().to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)

    struct_provider = BPPCache(args.rnafold_cache)

    results = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_examples": len(examples),
        "binding": {},
        "mod": {},
        "structure": {},
        "mask": {},
    }

    for role in ROLE_NAMES:
        results["binding"][role] = evaluate_binding(
            model=model,
            examples=examples,
            role=role,
            struct_provider=struct_provider,
            boundaries=boundaries,
            batch_token_budget=args.batch_token_budget,
            device=device,
        )

    results["mod"] = evaluate_mod(
        model=model,
        examples=examples,
        struct_provider=struct_provider,
        boundaries=boundaries,
        batch_token_budget=args.batch_token_budget,
        device=device,
    )

    results["structure"] = evaluate_structure(
        model=model,
        examples=examples,
        struct_provider=struct_provider,
        boundaries=boundaries,
        batch_token_budget=args.batch_token_budget,
        max_struct_len=args.max_struct_len,
        device=device,
    )

    results["mask"] = evaluate_mask(
        model=model,
        examples=examples,
        struct_provider=struct_provider,
        boundaries=boundaries,
        batch_token_budget=args.batch_token_budget,
        device=device,
    )

    if args.output:
        save_json(args.output, results)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
