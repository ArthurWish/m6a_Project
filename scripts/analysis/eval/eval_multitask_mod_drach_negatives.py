#!/usr/bin/env python3
"""评估多任务 checkpoint 在 DRACH 约束下的 m6A 识别能力。

本脚本固定的二分类定义是：
- 正样本：DRACH 基序上的 m6A 位点

在这个固定正样本定义下，分别构造 3 套负样本：
1. 非 DRACH 的 m6A
2. 非 DRACH 的 A（包含普通 A 和 m6A）
3. 非 DRACH 的普通 A（不含 m6A）
4. DRACH 但非 m6A 的普通 A

输出：
- n_pos
- n_neg_sampled
- AUROC
- AUPRC
- F1

- 这里评估的是“模型能否把 DRACH-m6A 和某类非 DRACH 位点区分开”。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

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

from models.etd_multitask.data import build_length_bucketed_batches, load_examples
from models.etd_multitask.evaluate import _encode_batch_raw, _forward_mod
from models.etd_multitask.metrics import binary_auprc, binary_auroc, binary_f1
from models.etd_multitask.model import ETDMultiTaskModel


D_SET = {"A", "G", "U"}
R_SET = {"A", "G"}
H_SET = {"A", "C", "U"}
DRH_COMBOS = [f"{d}{r}AC{h}" for d in ("A", "G", "U") for r in ("A", "G") for h in ("A", "C", "U")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multitask mod checkpoint with DRACH-m6A positives",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "outputs/etd_multitask/no-struct3171040/no-struct312/epoch_010.pt"),
        help="checkpoint 文件，或包含 epoch_*.pt 的实验目录",
    )
    parser.add_argument("--sites", default=str(REPO_ROOT / "data/processed/all_multitask_sites.parquet"))
    parser.add_argument("--transcripts", default=str(REPO_ROOT / "data/processed/all_multitask_transcripts.parquet"))
    parser.add_argument("--splits", default=str(REPO_ROOT / "data/processed/all_multitask_splits.json"))
    parser.add_argument("--split", nargs="+", default=["test"], help="可同时评估多个 split，如 --split val test")
    parser.add_argument("--max-len", type=int, default=12000)
    parser.add_argument("--batch-token-budget", type=int, default=24000)
    parser.add_argument("--bucket-boundaries", default="1024,2048,4096,8192,12000")
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=1.0,
        help="每条样本负样本采样倍率，相对该条样本的正样本数（正样本固定为 DRACH-m6A）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-interval", type=int, default=50, help="每多少个 batch 打印一次进度；<=0 表示不打印")
    parser.add_argument("--output-json", default=str(REPO_ROOT / "outputs/analysis/mod_drach_negative_eval.json"))
    parser.add_argument("--output-md", default=str(REPO_ROOT / "outputs/analysis/mod_drach_negative_eval.md"))
    return parser.parse_args()


def resolve_checkpoint(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("epoch_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"目录里没找到 epoch_*.pt: {path}")
        return candidates[-1]
    raise FileNotFoundError(path)


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


def drh_combo_at(seq: str, pos: int) -> str | None:
    if not is_drach_center(seq, pos):
        return None
    return f"{seq[pos - 2]}{seq[pos - 1]}AC{seq[pos + 2]}"


def sample_positions(rng: np.random.Generator, positions: np.ndarray, k: int) -> np.ndarray:
    if positions.size == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64)
    if positions.size <= k:
        return positions
    idx = rng.choice(positions.size, size=k, replace=False)
    return np.asarray(positions[idx], dtype=np.int64)


def evaluate_split(
    model: ETDMultiTaskModel,
    examples,
    boundaries: list[int],
    batch_token_budget: int,
    device: torch.device,
    neg_ratio: float,
    seed: int,
    progress_interval: int,
) -> dict[str, dict[str, float]]:
    batches = build_length_bucketed_batches(
        examples=examples,
        batch_token_budget=batch_token_budget,
        boundaries=boundaries,
        shuffle=False,
        seed=0,
    )

    schemes = {
        "drach_m6a_vs_non_drach_m6a": {"y_true": [], "y_prob": [], "n_pos": 0, "n_neg_sampled": 0},
        "drach_m6a_vs_non_drach_all_a": {"y_true": [], "y_prob": [], "n_pos": 0, "n_neg_sampled": 0},
        "drach_m6a_vs_non_drach_plain_a": {"y_true": [], "y_prob": [], "n_pos": 0, "n_neg_sampled": 0},
        "drach_m6a_vs_drach_plain_a": {"y_true": [], "y_prob": [], "n_pos": 0, "n_neg_sampled": 0},
    }
    combo_stats = {combo: {"all_drach_a": 0, "drach_m6a": 0} for combo in DRH_COMBOS}
    rng = np.random.default_rng(seed)
    print(f"[eval] n_examples={len(examples)} n_batches={len(batches)}", flush=True)

    with torch.no_grad():
        for batch_idx, batch_examples in enumerate(batches, start=1):
            if progress_interval > 0 and (batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == len(batches)):
                print(f"[eval] batch {batch_idx}/{len(batches)}", flush=True)
            base_inputs = _encode_batch_raw(batch_examples, device)
            probs_np = _forward_mod(model, base_inputs, mod_type="m6A", device=device)

            for i, item in enumerate(batch_examples):
                seq = item.sequence
                m6a_positions = np.asarray(item.mod_positions.get("m6A", np.zeros(0, dtype=np.int64)), dtype=np.int64)
                m6a_set = set(int(x) for x in m6a_positions.tolist())

                drach_m6a = np.asarray([p for p in m6a_positions if is_drach_center(seq, int(p))], dtype=np.int64)
                non_drach_m6a = np.asarray([p for p in m6a_positions if not is_drach_center(seq, int(p))], dtype=np.int64)

                # 所有 A 位点，用来构造 case 2/3 两类非 DRACH 负样本。
                all_a = np.asarray([idx for idx, base in enumerate(seq) if base == "A"], dtype=np.int64)
                drach_all_a = np.asarray([p for p in all_a if is_drach_center(seq, int(p))], dtype=np.int64)
                non_drach_all_a = np.asarray([p for p in all_a if not is_drach_center(seq, int(p))], dtype=np.int64)
                non_drach_plain_a = np.asarray([p for p in non_drach_all_a if int(p) not in m6a_set], dtype=np.int64)
                drach_plain_a = np.asarray([p for p in drach_all_a if int(p) not in m6a_set], dtype=np.int64)

                for p in drach_all_a:
                    combo = drh_combo_at(seq, int(p))
                    if combo is not None:
                        combo_stats[combo]["all_drach_a"] += 1
                for p in drach_m6a:
                    combo = drh_combo_at(seq, int(p))
                    if combo is not None:
                        combo_stats[combo]["drach_m6a"] += 1

                if drach_m6a.size == 0:
                    continue

                pos_probs = probs_np[i, drach_m6a]
                n_pos = drach_m6a.size
                n_neg_target = max(1, int(round(n_pos * neg_ratio)))

                neg_map = {
                    "drach_m6a_vs_non_drach_m6a": non_drach_m6a,
                    "drach_m6a_vs_non_drach_all_a": non_drach_all_a,
                    "drach_m6a_vs_non_drach_plain_a": non_drach_plain_a,
                    "drach_m6a_vs_drach_plain_a": drach_plain_a,
                }

                for name, neg_pool in neg_map.items():
                    neg_sampled = sample_positions(rng, neg_pool, n_neg_target)
                    if neg_sampled.size == 0:
                        continue
                    neg_probs = probs_np[i, neg_sampled]

                    schemes[name]["y_true"].append(np.ones(n_pos, dtype=np.int64))
                    schemes[name]["y_prob"].append(pos_probs)
                    schemes[name]["y_true"].append(np.zeros(neg_sampled.size, dtype=np.int64))
                    schemes[name]["y_prob"].append(neg_probs)
                    schemes[name]["n_pos"] += n_pos
                    schemes[name]["n_neg_sampled"] += int(neg_sampled.size)

    results: dict[str, dict[str, float]] = {}
    for name, col in schemes.items():
        if not col["y_true"]:
            results[name] = {
                "n_pos": 0,
                "n_neg_sampled": 0,
                "auroc": float("nan"),
                "auprc": float("nan"),
                "f1": float("nan"),
            }
            continue
        y_true = np.concatenate(col["y_true"])
        y_prob = np.concatenate(col["y_prob"])
        results[name] = {
            "n_pos": int(col["n_pos"]),
            "n_neg_sampled": int(col["n_neg_sampled"]),
            "auroc": float(binary_auroc(y_true, y_prob)),
            "auprc": float(binary_auprc(y_true, y_prob)),
            "f1": float(binary_f1(y_true, y_prob)),
        }
    total_all_drach_a = sum(v["all_drach_a"] for v in combo_stats.values())
    total_drach_m6a = sum(v["drach_m6a"] for v in combo_stats.values())
    combo_results = {}
    for combo in DRH_COMBOS:
        counts = combo_stats[combo]
        combo_results[combo] = {
            "all_drach_a": int(counts["all_drach_a"]),
            "all_drach_a_ratio": float(counts["all_drach_a"] / total_all_drach_a) if total_all_drach_a > 0 else float("nan"),
            "drach_m6a": int(counts["drach_m6a"]),
            "drach_m6a_ratio": float(counts["drach_m6a"] / total_drach_m6a) if total_drach_m6a > 0 else float("nan"),
        }
    return {
        "schemes": results,
        "drh_combo_stats": {
            "total_all_drach_a": int(total_all_drach_a),
            "total_drach_m6a": int(total_drach_m6a),
            "combos": combo_results,
        },
    }


def save_outputs(results: dict, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    lines = [
        "# 多任务 m6A mod 的 DRACH 负样本评估",
        "",
        "固定正样本定义：**DRACH 基序上的 m6A 位点**。",
        "",
        "负样本取 4 套定义：",
        "1. 非 DRACH 的 m6A",
        "2. 非 DRACH 的 A（包含普通 A 和 m6A）",
        "3. 非 DRACH 的普通 A",
        "4. DRACH 但非 m6A 的普通 A",
        "",
    ]
    for split_name, split_result in results["results"].items():
        lines.extend(
            [
                f"## {split_name}",
                "",
                "| scheme | n_pos | n_neg_sampled | AUROC | AUPRC | F1 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for scheme, metrics in split_result["schemes"].items():
            lines.append(
                f"| {scheme} | {metrics['n_pos']} | {metrics['n_neg_sampled']} | "
                f"{metrics['auroc']:.4f} | {metrics['auprc']:.4f} | {metrics['f1']:.4f} |"
            )
        combo_block = split_result["drh_combo_stats"]
        lines.extend(
            [
                "",
                f"DRH 18 种组合占比（总 DRACH A={combo_block['total_all_drach_a']}，总 DRACH m6A={combo_block['total_drach_m6a']}）",
                "",
                "| combo | all_drach_a | all_drach_a_ratio | drach_m6a | drach_m6a_ratio |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for combo in DRH_COMBOS:
            metrics = combo_block["combos"][combo]
            lines.append(
                f"| {combo} | {metrics['all_drach_a']} | {metrics['all_drach_a_ratio']:.4%} | "
                f"{metrics['drach_m6a']} | {metrics['drach_m6a_ratio']:.4%} |"
            )
        lines.append("")
    output_md.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    ckpt = resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    boundaries = [int(x.strip()) for x in args.bucket_boundaries.split(",") if x.strip()]
    if not boundaries:
        boundaries = [1024, 2048, 4096, 8192, 12000]

    model = ETDMultiTaskModel().to(device)
    payload = torch.load(ckpt, map_location=device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    model.eval()

    final = {
        "checkpoint": str(ckpt),
        "splits": list(args.split),
        "neg_ratio": args.neg_ratio,
        "results": {},
    }

    print(f"[checkpoint] {ckpt}")
    print(f"[neg_ratio] {args.neg_ratio}")
    for split_name in args.split:
        examples = load_examples(
            sites_path=args.sites,
            transcripts_path=args.transcripts,
            splits_path=args.splits,
            split_names=[split_name],
            max_len=args.max_len,
            smoke_ratio=1.0,
            seed=args.seed,
        )
        split_result = evaluate_split(
            model=model,
            examples=examples,
            boundaries=boundaries,
            batch_token_budget=args.batch_token_budget,
            device=device,
            neg_ratio=args.neg_ratio,
            seed=args.seed,
            progress_interval=args.progress_interval,
        )
        final["results"][split_name] = split_result

        print(f"\n[{split_name}]")
        for scheme, metrics in split_result["schemes"].items():
            print(
                f"  {scheme}: n_pos={metrics['n_pos']} n_neg_sampled={metrics['n_neg_sampled']} "
                f"AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} F1={metrics['f1']:.4f}"
            )

    save_outputs(final, Path(args.output_json), Path(args.output_md))
    print(f"\njson saved to: {args.output_json}")
    print(f"md saved to: {args.output_md}")


if __name__ == "__main__":
    main()
