#!/usr/bin/env python3
import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from rinalmo.pretrained import get_pretrained_model

REPO_ROOT = Path(__file__).resolve().parents[2]


def extract_window(seq: str, pos: int, window: int) -> tuple[str, int]:
    if window <= 0:
        return seq, 0
    n = len(seq)
    if window >= n:
        return seq, 0
    half = window // 2
    start = max(0, pos - half)
    end = start + window
    if end > n:
        end = n
        start = max(0, end - window)
    return seq[start:end], start


def count_labels(csv_path: Path, label_col: str) -> dict:
    counts = {}
    with csv_path.open(newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            lbl = row[label_col]
            counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def reservoir_sample_indices(csv_path: Path, label_col: str, majority_label: str, target: int, seed: int) -> set:
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    with csv_path.open(newline='') as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if row[label_col] != majority_label:
                continue
            seen += 1
            if len(reservoir) < target:
                reservoir.append(i)
            else:
                j = rng.randrange(seen)
                if j < target:
                    reservoir[j] = i
    return set(reservoir)


def main():
    ap = argparse.ArgumentParser(description="Generate balanced RiNALMo representations for m6A/YTH dataset")
    ap.add_argument("--csv", default=str(REPO_ROOT / "data/processed/m6A_YTH_dataset.csv"))
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "representations/rinalmo_giga_m6a_balanced"),
    )
    ap.add_argument("--model_name", default="giga-v1")
    ap.add_argument("--window", type=int, default=0, help="Window length around position; 0 means full sequence.")
    ap.add_argument("--max_len", type=int, default=4096, help="Max full length allowed; longer sequences will be windowed.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--seq_col", default="full_sequence")
    ap.add_argument("--pos_col", default="m6A_position_index")
    ap.add_argument("--modid_col", default="modId")
    ap.add_argument("--transcript_col", default="transcriptId")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = count_labels(csv_path, args.label_col)
    if "0" not in counts or "1" not in counts:
        raise ValueError(f"Expected labels '0' and '1', got: {counts}")
    target = min(counts["0"], counts["1"])
    majority_label = "0" if counts["0"] > counts["1"] else "1"
    minority_label = "1" if majority_label == "0" else "0"

    majority_indices = reservoir_sample_indices(csv_path, args.label_col, majority_label, target, args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = get_pretrained_model(model_name=args.model_name)
    model = model.to(device=device)
    model.eval()
    pad_idx = alphabet.pad_idx

    total = target * 2
    emb_path = out_dir / f"embeddings_window{args.window}.npy"
    labels_path = out_dir / f"labels_window{args.window}.npy"
    meta_path = out_dir / f"metadata_window{args.window}.csv"

    # We don't know embedding dim until first forward; use a placeholder and resize after first batch.
    embeddings = None
    labels = np.memmap(labels_path, dtype=np.int64, mode="w+", shape=(total,))

    def ensure_embeddings(dim: int):
        nonlocal embeddings
        if embeddings is None:
            embeddings = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(total, dim))

    idx = 0
    batch_seqs = []
    batch_meta = []
    processed = 0

    with meta_path.open("w", newline="") as mf:
        mw = csv.writer(mf)
        mw.writerow([args.modid_col, args.transcript_col, args.pos_col, args.label_col, "window_start", "window_seq"])
        with csv_path.open(newline='') as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                lbl = row[args.label_col]
                use = (lbl == minority_label) or (i in majority_indices)
                if not use:
                    continue
                seq = row[args.seq_col]
                try:
                    pos = int(row[args.pos_col])
                except ValueError:
                    pos = 0
                if args.window <= 0 and len(seq) > args.max_len:
                    window_seq, window_start = extract_window(seq, pos, args.max_len)
                else:
                    window_seq, window_start = extract_window(seq, pos, args.window)

                row["_window_start"] = window_start
                row["_window_seq"] = window_seq
                row["_pos_in_window"] = max(0, pos - window_start)

                batch_seqs.append(window_seq)
                batch_meta.append(row)

                if len(batch_seqs) >= args.batch_size:
                    tokens = torch.tensor(alphabet.batch_tokenize(batch_seqs), dtype=torch.int64, device=device)
                    with torch.no_grad(), torch.amp.autocast(device_type=device if device != "cpu" else "cpu"):
                        outputs = model(tokens)
                    rep = outputs["representation"]  # (B, L, D)
                    if embeddings is None:
                        ensure_embeddings(rep.shape[-1])
                    pooled = []
                    for b in range(rep.shape[0]):
                        pos_in_window = int(batch_meta[b].get("_pos_in_window", 0))
                        token_pos = pos_in_window + 1  # CLS offset
                        token_pos = min(token_pos, rep.shape[1] - 2)  # keep before EOS
                        pooled.append(rep[b, token_pos].detach().cpu().numpy().astype(np.float32))
                    pooled = np.stack(pooled, axis=0)

                    bsz = pooled.shape[0]
                    embeddings[idx:idx+bsz] = pooled
                    for j in range(bsz):
                        labels[idx+j] = int(batch_meta[j][args.label_col])
                        mw.writerow([
                            batch_meta[j][args.modid_col],
                            batch_meta[j][args.transcript_col],
                            batch_meta[j][args.pos_col],
                            batch_meta[j][args.label_col],
                            batch_meta[j]["_window_start"],
                            batch_meta[j]["_window_seq"],
                        ])
                    idx += bsz
                    processed = idx
                    if processed % 1000 == 0:
                        print(f\"processed {processed}/{total}\", flush=True)
                    batch_seqs.clear()
                    batch_meta.clear()

        # flush remaining
        if batch_seqs:
            tokens = torch.tensor(alphabet.batch_tokenize(batch_seqs), dtype=torch.int64, device=device)
            with torch.no_grad(), torch.amp.autocast(device_type=device if device != "cpu" else "cpu"):
                outputs = model(tokens)
            rep = outputs["representation"]
            if embeddings is None:
                ensure_embeddings(rep.shape[-1])
            pooled = []
            for b in range(rep.shape[0]):
                pos_in_window = int(batch_meta[b].get("_pos_in_window", 0))
                token_pos = pos_in_window + 1  # CLS offset
                token_pos = min(token_pos, rep.shape[1] - 2)  # keep before EOS
                pooled.append(rep[b, token_pos].detach().cpu().numpy().astype(np.float32))
            pooled = np.stack(pooled, axis=0)

            bsz = pooled.shape[0]
            embeddings[idx:idx+bsz] = pooled
            for j in range(bsz):
                labels[idx+j] = int(batch_meta[j][args.label_col])
                mw.writerow([
                    batch_meta[j][args.modid_col],
                    batch_meta[j][args.transcript_col],
                    batch_meta[j][args.pos_col],
                    batch_meta[j][args.label_col],
                    batch_meta[j]["_window_start"],
                    batch_meta[j]["_window_seq"],
                ])
            idx += bsz
            processed = idx
            if processed % 1000 == 0:
                print(f\"processed {processed}/{total}\", flush=True)

    if idx != total:
        print(f"Warning: expected {total} samples, wrote {idx}.")


if __name__ == "__main__":
    main()
