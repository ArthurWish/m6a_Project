"""统计 DRACH 基序覆盖情况。

目标：
1. 在所有转录本序列里，统计所有 A 位点中有多少满足 DRACH 基序。
2. 在真实 m6A 位点里，统计有多少满足 DRACH 基序。
3. 分别输出 full / train / val / test 四套统计。

这里采用的数据口径：
- 序列来自 `all_multitask_transcripts.parquet`
- train/val/test 划分来自 `all_multitask_splits.json`
- 真实 m6A 位点来自 transcripts parquet 中的 `mod_positions + mod_types`

DRACH 定义（5'->3'）：
- D = A/G/U （非 C）
- R = A/G
- A = 中心位点（待检测的 A）
- C = 固定 C
- H = A/C/U （非 G）

也就是对中心位点 i：
- seq[i-2] in {A, G, U}
- seq[i-1] in {A, G}
- seq[i]   == 'A'
- seq[i+1] == 'C'
- seq[i+2] in {A, C, U}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


D_SET = {"A", "G", "U"}
R_SET = {"A", "G"}
H_SET = {"A", "C", "U"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 all/train/val/test 中 DRACH 基序与 m6A 的关系",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        default="data/processed/all_multitask_transcripts.parquet",
        help="transcripts parquet 路径",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/processed/all_multitask_splits.json",
        help="train/val/test 划分文件",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/analysis/drach_motif_stats.json",
        help="统计结果 json 输出路径",
    )
    return parser.parse_args()


def is_drach_center(seq: str, pos: int) -> bool:
    """判断给定 0-based 位置 pos 是否是 DRACH 的中心 A。"""
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


def load_split_ids(splits_path: Path) -> dict[str, set[str]]:
    raw = json.loads(splits_path.read_text())
    return {
        "train": set(raw.get("train", [])),
        "val": set(raw.get("val", [])),
        "test": set(raw.get("test", [])),
    }


def iter_mod_types(mod_types_obj: object) -> Iterable[str]:
    """把 parquet 里的 object/ndarray/list 统一展开成字符串迭代器。"""
    if mod_types_obj is None:
        return ()
    # ndarray/list/tuple 都走这里；字符串单独排除，避免被逐字符遍历。
    if isinstance(mod_types_obj, (list, tuple)):
        return (str(x) for x in mod_types_obj)
    try:
        # numpy.ndarray 会有 tolist
        if hasattr(mod_types_obj, "tolist"):
            v = mod_types_obj.tolist()
            if isinstance(v, list):
                return (str(x) for x in v)
    except Exception:
        pass
    return (str(mod_types_obj),)


def empty_stats() -> dict[str, int]:
    return {
        "n_transcripts": 0,
        "n_all_a": 0,
        "n_all_a_in_drach": 0,
        "n_m6a": 0,
        "n_m6a_in_drach": 0,
    }


def finalize(stats: dict[str, int]) -> dict[str, float | int]:
    n_all_a = stats["n_all_a"]
    n_m6a = stats["n_m6a"]
    return {
        **stats,
        "all_a_drach_ratio": (stats["n_all_a_in_drach"] / n_all_a) if n_all_a else 0.0,
        "m6a_drach_ratio": (stats["n_m6a_in_drach"] / n_m6a) if n_m6a else 0.0,
    }


def main() -> None:
    args = parse_args()
    transcripts_path = Path(args.transcripts)
    splits_path = Path(args.splits)
    output_path = Path(args.output)

    split_ids = load_split_ids(splits_path)
    df = pd.read_parquet(transcripts_path)

    buckets: dict[str, dict[str, int]] = {
        "full": empty_stats(),
        "train": empty_stats(),
        "val": empty_stats(),
        "test": empty_stats(),
    }

    for row in df.itertuples(index=False):
        tid = str(row.transcript_id)
        seq = str(row.full_sequence).upper().replace("T", "U")

        target_splits = ["full"]
        if tid in split_ids["train"]:
            target_splits.append("train")
        if tid in split_ids["val"]:
            target_splits.append("val")
        if tid in split_ids["test"]:
            target_splits.append("test")

        # 先统计所有 A 位点。
        a_positions = [i for i, base in enumerate(seq) if base == "A"]
        drach_a_positions = {i for i in a_positions if is_drach_center(seq, i)}

        # 再统计真实 m6A 位点。
        m6a_positions: set[int] = set()
        mod_positions = row.mod_positions
        mod_types = row.mod_types
        if mod_positions is not None and mod_types is not None:
            for pos, mods_at_pos in zip(mod_positions, mod_types):
                try:
                    pos_int = int(pos)
                except Exception:
                    continue
                mods = set(iter_mod_types(mods_at_pos))
                if "m6A" in mods:
                    # 只保留序列里确实是 A 的位置，避免脏数据影响统计。
                    if 0 <= pos_int < len(seq) and seq[pos_int] == "A":
                        m6a_positions.add(pos_int)

        for split_name in target_splits:
            item = buckets[split_name]
            item["n_transcripts"] += 1
            item["n_all_a"] += len(a_positions)
            item["n_all_a_in_drach"] += len(drach_a_positions)
            item["n_m6a"] += len(m6a_positions)
            item["n_m6a_in_drach"] += sum(1 for pos in m6a_positions if pos in drach_a_positions)

    final = {name: finalize(stats) for name, stats in buckets.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final, ensure_ascii=False, indent=2))

    print("DRACH motif statistics")
    print("=" * 80)
    for name in ["full", "train", "val", "test"]:
        item = final[name]
        print(
            f"[{name}] transcripts={item['n_transcripts']} | "
            f"all_A={item['n_all_a']} drach_A={item['n_all_a_in_drach']} "
            f"ratio={item['all_a_drach_ratio']:.4%}"
        )
        print(
            f"       m6A={item['n_m6a']} drach_m6A={item['n_m6a_in_drach']} "
            f"ratio={item['m6a_drach_ratio']:.4%}"
        )

    print(f"\njson saved to: {output_path}")


if __name__ == "__main__":
    main()
