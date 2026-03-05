"""Online RNAfold provider for sequence-specific structure features.

用途：
- 按“当前输入序列”实时调用 RNAfold，得到配对概率。
- 支持序列级缓存，避免重复计算完全相同的序列。
- 提供与训练数据构造直接相关的两个接口：
  1) 逐位点统计特征（pair_prob / pair_count）
  2) 下采样结构目标矩阵（用于 struct 分支）
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import numpy as np

from .rnafold import parse_dot_ps_ubox


class OnlineRNAfoldProvider:
    """基于 RNAfold 命令行的在线结构提供器。"""

    def __init__(
        self,
        rnafold_bin: str = "RNAfold",
        timeout_seconds: int = 240,
        cache_size: int = 2048,
    ) -> None:
        self.rnafold_bin = str(rnafold_bin)
        self.timeout_seconds = int(timeout_seconds)
        self.cache_size = max(1, int(cache_size))

        # LRU: seq -> pair_map[(i,j)] = p
        self._pair_cache: OrderedDict[str, dict[tuple[int, int], float]] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def _sanitize_sequence(seq: str) -> str:
        seq = str(seq).upper().replace("T", "U")
        allowed = {"A", "C", "G", "U", "N", "6"}
        return "".join(ch if ch in allowed else "N" for ch in seq)

    def _cache_get(self, seq: str) -> dict[tuple[int, int], float] | None:
        with self._lock:
            payload = self._pair_cache.get(seq)
            if payload is None:
                return None
            # move-to-end for LRU
            self._pair_cache.move_to_end(seq)
            return payload

    def _cache_set(self, seq: str, payload: dict[tuple[int, int], float]) -> None:
        with self._lock:
            self._pair_cache[seq] = payload
            self._pair_cache.move_to_end(seq)
            while len(self._pair_cache) > self.cache_size:
                self._pair_cache.popitem(last=False)

    def _run_rnafold(self, seq: str) -> dict[tuple[int, int], float]:
        seq = self._sanitize_sequence(seq)
        with tempfile.TemporaryDirectory(prefix="rnafold_online_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_text = f">seq\n{seq}\n"
            cmd = [self.rnafold_bin, "-p", "--noLP", "-d2", "--modifications"]

            proc = subprocess.run(
                cmd,
                input=input_text,
                text=True,
                cwd=tmp_path,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(f"RNAfold failed: {err}")

            dot_ps = tmp_path / "seq_dp.ps"
            if not dot_ps.exists():
                raise RuntimeError("RNAfold did not generate seq_dp.ps")
            return parse_dot_ps_ubox(dot_ps)

    def _get_pair_map(self, seq: str) -> dict[tuple[int, int], float]:
        seq = self._sanitize_sequence(seq)
        cached = self._cache_get(seq)
        if cached is not None:
            return cached
        pair_map = self._run_rnafold(seq)
        self._cache_set(seq, pair_map)
        return pair_map

    def get_positional_stats(self, seq: str) -> tuple[np.ndarray, np.ndarray]:
        """返回逐位点配对统计。

        输出：
        - pair_prob[i]: 位点 i 的配对概率和
        - pair_count[i]: 位点 i 参与配对的边数
        """
        seq = self._sanitize_sequence(seq)
        length = len(seq)
        pair_prob = np.zeros(length, dtype=np.float32)
        pair_count = np.zeros(length, dtype=np.float32)

        pair_map = self._get_pair_map(seq)
        if not pair_map:
            return pair_prob, pair_count

        for (i, j), p in pair_map.items():
            if i < 0 or j < 0 or i >= length or j >= length:
                continue
            prob = float(p)
            pair_prob[i] += prob
            pair_prob[j] += prob
            pair_count[i] += 1.0
            pair_count[j] += 1.0

        pair_prob = np.clip(pair_prob, 0.0, 1.0)
        return pair_prob, pair_count

    def get_downsampled_target(self, seq: str, factor: int = 16) -> np.ndarray:
        """返回下采样后的结构目标矩阵。"""
        seq = self._sanitize_sequence(seq)
        length = len(seq)
        factor = max(1, int(factor))
        l_prime = int(math.ceil(length / factor))

        mat = np.zeros((l_prime, l_prime), dtype=np.float32)
        counts = np.zeros((l_prime, l_prime), dtype=np.float32)

        pair_map = self._get_pair_map(seq)
        for (i, j), p in pair_map.items():
            if i < 0 or j < 0 or i >= length or j >= length:
                continue
            bi = i // factor
            bj = j // factor
            mat[bi, bj] += float(p)
            counts[bi, bj] += 1.0

        nz = counts > 0
        mat[nz] = mat[nz] / counts[nz]
        mat = mat + mat.T
        np.fill_diagonal(mat, 0.0)
        return mat

    # 与 data.collate_batch 的在线分支约定接口名保持一致，
    # 这样训练主流程仍可统一用 `bpp_cache` 变量，无需改调度逻辑。
    def get_positional_stats_from_sequence(self, seq: str) -> tuple[np.ndarray, np.ndarray]:
        return self.get_positional_stats(seq)

    def get_downsampled_target_from_sequence(self, seq: str, factor: int = 16) -> np.ndarray:
        return self.get_downsampled_target(seq, factor=factor)

    def clear(self) -> None:
        """清空内存缓存。"""
        with self._lock:
            self._pair_cache.clear()
