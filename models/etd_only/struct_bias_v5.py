

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus

import numpy as np
import torch

from models.etd_multitask.constants import BASE_TO_ID, PAD_TOKEN_ID

ID_TO_BASE = {
    int(BASE_TO_ID["A"]): "A",
    int(BASE_TO_ID["C"]): "C",
    int(BASE_TO_ID["G"]): "G",
    int(BASE_TO_ID["U"]): "U",
    int(BASE_TO_ID["N"]): "N",
}


def tokens_to_sequence(tokens_1d: np.ndarray, attn_mask_1d: np.ndarray) -> str:
    """把一条带 padding 的 token 序列还原成 RNA 字符串。"""
    chars: list[str] = []
    for tok, keep in zip(tokens_1d.tolist(), attn_mask_1d.tolist()):
        if not keep or int(tok) == PAD_TOKEN_ID:
            continue
        chars.append(ID_TO_BASE.get(int(tok), "N"))
    return "".join(chars)


def pair_map_to_dense(pair_map: dict[tuple[int, int], float], length: int) -> np.ndarray:
    """把稀疏配对概率转成对称稠密矩阵。"""
    mat = np.zeros((length, length), dtype=np.float32)
    for (i, j), p in pair_map.items():
        if 0 <= i < length and 0 <= j < length:
            mat[i, j] = float(p)
    mat = mat + mat.T
    np.fill_diagonal(mat, 0.0)
    return mat


def downsample_pair_bias(mat: np.ndarray, factor: int) -> np.ndarray:
    """把原始分辨率的配对矩阵下采样到 bottleneck 长度。"""
    length = int(mat.shape[0])
    factor = max(1, int(factor))
    out_len = (length + factor - 1) // factor
    out = np.zeros((out_len, out_len), dtype=np.float32)

    for bi in range(out_len):
        i0 = bi * factor
        i1 = min(length, (bi + 1) * factor)
        for bj in range(out_len):
            j0 = bj * factor
            j1 = min(length, (bj + 1) * factor)
            block = mat[i0:i1, j0:j1]
            if block.size > 0:
                out[bi, bj] = float(block.mean())

    np.fill_diagonal(out, 0.0)
    return out


def _block_sizes(length: int, factor: int) -> np.ndarray:
    out_len = (length + factor - 1) // factor
    sizes = np.zeros(out_len, dtype=np.float32)
    for bi in range(out_len):
        i0 = bi * factor
        i1 = min(length, (bi + 1) * factor)
        sizes[bi] = float(max(i1 - i0, 1))
    return sizes


def sparse_edges_to_downsampled_bias(
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_p: np.ndarray,
    window_len: int,
    factor: int,
    scale: float = 1.0,
) -> np.ndarray:
    """直接把稀疏边投影成下采样后的 bias。"""
    factor = max(1, int(factor))
    out_len = (window_len + factor - 1) // factor
    out = np.zeros((out_len, out_len), dtype=np.float32)
    sizes = _block_sizes(window_len, factor)

    if edge_i.size > 0:
        bi = (edge_i.astype(np.int64) // factor).clip(0, out_len - 1)
        bj = (edge_j.astype(np.int64) // factor).clip(0, out_len - 1)
        p = edge_p.astype(np.float32)
        np.add.at(out, (bi, bj), p)
        np.add.at(out, (bj, bi), p)

    denom = sizes[:, None] * sizes[None, :]
    out = out / np.maximum(denom, 1.0)
    np.fill_diagonal(out, 0.0)
    return out * float(scale)



_HAS_RNA_API = False
try:
    import RNA as _RNA_mod
    _HAS_RNA_API = True
except ImportError:
    _RNA_mod = None


def _pairprob_via_python_api(seq: str) -> dict[tuple[int, int], float]:
   
    fc = _RNA_mod.fold_compound(seq)
    fc.pf()
    # bpp() 返回 (n+1) x (n+1) 的上三角列表，1-indexed
    bpp = fc.bpp()
    n = len(seq)
    pair_map: dict[tuple[int, int], float] = {}
    # 只遍历上三角，跳过概率极低的配对
    for i in range(1, n + 1):
        row = bpp[i]
        for j in range(i + 1, n + 1):
            p = row[j]
            if p > 1e-4:
                pair_map[(i - 1, j - 1)] = p
    return pair_map


def _pairprob_via_subprocess(
    seq: str,
    rnafold_bin: str = "RNAfold",
    timeout: int = 240,
) -> dict[tuple[int, int], float]:
    """通过 subprocess 调用 RNAfold -p，解析 dot plot 输出。

    这是 fallback 路径，每次调用 ~1-2s。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_text = f">seq\n{seq}\n"
        result = subprocess.run(
            [rnafold_bin, "-p"],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmpdir,
        )
        # RNAfold -p 会在 cwd 生成 seq_dp.ps（dot plot 文件）
        dp_file = os.path.join(tmpdir, "seq_dp.ps")
        if os.path.exists(dp_file):
            return _parse_dp_ps(dp_file)

        # fallback：尝试从 stdout 解析
        return _parse_rnafold_stdout(result.stdout, len(seq))


def _parse_dp_ps(path: str) -> dict[tuple[int, int], float]:
    """解析 RNAfold 输出的 PostScript dot plot 文件。"""
    pair_map: dict[tuple[int, int], float] = {}
    ubox_re = re.compile(r"(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+ubox")
    with open(path) as f:
        for line in f:
            m = ubox_re.search(line)
            if m:
                i = int(m.group(1)) - 1  # 1-indexed → 0-indexed
                j = int(m.group(2)) - 1
                p_sqrt = float(m.group(3))
                p = p_sqrt * p_sqrt       # dp.ps stores sqrt(prob)
                if p > 1e-4:
                    pair_map[(min(i, j), max(i, j))] = p
    return pair_map


def _parse_rnafold_stdout(stdout: str, seq_len: int) -> dict[tuple[int, int], float]:
    """从 RNAfold stdout 尝试提取碱基配对概率（某些版本格式）。"""
    # 如果解析失败，返回空 dict（相当于没有结构先验）
    return {}


class _LRUCache:

    def __init__(self, maxsize: int = 4096):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
        self._cache[key] = value


# 每个 DataLoader worker 进程会有自己独立的全局 cache 实例
_worker_cache = _LRUCache(maxsize=8192)


def compute_single_bias(
    seq: str,
    valid_indices: np.ndarray,
    window_size: int,
    factor: int,
    scale: float = 1.0,
    rnafold_bin: str = "RNAfold",
    rnafold_timeout: int = 240,
) -> np.ndarray:
   
    out_len = (window_size + factor - 1) // factor

    if not seq or len(seq) == 0:
        return np.zeros((out_len, out_len), dtype=np.float32)

    # LRU cache lookup（key = seq hash + factor + window_size）
    cache_key = hashlib.md5(f"{seq}|{window_size}|{factor}".encode()).hexdigest()
    cached = _worker_cache.get(cache_key)
    if cached is not None:
        # cached 是 (dense_of_seq,)，需要根据 valid_indices 重新映射
        dense_seq = cached["dense"]
        return _embed_and_downsample(dense_seq, valid_indices, window_size, factor, scale)

    # ---- 计算碱基配对概率 ----
    try:
        if _HAS_RNA_API:
            pair_map = _pairprob_via_python_api(seq)
        else:
            pair_map = _pairprob_via_subprocess(seq, rnafold_bin, rnafold_timeout)
        
    except Exception:

        return np.zeros((out_len, out_len), dtype=np.float32)

    dense_seq = pair_map_to_dense(pair_map, len(seq))
    _worker_cache.put(cache_key, {"dense": dense_seq})

    return _embed_and_downsample(dense_seq, valid_indices, window_size, factor, scale)


def _embed_and_downsample(
    dense_seq: np.ndarray,
    valid_indices: np.ndarray,
    window_size: int,
    factor: int,
    scale: float,
) -> np.ndarray:
    """把 seq 尺度的稠密矩阵嵌入到 window 尺度，然后下采样。"""
    full_dense = np.zeros((window_size, window_size), dtype=np.float32)
    if valid_indices.size > 0 and dense_seq.size > 0:
        n = min(len(valid_indices), dense_seq.shape[0])
        idx = valid_indices[:n]
        full_dense[np.ix_(idx, idx)] = dense_seq[:n, :n]
    return downsample_pair_bias(full_dense, factor=factor) * scale



class OfflineStructBiasCache:
    """按 transcript 懒加载离线结构 cache。"""
    def __init__(self, cache_dir: str | Path, max_transcripts: int = 256):
        self.cache_dir = Path(cache_dir)
        self.max_transcripts = max(1, int(max_transcripts))
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    @staticmethod
    def _transcript_filename(transcript_id: str) -> str:
        return quote_plus(str(transcript_id), safe="") + ".npz"

    def _load_transcript(self, transcript_id: str) -> dict[str, np.ndarray]:
        key = str(transcript_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = self.cache_dir / self._transcript_filename(key)
        with np.load(path, allow_pickle=False) as payload:
            data = {
                name: payload[name]
                for name in payload.files
                if name != "transcript_id"
            }
        if len(self._cache) >= self.max_transcripts:
            self._cache.popitem(last=False)
        self._cache[key] = data
        return data

    def get_downsampled_bias(
        self,
        transcript_id: str,
        site_pos: int,
        window_len: int,
        factor: int,
        scale: float = 1.0,
    ) -> np.ndarray:
        data = self._load_transcript(transcript_id)
        site_positions = data["site_positions"]
        idxs = np.where(site_positions == int(site_pos))[0]
        if idxs.size == 0:
            raise KeyError(
                f"site_pos={site_pos} not found in cache for transcript_id={transcript_id}"
            )
        idx = int(idxs[0])
        offsets = data["edge_offsets"]
        lo = int(offsets[idx])
        hi = int(offsets[idx + 1])
        return sparse_edges_to_downsampled_bias(
            edge_i=data["edge_i"][lo:hi],
            edge_j=data["edge_j"][lo:hi],
            edge_p=data["edge_p"][lo:hi],
            window_len=window_len,
            factor=factor,
            scale=scale,
        )
