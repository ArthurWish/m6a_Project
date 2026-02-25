"""Metrics for ETD multi-task evaluation."""

from __future__ import annotations

import numpy as np


def _safe_nan(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return float("nan")
    return float(value)


def binary_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float64)
    pos = y_true.sum()
    neg = (y_true == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-y_prob)
    y = y_true[order]

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    tpr = tps / pos
    fpr = fps / neg

    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])

    return _safe_nan(np.trapz(tpr, fpr))


def binary_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_prob = y_prob.astype(np.float64)
    pos = y_true.sum()
    if pos == 0:
        return float("nan")

    order = np.argsort(-y_prob)
    y = y_true[order]

    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / pos

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    return _safe_nan(np.trapz(precision, recall))


def binary_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = (2 * tp + fp + fn)
    if denom == 0:
        return float("nan")
    return _safe_nan((2 * tp) / denom)


def binary_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_prob >= threshold).astype(np.int64)
    if y_true.size == 0:
        return float("nan")
    return _safe_nan((y_pred == y_true).mean())


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    if y_true.size == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lo = bins[idx]
        hi = bins[idx + 1]
        if idx == n_bins - 1:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        else:
            in_bin = (y_prob >= lo) & (y_prob < hi)
        if not np.any(in_bin):
            continue
        conf = y_prob[in_bin].mean()
        acc = y_true[in_bin].mean()
        ece += np.abs(acc - conf) * (in_bin.sum() / y_true.size)

    return _safe_nan(ece)


def binary_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    if y_true.size == 0:
        return float("nan")
    p = np.clip(y_prob, eps, 1.0 - eps)
    nll = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).mean()
    return _safe_nan(nll)


def structure_f1(pred_prob: np.ndarray, target: np.ndarray, valid_len: int, threshold: float = 0.5, min_sep: int = 1) -> float:
    l = int(valid_len)
    if l <= 1:
        return float("nan")

    pred = pred_prob[:l, :l] >= threshold
    true = target[:l, :l] >= 0.5

    idx = np.arange(l)
    tri = idx[:, None] < idx[None, :]
    if min_sep > 0:
        tri &= (idx[:, None] + min_sep) < idx[None, :]

    pred = pred[tri]
    true = true[tri]

    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)

    denom = 2 * tp + fp + fn
    if denom == 0:
        return float("nan")
    return _safe_nan((2 * tp) / denom)
