"""Multi-task loss & metrics v4.

Loss = L_m6a + L_rbp_pos + α·L_rbp_uncertain + β·L_rbp_clean

Evaluation:
  m6A detection: AUPRC (all A), AUPRC (DRACH only)
  RBP binding: per-RBP AUPRC, top-k, LRAP, sample_F1, neg metrics
  Gated: P(RBP) = P(m6A) × P(RBP|m6A)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from models.etd_multitask.constants import (
    NUM_INDIVIDUAL_RBPS,
    INDIVIDUAL_RBPS,
    FAMILY_TO_IDXS,
)


# =====================================================================
# ASL (unchanged)
# =====================================================================

def asymmetric_loss(logits, targets, gamma_pos=0.0, gamma_neg=4.0,
                    clip=0.05, pos_weight=None, reduction="mean"):
    probs = torch.sigmoid(logits)
    probs_neg = (probs - clip).clamp(min=0.0)
    pc = targets * probs + (1 - targets) * probs_neg
    ce = -targets * torch.log(pc.clamp(min=1e-8)) \
         - (1 - targets) * torch.log((1 - pc).clamp(min=1e-8))
    if gamma_pos > 0 or gamma_neg > 0:
        fw_pos = (1 - pc).pow(gamma_pos) if gamma_pos > 0 else torch.ones_like(pc)
        fw_neg = pc.pow(gamma_neg) if gamma_neg > 0 else torch.ones_like(pc)
        ce = ce * (targets * fw_pos + (1 - targets) * fw_neg)
    if pos_weight is not None:
        pw = pos_weight.to(device=logits.device, dtype=logits.dtype)
        ce = ce * (targets * pw.unsqueeze(0).expand_as(targets) + (1 - targets))
    return ce.mean() if reduction == "mean" else (ce.sum() if reduction == "sum" else ce)


# =====================================================================
# Multi-task loss
# =====================================================================

def compute_multitask_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    rbp_pos_weight: torch.Tensor | None = None,
    # ASL params
    use_asl: bool = True,
    gamma_neg: float = 3.0,
    asl_clip: float = 0.05,
    # Loss weights
    m6a_loss_weight: float = 0.5,
    m6a_pos_weight: float = 20.0,
    rbp_uncertain_weight: float = 0.1,
    rbp_clean_weight: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Returns dict with total_loss and individual components for logging."""
    device = out["center_bind_logits"].device
    losses = {}

    # ---- 1) m6A detection loss ----
    if "m6a_det_logits" in out and out.get("m6a_det_mask", None) is not None:
        mask = out["m6a_det_mask"]
        if mask.any():
            logits = out["m6a_det_logits"][mask]
            targets = batch["m6a_det_targets"].to(device)[mask]
            pw = torch.tensor([m6a_pos_weight], device=device)
            losses["m6a"] = m6a_loss_weight * F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw)

    # ---- 2) RBP positive loss (center + extra positives) ----
    is_pos = batch["is_positive"].to(device)
    center_logits = out["center_bind_logits"]    # [B, 17]
    center_targets = batch["rbp_targets"].to(device)  # [B, 17]

    # Collect all positive logits and targets
    all_pos_logits = []
    all_pos_targets = []

    if is_pos.any():
        all_pos_logits.append(center_logits[is_pos])
        all_pos_targets.append(center_targets[is_pos])

    if "extra_pos_bind_logits" in out and out.get("extra_pos_mask", None) is not None:
        ep_mask = out["extra_pos_mask"]
        if ep_mask.any():
            all_pos_logits.append(out["extra_pos_bind_logits"][ep_mask])
            all_pos_targets.append(batch["extra_pos_targets"].to(device)[ep_mask])

    if all_pos_logits:
        pos_l = torch.cat(all_pos_logits, dim=0)
        pos_t = torch.cat(all_pos_targets, dim=0)
        if use_asl:
            losses["rbp_pos"] = asymmetric_loss(
                pos_l, pos_t, gamma_neg=gamma_neg,
                clip=asl_clip, pos_weight=rbp_pos_weight)
        else:
            losses["rbp_pos"] = F.binary_cross_entropy_with_logits(
                pos_l, pos_t, pos_weight=rbp_pos_weight)

    # ---- 3) RBP uncertain neg (m6A, no reader) ----
    if "m6a_neg_bind_logits" in out and out.get("m6a_neg_mask", None) is not None:
        mask = out["m6a_neg_mask"]
        if mask.any() and rbp_uncertain_weight > 0:
            neg_l = out["m6a_neg_bind_logits"][mask]
            losses["rbp_uncertain"] = rbp_uncertain_weight * \
                F.binary_cross_entropy_with_logits(neg_l, torch.zeros_like(neg_l))

    # ---- 4) RBP clean neg (non-m6A A) ----
    if "clean_neg_bind_logits" in out and out.get("clean_neg_mask", None) is not None:
        mask = out["clean_neg_mask"]
        if mask.any() and rbp_clean_weight > 0:
            neg_l = out["clean_neg_bind_logits"][mask]
            losses["rbp_clean"] = rbp_clean_weight * \
                F.binary_cross_entropy_with_logits(neg_l, torch.zeros_like(neg_l))

    # Total
    total = torch.tensor(0.0, device=device)
    for v in losses.values():
        total = total + v
    losses["total"] = total

    return losses


# =====================================================================
# Training-step metrics (lightweight)
# =====================================================================

def compute_step_metrics(out, batch, device) -> dict[str, float]:
    is_pos = batch["is_positive"].to(device)
    result = {"batch_size": int(is_pos.shape[0]), "n_pos": int(is_pos.sum().item())}

    if is_pos.any():
        probs = torch.sigmoid(out["center_bind_logits"][is_pos])
        targets = batch["rbp_targets"].to(device)[is_pos] > 0.5
        preds = probs >= 0.5
        result["bind_acc"] = float((preds == targets).float().mean().item())
        n_positive = int(targets.sum().item())
        result["bind_recall"] = (
            float(preds[targets].float().mean().item()) if n_positive > 0 else 0.0)

    if "m6a_det_logits" in out and out.get("m6a_det_mask", None) is not None:
        mask = out["m6a_det_mask"]
        if mask.any():
            m6a_probs = torch.sigmoid(out["m6a_det_logits"][mask])
            m6a_targets = batch["m6a_det_targets"].to(device)[mask]
            m6a_preds = m6a_probs >= 0.5
            m6a_true = m6a_targets > 0.5
            result["m6a_acc"] = float((m6a_preds == m6a_true).float().mean().item())

    return result


# =====================================================================
# Top-k helpers
# =====================================================================

def _topk_metrics(probs, targets, k):
    N, C = probs.shape
    if N == 0 or k <= 0:
        return {"topk_recall": 0.0, "topk_precision": 0.0, "topk_site_hit": 0.0}
    k = min(k, C)
    topk_idx = np.argpartition(-probs, k, axis=1)[:, :k]
    topk_mask = np.zeros_like(targets, dtype=bool)
    topk_mask[np.repeat(np.arange(N), k), topk_idx.ravel()] = True
    hits = topk_mask & (targets > 0.5)
    total_pos = max(float((targets > 0.5).sum()), 1e-8)
    has_pos = (targets > 0.5).any(axis=1)
    site_hit = hits.any(axis=1)
    n_pos_sites = max(int(has_pos.sum()), 1)
    return {
        "topk_recall": float(hits.sum()) / total_pos,
        "topk_precision": float(hits.sum()) / float(N * k),
        "topk_site_hit": float((site_hit & has_pos).sum()) / n_pos_sites,
    }


def _optimize_thresholds(probs, targets, n_steps=50):
    C = probs.shape[1]
    best_t = np.full(C, 0.5)
    for c in range(C):
        yt, yp = targets[:, c], probs[:, c]
        if yt.sum() == 0:
            continue
        best_f1 = 0.0
        for t in np.linspace(0.01, 0.95, n_steps):
            pred = (yp >= t).astype(np.float64)
            tp = (pred * yt).sum()
            p = tp / max(tp + (pred * (1 - yt)).sum(), 1e-8)
            r = tp / max(tp + ((1 - pred) * yt).sum(), 1e-8)
            f1 = 2 * p * r / max(p + r, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t[c] = t
    return best_t


# =====================================================================
# Full epoch evaluation
# =====================================================================

def full_evaluate(
    # gated probs: P(m6A) × P(RBP|m6A)
    gated_probs: np.ndarray,       # [N, 17]
    targets: np.ndarray,           # [N, 17]
    is_pos: np.ndarray,            # [N]
    # m6A detection (optional)
    m6a_probs: np.ndarray | None = None,  # [N]  center site m6A prob
    # raw bind probs (without gate, for diagnostics)
    raw_bind_probs: np.ndarray | None = None,  # [N, 17]
    m6a_det_probs: np.ndarray | None = None,  
    m6a_det_targets: np.ndarray | None = None, 
    m6a_det_keys: np.ndarray | None = None, 
    top_ks=(3, 5),
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score, label_ranking_average_precision_score

    N, C = gated_probs.shape
    result = {"n_total": float(N), "n_positive_sites": float(is_pos.sum())}

    # ---- 1) Per-RBP metrics (on gated probs, all samples) ----
    auprcs, aurocs = [], []
    for i, name in enumerate(INDIVIDUAL_RBPS):
        yt = targets[:, i]
        ys = gated_probs[:, i]
        n_pos = int(yt.sum())
        result[f"{name}_n_pos"] = float(n_pos)
        if n_pos == 0 or int((yt < 0.5).sum()) == 0:
            result[f"{name}_auprc"] = float("nan")
            result[f"{name}_auroc"] = float("nan")
            continue
        ap = float(average_precision_score(yt, ys))
        ar = float(roc_auc_score(yt, ys))
        result[f"{name}_auprc"] = ap
        result[f"{name}_auroc"] = ar
        auprcs.append(ap)
        aurocs.append(ar)

    result["macro_auprc"] = float(np.mean(auprcs)) if auprcs else float("nan")
    result["macro_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")

    # ---- 2) Family aggregated ----
    fam_ap = []
    for fn in sorted(FAMILY_TO_IDXS.keys()):
        member_ap = [result.get(f"{INDIVIDUAL_RBPS[i]}_auprc", float("nan"))
                     for i in FAMILY_TO_IDXS[fn]]
        member_ap = [v for v in member_ap if not np.isnan(v)]
        v = float(np.mean(member_ap)) if member_ap else float("nan")
        result[f"family_{fn}_auprc"] = v
        if not np.isnan(v):
            fam_ap.append(v)
    result["family_macro_auprc"] = float(np.mean(fam_ap)) if fam_ap else float("nan")

    # ---- 3) F1 opt ----
    opt_t = _optimize_thresholds(gated_probs, targets)
    f1_opts = []
    for i, name in enumerate(INDIVIDUAL_RBPS):
        yt = targets[:, i]
        if int(yt.sum()) == 0:
            result[f"{name}_f1_opt"] = float("nan")
            continue
        pred = (gated_probs[:, i] >= opt_t[i]).astype(np.float64)
        tp = (pred * yt).sum()
        p = tp / max(tp + (pred * (1 - yt)).sum(), 1e-8)
        r = tp / max(tp + ((1 - pred) * yt).sum(), 1e-8)
        f1 = float(2 * p * r / max(p + r, 1e-8))
        result[f"{name}_f1_opt"] = f1
        f1_opts.append(f1)
    result["macro_f1_opt"] = float(np.mean(f1_opts)) if f1_opts else float("nan")

    # ---- 4) Top-k, LRAP, sample F1/IoU (on positive sites) ----
    pm = is_pos.astype(bool)
    if pm.sum() > 0:
        pp = gated_probs[pm]
        pt = targets[pm]
        for k in top_ks:
            tk = _topk_metrics(pp, pt, k)
            result[f"top{k}_recall"] = tk["topk_recall"]
            result[f"top{k}_precision"] = tk["topk_precision"]
            result[f"top{k}_site_hit"] = tk["topk_site_hit"]

        result["lrap"] = float(label_ranking_average_precision_score(pt, pp))

        # Sample F1 & IoU
        pos_preds = (pp >= opt_t[np.newaxis, :]).astype(np.float64)
        sf1s, sjacs = [], []
        for i in range(pos_preds.shape[0]):
            pred, true = pos_preds[i], pt[i]
            tp = (pred * true).sum()
            fp = (pred * (1 - true)).sum()
            fn = ((1 - pred) * true).sum()
            p = tp / max(tp + fp, 1e-8)
            r = tp / max(tp + fn, 1e-8)
            sf1s.append(2 * p * r / max(p + r, 1e-8))
            sjacs.append(tp / max(tp + fp + fn, 1e-8))
        result["sample_f1"] = float(np.mean(sf1s))
        result["sample_jaccard"] = float(np.mean(sjacs))

    # ---- 5) Negative site metrics (on gated probs) ----
    nm = ~pm
    if nm.sum() > 0:
        neg_p = gated_probs[nm]
        neg_max = neg_p.max(axis=1)
        result["neg_silence_rate"] = float((neg_max < 0.5).mean())
        result["neg_max_prob_mean"] = float(neg_max.mean())
        top3i = np.argpartition(-neg_p, min(3, neg_p.shape[1]), axis=1)[:, :3]
        top3p = np.take_along_axis(neg_p, top3i, axis=1)
        result["neg_false_alarm_top3"] = float((top3p.max(axis=1) > 0.3).mean())
        result["n_negative_sites"] = int(nm.sum())

    # ---- 6) m6A head metrics (center site) ----
    if m6a_probs is not None:
        m6a_labels = pm.astype(np.int64)  # approximation: positive sites ≈ m6A
        # Actually all center sites are m6A, but presence of reader is the label
        # For m6A head evaluation we use: all center sites are m6A (label=1)
        # vs negative center sites which are also m6A but without reader
        # So m6a_head should output high for all centers — check that:
        result["m6a_center_mean_prob"] = float(m6a_probs.mean())
        result["m6a_center_pos_mean"] = float(m6a_probs[pm].mean()) if pm.sum() > 0 else 0.0
        result["m6a_center_neg_mean"] = float(m6a_probs[nm].mean()) if nm.sum() > 0 else 0.0

    # ---- m6A detection evaluation ----
    if m6a_det_probs is not None and m6a_det_targets is not None and m6a_det_keys is not None:
        unique = {}
        for i in range(len(m6a_det_keys)):
            k = int(m6a_det_keys[i])
            if k not in unique:
                unique[k] = {"probs": [], "target": float(m6a_det_targets[i])}
            unique[k]["probs"].append(float(m6a_det_probs[i]))

        dedup_probs = np.array([np.mean(v["probs"]) for v in unique.values()])
        dedup_targets = np.array([v["target"] for v in unique.values()])

        m6a_binary = (dedup_targets > 0.5).astype(np.int64)
        n_m6a_pos = int(m6a_binary.sum())
        n_m6a_neg = int((m6a_binary == 0).sum())
        result["m6a_det_n_pos"] = float(n_m6a_pos)
        result["m6a_det_n_neg"] = float(n_m6a_neg)
        result["m6a_det_n_total"] = float(len(m6a_binary))
        result["m6a_det_n_raw"] = float(len(m6a_det_keys))          # 去重前的数量
        result["m6a_det_dedup_ratio"] = float(len(m6a_binary)) / max(float(len(m6a_det_keys)), 1)

        if n_m6a_pos > 0 and n_m6a_neg > 0:
            result["m6a_det_auprc"] = float(average_precision_score(m6a_binary, dedup_probs))
            result["m6a_det_auroc"] = float(roc_auc_score(m6a_binary, dedup_probs))

            for t in [0.1, 0.3, 0.5]:
                preds = (dedup_probs >= t).astype(np.int64)
                result[f"m6a_det_recall@{t}"] = float(preds[m6a_binary == 1].mean())
                result[f"m6a_det_precision@{t}"] = float(
                    m6a_binary[preds == 1].mean() if preds.sum() > 0 else 0.0)

        result["m6a_det_pos_prob_mean"] = float(dedup_probs[m6a_binary == 1].mean()) if n_m6a_pos > 0 else 0.0
        result["m6a_det_neg_prob_mean"] = float(dedup_probs[m6a_binary == 0].mean()) if n_m6a_neg > 0 else 0.0

    return result


def format_eval_table(result: dict[str, float], top_ks=(3, 5)) -> str:
    lines = []

    # Per-RBP
    lines.append(f"  {'RBP':12s} {'n_pos':>7s} {'AUPRC':>7s} {'AUROC':>7s} {'F1opt':>7s}")
    lines.append("  " + "-" * 46)
    for name in INDIVIDUAL_RBPS:
        n = int(result.get(f"{name}_n_pos", 0))
        ap = result.get(f"{name}_auprc", float("nan"))
        ar = result.get(f"{name}_auroc", float("nan"))
        f1 = result.get(f"{name}_f1_opt", float("nan"))
        lines.append(f"  {name:12s} {n:7d} {ap:7.4f} {ar:7.4f} {f1:7.4f}")
    lines.append("  " + "-" * 46)
    lines.append(f"  {'MACRO':12s} {'':7s} "
                 f"{result.get('macro_auprc', 0):7.4f} "
                 f"{result.get('macro_auroc', 0):7.4f} "
                 f"{result.get('macro_f1_opt', 0):7.4f}")

    # Family
    lines.append(f"  {'Family':12s} {'AUPRC':>7s}")
    lines.append("  " + "-" * 22)
    for fn in sorted(FAMILY_TO_IDXS.keys()):
        lines.append(f"  {fn:12s} {result.get(f'family_{fn}_auprc', 0):7.4f}")
    lines.append(f"  {'FAM MACRO':12s} {result.get('family_macro_auprc', 0):7.4f}")

    # Top-k
    for k in top_ks:
        r = result.get(f"top{k}_recall", 0)
        p = result.get(f"top{k}_precision", 0)
        h = result.get(f"top{k}_site_hit", 0)
        lines.append(f"  Top-{k}: recall={r:.4f}  precision={p:.4f}  site_hit={h:.4f}")

    # Site-level
    lines.append(f"  Site-level: LRAP={result.get('lrap', 0):.4f}  "
                 f"sample_F1={result.get('sample_f1', 0):.4f}  "
                 f"sample_IoU={result.get('sample_jaccard', 0):.4f}")

    # Neg
    lines.append(f"  Neg sites ({int(result.get('n_negative_sites', 0))}): "
                 f"silence={result.get('neg_silence_rate', 0):.4f}  "
                 f"avg_max_p={result.get('neg_max_prob_mean', 0):.4f}  "
                 f"false_alarm@3={result.get('neg_false_alarm_top3', 0):.4f}")

    # m6A head
    lines.append(f"  m6A head: center_avg={result.get('m6a_center_mean_prob', 0):.4f}  "
                 f"pos_avg={result.get('m6a_center_pos_mean', 0):.4f}  "
                 f"neg_avg={result.get('m6a_center_neg_mean', 0):.4f}")

    # m6A detection
    lines.append(
        f"  m6A detection ({int(result.get('m6a_det_n_total', 0))} unique / "
        f"{int(result.get('m6a_det_n_raw', 0))} raw, "
        f"{int(result.get('m6a_det_n_pos', 0))} pos): "
        f"AUPRC={result.get('m6a_det_auprc', 0):.4f}  "
        f"AUROC={result.get('m6a_det_auroc', 0):.4f}  "
        f"pos_prob={result.get('m6a_det_pos_prob_mean', 0):.4f}  "
        f"neg_prob={result.get('m6a_det_neg_prob_mean', 0):.4f}")
    lines.append(
        f"  m6A recall: @0.1={result.get('m6a_det_recall@0.1', 0):.4f}  "
        f"@0.3={result.get('m6a_det_recall@0.3', 0):.4f}  "
        f"@0.5={result.get('m6a_det_recall@0.5', 0):.4f}")
    return "\n".join(lines)