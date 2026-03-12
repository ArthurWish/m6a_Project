

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
import numpy as np
import torch

from models.etd_multitask.constants import (
    BASE_TO_ID,
    COND_BASE_IDS,
    MOD_BASE_CHANNEL,
    MOD_BASE_MAP,
    MOD_TYPE_IDS,
    MOD_TYPE_NAMES,
    PAD_TOKEN_ID,
    ROLE_IDS,
    TASK_IDS,
    ROLE_IDS,
    ROLE_NAMES,
    MOD_TOKEN_IDS,
)
from models.etd_multitask.data import (
    TranscriptExample,
    build_length_bucketed_batches,
)
from models.etd_multitask.metrics import (
    binary_auprc,
    binary_auroc,
    binary_f1,
    expected_calibration_error,
)
from models.etd_multitask.model import ETDMultiTaskModel
from models.etd_multitask.utils import encode_sequence




def _encode_batch_raw(
    examples: list[TranscriptExample],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """把一组 examples 编码成模型基础输入（无 A' 替换、无任务标签）。

    返回：
    - tokens [B, L]
    - attn_mask [B, L]
    - struct_feats [B, L, 8]
    """
    batch_size = len(examples)
    max_len = max(item.seq_len for item in examples)

    tokens = np.full((batch_size, max_len), PAD_TOKEN_ID, dtype=np.int64)
    attn_mask = np.zeros((batch_size, max_len), dtype=bool)

    for i, item in enumerate(examples):
        seq_ids = np.asarray(encode_sequence(item.sequence), dtype=np.int64)
        length = seq_ids.shape[0]
        tokens[i, :length] = seq_ids
        attn_mask[i, :length] = True


    return {
        "tokens": torch.tensor(tokens, dtype=torch.long, device=device),
        "attn_mask": torch.tensor(attn_mask, dtype=torch.bool, device=device),
    }


def _forward_mod(
    model: ETDMultiTaskModel,
    base_inputs: dict[str, torch.Tensor],
    mod_type: str,
    device: torch.device,
) -> np.ndarray:
    """对一个 batch 做 mod 前向，返回指定 mod_type 通道的 sigmoid 概率。

    返回 shape: [B, L] numpy array
    """
    bsz = base_inputs["tokens"].shape[0]
    channel = MOD_BASE_CHANNEL[mod_type]
    eval_base = MOD_BASE_MAP[mod_type]

    cond_task = torch.full((bsz,), TASK_IDS["mod"], device=device, dtype=torch.long)
    cond_role = torch.full((bsz,), ROLE_IDS["none"], device=device, dtype=torch.long)
    cond_base = torch.full((bsz,), COND_BASE_IDS[eval_base], device=device, dtype=torch.long)
    cond_mod_type = torch.full((bsz,), MOD_TYPE_IDS[mod_type], device=device, dtype=torch.long)

    outputs = model(
        tokens=base_inputs["tokens"],
        cond_task=cond_task,
        cond_role=cond_role,
        cond_base=cond_base,
        cond_mod_type=cond_mod_type,
        attn_mask=base_inputs["attn_mask"],
        compute_struct=False,
    )

    probs = torch.sigmoid(outputs["mod_logits_acu"][..., channel])
    return probs.detach().cpu().numpy()



def evaluate_mod_all_types(
    model: ETDMultiTaskModel,
    examples: list[TranscriptExample],
    batch_token_budget: int,
    boundaries: list[int],
    device: torch.device,
    neg_ratio: float = 1.0,
    seed: int = 0,
) -> dict[str, float]:
    """评估所有修饰类型的 mod 预测能力。

    流程：
    1. 按长度分桶组 batch
    2. 每个 batch 编码一次原始序列（共享输入）
    3. 对每种 mod_type 做一次前向，收集对应位置的预测
    4. 正集 = 该 mod_type 的全部 positive 位点（全覆盖）
    5. 负集 = 从 unlabeled 位点中按 neg_ratio 采样（相对于正集大小）
    6. 汇报 per-mod_type 指标 + 加权平均

    参数：
    - neg_ratio: 负/正采样比例，1.0 = 1:1 平衡采样
    - seed: 负采样的随机种子
    """
    model.eval()

    batches = build_length_bucketed_batches(
        examples=examples,
        batch_token_budget=batch_token_budget,
        boundaries=boundaries,
        shuffle=False,
        seed=0,
    )

    collectors: dict[str, dict] = {
        mt: {"y_true": [], "y_prob": [], "n_pos": 0, "n_unl": 0, "n_unl_sampled": 0}
        for mt in MOD_TYPE_NAMES
    }
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for batch_examples in batches:
            base_inputs = _encode_batch_raw(batch_examples, device)
            for mt in MOD_TYPE_NAMES:
                probs_np = _forward_mod(model, base_inputs, mt, device)
                col = collectors[mt]

                for i, item in enumerate(batch_examples):
                    pos_positions = item.mod_positions.get(mt)
                    unl_positions = item.unlabeled_positions.get(mt)

                    if pos_positions is None or pos_positions.size == 0:
                        continue

                    pos_probs = probs_np[i, pos_positions]
                    col["y_true"].append(np.ones(pos_positions.size, dtype=np.int64))
                    col["y_prob"].append(pos_probs)
                    col["n_pos"] += pos_positions.size

                    if unl_positions is not None and unl_positions.size > 0:
                        target_neg = max(1, int(round(pos_positions.size * neg_ratio)))
                        k = min(target_neg, unl_positions.size)
                        sampled_unl = rng.choice(unl_positions, size=k, replace=False)

                        unl_probs = probs_np[i, sampled_unl]
                        col["y_true"].append(np.zeros(k, dtype=np.int64))
                        col["y_prob"].append(unl_probs)
                        col["n_unl"] += unl_positions.size
                        col["n_unl_sampled"] += k

    results: dict[str, float] = {}
    total_pos_all = 0
    weighted_auroc = 0.0
    weighted_auprc = 0.0

    for mt in MOD_TYPE_NAMES:
        col = collectors[mt]
        prefix = f"mod_{mt}"

        if not col["y_true"] or col["n_pos"] == 0:
            results[f"{prefix}_auroc"] = float("nan")
            results[f"{prefix}_auprc"] = float("nan")
            results[f"{prefix}_f1"] = float("nan")
            results[f"{prefix}_ece"] = float("nan")
            results[f"{prefix}_n_pos"] = 0
            results[f"{prefix}_n_unl_sampled"] = 0
            continue

        y_true = np.concatenate(col["y_true"])
        y_prob = np.concatenate(col["y_prob"])

        auroc = binary_auroc(y_true, y_prob)
        auprc = binary_auprc(y_true, y_prob)
        f1 = binary_f1(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob)

        results[f"{prefix}_auroc"] = auroc
        results[f"{prefix}_auprc"] = auprc
        results[f"{prefix}_f1"] = f1
        results[f"{prefix}_ece"] = ece
        results[f"{prefix}_n_pos"] = col["n_pos"]
        results[f"{prefix}_n_unl_sampled"] = col["n_unl_sampled"]

        if np.isfinite(auroc):
            weighted_auroc += auroc * col["n_pos"]
            total_pos_all += col["n_pos"]
        if np.isfinite(auprc):
            weighted_auprc += auprc * col["n_pos"]

        print(
            f"  [{mt}] pos={col['n_pos']}, unl_sampled={col['n_unl_sampled']} | "
            f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}, F1={f1:.4f}, ECE={ece:.4f}",
            flush=True,
        )

    # 加权平均（按正样本数量加权）
    if total_pos_all > 0:
        results["mod_avg_auroc"] = weighted_auroc / total_pos_all
        results["mod_avg_auprc"] = weighted_auprc / total_pos_all
    else:
        results["mod_avg_auroc"] = float("nan")
        results["mod_avg_auprc"] = float("nan")

    return results



@dataclass
class _BindEvalSample:
    """一个待评估的 bind 样本（一条序列 + 一个揭示位点）。"""
    item: TranscriptExample
    site_pos: int
    label: int         # 1=positive, 0=negative
    mod_type: str
    seq_len: int



def _collect_bind_eval_samples(
    examples: list[TranscriptExample],
    mod_type: str,
    role: str,
    neg_ratio: float,
    rng: np.random.Generator,
) -> list[_BindEvalSample]:
    """收集指定 (mod_type, role) 的全部正样本和采样的负样本。

    正样本：role_labels[mod_type][role] == 1 的修饰位点（全覆盖）
    负样本：role_labels[mod_type][role] == -1 的修饰位点（按 neg_ratio 采样）
    """
    samples: list[_BindEvalSample] = []

    for item in examples:
        positions = item.mod_positions.get(mod_type)
        labels_arr = item.role_labels.get(mod_type, {}).get(role)
        if positions is None or positions.size == 0 or labels_arr is None:
            continue

        n = min(positions.size, labels_arr.size)
        pos_indices = np.where(labels_arr[:n] == 1)[0]
        neg_indices = np.where(labels_arr[:n] == -1)[0]

        # 全部正样本
        for idx in pos_indices:
            samples.append(_BindEvalSample(
                item=item,
                site_pos=int(positions[idx]),
                label=1,
                mod_type=mod_type,
                seq_len=item.seq_len,
            ))

        # 按比例采样负样本
        if neg_indices.size > 0 and pos_indices.size > 0:
            target_neg = max(1, int(round(pos_indices.size * neg_ratio)))
            k = min(target_neg, neg_indices.size)
            chosen_neg = rng.choice(neg_indices, size=k, replace=False)
            for idx in chosen_neg:
                samples.append(_BindEvalSample(
                    item=item,
                    site_pos=int(positions[idx]),
                    label=0,
                    mod_type=mod_type,
                    seq_len=item.seq_len,
                ))

    return samples



def _encode_single_site_batch(
    batch_samples: list[_BindEvalSample],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """把一批 _BindEvalSample 编码成模型输入。

    优化：同一批内的样本来自同一条序列（由调用方保证），
    只编码一次 base 序列再 tile，每份只改一个位点的 token。
    """
    bsz = len(batch_samples)
    ref = batch_samples[0]
    seq_ids_base = np.asarray(encode_sequence(ref.item.sequence), dtype=np.int64)
    length = seq_ids_base.shape[0]

    # tile：复制 bsz 份完全相同的序列
    tokens = np.tile(seq_ids_base, (bsz, 1))  # [bsz, length]，零 padding

    # 逐样本只替换一个位点
    for i, s in enumerate(batch_samples):
        tokens[i, s.site_pos] = MOD_TOKEN_IDS[s.mod_type]

    attn_mask = np.ones((bsz, length), dtype=bool)
    site_positions = np.array([[s.site_pos] for s in batch_samples], dtype=np.int64)
    site_mask = np.ones((bsz, 1), dtype=bool)

    return {
        "tokens": torch.tensor(tokens, dtype=torch.long, device=device),
        "attn_mask": torch.tensor(attn_mask, dtype=torch.bool, device=device),
        "site_positions": torch.tensor(site_positions, dtype=torch.long, device=device),
        "site_mask": torch.tensor(site_mask, dtype=torch.bool, device=device),
    }


def _forward_bind_batch(
    model: ETDMultiTaskModel,
    inputs: dict[str, torch.Tensor],
    mod_type: str,
    role: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """前向一个 batch，返回 bind 概率和不确定度。

    返回：
    - probs: [B] 每个样本的 bind sigmoid 概率
    - uncertainties: [B] 每个样本的不确定度
    """
    bsz = inputs["tokens"].shape[0]
    eval_base = MOD_BASE_MAP[mod_type]

    cond_task = torch.full((bsz,), TASK_IDS["bind"], device=device, dtype=torch.long)
    cond_role = torch.full((bsz,), ROLE_IDS[role], device=device, dtype=torch.long)
    cond_base = torch.full((bsz,), COND_BASE_IDS[eval_base], device=device, dtype=torch.long)
    cond_mod_type = torch.full((bsz,), MOD_TYPE_IDS[mod_type], device=device, dtype=torch.long)

    outputs = model(
        tokens=inputs["tokens"],
        cond_task=cond_task,
        cond_role=cond_role,
        cond_base=cond_base,
        cond_mod_type=cond_mod_type,
        attn_mask=inputs["attn_mask"],
        site_positions=inputs["site_positions"],
        site_mask=inputs["site_mask"],
        compute_struct=False,
    )

    # site 维度只有 1（每个样本只评估一个位点），取 [:, 0]
    probs = torch.sigmoid(outputs["bind_logits"][:, 0]).detach().cpu().numpy()
    uncertainties = outputs["bind_uncertainty"][:, 0].detach().cpu().numpy()
    return probs, uncertainties


def evaluate_bind_all_types(
    model: ETDMultiTaskModel,
    examples: list[TranscriptExample],
    batch_token_budget: int,
    device: torch.device,
    neg_ratio: float = 1.0,
    seed: int = 0,
) -> dict[str, float]:
    """评估所有 (mod_type, role) 组合的 bind 预测能力。
    流程：
    1. 对每种 (mod_type, role) 收集正/负评估样本
    2. 按序列长度分批（token budget 控制显存）
    3. 每个样本仅替换一个位点，独立预测
    4. 汇报 per-(mod_type, role) 指标 + 加权平均

    参数：
    - neg_ratio: 负/正采样比例
    - batch_token_budget: 控制每批的 token 总量
    """
    model.eval()
    rng = np.random.default_rng(seed)
    results: dict[str, float] = {}

    total_pos_global = 0
    weighted_auroc_global = 0.0
    weighted_auprc_global = 0.0

    with torch.no_grad():
        for mt in ["m6A"]:
            for role in ROLE_NAMES:
                samples = _collect_bind_eval_samples(
                    examples, mod_type=mt, role=role, neg_ratio=neg_ratio, rng=rng,
                )
                if role == "reader" and len(samples) > 0:
                    pos_samples = [s for s in samples if s.label == 1]
                    neg_samples = [s for s in samples if s.label == 0]

                    pos_k = max(1, len(pos_samples) // 10) if pos_samples else 0
                    neg_k = max(1, len(neg_samples) // 10) if neg_samples else 0

                    pos_idx = rng.choice(len(pos_samples), size=pos_k, replace=False) if pos_k > 0 else []
                    neg_idx = rng.choice(len(neg_samples), size=neg_k, replace=False) if neg_k > 0 else []

                    samples = [pos_samples[i] for i in pos_idx] + [neg_samples[i] for i in neg_idx]
                    rng.shuffle(samples)
                
                n_pos = sum(1 for s in samples if s.label == 1)
                n_neg = sum(1 for s in samples if s.label == 0)
                prefix = f"bind_{mt}_{role}"

                if n_pos == 0:
                    results[f"{prefix}_auroc"] = float("nan")
                    results[f"{prefix}_auprc"] = float("nan")
                    results[f"{prefix}_f1"] = float("nan")
                    results[f"{prefix}_ece"] = float("nan")
                    results[f"{prefix}_n_pos"] = 0
                    results[f"{prefix}_n_neg"] = 0
                    continue

                from collections import defaultdict
                groups = defaultdict(list)
                for s in samples:
                    groups[s.item.transcript_id].append(s)

                batches: list[list[_BindEvalSample]] = []
                for tid, group_samples in groups.items():
                    seq_len = group_samples[0].seq_len
                    max_per_batch = max(1, batch_token_budget // seq_len)
                    for start in range(0, len(group_samples), max_per_batch):
                        batches.append(group_samples[start:start + max_per_batch])

                y_true_all: list[np.ndarray] = []
                y_prob_all: list[np.ndarray] = []
                unc_all: list[np.ndarray] = []

                for batch_samples in batches:
                    inputs = _encode_single_site_batch(batch_samples, device)
                    probs, uncertainties = _forward_bind_batch(
                        model, inputs, mod_type=mt, role=role, device=device,
                    )
                    labels = np.array([s.label for s in batch_samples], dtype=np.int64)
                    y_true_all.append(labels)
                    y_prob_all.append(probs)
                    unc_all.append(uncertainties)

                y_true = np.concatenate(y_true_all)
                y_prob = np.concatenate(y_prob_all)
                unc = np.concatenate(unc_all)

                auroc = binary_auroc(y_true, y_prob)
                auprc = binary_auprc(y_true, y_prob)
                f1 = binary_f1(y_true, y_prob)
                ece = expected_calibration_error(y_true, y_prob)

                pos_mask = y_true == 1
                neg_mask = y_true == 0
                prob_pos = float(y_prob[pos_mask].mean()) if pos_mask.any() else float("nan")
                prob_neg = float(y_prob[neg_mask].mean()) if neg_mask.any() else float("nan")
                unc_pos = float(unc[pos_mask].mean()) if pos_mask.any() else float("nan")
                unc_neg = float(unc[neg_mask].mean()) if neg_mask.any() else float("nan")

                results[f"{prefix}_auroc"] = auroc
                results[f"{prefix}_auprc"] = auprc
                results[f"{prefix}_f1"] = f1
                results[f"{prefix}_ece"] = ece
                results[f"{prefix}_n_pos"] = n_pos
                results[f"{prefix}_n_neg"] = n_neg
                results[f"{prefix}_prob_pos"] = prob_pos
                results[f"{prefix}_prob_neg"] = prob_neg
                results[f"{prefix}_unc_pos"] = unc_pos
                results[f"{prefix}_unc_neg"] = unc_neg

                print(
                    f"  [{mt}/{role}] pos={n_pos}, neg={n_neg} | "
                    f"AUROC={auroc:.4f}, AUPRC={auprc:.4f}, F1={f1:.4f} | "
                    f"prob: pos={prob_pos:.3f} neg={prob_neg:.3f} | "
                    f"unc: pos={unc_pos:.3f} neg={unc_neg:.3f}",
                    flush=True,
                )

                if np.isfinite(auroc):
                    weighted_auroc_global += auroc * n_pos
                    weighted_auprc_global += auprc * n_pos
                    total_pos_global += n_pos

    if total_pos_global > 0:
        results["bind_avg_auroc"] = weighted_auroc_global / total_pos_global
        results["bind_avg_auprc"] = weighted_auprc_global / total_pos_global
    else:
        results["bind_avg_auroc"] = float("nan")
        results["bind_avg_auprc"] = float("nan")

    return results
