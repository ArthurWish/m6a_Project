from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

# ======== spawn 模式：仅在 ViennaRNA Python API 不可用、需 subprocess fallback 时必要 ========
import torch.multiprocessing as mp

# DDP + many DataLoader workers + forced spawn is prone to semaphore/resource_tracker
# shutdown issues on this codepath. Keep Linux default unless explicitly requested.
_requested_mp_start = os.environ.get("ETD_MP_START_METHOD", "").strip().lower()
if _requested_mp_start:
    try:
        mp.set_start_method(_requested_mp_start, force=True)
    except RuntimeError:
        pass

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ======== 2. 新版 DataLoader：struct bias 在 __getitem__ 里并行计算 ========
from models.etd_only.bind_dataloader_v5 import (
    MultiTaskDataset,
    ETDCollate,
    StructBiasConfig,
    load_site_metas_and_seqs,
)
from models.etd_multitask.constants import (
    NUM_INDIVIDUAL_RBPS,
    INDIVIDUAL_RBPS,
    FAMILY_TO_IDXS,
)
from models.etd_only.bind_loss_v4 import (
    compute_multitask_loss,
    compute_step_metrics,
    full_evaluate,
    format_eval_table,
)
from models.etd_only.etd_bind_bias import MultiTaskBindBiasModel, MultiTaskBiasConfig
from models.etd_only.etd_bind_v4 import MultiTaskBindModel, MultiTaskConfig
from scripts.training.configs.etd_together_offline_bias import CONFIG as BIAS_CONFIG
from scripts.training.configs.etd_together_baseline import CONFIG as BASELINE_CONFIG

# =====================================================================
# Utilities
# =====================================================================

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _compute_rbp_pos_weight(metas, alpha=0.5, max_w=3.0):
    pos = [m for m in metas if m.is_positive]
    if not pos: return np.ones(NUM_INDIVIDUAL_RBPS, dtype=np.float32)
    tgt = np.stack([m.rbp_targets for m in pos])
    n_pos = tgt.sum(axis=0)
    n_neg = float(tgt.shape[0]) - n_pos
    ratio = n_neg / np.maximum(n_pos, 1.0)
    return np.clip(np.power(ratio, alpha), 1.0, max_w).astype(np.float32)

def _shuffled_transcript_ids(transcript_ids, seed: int) -> list[str]:
    tids = sorted({str(tid) for tid in transcript_ids})
    rng = random.Random(int(seed))
    rng.shuffle(tids)
    return tids

def _parse_shard_spec(spec: str, num_shards: int) -> set[int]:
    spec = str(spec).strip()
    if not spec: return set(range(int(num_shards)))
    selected: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token: continue
        if "-" in token:
            lo_str, hi_str = token.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            if hi < lo: lo, hi = hi, lo
            for idx in range(lo, hi + 1):
                if 0 <= idx < int(num_shards): selected.add(idx)
        else:
            idx = int(token)
            if 0 <= idx < int(num_shards): selected.add(idx)
    return selected

def _filter_metas_by_transcript_shards(train_metas, val_metas, num_shards: int, include_spec: str, seed: int):
    num_shards = int(num_shards)
    if num_shards <= 1:
        return train_metas, val_metas, {
            "enabled": False, "kept_train": len(train_metas),
            "kept_val": len(val_metas), "kept_transcripts": len({m.transcript_id for m in train_metas + val_metas}),
        }

    include = _parse_shard_spec(include_spec, num_shards)
    all_tids = _shuffled_transcript_ids(
        transcript_ids=[m.transcript_id for m in train_metas + val_metas], seed=seed)
    keep_tids = {tid for i, tid in enumerate(all_tids) if i % num_shards in include}
    train_filtered = [m for m in train_metas if m.transcript_id in keep_tids]
    val_filtered = [m for m in val_metas if m.transcript_id in keep_tids]
    return train_filtered, val_filtered, {
        "enabled": True, "num_shards": num_shards, "include_spec": include_spec,
        "n_selected_shards": len(include), "seed": int(seed),
        "kept_train": len(train_filtered), "kept_val": len(val_filtered),
        "kept_transcripts": len(keep_tids),
    }

class DistributedWeightedSampler(DistributedSampler):
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

def gather_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized(): return t
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return torch.cat(gathered, dim=0)

# =====================================================================
# Evaluate
# =====================================================================

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
    rbp_pos_weight: torch.Tensor | None = None,
    is_main: bool = True,
) -> tuple[float, dict[str, float]]:
    model.eval()

    all_gated_probs, all_raw_bind_probs, all_m6a_probs = [], [], []
    all_targets, all_is_pos = [], []
    all_m6a_det_probs, all_m6a_det_targets, all_m6a_det_keys = [], [], []
    total_loss = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0, device=device)

    for step, batch in enumerate(dataloader):
        if is_main and step % config.get("log_interval", 20) == 0:
            print(f"  [Eval] processing batch {step}/{len(dataloader)}...", flush=True)
            
        tokens = batch["tokens"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        center_indices = batch["center_indices"].to(device)
        
        attn_bias = batch.get("attn_bias", None)
        if attn_bias is not None:
            attn_bias = attn_bias.to(device)

        model_kwargs = dict(
            tokens=tokens,
            attn_mask=attn_mask,
            center_indices=center_indices,
            m6a_det_positions=batch["m6a_det_positions"].to(device),
            m6a_det_mask=batch["m6a_det_mask"].to(device),
        )
        if config.get("use_struct_bias", False):
            model_kwargs["attn_bias"] = attn_bias

        out = model(**model_kwargs)

        losses = compute_multitask_loss(
            out, batch,
            rbp_pos_weight=rbp_pos_weight,
            use_asl=config["use_asl"],
            gamma_neg=config["gamma_neg"],
            asl_clip=config["asl_clip"],
            m6a_loss_weight=config["m6a_loss_weight"],
            m6a_pos_weight=config["m6a_pos_weight"],
            rbp_uncertain_weight=0.0,
            rbp_clean_weight=0.0,
        )

        bs = tokens.shape[0]
        total_loss += losses["total"] * bs
        total_samples += bs

        m6a_gate = torch.sigmoid(out["center_m6a_logit"]).unsqueeze(-1)
        bind_probs = torch.sigmoid(out["center_bind_logits"])
        gated = m6a_gate * bind_probs

        all_gated_probs.append(gated)
        all_raw_bind_probs.append(bind_probs)
        all_m6a_probs.append(torch.sigmoid(out["center_m6a_logit"]))
        all_targets.append(batch["rbp_targets"].to(device))
        all_is_pos.append(batch["is_positive"].to(device))

        mask = out["m6a_det_mask"]     
        det_probs = torch.sigmoid(out["m6a_det_logits"])  
        det_targets = batch["m6a_det_targets"].to(device) 
        global_pos = batch["m6a_det_global_pos"].to(device)
        seq_idx = batch["m6a_det_seq_idx"].to(device)  
        seq_expanded = seq_idx.unsqueeze(1).expand_as(mask)
        keys = seq_expanded * 100000 + global_pos

        all_m6a_det_probs.append(det_probs[mask])  
        all_m6a_det_targets.append(det_targets[mask])
        all_m6a_det_keys.append(keys[mask])

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        total_samples_sync = total_samples.clone()
        dist.all_reduce(total_samples_sync, op=dist.ReduceOp.SUM)
    else:
        total_samples_sync = total_samples
    avg_loss = (total_loss / total_samples_sync.clamp(min=1)).item()

    gated_cat = gather_tensor(torch.cat(all_gated_probs), device)
    raw_cat = gather_tensor(torch.cat(all_raw_bind_probs), device)
    m6a_cat = gather_tensor(torch.cat(all_m6a_probs), device)
    tgt_cat = gather_tensor(torch.cat(all_targets), device)
    pos_cat = gather_tensor(torch.cat(all_is_pos), device)
    det_probs_cat = gather_tensor(torch.cat(all_m6a_det_probs), device)
    det_targets_cat = gather_tensor(torch.cat(all_m6a_det_targets), device)
    det_keys_cat = gather_tensor(torch.cat(all_m6a_det_keys), device)
    
    if not is_main: return avg_loss, {}

    result = full_evaluate(
        gated_probs=gated_cat.cpu().numpy(), targets=tgt_cat.cpu().numpy(),
        is_pos=pos_cat.cpu().numpy(), m6a_probs=m6a_cat.cpu().numpy(),
        raw_bind_probs=raw_cat.cpu().numpy(), m6a_det_probs=det_probs_cat.cpu().numpy(),
        m6a_det_targets=det_targets_cat.cpu().numpy(), m6a_det_keys=det_keys_cat.cpu().numpy(),
        top_ks=(3, 5),
    )
    return avg_loss, result


# =====================================================================
# Main
# =====================================================================

def main():
    use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main = global_rank == 0
    else:
        local_rank = global_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    def mprint(*args, **kw):
        if is_main: print(*args, **kw)

    config_name = os.environ.get("ETD_TOGETHER_CONFIG", "bias").strip().lower()
    if config_name == "baseline": config = dict(BASELINE_CONFIG)
    elif config_name == "bias": config = dict(BIAS_CONFIG)
    else: raise ValueError(f"Unsupported ETD_TOGETHER_CONFIG={config_name!r}")
    
    _set_seed(config["seed"] + global_rank)

    out_dir = Path(str(config["output_root"])) / str(config["exp_name"])
    out_dir = out_dir.resolve()
    ckpt_dir = out_dir / "checkpoints"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "run_config.json").open("w") as f: json.dump(config, f, indent=2)
        writer = SummaryWriter(log_dir=str(out_dir / "logs"))
        mprint(f"[config] mode={config_name} exp={config['exp_name']}")
        mprint(f"[config] output_dir={out_dir}")

    enc_ch = tuple(int(x) for x in config["encoder_channels"].split(","))
    n_downsample = sum(1 for i in range(len(enc_ch)) if i < len(enc_ch) - 1)

    mprint("[data] loading train metas ...")
    train_metas, train_store = load_site_metas_and_seqs(
        sites_path=config["sites_path"], transcripts_path=config["transcripts_path"],
        splits_path=config["splits_path"], split_names=["train"], mod_type=config["mod_type"],
        role_name=config["role_name"], max_len=config["max_len"], neg_ratio=config["neg_ratio"],
        smoke_ratio=config["smoke_ratio"], seed=config["seed"] + global_rank,
    )
    mprint(f"[data] train metas loaded: n={len(train_metas)}")
    
    mprint("[data] loading val metas ...")
    val_metas, val_store = load_site_metas_and_seqs(
        sites_path=config["sites_path"], transcripts_path=config["transcripts_path"],
        splits_path=config["splits_path"], split_names=["val"], mod_type=config["mod_type"],
        role_name=config["role_name"], max_len=config["max_len"], neg_ratio=config["neg_ratio"],
        smoke_ratio=config["smoke_ratio"], seed=config["seed"],
    )
    mprint(f"[data] val metas loaded: n={len(val_metas)}")

    subset_num_shards = int(config.get("subset_num_shards", 1))
    subset_include_shards = str(config.get("subset_include_shards", "")).strip()
    train_metas, val_metas, subset_info = _filter_metas_by_transcript_shards(
        train_metas=train_metas, val_metas=val_metas, num_shards=subset_num_shards,
        include_spec=subset_include_shards, seed=int(config["seed"]),
    )
    if is_main and subset_info.get("enabled", False):
        mprint(f"[data] subset filter enabled: {subset_info['kept_transcripts']} transcripts kept.")

    # ======== 构建 struct bias config ========
    use_struct = config.get("use_struct_bias", False)
    sb_cfg = StructBiasConfig(
        enabled=use_struct,
        mode=str(config.get("struct_bias_backend", "online")).strip().lower(),
        scale=float(config.get("struct_bias_scale", 1.0)),
        n_downsample=n_downsample,
        window_size=2 * config["half_window"] + 1,
        rnafold_bin=config.get("online_rnafold_bin", "RNAfold"),
        rnafold_timeout=int(config.get("online_rnafold_timeout_seconds", 240)),
        cache_dir=config.get("offline_struct_cache_dir", ""),
        offline_cache_max_transcripts=int(config.get("offline_cache_max_transcripts", 256)),
    )

    train_ds = MultiTaskDataset(
        train_metas, train_store,
        half_window=config["half_window"], max_jitter=config["max_jitter"], training=True,
        n_extra_pos=config["n_extra_pos"], n_m6a_neg=config["n_m6a_neg"],
        n_clean_neg=config["n_clean_neg"], n_m6a_eval_a=config["n_m6a_eval_a"],
        m6a_neg_smooth=config["m6a_neg_smooth"],
        struct_bias_cfg=sb_cfg,
    )
    # validation 不需要 struct bias（可选），如果需要也传进去
    val_sb_cfg = StructBiasConfig(
        enabled=use_struct,
        mode=sb_cfg.mode,
        scale=sb_cfg.scale,
        n_downsample=n_downsample,
        window_size=sb_cfg.window_size,
        rnafold_bin=sb_cfg.rnafold_bin,
        rnafold_timeout=sb_cfg.rnafold_timeout,
        cache_dir=sb_cfg.cache_dir,
        offline_cache_max_transcripts=sb_cfg.offline_cache_max_transcripts,
    )
    val_ds = MultiTaskDataset(
        val_metas, val_store,
        half_window=config["half_window"], max_jitter=0, training=False,
        n_extra_pos=0, n_m6a_neg=0, n_clean_neg=0, n_m6a_eval_a=config["n_m6a_eval_a"],
        m6a_neg_smooth=config["m6a_neg_smooth"],
        struct_bias_cfg=val_sb_cfg,
    )

    rbp_counts = np.zeros(NUM_INDIVIDUAL_RBPS, dtype=np.float64)
    for m in train_metas: rbp_counts += m.rbp_targets.astype(np.float64)
    rbp_w = 1.0 / np.sqrt(np.maximum(rbp_counts, 1.0))
    pos_ws = [float(rbp_w[m.rbp_targets > 0.5].max()) for m in train_metas if m.is_positive and (m.rbp_targets > 0.5).any()]
    avg_pw = float(np.mean(pos_ws)) if pos_ws else 1.0

    sample_weights = []
    for m in train_metas:
        if m.is_positive and (m.rbp_targets > 0.5).any():
            sample_weights.append(float(rbp_w[m.rbp_targets > 0.5].max()))
        else: sample_weights.append(avg_pw)

    if use_ddp:
        train_sampler = DistributedWeightedSampler(train_ds, weights=sample_weights)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else: train_sampler, val_sampler = None, None

    # ======== DataLoader：struct bias 已在 __getitem__ 里算好，collate 只做 stack ========
    num_workers = int(config.get("dataloader_workers", 16))
    if use_ddp:
        ddp_default_workers = 0 if use_struct else min(num_workers, 2)
        num_workers = int(config.get("ddp_dataloader_workers", ddp_default_workers))
    prefetch_factor = int(config.get("prefetch_factor", 4))
    if use_ddp:
        prefetch_factor = int(config.get("ddp_prefetch_factor", min(prefetch_factor, 2)))
    persistent_workers = bool(config.get("persistent_workers", num_workers > 0))
    if use_ddp:
        persistent_workers = bool(config.get("ddp_persistent_workers", False))

    collate_obj = ETDCollate(use_struct_bias=use_struct)
    train_loader_kwargs = dict(
        dataset=train_ds, batch_size=config["batch_size"], sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collate_obj,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=persistent_workers,
    )
    val_loader_kwargs = dict(
        dataset=val_ds, batch_size=config["val_batch_size"], sampler=val_sampler,
        shuffle=False, collate_fn=collate_obj,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
        val_loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)
    mprint(f"[data] train_loader={len(train_loader)} val_loader={len(val_loader)}")
    mprint(
        f"[data] dataloader_workers={num_workers} prefetch_factor="
        f"{prefetch_factor if num_workers > 0 else 0} persistent_workers={persistent_workers}"
    )

    if config.get("use_struct_bias", False):
        model_cfg = MultiTaskBiasConfig(
            d_model=config["d_model"], encoder_channels=enc_ch,
            n_transformer_layers=config["n_transformer_layers"], n_heads=config["n_heads"],
            ff_mult=config["ff_mult"], dropout=config["dropout"], head_dropout=config["head_dropout"],
        )
        model = MultiTaskBindBiasModel(model_cfg).to(device)
    else:
        model_cfg = MultiTaskConfig(
            d_model=config["d_model"], encoder_channels=enc_ch,
            n_transformer_layers=config["n_transformer_layers"], n_heads=config["n_heads"],
            ff_mult=config["ff_mult"], dropout=config["dropout"], head_dropout=config["head_dropout"],
        )
        model = MultiTaskBindModel(model_cfg).to(device)
    if use_ddp: model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    mprint(f"[model] parameters: {_count_parameters(model):,}")

    rbp_pw_tensor = torch.tensor(_compute_rbp_pos_weight(train_metas, max_w=3.0), device=device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = OneCycleLR(
        optimizer, max_lr=config["lr"], total_steps=len(train_loader) * config["epochs"],
        pct_start=config["pct_start"], div_factor=config["div_factor"], final_div_factor=config["final_div_factor"],
    )

    best_val_auprc = 0.0
    global_step = 0

    for epoch in range(1, config["epochs"] + 1):
        if use_ddp and train_sampler is not None: train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss, epoch_m6a_loss, epoch_rbp_pos_loss = 0.0, 0.0, 0.0

        mprint(f"============================================================")
        if config.get("use_struct_bias", False):
            mprint(f"[E{epoch:02d}] 🚀 struct bias 由 {num_workers} 个 DataLoader workers 并行准备。")
        else:
            mprint(f"[E{epoch:02d}] 🚀 baseline 数据由 {num_workers} 个 DataLoader workers 并行准备。")
        mprint(f"============================================================")

        for step, batch in enumerate(train_loader, 1):
            if step == 1:
                mprint(f"✅ 第一批数据已就绪，GPU 开始全速训练！")
            
            global_step += 1

            tokens = batch["tokens"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            center_indices = batch["center_indices"].to(device)
            
            attn_bias = batch.get("attn_bias", None)
            if attn_bias is not None:
                attn_bias = attn_bias.to(device)

            model_kwargs = dict(
                tokens=tokens,
                attn_mask=attn_mask,
                center_indices=center_indices,
                m6a_det_positions=batch["m6a_det_positions"].to(device),
                m6a_det_mask=batch["m6a_det_mask"].to(device),
                extra_pos_positions=batch["extra_pos_positions"].to(device),
                extra_pos_mask=batch["extra_pos_mask"].to(device),
                m6a_neg_positions=batch["m6a_neg_positions"].to(device),
                m6a_neg_mask=batch["m6a_neg_mask"].to(device),
                clean_neg_positions=batch["clean_neg_positions"].to(device),
                clean_neg_mask=batch["clean_neg_mask"].to(device),
            )
            if config.get("use_struct_bias", False):
                model_kwargs["attn_bias"] = attn_bias

            out = model(**model_kwargs)

            losses = compute_multitask_loss(
                out, batch, rbp_pos_weight=rbp_pw_tensor, use_asl=config["use_asl"],
                gamma_neg=config["gamma_neg"], asl_clip=config["asl_clip"],
                m6a_loss_weight=config["m6a_loss_weight"], m6a_pos_weight=config["m6a_pos_weight"],
                rbp_uncertain_weight=config["rbp_uncertain_weight"], rbp_clean_weight=config["rbp_clean_weight"],
            )

            loss = losses["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_m6a_loss += losses.get("m6a", torch.tensor(0.0)).item()
            epoch_rbp_pos_loss += losses.get("rbp_pos", torch.tensor(0.0)).item()

            if step % config["log_interval"] == 0:
                with torch.no_grad(): step_m = compute_step_metrics(out, batch, device)
                mprint(
                    f"[E{epoch:02d}][{step:04d}/{len(train_loader)}] loss={loss.item():.4f} "
                    f"L_m6a={losses.get('m6a', torch.tensor(0.0)).item():.4f} L_rbp={losses.get('rbp_pos', torch.tensor(0.0)).item():.4f} "
                    f"bind_acc={step_m.get('bind_acc', 0):.3f} m6a_acc={step_m.get('m6a_acc', 0):.3f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
                if is_main:
                    writer.add_scalar("Train/Loss_total", loss.item(), global_step)
                    writer.add_scalar("Train/Loss_m6a", losses.get("m6a", torch.tensor(0.0)).item(), global_step)
                    writer.add_scalar("Train/Loss_rbp_pos", losses.get("rbp_pos", torch.tensor(0.0)).item(), global_step)
                    writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], global_step)

    
        val_loss, val_metrics = evaluate_epoch(
            model, val_loader, device, config, rbp_pos_weight=rbp_pw_tensor, is_main=is_main)

        if is_main:
            train_loss = epoch_loss / len(train_loader)
            train_m6a = epoch_m6a_loss / len(train_loader)
            train_rbp = epoch_rbp_pos_loss / len(train_loader)
            val_auprc = val_metrics.get("macro_auprc", 0.0)
            
            log_str = (
                f"\nEpoch {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"AUPRC={val_auprc:.4f} AUROC={val_metrics.get('macro_auroc', 0.0):.4f}\n"
            ) + format_eval_table(val_metrics, top_ks=(3, 5)) + "\n"
            print(log_str, end="")
            with open(out_dir / "val_metrics.log", "a") as f: f.write(log_str)

            writer.add_scalar("Loss/Train_Epoch", train_loss, epoch)
            writer.add_scalar("Loss/Val_Epoch", val_loss, epoch)
            writer.add_scalar("Metrics/Macro_AUPRC", val_auprc, epoch)
            
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, ckpt_dir / f"epoch_{epoch:02d}.pt")
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                torch.save(state_dict, ckpt_dir / "best.pt")

    if is_main: writer.close()
    if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
