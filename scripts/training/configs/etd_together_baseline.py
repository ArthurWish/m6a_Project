"""Current ETD together baseline training config.

This is the canonical config for the maintained ETD-together baseline path.
"""

from __future__ import annotations


CONFIG = {
    "exp_name": "v4_multitask_m6a_rbp_baseline_hw256_bs64_e1_rand33_330",
    "seed": 42,
    # Data
    "sites_path": "data/processed/all_multitask_sites.parquet",
    "transcripts_path": "data/processed/all_multitask_transcripts.parquet",
    "splits_path": "data/processed/all_multitask_splits.json",
    "mod_type": "m6A",
    "role_name": "reader",
    "max_len": 12000,
    "half_window": 256,
    "max_jitter": 0,
    "neg_ratio": 0.3,
    "smoke_ratio": 1.0,
    "subset_num_shards": 32,
    "subset_include_shards": "0-10",
    "align_to_struct_cache_dir": "data/processed/etd_together_struct_bias_hw256_part33",
    # Sampling per window
    "n_extra_pos": 5,
    "n_m6a_neg": 3,
    "n_clean_neg": 8,
    "n_m6a_eval_a": 32,
    "m6a_neg_smooth": 0.002,
    # Model
    "d_model": 256,
    "encoder_channels": "256,384,512",
    "n_transformer_layers": 4,
    "n_heads": 8,
    "ff_mult": 4,
    "dropout": 0.15,
    "head_dropout": 0.25,
    # Training
    "batch_size": 64,
    "val_batch_size": 64,
    "epochs": 1,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "grad_clip_norm": 1.0,
    # Loss
    "use_asl": True,
    "gamma_neg": 3.0,
    "asl_clip": 0.05,
    "m6a_loss_weight": 1.0,
    "m6a_pos_weight": 5,
    "rbp_uncertain_weight": 0.5,
    "rbp_clean_weight": 0.2,
    # Structure bias disabled in baseline
    "use_struct_bias": False,
    "online_rnafold_bin": "RNAfold",
    "online_rnafold_timeout_seconds": 240,
    "online_rnafold_cache_size": 4096,
    "struct_bias_scale": 1.0,
    # Scheduler
    "pct_start": 0.15,
    "div_factor": 25.0,
    "final_div_factor": 20.0,
    # Misc
    "use_balanced_sampler": True,
    "log_interval": 20,
    "dataloader_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "ddp_dataloader_workers": 2,
    "ddp_prefetch_factor": 2,
    "ddp_persistent_workers": False,
    "output_root": "outputs/etd_bind_baseline/aligned_subset",
}
