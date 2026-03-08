#!/usr/bin/env bash
set -euo pipefail

cd /media/scw-workspace/m6a_dataset

/root/miniconda3/envs/m6a/bin/python scripts/dataset/build_rnafold_single_site_dense_cache.py \
  --max-len 12000 \
  --jobs 24 \
  --num-shards 4 \
  --shard-index 1 \
  --tmp-dir /tmp \
  --rnafold-bin /root/miniconda3/envs/m6a/bin/RNAfold \
  --output-dir data/processed/rnafold_single_site_dense \
  --manifest-out data/processed/rnafold_single_site_dense_manifest_shard1.json
