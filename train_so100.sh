#!/bin/bash

# ==============================================================================
# ACT Policy Training Script for SO100 (LOCAL DATASET VERSION)
# ==============================================================================

export HF_USER="team-11"

echo "Starting ACT Training using LOCAL dataset..."
echo "Policy will be pushed to: ${HF_USER}/my_policy"

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 lerobot-train \
  --dataset.repo_id=local_dataset \
  --dataset.root="/home/g1/Developer/Physical Agents/data/imitation_dataset" \
  --policy.type=act \
  --output_dir="/home/g1/Developer/Physical Agents/outputs/train/act_so100_local" \
  --job_name=act_so100_local \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=80000 \
  --save_freq=10000 \
  --log_freq=100