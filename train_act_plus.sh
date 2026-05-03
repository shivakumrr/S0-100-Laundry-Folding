#!/bin/bash
# ===========================================================
# ACT++ Training Launcher
# ===========================================================
# Usage: ./train_act_plus.sh
# ===========================================================

OUTPUT_DIR="outputs/act_plus_plus_ckpt_$(date +%Y%m%d_%H%M%S)"

echo "============================================="
echo "  ACT++ Training"
echo "  Output  : $OUTPUT_DIR"
echo "============================================="

cd act-plus-plus
~/miniconda3/envs/lerobot/bin/python imitate_episodes.py \
  ct1--task_name so100_teleop \
  --ckpt_dir "../$OUTPUT_DIR" \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_steps 50000 \
  --lr 1e-5 \
  --seed 0
