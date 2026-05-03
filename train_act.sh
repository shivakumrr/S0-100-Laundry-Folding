#!/bin/bash
# ===========================================================
# ACT Training Launcher for Dual-Arm Hand Teleop Dataset
# ===========================================================
# Usage:
#   ./train_act.sh                        (uses default dataset path)
#   ./train_act.sh data/hand_teleop_imitation_1773977954
# ===========================================================

DATASET_PATH="${1:-data/hand_teleop_imitation_1773977954}"
OUTPUT_DIR="outputs/act_dual_arm_$(date +%Y%m%d_%H%M%S)"

echo "============================================="
echo "  LeRobot ACT Training"
echo "  Dataset : $DATASET_PATH"
echo "  Output  : $OUTPUT_DIR"
echo "============================================="

~/miniconda3/envs/lerobot/bin/python lerobot/src/lerobot/scripts/lerobot_train.py \
  --policy.type=act \
  --dataset.repo_id=local_teleop/imitation_dataset \
  --dataset.root="$DATASET_PATH" \
  --output_dir="$OUTPUT_DIR" \
  --policy.push_to_hub=false \
  --dataset.video_backend=pyav \
  --batch_size=16 \
  --steps=100000 \
  --log_freq=100 \
  --save_freq=5000 \
  --num_workers=4 \
