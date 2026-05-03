#!/bin/bash
/home/g1/miniconda3/envs/lerobot/bin/lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_arm \
  --robot.cameras '{"camera_2": {"type": "opencv", "index_or_path": 4, "fps": 30, "width": 640, "height": 480}}' \
  --dataset.repo_id=local_teleop/imitation_dataset \
  --dataset.single_task="puppet_teleoperation" \
  --dataset.fps=30 \
  --dataset.episode_time_s=30 \
  --dataset.push_to_hub=false \
  --dataset.root="data/imitation_dataset" \
  --dataset.num_episodes=10 \
  --dataset.reset_time_s=10
  # --play_sounds=false \
  
