#!/home/g1/miniconda3/envs/lerobot/bin/python
"""
run_act_plus.py — Deploy an ACT++ trained policy on real SO-100 arms.

Usage:
    python run_act_plus.py --ckpt_dir outputs/act_plus_plus_ckpt_XXXX
"""

import sys
import os
import argparse
from pathlib import Path
import pickle
import cv2
import numpy as np
import torch

from act_plus_plus.policy import ACTPolicy

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default=None)
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--step", type=int, default=500)
args = parser.parse_args()

# ── find latest checkpoint automatically ─────────────────────────────────────
def find_latest_ckpt_dir():
    candidates = sorted(Path("outputs").glob("act_plus_plus_ckpt_*"))
    if not candidates:
        raise FileNotFoundError("No act_plus_plus_ckpt_ folder found under outputs/")
    return candidates[-1]

ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else find_latest_ckpt_dir()
ckpt_path = ckpt_dir / f"policy_step_{args.step}_seed_0.ckpt"
if getattr(ckpt_path, "exists") and not ckpt_path.exists():
    ckpt_path = ckpt_dir / "policy_last.ckpt"

print(f"\n[POLICY] Loading ACT++ checkpoint: {ckpt_path}")

# Load properties from config
with open(ckpt_dir / "config.pkl", "rb") as f:
    config = pickle.load(f)

with open(ckpt_dir / "dataset_stats.pkl", "rb") as f:
    stats = pickle.load(f)

# Initialize policy
policy_config = config['policy_config']
policy = ACTPolicy(policy_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loading_status = policy.deserialize(torch.load(ckpt_path, map_location=device))
print(f"[POLICY] Checkpoint loaded {loading_status}")
policy.to(device)
policy.eval()

# Helper lambdas for norm/unnorm
pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
post_process = lambda a: a * stats['action_std'] + stats['action_mean']

print(f"[POLICY] ACT++ policy loaded on {device}  ✓")

# ── connect robots ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "hand_teleop"))
import hand_teleop.config as teleop_config
from hand_teleop.robot_manager import RobotManager

print("\n[ROBOT] Connecting to arms...")
rm = RobotManager()
if not rm.robot_map:
    print("[ROBOT] WARNING: No robots found — running camera-only dry run.")

# ── camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  teleop_config.CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, teleop_config.CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

def preprocess_frame(bgr_frame):
    # ACT++ expects image (1, C, H, W) in range [0, 1] no imagenet norm inside inference script (it's inside __call__)
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, H, W)
    # the policy __call__ expects (1, 1, 3, H, W) because dim 1 is num_cams
    return t.unsqueeze(1)

# ── read current joint state from arms ───────────────────────────────────────
def read_arm_state():
    state = np.zeros(6, dtype=np.float32)
    for side, base in [("Left", 0), ("Right", 3)]:
        robot = rm.robot_map.get(side)
        if robot:
            try:
                obs = robot.get_observation()
                state[base + 0] = obs.get("shoulder_pan.pos", 0.0)
                state[base + 1] = obs.get("elbow_flex.pos",   0.0)
                state[base + 2] = obs.get("gripper.pos",      0.0)
            except Exception:
                pass
    return state

# ── send action to arms ────────────────────────────────────────────────────────
def send_action(action_np):
    # Clamp to safe hardware ranges before sending
    action_np[0] = np.clip(action_np[0], teleop_config.SAFE_PAN_MIN,  teleop_config.SAFE_PAN_MAX)
    action_np[1] = np.clip(action_np[1], teleop_config.SAFE_LIFT_MIN, teleop_config.SAFE_LIFT_MAX)
    action_np[3] = np.clip(action_np[3], teleop_config.SAFE_PAN_MIN,  teleop_config.SAFE_PAN_MAX)
    action_np[4] = np.clip(action_np[4], teleop_config.SAFE_LIFT_MIN, teleop_config.SAFE_LIFT_MAX)
    action_np[2] = np.clip(action_np[2], 0.0, 100.0)
    action_np[5] = np.clip(action_np[5], 0.0, 100.0)

    for side, base in [("Left", 0), ("Right", 3)]:
        robot = rm.robot_map.get(side)
        if robot:
            try:
                robot.send_action({
                    "shoulder_pan.pos": float(action_np[base + 0]),
                    "elbow_flex.pos":   float(action_np[base + 1]),
                    "gripper.pos":      float(action_np[base + 2]),
                })
            except Exception as e:
                print(f"  [WARN] {side} send failed: {e}")

# ── inference loop ────────────────────────────────────────────────────────────
print("\n[POLICY] Running!  SPACE = pause/resume  |  Q = quit")

paused = False
step   = 0

# For temporal aggregation / chunking
chunk_size = policy_config['num_queries']
query_frequency = chunk_size # default no temporal ensembling
t = 0
action_queue = []

with torch.inference_mode():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("[POLICY] PAUSED" if paused else "[POLICY] RESUMED")

        if paused:
            cv2.putText(display, "PAUSED — press SPACE to resume",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("ACT++ Policy Inference", display)
            continue

        raw_state = read_arm_state()
        
        # ACT++ evaluation
        if t % query_frequency == 0 or len(action_queue) == 0:
            qpos_np = pre_process(raw_state)
            qpos = torch.from_numpy(qpos_np).float().to(device).unsqueeze(0)
            curr_image = preprocess_frame(frame)
            
            # shape check: qpos (1, 6), curr_image (1, 1, 3, 480, 640)
            all_actions = policy(qpos, curr_image) # prediction chunk [1, chunk_size, 8]
            
            # Post process and store chunk
            raw_actions_np = all_actions.squeeze(0).cpu().numpy()
            action_queue = []
            for i in range(chunk_size):
                action_unnorm = post_process(raw_actions_np[i])
                target_qpos = action_unnorm[:6] # [0:6] since model was trained with action_dim 8 but only 6 is our qpos!
                # Wait, in imitation episodes we had state_dim=6, action_dim=8!
                # The raw_action has dim 8.
                action_queue.append(target_qpos.copy())

        # execution step
        action_np = action_queue.pop(0)
        
        # Smooth/clamping logic from previous
        send_action(action_np)
        
        t += 1
        step += 1

        # HUD overlay
        cv2.putText(display, f"[POLICY RUNNING]  step {step}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display,
                    f"L: pan={action_np[0]:.1f} lift={action_np[1]:.1f} grip={action_np[2]:.1f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(display,
                    f"R: pan={action_np[3]:.1f} lift={action_np[4]:.1f} grip={action_np[5]:.1f}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imshow("ACT++ Policy Inference", display)

# ── cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
rm.disconnect_all()
print("\n[POLICY] Session ended.")
