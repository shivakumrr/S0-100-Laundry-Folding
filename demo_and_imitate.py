#!/home/g1/miniconda3/envs/lerobot/bin/python
"""
demo_and_imitate.py  —  Record ONE demo, train ACT, then replay autonomously.

Usage:
    python demo_and_imitate.py

Phases:
  1. RECORD  — Teleop window opens.  Use your hands to operate the robot.
               Press  R  to start recording the episode.
               Press  S  to stop recording (saves dataset).
               Press  Q  to close the window and advance to training.
  2. TRAIN   — ACT policy trains automatically on the recorded dataset.
               Training prints progress; no interaction needed.
               (Ctrl+C to abort training early — inference will still attempt
               to load whatever checkpoint exists.)
  3. IMITATE — ACT policy runs on the robot continuously.
               Press  SPACE  to pause/resume.
               Press  Q  to quit.
"""

import sys
import os
import subprocess
import signal
import time
from pathlib import Path

# ── add hand_teleop to module search path ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "hand_teleop"))

# ==============================================================================
# PHASE 1 — RECORD ONE EPISODE
# ==============================================================================

def phase_record() -> Path:
    """
    Opens the teleop window.  Operator uses hands to drive the robot.
    R = start recording | S = stop & save | Q = quit recording window.
    Returns the path to the saved dataset directory.
    """
    import cv2
    import config
    from robot_manager import RobotManager
    from vision_tracker import HandTracker
    from arm_controller import ArmController
    from dataset_recorder import ImitationRecorder

    print("\n" + "=" * 60)
    print("  PHASE 1 — RECORD DEMO")
    print("  R = start recording  |  S = stop recording  |  Q = done")
    print("=" * 60 + "\n")

    rm = RobotManager()
    tracker  = HandTracker()
    controller = ArmController()
    recorder = ImitationRecorder()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    print("Teleop window is open. Perform your demo motion, then press Q.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image, current_hands = tracker.process_frame(frame)
        current_time = time.time()

        global_pause = False
        for side, landmarks in current_hands.items():
            tracker.draw_landmarks(image, landmarks)
            if tracker.is_fist(landmarks):
                global_pause = True

        if global_pause:
            cv2.putText(image, ">>> SYSTEM PAUSED (FIST) <<<",
                        (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        for side in ["Left", "Right"]:
            landmarks = current_hands.get(side)
            render_data = controller.process_arm(
                side, landmarks, current_time, rm.robot_map, global_pause,
                image.shape, gripper_bounds=rm.gripper_bounds[side]
            )
            line_color = (0, 165, 255) if side == "Left" else (255, 0, 255)
            y_pos = 50 if side == "Left" else 90
            if render_data["status"] == "engaged":
                cv2.line(image, render_data["thumb_pt"], render_data["index_pt"], line_color, 3)
                cv2.putText(image, f"{side.upper()} [ACTIVE]: {render_data['gripper']:.0f}%",
                            (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif render_data["status"] == "waiting":
                cv2.line(image, render_data["thumb_pt"], render_data["index_pt"], line_color, 3)
                cv2.putText(image, f"{side.upper()} [WAIT]: {render_data['countdown']:.1f}s",
                            (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(image, f"{side.upper()} [MISSING]",
                            (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if recorder.is_recording:
            recorder.capture_telemetry(frame, controller.emas)
            cv2.putText(image, f"[REC] FRAMES: {recorder.frames_recorded}",
                        (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            rec_hint = "RECORDING STOPPED — press Q to train" if recorder.episode_idx > 0 \
                       else "R=record  S=stop  Q=done"
            cv2.putText(image, rec_hint,
                        (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Phase 1 — Record Demo", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recorder.start_recording("demo_and_imitate")
        elif key == ord('s'):
            recorder.stop_recording()

    cap.release()
    cv2.destroyAllWindows()
    rm.disconnect_all()

    if recorder.episode_idx == 0:
        print("\n[WARN] No episode was saved!  Exiting.")
        recorder.shutdown()
        sys.exit(1)

    # Grab the dataset path that was created during this session
    dataset_path = Path(recorder.dataset.root)
    recorder.shutdown()

    print(f"\n[PHASE 1 COMPLETE] Dataset saved → {dataset_path}\n")
    return dataset_path


# ==============================================================================
# PHASE 2 — TRAIN ACT POLICY
# ==============================================================================

def phase_train(dataset_path: Path) -> Path:
    """
    Calls train_act.sh with the recorded dataset path.
    Streams training output live.  Returns the output directory.
    """
    print("=" * 60)
    print("  PHASE 2 — TRAINING ACT POLICY")
    print(f"  Dataset : {dataset_path}")
    print("  (Ctrl+C will abort training — inference will try the latest checkpoint)")
    print("=" * 60 + "\n")

    script = Path(__file__).parent / "train_act.sh"
    if not script.exists():
        print(f"[ERROR] train_act.sh not found at {script}")
        sys.exit(1)

    # Stream output live so operator can watch loss decrease
    proc = subprocess.Popen(
        ["bash", str(script), str(dataset_path)],
        cwd=str(Path(__file__).parent),
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid,   # own process group so Ctrl+C works cleanly
    )

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[TRAIN] Interrupted — attempting to use latest partial checkpoint...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()

    if proc.returncode not in (0, -signal.SIGTERM):
        print(f"\n[ERROR] Training exited with code {proc.returncode}")
        sys.exit(proc.returncode)

    # Find the output dir that was just created (most recent)
    outputs_root = Path(__file__).parent / "outputs"
    candidates = sorted(outputs_root.glob("act_dual_arm_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print("[ERROR] No output directory found after training!")
        sys.exit(1)

    output_dir = candidates[-1]
    print(f"\n[PHASE 2 COMPLETE] Training output → {output_dir}\n")
    return output_dir


# ==============================================================================
# PHASE 3 — IMITATE (ACT INFERENCE)
# ==============================================================================

def phase_imitate(output_dir: Path):
    """
    Finds the best checkpoint and runs ACT inference on the real robot.
    Mirrors run_policy.py logic, embedded here so it shares the same process.
    """
    print("=" * 60)
    print("  PHASE 3 — IMITATING DEMO  (ACT INFERENCE)")
    print("  SPACE = pause/resume  |  Q = quit")
    print("=" * 60 + "\n")

    import cv2
    import numpy as np
    import torch

    # ── find latest checkpoint ────────────────────────────────────────────────
    candidates = sorted(output_dir.glob("checkpoints/*/pretrained_model"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        print("[ERROR] No pretrained_model checkpoint found!")
        sys.exit(1)

    checkpoint_path = candidates[-1]
    print(f"[POLICY] Loading checkpoint: {checkpoint_path}")

    # ── load policy ───────────────────────────────────────────────────────────
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint_path))
    policy_cfg.pretrained_path = checkpoint_path
    policy = ACTPolicy.from_pretrained(pretrained_name_or_path=checkpoint_path,
                                        config=policy_cfg)
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint_path),
    )
    print(f"[POLICY] Loaded on {device}  ✓\n")

    # ── connect robots ────────────────────────────────────────────────────────
    import config
    from robot_manager import RobotManager

    rm = RobotManager()
    if not rm.robot_map:
        print("[WARN] No robots found — camera-only dry run.")

    # ── camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    def preprocess_frame(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t   = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
        return t

    def read_arm_state():
        state = np.zeros(6, dtype=np.float32)
        robot = rm.robot_map.get("Left")  # The follower arm is mapped to "Left" (ACM1)
        if robot:
            try:
                obs = robot.get_observation()
                state[0] = obs.get("shoulder_pan.pos", 0.0)
                state[1] = obs.get("shoulder_lift.pos", 0.0)
                state[2] = obs.get("elbow_flex.pos", 0.0)
                state[3] = obs.get("wrist_flex.pos", 0.0)
                state[4] = obs.get("wrist_roll.pos", 0.0)
                state[5] = obs.get("gripper.pos", 0.0)
            except Exception:
                pass
        return torch.from_numpy(state).unsqueeze(0).to(device)

    # ── EMA smoother ──────────────────────────────────────────────────────────
    ema_action = None
    EMA_ALPHA  = 0.05
    MAX_SPEED  = 5.0

    def smooth_and_clamp(raw_np):
        nonlocal ema_action
        if ema_action is None:
            ema_action = raw_np.copy()
        else:
            ema_action = EMA_ALPHA * raw_np + (1.0 - EMA_ALPHA) * ema_action
        return ema_action.copy()

    def send_action(action_np):
        action_np[0] = np.clip(action_np[0], config.SAFE_PAN_MIN,  config.SAFE_PAN_MAX)
        action_np[1] = np.clip(action_np[1], config.SAFE_LIFT_MIN, config.SAFE_LIFT_MAX)
        action_np[5] = np.clip(action_np[5], 0.0, 100.0)
        
        robot = rm.robot_map.get("Left")
        if robot:
            try:
                robot.send_action({
                    "shoulder_pan.pos":  float(action_np[0]),
                    "shoulder_lift.pos": float(action_np[1]),
                    "elbow_flex.pos":    float(action_np[2]),
                    "wrist_flex.pos":    float(action_np[3]),
                    "wrist_roll.pos":    float(action_np[4]),
                    "gripper.pos":       float(action_np[5]),
                })
            except Exception as e:
                print(f"  [WARN] Follower send failed: {e}")

    # ── inference loop ────────────────────────────────────────────────────────
    paused = False
    step   = 0

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
                cv2.putText(display, "PAUSED — SPACE to resume",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Phase 3 — Imitating Demo", display)
                continue

            obs = {
                "observation.images.camera_2": preprocess_frame(frame),
                "observation.state":         read_arm_state(),
            }
            obs_proc   = preprocessor(obs)
            raw_action = policy.select_action(obs_proc)

            if isinstance(raw_action, dict):
                action_t = raw_action.get("action", next(iter(raw_action.values())))
            else:
                action_t = raw_action

            raw_np    = action_t.squeeze(0).cpu().float().numpy()
            action_np = smooth_and_clamp(raw_np)
            send_action(action_np)
            step += 1

            cv2.putText(display, f"[IMITATING]  step {step}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display,
                        f"Pan:{action_np[0]:.0f} Lift:{action_np[1]:.0f} Elbow:{action_np[2]:.0f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(display,
                        f"W_Flex:{action_np[3]:.0f} W_Roll:{action_np[4]:.0f} Grip:{action_np[5]:.0f}",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(display, "SPACE=pause  Q=quit",
                        (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("Phase 3 — Imitating Demo", display)

    cap.release()
    cv2.destroyAllWindows()
    rm.disconnect_all()
    print("\n[PHASE 3 COMPLETE] Imitation session ended.")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record → Train → Imitate (no prompts)")
    parser.add_argument("--skip-record",   metavar="DATASET_PATH", default=None,
                        help="Skip recording; use an existing dataset path.")
    parser.add_argument("--skip-train",    metavar="OUTPUT_DIR",   default=None,
                        help="Skip training; use an existing output directory.")
    parser.add_argument("--steps",         type=int, default=None,
                        help="Override training steps (default: from train_act.sh, 50000).")
    args = parser.parse_args()

    # ── optionally patch training steps into env so train_act.sh picks it up ─
    if args.steps is not None:
        os.environ["TRAIN_STEPS"] = str(args.steps)
        print(f"[CONFIG] Training steps overridden to {args.steps}")

    # Phase 1
    if args.skip_record:
        dataset_path = Path(args.skip_record)
        print(f"[SKIP RECORD] Using existing dataset: {dataset_path}")
    else:
        dataset_path = phase_record()

    # Phase 2
    if args.skip_train:
        output_dir = Path(args.skip_train)
        print(f"[SKIP TRAIN] Using existing output: {output_dir}")
    else:
        output_dir = phase_train(dataset_path)

    # Phase 3
    phase_imitate(output_dir)
