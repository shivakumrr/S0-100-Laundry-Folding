#!/usr/bin/env python3
"""
eval_model.py

Continuously monitors a camera feed for a blue/purple block. When detected,
launches lerobot-record with num_episodes=1000 so the policy loads ONCE and
stays in memory for the entire session.

Detection triggers the launch once. After that, lerobot handles all episodes
internally — our script just monitors that the process is still alive.

Logs are stored in the ./logs/ folder next to this script.

Usage:
    python eval_model.py                  # rerun viewer ON by default
    python eval_model.py --no-rerun       # disable rerun viewer
    python eval_model.py --visualize      # show HSV detection window
    python eval_model.py --tune           # interactive HSV tuner
    python eval_model.py --max-episodes 20
"""

import os
import re
import signal
import subprocess
import time
import sys
import cv2
import numpy as np
import argparse
import logging
from datetime import datetime
import threading

import app as flask_app_module

def start_flask():
    flask_app_module.app.config['TEMPLATES_AUTO_RELOAD'] = True
    flask_app_module.app.jinja_env.auto_reload = True
    try:
        flask_app_module.app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to start web server wrapper! {e}\nIs Port 8000 already in use?", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)

threading.Thread(target=start_flask, daemon=True).start()

# ─── Logs folder ──────────────────────────────────────────────────────────────
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ─── Logging setup ────────────────────────────────────────────────────────────
_timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_filename = os.path.join(LOGS_DIR, f"eval_model_{_timestamp_str}.log")
_csv_filename = os.path.join(LOGS_DIR, f"eval_model_{_timestamp_str}.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_filename),
    ],
)
log = logging.getLogger(__name__)

# Initialize CSV log with header
with open(_csv_filename, "w") as f:
    f.write("timestamp,episode_index,policy,task,trigger_area_px\n")

# ─── Configuration ────────────────────────────────────────────────────────────

PREVIEW_CAMERA_URL = "http://192.168.1.68:8080/video"   # IP Camera URL for web preview

# HSV range — run --tune to adjust for your lighting
PURPLE_HSV_LOWER = np.array([110, 168, 0])
PURPLE_HSV_UPPER = np.array([179, 255, 255])

MIN_CONTOUR_AREA = 3580    # px²  — run --tune to recalibrate
POLL_INTERVAL_SEC           = 0.5    # detection check interval
POST_EPISODE_COOLDOWN_SEC   = 3.0   # wait after episode before re-detecting
# After a real episode finishes, suppress block detection for this many seconds.
# This prevents the block sitting in the pot from immediately re-triggering
# the next episode before the user has a chance to physically reset.
BLOCK_DETECTION_SUPPRESS_SEC = 10.0

# HSV values are tuned for the DETECTION camera (video6, overhead view).
# Run:  python eval_model.py --tune   to re-tune interactively.
# NOTE: --tune now opens video6 (same camera used for detection).
MAX_TOTAL_EPISODES      = None   # None = run forever
EPISODE_TIME_S = 120   # seconds per episode

# How many episodes lerobot manages per process (keeps policy in memory)
EPISODES_PER_LAUNCH     = 1

# Seconds to wait for lerobot to finish loading before resuming detection
# (covers model download + GPU load time — adjust if your machine is slower)
LEROBOT_STARTUP_WAIT_SEC = 50.0

# ─── Policy config ────────────────────────────────────────────────────────────

ACT_POLICY_PATH      = "team-11/my_policy"
SMOLVLA_POLICY_PATH  = "team-11/smolvla_pickplace"
ACT_DATASET_REPO     = "team-11/eval_record-test"
SMOLVLA_DATASET_REPO = "team-11/eval_smolvla_pickplace"
DEFAULT_TASK         = "Pick up the purple cube and place in the pot"

ROBOT_ARGS = [
    "--robot.type=so100_follower",
    "--robot.port=/dev/ttyACM1",
    "--robot.id=follower_arm",
    "--robot.cameras={ cam_0: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, "
                     "cam_3: {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}}",
]

_lerobot_proc = None
LEROBOT_CMD   = []


# ─── Graceful shutdown ────────────────────────────────────────────────────────

def _shutdown(signum=None, frame=None):
    global _lerobot_proc
    log.info("\nShutting down gracefully …")
    if _lerobot_proc is not None and _lerobot_proc.poll() is None:
        log.info("Sending SIGINT to lerobot-record so it can release motors …")
        # SIGINT triggers lerobot's own cleanup: robot.disconnect() → motors go limp
        _lerobot_proc.send_signal(signal.SIGINT)
        try:
            # Give lerobot up to 30s to save dataset and release motors
            _lerobot_proc.wait(timeout=30)
            log.info("✔ lerobot-record exited cleanly — motors released.")
        except subprocess.TimeoutExpired:
            log.warning("lerobot-record did not exit in 30s — sending SIGTERM …")
            _lerobot_proc.terminate()
            try:
                _lerobot_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("Force-killing lerobot-record.")
                _lerobot_proc.kill()
    os.kill(os.getpid(), signal.SIGKILL)


# ─── Policy selection prompt ──────────────────────────────────────────────────

def prompt_policy_mode(policy_mode="smolvla"):
    global LEROBOT_CMD

    if policy_mode == "act":
        policy_path  = ACT_POLICY_PATH
        dataset_repo = ACT_DATASET_REPO
        task         = DEFAULT_TASK
        print(f"\n✔ ACT selected | task: {task}")
    else:
        policy_path  = SMOLVLA_POLICY_PATH
        dataset_repo = SMOLVLA_DATASET_REPO
        task         = DEFAULT_TASK
        print(f"\n✔ SmolVLA selected | default task: {task}")

    print("═" * 50 + "\n")

    LEROBOT_CMD = [
        "lerobot-record",
        *ROBOT_ARGS,
        f"--policy.path={policy_path}",
        f"--dataset.repo_id={dataset_repo}",
        f"--dataset.num_episodes={EPISODES_PER_LAUNCH}",
        f"--dataset.single_task={task}",
        "--dataset.streaming_encoding=true",
        "--dataset.encoder_threads=2",
        "--dataset.push_to_hub=false",
        f"--dataset.episode_time_s={EPISODE_TIME_S}",
        "--dataset.reset_time_s=5",
        "--play_sounds=false",
    ]

    return task


# ─── HSV Tuner ────────────────────────────────────────────────────────────────

def run_hsv_tuner(detection_cam_index: int):
    """Interactive HSV tuner — shows both cameras, press Q to save values into this script."""
    other_index = DETECTION_CAMERA_INDEX_PHASE2 if detection_cam_index == DETECTION_CAMERA_INDEX else DETECTION_CAMERA_INDEX

    cap_detect = cv2.VideoCapture(detection_cam_index)
    cap_other  = cv2.VideoCapture(other_index)

    if not cap_detect.isOpened():
        log.critical(f"Cannot open detection camera {detection_cam_index}")
        sys.exit(1)

    print(f"\n[TUNER] Detection cam: /dev/video{detection_cam_index}  |  Other cam: /dev/video{other_index}")
    print("[TUNER] Bottom row = Detection cam view + its mask (used for block detection)")
    print("[TUNER] Top row    = Other cam view (display only)")
    print("[TUNER] Adjust trackbars until ONLY the block is white in the Mask panel.")
    print("[TUNER] Press Q or Escape when done.\n")

    win = "HSV Tuner (press Q to save and quit)"
    cv2.namedWindow(win)

    h_lo, s_lo, v_lo = PURPLE_HSV_LOWER
    h_hi, s_hi, v_hi = PURPLE_HSV_UPPER

    cv2.createTrackbar("H low",        win, int(h_lo), 179, lambda x: None)
    cv2.createTrackbar("H high",       win, int(h_hi), 179, lambda x: None)
    cv2.createTrackbar("S low",        win, int(s_lo), 255, lambda x: None)
    cv2.createTrackbar("S high",       win, int(s_hi), 255, lambda x: None)
    cv2.createTrackbar("V low",        win, int(v_lo), 255, lambda x: None)
    cv2.createTrackbar("V high",       win, int(v_hi), 255, lambda x: None)
    cv2.createTrackbar("Min area /10", win, MIN_CONTOUR_AREA // 10, 5000, lambda x: None)

    W, H = 640, 360  # display size per panel

    while True:
        ret_d, frame_d = cap_detect.read()
        ret_o, frame_o = cap_other.read()

        if not ret_d:
            continue

        hl = cv2.getTrackbarPos("H low",  win)
        hh = cv2.getTrackbarPos("H high", win)
        sl = cv2.getTrackbarPos("S low",  win)
        sh = cv2.getTrackbarPos("S high", win)
        vl = cv2.getTrackbarPos("V low",  win)
        vh = cv2.getTrackbarPos("V high", win)
        min_area = cv2.getTrackbarPos("Min area /10", win) * 10

        lower = np.array([hl, sl, vl])
        upper = np.array([hh, sh, vh])

        # --- Detection cam analysis ---
        hsv   = cv2.cvtColor(frame_d, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = max((cv2.contourArea(c) for c in contours), default=0)
        detected = largest_area >= min_area
        color = (0, 255, 0) if detected else (0, 0, 255)
        label = (f"DETECTION CAM  DETECTED ({largest_area:.0f}px²)"
                 if detected else f"DETECTION CAM  No match ({largest_area:.0f}px²)")

        panel_detect = cv2.resize(frame_d.copy(), (W, H))
        if contours:
            # Scale contours to panel size
            sx = W / frame_d.shape[1]
            sy = H / frame_d.shape[0]
            scaled = [np.multiply(c, [sx, sy]).astype(np.int32) for c in contours]
            cv2.drawContours(panel_detect, [max(scaled, key=cv2.contourArea)], -1, color, 2)
        cv2.putText(panel_detect, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        mask_rgb  = cv2.cvtColor(cv2.resize(mask, (W, H)), cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_rgb, "MASK (detection cam)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        # --- Other cam panel ---
        if ret_o and frame_o is not None:
            panel_other = cv2.resize(frame_o, (W, H))
        else:
            panel_other = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.putText(panel_other, f"OTHER CAM (video{other_index})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)

        # Compose: top row = [other cam | blank], bottom row = [detect cam | mask]
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        top    = np.concatenate((panel_other, blank),    axis=1)
        bottom = np.concatenate((panel_detect, mask_rgb), axis=1)
        composite = np.concatenate((top, bottom), axis=0)
        cv2.imshow(win, composite)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("\n──── Saving tuned values into eval_model.py ────")
            script_path = os.path.abspath(__file__)
            with open(script_path, "r") as f:
                source = f.read()
            source = re.sub(r"PURPLE_HSV_LOWER\s*=\s*np\.array\(\[.*?\]\)",
                            f"PURPLE_HSV_LOWER = np.array([110, 168, 0])", source)
            source = re.sub(r"PURPLE_HSV_UPPER\s*=\s*np\.array\(\[.*?\]\)",
                            f"PURPLE_HSV_UPPER = np.array([179, 255, 255])", source)
            source = re.sub(r"MIN_CONTOUR_AREA\s*=\s*\d+",
                            f"MIN_CONTOUR_AREA = {min_area}", source)
            with open(script_path, "w") as f:
                f.write(source)
            print(f"PURPLE_HSV_LOWER = np.array([110, 168, 0])")
            print(f"PURPLE_HSV_UPPER = np.array([179, 255, 255])")
            print(f"MIN_CONTOUR_AREA = {min_area}")
            print(f"✔ Saved to {script_path}")
            print("────────────────────────────────────────\n")
            break

    cap_detect.release()
    cap_other.release()
    cv2.destroyAllWindows()




# ─── Block Detection ──────────────────────────────────────────────────────────

class AsyncCameraGrabber:
    """Reads camera frames in a background thread.
    Retries opening the camera every few seconds if it is initially busy."""
    def __init__(self, index):
        self.index   = index
        self.cap     = None
        self.ret     = False
        self.frame   = None
        self.running = True
        self.thread  = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _try_open(self):
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            self.cap = cap
            log.info(f"[Cam{self.index}] Opened successfully.")
            return True
        cap.release()
        return False

    def _update(self):
        while self.running:
            # Try to open if not yet open
            if self.cap is None or not self.cap.isOpened():
                if not self._try_open():
                    log.debug(f"[Cam{self.index}] Busy — retrying in 3s…")
                    time.sleep(3.0)
                    continue
            try:
                ret, frame = self.cap.read()
                self.ret = ret
                if ret:
                    self.frame = frame
                elif not self.cap.isOpened():
                    # Camera disconnected — reset and retry
                    self.cap = None
                    self.ret = False
            except Exception as e:
                log.debug(f"[Cam{self.index}] read error: {e}")
                self.ret = False
                self.cap = None
            time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join(timeout=2.0)
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass

def update_preview_frame(f1):
    if f1 is None:
        f1 = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(f1, "IP Camera Offline/In Use", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    f1 = cv2.resize(f1, (640, 480))
    try:
        flask_app_module.latest_frame = f1
    except Exception:
        pass

def _update_from_lerobot_tmp_or_blank():
    """Read lerobot's written camera frames from /tmp and push to web UI.
    Falls back to a 'Loading…' placeholder when files aren't ready yet."""
    lf0 = cv2.imread("/tmp/lerobot_cam_0.jpg") if os.path.exists("/tmp/lerobot_cam_0.jpg") else None
    lf1 = cv2.imread("/tmp/lerobot_cam_1.jpg") if os.path.exists("/tmp/lerobot_cam_1.jpg") else None
    update_dual_frame(lf0, lf1)

def detect_block(frame, visualize: bool = False):
    if frame is None:
        log.warning("No frame provided for detection.")
        return False, 0.0

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PURPLE_HSV_LOWER, PURPLE_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = max((cv2.contourArea(c) for c in contours), default=0.0)
    detected = largest_area >= MIN_CONTOUR_AREA

    if detected:
        log.info(f"[detect] area={largest_area:.0f}px² threshold={MIN_CONTOUR_AREA} → ✔ DETECTED")

    if visualize:
        display = frame.copy()
        color = (0, 255, 0) if detected else (0, 0, 255)
        label = (f"DETECTED ({largest_area:.0f}px²)" if detected
                 else f"Not detected ({largest_area:.0f}px²)")
        if detected and contours:
            cv2.drawContours(display, [max(contours, key=cv2.contourArea)], -1, color, 2)
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Object Detection", display)
        cv2.waitKey(1)

    return detected, largest_area


# ─── lerobot launcher ──────────────────────────────────────────────────────

def launch_lerobot(display_data: bool = False) -> subprocess.Popen:
    global _lerobot_proc

    import shutil, pathlib
    repo_id = next((a.split("=", 1)[1] for a in LEROBOT_CMD
                    if a.startswith("--dataset.repo_id=")), None)
    if repo_id:
        cache_path = pathlib.Path.home() / ".cache/huggingface/lerobot" / repo_id
        if cache_path.exists():
            shutil.rmtree(cache_path)
            log.info(f"Cleared stale cache: {cache_path}")

    cmd = list(LEROBOT_CMD)
    # Always disable rerun viewer — it opens a window, hogs bandwidth, and
    # crashes with SIGABRT when the process is interrupted via Ctrl+C.
    cmd.append("--display_data=false")

    trigger_file = "/tmp/lerobot_trigger"
    os.environ["LEROBOT_WAIT_TRIGGER"] = trigger_file

    log.info("Launching lerobot-record …")
    _lerobot_proc = subprocess.Popen(cmd, env=os.environ.copy())
    return _lerobot_proc


# ─── Block-detection daemon thread ───────────────────────────────────────────

def _block_detection_worker(cap_c1_ref: list):
    """
    Daemon thread:
    - Reads the IP camera for the web UI display.
    """
    block_first_seen = None
    post_episode_suppress_until = 0.0
    was_in_episode = False

    log.info("[PreviewThread] started.")
    while True:
        try:
            cap1 = cap_c1_ref[0]
            if cap1 is not None:
                # Camera is ours — read directly
                _, lf0 = cap1.read()
            else:
                # Lerobot owns the cameras — read from its written frames just in case
                # Look for cam_ip.jpg first, fallback to cam_0.jpg
                if os.path.exists("/tmp/lerobot_cam_2.jpg"):
                    lf0 = cv2.imread("/tmp/lerobot_cam_2.jpg")
                elif os.path.exists("/tmp/lerobot_cam_0.jpg"):
                    lf0 = cv2.imread("/tmp/lerobot_cam_0.jpg")
                else:
                    lf0 = None

            update_preview_frame(lf0)

            # Detection uses cam0 frame only
            detect_frame = lf0

            # Track episode end → apply post-episode suppression
            in_episode = os.path.exists("/tmp/lerobot_in_episode")
            if was_in_episode and not in_episode:
                post_episode_suppress_until = time.time() + BLOCK_DETECTION_SUPPRESS_SEC
                block_first_seen = None
                log.info(f"[BlockThread] Episode ended — suppressing for {BLOCK_DETECTION_SUPPRESS_SEC:.0f}s.")
            was_in_episode = in_episode

            if detect_frame is None:
                time.sleep(0.05)
                continue

            detected, _ = detect_block(detect_frame)
            now = time.time()

            if detected:
                if block_first_seen is None:
                    block_first_seen = now
                elif (now - block_first_seen) >= 1.0 and now >= post_episode_suppress_until:
                    if not os.path.exists("/tmp/lerobot_block_ready"):
                        log.info("[BlockThread] Block confirmed for 1s — writing /tmp/lerobot_block_ready.")
                        with open("/tmp/lerobot_block_ready", "w") as f:
                            f.write("1")
            else:
                block_first_seen = None

        except Exception as e:
            log.debug(f"[BlockThread] error: {e}")
        time.sleep(0.05)



def main(visualize: bool = False, display_data: bool = False, policy_mode: str = "act"):
    global _lerobot_proc

    prompt_policy_mode(policy_mode)

    log.info("Starting IP camera connection for web preview…")
    cap_c1 = AsyncCameraGrabber(PREVIEW_CAMERA_URL)
    cap_c1_ref = [cap_c1]
    threading.Thread(target=_block_detection_worker, args=(cap_c1_ref,), daemon=True).start()

    log.info("Ready — submit a task from the web UI to begin.\n")

    while True:
        log.info("Standby phase… Waiting for trigger.")
        # Clear stale tmp files
        for _f in ["/tmp/lerobot_cam_0.jpg", "/tmp/lerobot_cam_1.jpg", "/tmp/lerobot_cam_2.jpg",
                   "/tmp/lerobot_block_ready", "/tmp/lerobot_in_episode",
                   "/tmp/lerobot_abort", "/tmp/lerobot_running"]:
            if os.path.exists(_f): os.remove(_f)

        # 1. Wait for UI trigger
        while not os.path.exists("/tmp/lerobot_trigger"):
            time.sleep(0.1)

        # 2. Trigger received -> launching lerobot
        log.info("UI trigger received — launching lerobot…")
        # Ensure we don't release cap_c1 so the IP preview stays active on the website during the episode

        # 3. Launch single-episode process
        proc = launch_lerobot(display_data=display_data)
        
        # 4. Wait for episode to complete and process to exit
        try:
            log.info("Policy running… waiting for completion.")
            proc.wait()  # Wait infinitely for the run to naturally complete
        except KeyboardInterrupt:
            _shutdown()
        except Exception as e:
            log.warning(f"Error while waiting for lerobot: {e}")
        
        # Clear the global reference so Ctrl+C doesn't try to shut down a dead process and hang for 40s
        global _lerobot_proc
        _lerobot_proc = None
        
        log.info("✔ lerobot-record episode completed and exited naturally.")
        
        # 5. Reset UI Trigger to return to 'Idle' state
        if os.path.exists("/tmp/lerobot_trigger"):
            os.remove("/tmp/lerobot_trigger")
            
        log.info("Returning to standby mode…\n")
        time.sleep(1.0)






# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy robot policy triggered by block detection. "
                    "Policy loads once — no reload between episodes."
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Open interactive HSV tuner (auto-saves to script).",
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Show live HSV detection preview window.",
    )
    parser.add_argument(
        "--no-rerun", action="store_true",
        help="Disable the rerun viewer window (default: rerun is ON).",
    )
    parser.add_argument(
        "--max-episodes", "-n", type=int, default=None,
        help="Stop after N block detections (default: run forever).",
    )
    parser.add_argument(
        "--policy_mode", "-p", type=str, choices=["act", "smolvla"], default="act",
        help="Which policy to run (act or smolvla). Default: act."
    )
    args = parser.parse_args()

    if args.tune:
        # Tune on the same camera used for detection (video0, robot arm cam)
        print(f"[Tuner] Opening detection camera /dev/video{DETECTION_CAMERA_INDEX} (video0)")
        print("[Tuner] Tune HSV so ONLY the purple block is white in the Mask window.")
        print("[Tuner] Tip: raise S_low to 60+ to reject low-saturation backgrounds.\n")
        run_hsv_tuner(DETECTION_CAMERA_INDEX)
        sys.exit(0)

    if args.max_episodes is not None:
        MAX_TOTAL_EPISODES = args.max_episodes

    main(
        visualize=args.visualize,
        display_data=not args.no_rerun,
        policy_mode=args.policy_mode
    )