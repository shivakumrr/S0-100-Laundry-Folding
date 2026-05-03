import time
import math
import config
import numpy as np

class ArmController:
    def __init__(self):
        """
        Encapsulates mathematical kinematics, exponential smoothing pipelines,
        engagement tracking, positional delta latches, and coordinate outputs.
        """
        # 0. Set up IK Solver
        self.kin_solver = None
        try:
            from lerobot.model.kinematics import RobotKinematics
            import os
            urdf_path = os.path.join(os.path.dirname(__file__), "models", "so101_no_mesh.urdf")
            self.kin_solver = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_frame_link",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            )
        except Exception as e:
            print(f"Failed to load IK Solver: {e}")

        # 1. Physical Connectivity State
        self.hand_states = {
            "Left":  { "engaged": False, "detected_since": None, "lost_since": None },
            "Right": { "engaged": False, "detected_since": None, "lost_since": None }
        }
        
        # 2. Physics Relative Latches
        self.latches = {
            "Left":  { "wrist_x": None, "wrist_y": None, "wrist_z": None, "ee_pose": None, "start_joints": None },
            "Right": { "wrist_x": None, "wrist_y": None, "wrist_z": None, "ee_pose": None, "start_joints": None }
        }
        
        # 3. Mathematical EMA Digital Low-Pass Trackers
        self.emas = {
            "Left":  { "gripper": None, "gui_gripper": None, "joints": None },
            "Right": { "gripper": None, "gui_gripper": None, "joints": None }
        }

    def compute_gripper_percentage(self, hand_landmarks, img_shape):
        """
        Predicts exactly how 'open' the user's hand is based on raw pixel coordinates.
        Clamps explicitly against configurably tuned constraints natively.
        """
        h, w, c = img_shape
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Map pixel coordinates onto Cartesian screen size matrix
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        
        distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
        
        clamped_dist = max(config.MIN_DIST, min(distance, config.MAX_DIST))
        
        percent = ((clamped_dist - config.MIN_DIST) / (config.MAX_DIST - config.MIN_DIST)) * 100.0
        
        return percent, (thumb_x, thumb_y), (index_x, index_y)
        
    def process_arm(self, side, landmarks, pos_3d, current_time, robot_map, global_pause, img_shape, gripper_bounds):
        """
        Renders mathematical positional offsets and directly controls FEETECH hardware.
        Called once mathematically per inference frame intrinsically per active arm.
        gripper_bounds: dict with keys 'closed' and 'open' from calibration JSON.
        """
        state = self.hand_states[side]
        
        # =============================================================
        # 1. State Machine: Manage Disconnection/Countdown Engagement
        # =============================================================
        if landmarks:
            state["lost_since"] = None  # Instantly reset the debounce dropout timer
            if global_pause:
                if state["engaged"]:
                    state["engaged"] = False
                    self.emas[side] = { "gripper": None, "gui_gripper": None, "joints": None }
                    print(f"\n>>> SYSTEM CLUTCH! {side.upper()} tracking safely suspended. <<<")
                
                # Prevent timer from instantly rolling over when fist releases
                state["detected_since"] = None
            else:
                if not state["engaged"]:
                    if state["detected_since"] is None:
                        state["detected_since"] = current_time
                    elif (current_time - state["detected_since"]) >= config.ENGAGEMENT_DELAY_SEC:
                        
                        # >> SWITCH TO ACTIVE <<
                        state["engaged"] = True
                        print(f"\n>>> {side.upper()} Arm System ENGAGED! Latching coordinates... <<<")
                        
                        wrist = landmarks.landmark[0]
                        if pos_3d is not None:
                            self.latches[side]["wrist_x"] = pos_3d[0]
                            self.latches[side]["wrist_y"] = pos_3d[1]
                            self.latches[side]["wrist_z"] = pos_3d[2]
                            self.latches[side]["wrist_pitch"] = pos_3d[3]
                            self.latches[side]["stereo"] = True
                        else:
                            self.latches[side]["wrist_x"] = max(0.0, min(1.0, wrist.x))
                            self.latches[side]["wrist_y"] = max(0.0, min(1.0, wrist.y))
                            self.latches[side]["wrist_z"] = wrist.z # Use depth inference mapping
                            self.latches[side]["wrist_pitch"] = 0.0
                            self.latches[side]["stereo"] = False
                        
                        target_robot = robot_map.get(side)
                        if target_robot and self.kin_solver:
                            try:
                                obs = target_robot.get_observation()
                                current_joints = []
                                for j in ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"]:
                                    current_joints.append(obs.get(j, 0.0))
                                
                                ee_pose = self.kin_solver.forward_kinematics(np.array(current_joints))
                                self.latches[side]["ee_pose"] = ee_pose
                                self.latches[side]["start_joints"] = np.array(current_joints)
                            except Exception:
                                pass
        else:
            if state["engaged"]:
                # Initialize debounce timer so we don't drop tracking for a 1-frame AI hiccup
                if state["lost_since"] is None:
                    state["lost_since"] = current_time
                elif (current_time - state["lost_since"]) > 0.6:  # 600ms grace period timeout
                    # >> SWITCH TO INACTIVE / ASLEEP <<
                    state["engaged"] = False
                    print(f"\n>>> {side.upper()} Tracking Signal LOST (>0.6s)! Safely locking joints. <<<")
                    
                    # Delete historically cached digital signals to prevent massive spikes
                    self.emas[side] = { "gripper": None, "gui_gripper": None, "joints": None }
                    state["detected_since"] = None
            else:
                state["detected_since"] = None

        # =============================================================
        # 2. Extract Data Coordinates (Rendering loop independently)
        # =============================================================
        if landmarks:
            raw_gripper_percent, thumb_pt, index_pt = self.compute_gripper_percentage(landmarks, img_shape)
            
            # --- MAP MATHEMATICAL PERCENT TO CALIBRATED HARDWARE BOUNDS ---
            # These come directly from the arm's calibration JSON (no more guessing!)
            closed_min = gripper_bounds["closed"]
            open_max   = gripper_bounds["open"]
            mapped_gripper_percent = closed_min + (raw_gripper_percent / 100.0) * (open_max - closed_min)
            
            wrist = landmarks.landmark[0]
            wrist_x = max(0.0, min(1.0, wrist.x))
            wrist_y = max(0.0, min(1.0, wrist.y))
            
            # --- COMPUTE HARDWARE PHYSICAL COMMANDS ---
            if state["engaged"]:
                latch = self.latches[side]
                
                target_joints = [0.0]*5
                if self.kin_solver and latch.get("ee_pose") is not None:
                    target_ee_pose = latch["ee_pose"].copy()
                    
                    if pos_3d is not None:
                        delta_x = pos_3d[0] - latch["wrist_x"] # X (Left/Right)
                        delta_y = pos_3d[1] - latch["wrist_y"] # Y (Up/Down)
                        delta_size = pos_3d[2] - latch["wrist_z"] # Palm Size!
                        delta_pitch = pos_3d[3] - latch["wrist_pitch"] # Pitch Angle!
                        
                        fwd_back_m = delta_size * config.IK_Z_SENSITIVITY
                        left_right_m = -(delta_x * config.IK_X_SENSITIVITY)
                        up_down_m = -(delta_y * config.IK_Y_SENSITIVITY)
                        
                        # Apply a harsh pre-IK Cartesian Low-Pass Filter to eradicate AI snapping
                        # We use the config EMA alpha (e.g. 0.09) so it's globally synced
                        ema = self.emas[side]
                        if ema.get("cartesian_delta") is None:
                            ema["cartesian_delta"] = [fwd_back_m, left_right_m, up_down_m]
                        else:
                            alpha = config.EMA_ALPHA * 0.5 # Double smooth the Cartesian input!
                            ema["cartesian_delta"][0] = (fwd_back_m * alpha) + (ema["cartesian_delta"][0] * (1.0 - alpha))
                            ema["cartesian_delta"][1] = (left_right_m * alpha) + (ema["cartesian_delta"][1] * (1.0 - alpha))
                            ema["cartesian_delta"][2] = (up_down_m * alpha) + (ema["cartesian_delta"][2] * (1.0 - alpha))
                        
                        stable_fwd = ema["cartesian_delta"][0]
                        stable_lr = ema["cartesian_delta"][1]
                        stable_ud = ema["cartesian_delta"][2]

                        # Print the real-time target displacement to terminal for debugging
                        # \r overwrites the line so it doesn't hopelessly spam the terminal buffer
                        import sys
                        sys.stdout.write(f"\r[{side.upper()}] FWD/BACK: {stable_fwd:+.3f}m | LEFT/RIGHT: {stable_lr:+.3f}m | UP/DOWN: {stable_ud:+.3f}m   ")
                        sys.stdout.flush()
                        
                        # Distance between hand and camera reduces -> palm size increases (delta_size > 0).
                        # Robot end effector must go front (+X direction in its coordinate space).
                        target_ee_pose[0, 3] += stable_fwd # Depth (Forward/Back)
                        target_ee_pose[1, 3] += stable_lr  # Left-Right
                        target_ee_pose[2, 3] += stable_ud  # Up-Down
                        current_deltas = np.array([stable_fwd, stable_lr, stable_ud])
                    else:
                        delta_x = wrist_x - latch["wrist_x"] if latch["wrist_x"] is not None else 0.0
                        delta_y = wrist_y - latch["wrist_y"] if latch["wrist_y"] is not None else 0.0
                        delta_z = wrist.z - latch["wrist_z"] if latch.get("wrist_z") is not None else 0.0
                        
                        target_ee_pose[0, 3] -= delta_z * config.IK_Z_SENSITIVITY    # Depth
                        target_ee_pose[1, 3] -= delta_x * config.IK_X_SENSITIVITY    # Left-Right
                        target_ee_pose[2, 3] -= delta_y * config.IK_Y_SENSITIVITY    # Up-Down
                        current_deltas = np.array([-delta_z * config.IK_Z_SENSITIVITY, -delta_x * config.IK_X_SENSITIVITY, -delta_y * config.IK_Y_SENSITIVITY])
                    
                    # --- CARTESIAN DEADBAND ---
                    # Only compute new IK targets if the user moved purposefully (>8mm)
                    ema = self.emas[side]
                    if ema.get("last_ik_deltas") is None:
                        ema["last_ik_deltas"] = current_deltas
                        target_joints = self.kin_solver.inverse_kinematics(
                            current_joint_pos=latch["start_joints"],
                            desired_ee_pose=target_ee_pose,
                            position_weight=1.0,
                            orientation_weight=0.01
                        )
                        ema["last_ik_joints"] = target_joints
                    else:
                        dist = np.linalg.norm(current_deltas - ema["last_ik_deltas"])
                        deadband_meters = config.IK_DEADZONE_CM / 100.0
                        if dist > deadband_meters:  # User-configured physical deadzone
                            target_joints = self.kin_solver.inverse_kinematics(
                                current_joint_pos=latch["start_joints"],
                                desired_ee_pose=target_ee_pose,
                                position_weight=1.0,
                                orientation_weight=0.01
                            )
                            ema["last_ik_deltas"] = current_deltas
                            ema["last_ik_joints"] = target_joints
                        else:
                            # User is holding still (or making micro-adjustments inside the deadzone)!
                            # Lock the mathematical target. The exact exact position will be held.
                            target_joints = list(ema["last_ik_joints"])
                            
                    # --- APPLY WRIST PITCH OVERRIDE ---
                    if pos_3d is not None:
                        # Append manual wrist flick on top of the Inverse Kinematics native bend compensation.
                        # Target joint [3] correlates directly to the SO100 Wrist Flex servo!
                        target_joints[3] += (delta_pitch * 1.5)
                
                # --- APPLY DIGITAL LOW PASS FILTER ---
                ema = self.emas[side]
                if ema["gripper"] is None:
                    ema["gripper"] = mapped_gripper_percent
                    ema["gui_gripper"] = raw_gripper_percent
                    ema["joints"] = list(target_joints)
                else:
                    ema["gripper"] = (mapped_gripper_percent * config.EMA_ALPHA) + (ema["gripper"] * (1.0 - config.EMA_ALPHA))
                    ema["gui_gripper"] = (raw_gripper_percent * config.EMA_ALPHA) + (ema["gui_gripper"] * (1.0 - config.EMA_ALPHA))
                    
                    for i in range(5):
                        ema["joints"][i] = (target_joints[i] * config.EMA_ALPHA) + (ema["joints"][i] * (1.0 - config.EMA_ALPHA))

                # --- TRANSMIT DATA ---
                # Hardware physical writes only trigger if fists aren't currently paused!
                target_robot = robot_map.get(side)
                if target_robot and not global_pause:
                    try:
                        action = {
                            "shoulder_pan.pos": ema["joints"][0],
                            "shoulder_lift.pos": ema["joints"][1],
                            "elbow_flex.pos": ema["joints"][2],
                            "wrist_flex.pos": ema["joints"][3],
                            "wrist_roll.pos": ema["joints"][4],
                            "gripper.pos": ema["gripper"]
                        }
                        target_robot.send_action(action)
                    except Exception:
                        pass
                
                # GUI data pipeline
                return {
                    "status": "engaged",
                    "gripper": ema["gui_gripper"], # Pure 0-100% Visual representation
                    "thumb_pt": thumb_pt,
                    "index_pt": index_pt
                }
            else:
                # Still counting down natively
                time_waiting = current_time - state["detected_since"] if state["detected_since"] else 0.0
                countdown = max(0.0, config.ENGAGEMENT_DELAY_SEC - time_waiting)
                return {
                    "status": "waiting",
                    "countdown": countdown,
                    "thumb_pt": thumb_pt,
                    "index_pt": index_pt
                }
                
        # Camera blind natively 
        return {"status": "missing"}
