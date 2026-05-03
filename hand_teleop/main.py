import cv2
import time
import config
from robot_manager import RobotManager
from vision_tracker import HandTracker
from arm_controller import ArmController
from dataset_recorder import ImitationRecorder

def main():
    print("=============================================================")
    print("INITIALIZING MODULAR DUAL-ARM HAND TELEOPERATION SYSTEM")
    print("=============================================================")
    
    # 1. Initialize Modular Components natively
    rm = RobotManager()
    if len(rm.robots) < 2:
        print("\nWARNING: Less than 2 robots physically assigned!")

    tracker = HandTracker()
    controller = ArmController()
    recorder = ImitationRecorder()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    print("\nSYSTEM ONLINE: Hand Detection & Tracking Armed!\n")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        image, current_hands, pos_3d = tracker.process_frame(frame)
        current_time = time.time()

        # -------------------------------------------------------------
        # GLOBAL FIST DETECTION (SYSTEM PAUSE)
        # Check instantly if the user is squeezing a fist heuristically
        # -------------------------------------------------------------
        global_pause = False
        for side, landmarks in current_hands.items():
            tracker.draw_landmarks(image, landmarks)
            if tracker.is_fist(landmarks):
                global_pause = True
                
        if global_pause:
            cv2.putText(
                image, 
                ">>> SYSTEM PAUSED (FIST DETECTED) <<<", 
                (50, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 0, 255), 
                3
            )

        # -------------------------------------------------------------
        # ISOLATED ARM PROCESSING LOOP
        # Compute tracking updates recursively without halting either side
        # -------------------------------------------------------------
        for side in ["Left", "Right"]:
            landmarks = current_hands.get(side)
            
            # Delegate heavily to Arm Controller logic internally
            render_data = controller.process_arm(
                side, landmarks, pos_3d.get(side), current_time, rm.robot_map, global_pause, image.shape,
                gripper_bounds=rm.gripper_bounds[side]
            )
            
            line_color = (0, 165, 255) if side == "Left" else (255, 0, 255)
            y_pos = 50 if side == "Left" else 90

            # Dynamic Video UI Output Renderer
            if render_data["status"] == "engaged":
                cv2.line(image, render_data["thumb_pt"], render_data["index_pt"], line_color, 3)
                txt = f"{side.upper()} Hand [ACTIVE]: {render_data['gripper']:.0f}%"
                cv2.putText(image, txt, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            elif render_data["status"] == "waiting":
                cv2.line(image, render_data["thumb_pt"], render_data["index_pt"], line_color, 3)
                txt = f"{side.upper()} Hand [WAIT]: {render_data['countdown']:.1f}s"
                cv2.putText(image, txt, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
            else:
                txt = f"{side.upper()} Hand [MISSING]"
                cv2.putText(image, txt, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # -------------------------------------------------------------
        # BEHAVIORAL CLONING RECORDER HOOK
        # Pass pristine image stream and kinematics mathematical metadata
        # -------------------------------------------------------------
        if recorder.is_recording:
            # We strictly pass 'frame' (the pristine unmodified image from center camera) rather 
            # than 'image' (which is completely littered with rendered cv2 lines)
            recorder.capture_telemetry(frame, controller.emas)
            cv2.putText(image, f"[REC] DATASET FRAMES: {recorder.frames_recorded}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Press 'r' to start recording, 's' to stop.", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw Desk/Horizon Leveling Line
        h, w, _ = image.shape
        cv2.line(image, (0, h//2), (w, h//2), (0, 120, 0), 2) # Faded Green Horizon Line
        cv2.putText(image, "DESK LEVEL", (w - 120, h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)

        cv2.imshow('Modular Dual Robot Teleop', image)

        # Hotkey Quit Override
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recorder.start_recording("teleop_human_dataset_demo")
        elif key == ord('s'):
            recorder.stop_recording()

    # Graceful Shutdown Process Pipeline natively
    cap.release()
    cv2.destroyAllWindows()
    rm.disconnect_all()
    recorder.shutdown()

if __name__ == "__main__":
    main()
