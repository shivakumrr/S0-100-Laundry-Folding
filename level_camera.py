import cv2
import mediapipe as mp
import sys
import os
import math

# Attempt to load user's configured telemetry settings
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'hand_teleop'))
    import config
    CAMERA_INDEX = config.CAMERA_INDEX
    CAMERA_WIDTH = config.CAMERA_WIDTH
    CAMERA_HEIGHT = config.CAMERA_HEIGHT
except ImportError:
    CAMERA_INDEX = 10
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    print("====================================")
    print(" CAMERA LEVELING & ALIGNMENT TOOL")
    print("====================================")
    print("1. Align your PHYSICAL DESK EDGE with the thick GREEN horizontal line.")
    print("2. Sit straight and align your nose with the thick BLUE vertical line.")
    print("3. Check the face-tilt read out in the top left to ensure the camera isn't twisted!")
    print("4. Press 'q' to quit.")
    print("====================================")

    ema_angle = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame")
            break

        frame = cv2.flip(frame, 1) # Mirror intuitively
        h, w, _ = frame.shape

        # Draw Rule of Thirds Grid for depth/perspective alignment
        cv2.line(frame, (0, h//3), (w, h//3), (200, 200, 200), 1)
        cv2.line(frame, (0, 2*h//3), (w, 2*h//3), (200, 200, 200), 1)
        cv2.line(frame, (w//3, 0), (w//3, h), (200, 200, 200), 1)
        cv2.line(frame, (2*w//3, 0), (2*w//3, h), (200, 200, 200), 1)

        # Draw Exact Center Crosshairs
        cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 2) # Horizontal Green
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 0, 0), 2) # Vertical Blue

        # Detect Face to check literal structural Camera Roll 
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                keypoints = detection.location_data.relative_keypoints
                if len(keypoints) >= 2:
                    # MediaPipe Face Detection Keypoints:
                    # Right Eye (0), Left Eye (1), Nose Tip (2), Mouth Center (3)
                    right_eye = keypoints[0]
                    left_eye = keypoints[1]
                    
                    rx, ry = int(right_eye.x * w), int(right_eye.y * h)
                    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
                    
                    cv2.line(frame, (rx, ry), (lx, ly), (0, 0, 255), 2)
                    
                    # Calculate roll (tilt) angle
                    dy = ly - ry
                    dx = lx - rx
                    raw_angle = math.degrees(math.atan2(dy, dx))
                    
                    if ema_angle is None:
                        ema_angle = raw_angle
                    else:
                        # Heavy low-pass filter to stop human-head natural fidgeting
                        ema_angle = (raw_angle * 0.05) + (ema_angle * 0.95)
                    
                    tilt = "PERFECT!" if abs(ema_angle) < 4.0 else "TILTED! ADJUST!"
                    color = (0, 255, 0) if tilt == "PERFECT!" else (0, 0, 255)
                    
                    cv2.putText(frame, f"Face Leveling: {abs(ema_angle):.1f} deg ({tilt})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw Nose tracker
                    nose = keypoints[2]
                    nx, ny = int(nose.x * w), int(nose.y * h)
                    cv2.circle(frame, (nx, ny), 5, (255, 0, 0), -1)
                    
                    break # Only process the primary first face

        # Instruction Hud
        cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "Match your DESK EDGE entirely with the GREEN LINE", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Camera Leveling Tool', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
