import cv2
import math
import mediapipe as mp
import config

class HandTracker:
    def __init__(self):
        """
        Encapsulates MediaPipe logic, inference modeling, and computer vision utilities.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,  # 0=Lite, 1=Full (Vastly more stable at the slight cost of performance)
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def _parse_hands(self, results):
        current_hands = {}
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_handedness_label = results.multi_handedness[idx].classification[0].label
                fixed_label = "Right" if mp_handedness_label == "Left" else "Left"
                if config.INVERT_HANDS:
                    fixed_label = "Right" if fixed_label == "Left" else "Left"
                current_hands[fixed_label] = hand_landmarks
        return current_hands

    def process_frame(self, frame):
        # 1. Image preparation natively flipped like a mirror
        image = cv2.flip(frame, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Run Deep Learning AI Inference Pipeline
        results = self.hands.process(rgb_image)
        
        # 3. Object Extraction Dictionary
        current_hands = self._parse_hands(results)
        
        # 4. Extract 3D Positions including AI-inferred Z Depth
        pos_3d = {}
        import numpy as np
        for side, landmarks in current_hands.items():
            lm = landmarks.landmark[0] # Wrist joint
            mcp = landmarks.landmark[9] # Middle finger base
            thumb = landmarks.landmark[4]
            index = landmarks.landmark[8]
            
            pinch_x = (thumb.x + index.x) / 2.0
            pinch_y = (thumb.y + index.y) / 2.0
            
            # Calculate the 2D Pitch Angle of the hand based on where the PINCH is physically pointing
            # relative to the anchored wrist joint.
            dy = pinch_y - lm.y
            dx = pinch_x - lm.x
            pitch_angle = math.atan2(dy, dx)
            
            # Calculate Depth Scale natively from physical constraints instead of AI Z estimates
            palm_size = math.hypot(lm.x - mcp.x, lm.y - mcp.y)
            
            # Anchor translation XYZ firmly to the PINCH coordinates (index + thumb)
            # So the math absolutely centers the grip position around the actual grabbing fingers!
            pos_3d[side] = np.array([pinch_x, pinch_y, palm_size, pitch_angle], dtype=np.float32)

        return image, current_hands, pos_3d
        
    def draw_landmarks(self, image, hand_landmarks):
        self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def is_fist(self, hand_landmarks):
        """
        Calculates heuristics across the 4 major fingertips to determine 
        if the user is actively squeezing their palm into a ball.
        """
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        
        # Calculate normative base distance of the user's flat palm metric length
        palm_size = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)
        
        if palm_size < 0.001:
            return False
            
        tips = [8, 12, 16, 20]
        for tip_idx in tips:
            tip = hand_landmarks.landmark[tip_idx]
            dist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            
            # If any finger is extended significantly beyond the palm, we are open!
            if dist > palm_size * 1.4:
                return False
                
        # If passed the threshold gauntlet natively, they are fully clenching their fist.
        return True
