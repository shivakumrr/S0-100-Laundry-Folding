import time
import numpy as np
import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import config

class ImitationRecorder:
    def __init__(self):
        """
        Manages the precise deep learning Hugging Face 'LeRobotDataset' writer API,
        storing synchronized raw CV2 image frames and matching mathematical coordinate
        metadata for Behavioral Cloning PyTorch training downstream.
        """
        self.dataset = None
        self.is_recording = False
        self.frames_recorded = 0
        self.episode_idx = 0

    def start_recording(self, task_name="puppet_teleoperation"):
        if self.is_recording:
            return
            
        self.task_name = task_name
        self.frames_recorded = 0
        
        # 1. Initialize the strict neural-net feature architecture
        if self.dataset is None:
            features = {
                "observation.images.webcam": {
                    "dtype": "video",
                    "shape": (3, config.CAMERA_HEIGHT, config.CAMERA_WIDTH), # Standard LeRobot format: Channels natively first
                    "names": ["channels", "height", "width"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (12,),  # 5 joints + 1 grip for Left, then Right
                    "names": ["motors"]
                },
                "action": {
                    "dtype": "float32",
                    "shape": (12,),
                    "names": ["motors"]
                }
            }
            
            # Locally caching into the `data` directory using HF API securely natively
            self.dataset = LeRobotDataset.create(
                repo_id="local_teleop/imitation_dataset",
                fps=min(30, config.RECORDING_FPS),
                features=features,
                use_videos=True,
                root=f"data/hand_teleop_imitation_{int(time.time())}"
            )
            
        self.is_recording = True
        print(f"\n[REC] >>> RECORDING STARTED! Task: '{self.task_name}' <<<")

    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        print(f"\n[REC] >>> RECORDING STOPPED! Processing {self.frames_recorded} frames into Parquet Dataset... <<<")
        
        try:
            # task is already stored per-frame in the buffer; just call save_episode()
            self.dataset.save_episode()
            self.episode_idx += 1
            print(f"[REC] >>> Episode {self.episode_idx} perfectly saved to HF Format! <<<")
        except Exception as e:
            print(f"[REC] ERROR saving episode dataset natively: {e}")

    def capture_telemetry(self, image, emas):
        """
        Saves exactly 1 synchronized frame of visual and kinematic mathematical data.
        """
        if not self.is_recording:
            return

        # Prepare CV2 image natively for torch/hf format (C, H, W layout required)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Reshape to explicitly match (3, H, W)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))

        # We construct the 12-axis hardware arrays explicitly for Left (0..5) and Right (6..11)
        state_array = np.zeros(12, dtype=np.float32)
        
        # Left Arm Mapping
        if emas["Left"]["joints"] is not None:
            for i in range(5):
                state_array[i] = float(emas["Left"]["joints"][i])
            state_array[5] = float(emas["Left"]["gripper"])
            
        # Right Arm Mapping
        if emas["Right"]["joints"] is not None:
            for i in range(5):
                state_array[6+i] = float(emas["Right"]["joints"][i])
            state_array[11] = float(emas["Right"]["gripper"])
            
        # For straightforward behavioral cloning, Action == State initially
        # (Assuming the system tracked precisely without physics slips).
        action_array = np.copy(state_array)

        # Ship strictly down to the HF Dataset API natively
        # 'task' must be injected per-frame, not into save_episode()
        frame_dict = {
            "observation.images.webcam": rgb_image,
            "observation.state": state_array,
            "action": action_array,
            "task": self.task_name,   # LeRobot stores task per-frame in the buffer
        }
        
        try:
            self.dataset.add_frame(frame_dict)
            self.frames_recorded += 1
        except Exception as e:
            print(f"[REC] Data Frame Insertion failed: {e}")

    def shutdown(self):
        if self.dataset:
            # Gracefully purge buffers preventing parquet fragmentation
            self.dataset.finalize()
