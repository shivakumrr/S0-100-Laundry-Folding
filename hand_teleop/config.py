# -------------------------------------------------------------
# Configuration Constants for Dual Hand Teleoperation
# -------------------------------------------------------------

# --- Camera & Vision Settings ---
CAMERA_INDEX = 8  # Switched to Camera 0 (/dev/video10)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
RECORDING_FPS = 30    # Framerate for logging Hughes Face datasets
INVERT_HANDS = False  # Set to True if your camera lens flip makes your Left hand map to your Right robot

# --- Engagement & Safety Configuration ---
ENGAGEMENT_DELAY_SEC = 2.0  # Dead man's switch timer before physical motors obey hand tracking

# --- Gripper Calibration ---
MIN_DIST = 25.0  # Pixel distance threshold defining a 'Fully Closed' hand
MAX_DIST = 65.0  # Lowered drastically so opening your fingers just a couple inches triggers 'fully open'

# --- Movement Sensitivities (Relative Delta Mapping) ---
PAN_SENSITIVITY = 500.0   # Sweeping sensitivity for shoulder base pan
LIFT_SENSITIVITY = 250.0  # Sweeping sensitivity for elbow flex height

# --- IK Movement Sensitivities (Meters) ---
IK_X_SENSITIVITY = 1.6   # Left/Right translation scale
IK_Y_SENSITIVITY = 3.0   # Up/Down translation scale
IK_Z_SENSITIVITY = 2.5   # Raised significantly because MP depth values are intrinsically tiny


# --- Anti-Jitter Filter ---
# EMA stands for Exponential Moving Average. It eradicates AI inference micro-jitter.
EMA_ALPHA = 0.09  # Lowered strictly to squash MediaPipe micro-jitter.

# --- Deadzone Settings ---
# How far physically the target MUST move in mathematical space before the robot bothers calculating
# a new path. If 15.0, the robot will literally only jump in 15cm jagged steps! 
IK_DEADZONE_CM = 15.0

# -------------------------------------------------------------
# HARDWARE SAFETY METRICS (New)
# -------------------------------------------------------------
# Absolute hardware ranges to prevent stripping servo gears
SAFE_PAN_MIN = -120.0
SAFE_PAN_MAX = 120.0
SAFE_LIFT_MIN = -120.0
SAFE_LIFT_MAX = 120.0

# Software velocity governor: maximum allowed degree change per camera frame
# Prevents violent snapping/whiplash if inference spikes anomaly data
MAX_DEGREE_SPEED = 12.0
