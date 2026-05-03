import glob
import json
import builtins
from pathlib import Path
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

FOLLOWER_CALIB_PATH = Path("/home/g1/Developer/Physical Agents/calibration/robots/so_follower/follower_arm.json")
LEADER_CALIB_PATH   = Path("/home/g1/Developer/Physical Agents/calibration/teleoperators/so_leader/leader_arm.json")

def load_gripper_bounds(calib_path: Path) -> tuple[float, float]:
    """
    Parse a LeRobot calibration JSON and return the gripper's
    (closed_pct, open_pct) in the normalized 0-100 space LeRobot uses.

    The gripper motor uses RANGE_0_100 normalization:
        norm = ((raw - range_min) / (range_max - range_min)) * 100

    The physical 'closed' state is at the raw_min end of the range,
    and 'open' is at the raw_max end.
    So in normalized space:  closed = 0.0,  open = 100.0
    BUT if the motor is mounted inverted (drive_mode=1), the mapping flips.
    """
    with open(calib_path) as f:
        calib = json.load(f)

    gripper = calib["gripper"]
    drive_mode = gripper.get("drive_mode", 0)

    # When drive_mode=1 the framework applies (100 - norm), so we must invert
    if drive_mode == 1:
        closed_pct = 100.0
        open_pct   = 0.0
    else:
        closed_pct = 0.0
        open_pct   = 100.0

    print(f"   [CAL] {calib_path.stem}: gripper drive_mode={drive_mode}  closed={closed_pct}%  open={open_pct}%")
    return closed_pct, open_pct

class RobotManager:
    def __init__(self):
        """
        Scans, discovers, auto-calibrates, and assigns physical robots.
        """
        # Load calibrated gripper bounds from JSON files first
        print("\n[CAL] Reading gripper bounds from calibration files...")
        follower_closed, follower_open = load_gripper_bounds(FOLLOWER_CALIB_PATH)
        leader_closed,   leader_open   = load_gripper_bounds(LEADER_CALIB_PATH)
        
        # ACM1 = Follower = Left hand, ACM0 = Leader = Right hand
        self.gripper_bounds = {
            "Left":  {"closed": follower_closed, "open": follower_open},
            "Right": {"closed": leader_closed,   "open": leader_open},
        }

        self.robots = self._find_all_robots()
        self.robot_map = self._map_robots()

    def _find_all_robots(self):
        ports = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*') + glob.glob('/dev/tty.usbmodem*')
        ports = sorted(ports)
        robots = []
        
        for port in ports:
            try:
                # Retain physical calibrated assignments organically
                if "ACM1" in port: arm_id = "follower_arm"
                elif "ACM0" in port: arm_id = "leader_arm"
                else: arm_id = f"arm_{port.split('/')[-1]}"
                    
                cfg = SO100FollowerConfig(port=port, id=arm_id, disable_torque_on_disconnect=True)
                rob = SO100Follower(cfg)
                
                # -------------------------------------------------------------
                # AUTO-BYPASS CALIBRATION PROMPT MAGIC
                # -------------------------------------------------------------
                original_input = builtins.input
                builtins.input = lambda _: ""
                try:
                    rob.connect()
                finally:
                    builtins.input = original_input

                _ = rob.get_observation()

                robots.append(rob)
                print(f" -> SUCCESS: Robot initialized at {port}!")
            except Exception as e:
                print(f" -> ERROR: Could not connect to {port} ({e})")
                pass
                
        return robots

    def _map_robots(self):
        # -------------------------------------------------------------
        # Physical assignments: Which port maps to which physical hand
        # -------------------------------------------------------------
        m = {}
        for rob in self.robots:
            if "ACM0" in rob.config.port:
                m["Right"] = rob  
            elif "ACM1" in rob.config.port:
                m["Left"] = rob   
        return m

    def disconnect_all(self):
        for robot in self.robots:
            print(f"Disconnecting hardware torque on {robot.config.port}...")
            try:
                robot.disconnect()
            except Exception as e:
                print(f"  (non-fatal disconnect error: {e})")
