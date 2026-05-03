import time
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode
# Initialize the Feetech motor bus
bus = FeetechMotorsBus(
    port="/dev/ttyACM0",
motors={f"motor_{i}": Motor(i, "sts3215", MotorNormMode.DEGREES) for i in range(1, 7)}
)

bus.connect()

# Disable torque so you can manually move the arm
for name in list(bus.motors.keys()):
    bus.write("Torque_Enable", name, 0)

print("---")
print("Torque disabled. Move your arm to the physical center.")
print("Look for the motor whose value is furthest from 2048 (likely near 0 or 4095).")
print("Press Ctrl+C to stop.")
print("---\n")

try:
    while True:
        output = []
        for name in list(bus.motors.keys()):
            # Read the raw position of each motor (0-4095)
            val = bus.read("Present_Position", name)
            
            # Convert to standard int if it returns a numpy/torch scalar
            val = val.item() if hasattr(val, "item") else val 
            output.append(f"ID {name.split('_')[1]}: {int(val):04d}")
            
        # Print on the same line to create a real-time monitor
        print(" | ".join(output), end="\r")
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nDisconnecting...")
    bus.disconnect()