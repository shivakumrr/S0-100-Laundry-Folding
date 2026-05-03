import cv2
import numpy as np
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Stereo Calibration for Dual-Camera IK Tracking")
    parser.add_argument("--cam0", type=int, default=0, help="Device index for Camera 0 (Side)")
    parser.add_argument("--cam1", type=int, default=1, help="Device index for Camera 1 (Center)")
    parser.add_argument("--board-w", type=int, default=9, help="Number of inner corners on the checkerboard width")
    parser.add_argument("--board-h", type=int, default=6, help="Number of inner corners on the checkerboard height")
    parser.add_argument("--square-size", type=float, default=0.025, help="Size of a square in meters (default 0.025 = 2.5cm)")
    args = parser.parse_args()

    CHECKERBOARD_SIZE = (args.board_w, args.board_h)
    SQUARE_SIZE = args.square_size

    cap0 = cv2.VideoCapture(args.cam0)
    cap1 = cv2.VideoCapture(args.cam1)
    
    # Set to uniform resolutions
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Termination criteria for subpixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare real-world metric 3D point array for the checkerboard corners
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = [] # 3D points in real world space
    imgpoints0 = [] # 2D points in image plane from cam 0
    imgpoints1 = [] # 2D points in image plane from cam 1

    print("\n" + "="*50)
    print(" STEREO CALIBRATION FOR DUAL-CAMERA IK")
    print("="*50)
    print("1. Please hold a printed checkerboard pattern so BOTH cameras can see it clearly.")
    print("2. The pattern should have a clear distinction of inner corners.")
    print(f"3. Expected Pattern: {args.board_w}x{args.board_h} inner corners, each square {args.square_size*100} cm.")
    print("4. Press [SPACE] to capture an image pair.")
    print("5. Try to capture from various angles and distances (15-20 captures recommended).")
    print("6. Press [Q] when finished to compute calibration.\n")

    capture_count = 0

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print("[ERROR] Failed to read from one or both cameras. Please check connections.")
            break

        viz0 = frame0.copy()
        viz1 = frame1.copy()

        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        # Fast check for visual display
        ret_fast0, corners_fast0 = cv2.findChessboardCorners(gray0, CHECKERBOARD_SIZE, cv2.CALIB_CB_FAST_CHECK)
        ret_fast1, corners_fast1 = cv2.findChessboardCorners(gray1, CHECKERBOARD_SIZE, cv2.CALIB_CB_FAST_CHECK)

        if ret_fast0: cv2.drawChessboardCorners(viz0, CHECKERBOARD_SIZE, corners_fast0, ret_fast0)
        if ret_fast1: cv2.drawChessboardCorners(viz1, CHECKERBOARD_SIZE, corners_fast1, ret_fast1)

        cv2.putText(viz0, f"Captures: {capture_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(viz1, f"[SPACE]=Capture  [Q]=Quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        combined = np.hstack((viz0, viz1))
        cv2.imshow("Stereo Calibration: Cam 0 (Side) | Cam 1 (Center)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Attempt highly accurate capture
            ret_chk0, corners0 = cv2.findChessboardCorners(gray0, CHECKERBOARD_SIZE, None)
            ret_chk1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD_SIZE, None)
            
            if ret_chk0 and ret_chk1:
                corners2_0 = cv2.cornerSubPix(gray0, corners0, (11,11), (-1,-1), criteria)
                corners2_1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
                
                objpoints.append(objp)
                imgpoints0.append(corners2_0)
                imgpoints1.append(corners2_1)
                
                capture_count += 1
                print(f"[SUCCESS] Pair {capture_count} captured!")
            else:
                print("[WARN] Could not find the checkerboard perfectly in both cameras. Adjust pattern and try again.")
        
        elif key == ord('q'):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

    if capture_count < 5:
        print("\n[ABORT] Not enough captures explicitly saved. You need at least 5 frames, but 15+ is recommended for stable 3D math.")
        return

    print(f"\nComputing physical calibration matrices for {capture_count} captures... This might take a few seconds.")
    img_size = (frame0.shape[1], frame0.shape[0])

    # 1. Calibrate single cameras individually
    ret0, mtx0, dist0, rvecs0, tvecs0 = cv2.calibrateCamera(objpoints, imgpoints0, img_size, None, None)
    ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, img_size, None, None)

    # 2. Stereo calibrate
    flags = cv2.CALIB_FIX_INTRINSIC
    stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    
    ret_stereo, CM0, dist0, CM1, dist1, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints0, imgpoints1, 
        mtx0, dist0,
        mtx1, dist1,
        img_size, criteria=stereo_criteria, flags=flags
    )

    print(f"\n[DONE] Stereo Calibration RMS error: {ret_stereo:.4f} pixels (lower is better, ideally < 1.0)")
    
    # 3. Save to file
    calib_data = {
        "cam0": { "mtx": CM0.tolist(), "dist": dist0.tolist() },
        "cam1": { "mtx": CM1.tolist(), "dist": dist1.tolist() },
        "stereo": { "R": R.tolist(), "T": T.tolist(), "E": E.tolist(), "F": F.tolist() }
    }

    os.makedirs("hand_teleop/calibration", exist_ok=True)
    out_file = "hand_teleop/calibration/stereo_calib.json"
    with open(out_file, "w") as f:
        json.dump(calib_data, f, indent=4)
        
    print(f"[SUCCESS] Spatial transformation matrices saved to -> {out_file}\nYou are now ready for pure 3D multi-camera triangulation tracking!")

if __name__ == "__main__":
    main()
