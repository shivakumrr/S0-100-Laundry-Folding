#!/usr/bin/env python3
"""
check_cameras.py — quick camera diagnostic tool.

Scans /dev/video0, /dev/video2, /dev/video4, /dev/video6, /dev/video8,
grabs one frame from each that opens successfully, and saves them to:
    /tmp/cam_check_<index>.jpg

Open those files in any image viewer to confirm which physical camera
maps to which device index.

Usage:
    python check_cameras.py
"""

import cv2
import os
import sys

INDICES_TO_CHECK = [0, 2, 4, 6, 8, 10]

print("=== Camera Inspector ===\n")
found = []
for idx in INDICES_TO_CHECK:
    path = f"/dev/video{idx}"
    if not os.path.exists(path):
        print(f"  {path} — not present, skipping")
        continue

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"  {path} (index {idx}) — could not open")
        cap.release()
        continue

    # Warm up: read a few frames
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"  {path} (index {idx}) — opened but no frame returned")
        continue

    h, w = frame.shape[:2]
    out_path = f"/tmp/cam_check_{idx}.jpg"

    # Label the frame so you know which one is which
    label = f"/dev/video{idx}  ({w}x{h})"
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, frame)
    print(f"  /dev/video{idx}  ({w}x{h})  →  saved to {out_path}")
    found.append(idx)

print(f"\nDone. Captured {len(found)} camera(s): {[f'/dev/video{i}' for i in found]}")
print("\nOpen the saved files to verify which camera is which:")
for idx in found:
    print(f"  eog /tmp/cam_check_{idx}.jpg   # or: display /tmp/cam_check_{idx}.jpg")
