#!/usr/bin/env python3

import cv2
import numpy as np
import sys


if len(sys.argv) < 2:
    print("Usage: python motiondetector.py <video.mp4>")
    sys.exit(1)


video = sys.argv[1]

cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print(f"Cannot open video: {video}")
    sys.exit(1)


fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 1.0


ret, old_frame = cap.read()

if not ret:
    print("Cannot read first frame")
    cap.release()
    sys.exit(1)


old_gray = cv2.cvtColor(
    old_frame,
    cv2.COLOR_BGR2GRAY
)

# Convert once to float32
old_gray = old_gray.astype(np.float32)

height, width = old_gray.shape

# Create Hann window to reduce border artifacts
window = cv2.createHanningWindow(
    (width, height),
    cv2.CV_32F
)

frame_id = 1
events = []

LOG_THRESHOLD = 0.2
EVENT_THRESHOLD = 0.2

print("Starting analysis...\n", flush=True)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    timestamp = frame_id / fps

    # Convert current frame to grayscale float32
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    # Estimate global translation between consecutive frames
    (dx, dy), response = cv2.phaseCorrelate(
        old_gray,
        gray,
        window
    )

    # Euclidean displacement in pixels
    movement = float(np.hypot(dx, dy))

    if movement > LOG_THRESHOLD:
        print(
            f"frame={frame_id} "
            f"time={timestamp:.2f}s "
            f"movement={movement:.3f}px "
            f"dx={dx:.3f}px "
            f"dy={dy:.3f}px "
            f"response={response:.3f}",
            flush=True
        )

    if movement > EVENT_THRESHOLD:
        events.append({
            "time": timestamp,
            "movement": movement,
            "dx": dx,
            "dy": dy,
            "response": response,
            "frame": frame_id
        })

    # Current frame becomes previous frame
    old_gray = gray

    frame_id += 1


cap.release()


print("\n\n=== DETECTED EVENTS (sorted by movement) ===")

if not events:
    print("No vibration detected")
else:
    events.sort(
        key=lambda x: x["movement"],
        reverse=True
    )

    for e in events:
        print(
            f"time={e['time']:.2f}s "
            f"movement={e['movement']:.3f}px "
            f"dx={e['dx']:.3f}px "
            f"dy={e['dy']:.3f}px "
            f"response={e['response']:.3f} "
            f"frame={e['frame']}"
        )
