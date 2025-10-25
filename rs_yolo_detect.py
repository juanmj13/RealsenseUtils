#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realsense_yolo_detect

YOLO detection viewer for Intel RealSense .bag files.
- Reads .bag files only (no webcams, no standalone images).
- Takes COLOR frames from the .bag, runs the YOLO (Ultralytics) model, and displays/saves the result.
- Optional incremental MP4 writing (no full RAM loading).

Requirements:
- pyrealsense2
- opencv-python
- numpy
- ultralytics (YOLO)

Usage examples:

python realsense_yolo_detect.py \
    --model runs/detect/train/weights/best.pt \
    --bag   capture.bag \
    --thresh 0.4 \
    --fps 30 \
    --resolution 1280x720 \
    --record out_demo.mp4

Keys:

q → quit
s → pause
p → save snapshot "capture.png"

Compatibility notes:
- This file is written for Python 3.7 to 3.9 compatibility.
- OpenCV displays in BGR; RealSense usually provides RGB8. We convert to BGR for
display and for YOLO input (Ultralytics handles BGR/uint8 just fine).
- If your .bag does not contain a color stream, no results will be displayed.

"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 is not installed. Please install librealsense/pyrealsense2")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics is not installed. Please install:  pip install ultralytics")
    sys.exit(1)


def open_playback(bag_path: Path):
    """load .bag in playback mode and prepare pipeline/profile."""
    pipeline = rs.pipeline()
    config = rs.config()
    # Load from file
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    profile = pipeline.start(config)
    dev = profile.get_device()
    playback = dev.as_playback()

    # Disable realtime and repeat to have full control
    if hasattr(playback, "set_real_time"):
        playback.set_real_time(False)
    if hasattr(playback, "set_repeat"):
        playback.set_repeat(False)

    return pipeline, profile


def next_color_frame(pipeline: rs.pipeline, timeout_ms: int) -> Optional[np.ndarray]:
    """Gets the next color frame as an np.ndarray (RGB or BGR depending on the SDK).
    Returns None when playback ends or on prolonged timeout
    """
    try:
        frames = pipeline.wait_for_frames(timeout_ms)
    except RuntimeError:
        return None

    if not frames:
        return None

    color_frame = frames.get_color_frame()
    if color_frame is None:
        return None

    color_np = np.asanyarray(color_frame.get_data())  # RGB usually delivers RGB8
    return color_np


def draw_detections(frame_bgr: np.ndarray, detections, labels, conf_thresh: float) -> int:
    """Draws bounding boxes and labels on the frame."""
    # Tableau 10 colors palette
    bbox_colors = [
        (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
        (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
    ]

    count = 0
    for i in range(len(detections)):
        conf = float(detections[i].conf.item())
        if conf < conf_thresh:
            continue

        # Coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        try:
            classname = labels.get(classidx, str(classidx))  # dict-like in ultralytics
        except AttributeError:
            classname = labels[classidx]

        color = bbox_colors[classidx % len(bbox_colors)]
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), color, 2)

        label = f"{classname}: {int(conf * 100)}%"
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, labelSize[1] + 10)
        cv2.rectangle(
            frame_bgr,
            (xmin, label_ymin - labelSize[1] - 10),
            (xmin + labelSize[0], label_ymin + baseLine - 10),
            color,
            cv2.FILLED,
        )
        cv2.putText(frame_bgr, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        count += 1

    return count


def parse_resolution(res_str: Optional[str]) -> Optional[Tuple[int, int]]:
    if not res_str:
        return None
    try:
        w, h = res_str.lower().split('x')
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("--resolution must be WxH, ej: 1280x720")


def main():
    parser = argparse.ArgumentParser(
        description="Detection viewer for Intel RealSense .bag files using YOLO models.",
    )
    parser.add_argument('--model', required=True, help='Path of the YOLO model file (.pt)')
    parser.add_argument('--bag', required=True, help='Path of the .bag file RealSense')
    parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--resolution', type=str, default=None, help='Resolution WxH para show/record (optional)')
    parser.add_argument('--timeout-ms', type=int, default=5000, help='timout for waiting frames (ms)')
    parser.add_argument('--every-n', type=int, default=1, help='Sampling 1 from N frames')
    parser.add_argument('--max-frames', type=int, default=0, help='Máximo de frames a procesar (0=todos)')
    parser.add_argument('--fps', type=int, default=30, help='FPS for recording (if using --record)')
    parser.add_argument('--record', type=str, default=None, help='Path of the mp4 output (optional)')
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.exists():
        print(f"ERROR: .bag file doesnt exist: {bag_path}")
        sys.exit(1)

    # load Yolo model
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        print(f"ERROR: YOLO model doesnt exist: {model_path}")
        sys.exit(1)
    model = YOLO(str(model_path), task='detect')
    labels = model.names

    # Open playback
    pipeline, _profile = open_playback(bag_path)

    # Desired Resolution (just for display/record)
    resize_tuple = parse_resolution(args.resolution) if args.resolution else None

    # Recording (optional)
    writer = None
    total_written = 0

    # FPS Statistics
    frame_rate_buffer = []
    fps_avg_len = 200
    avg_fps = 0.0

    # Main loop
    processed = 0
    frame_idx = 0

    try:
        while True:
            color_np = next_color_frame(pipeline, args.timeout_ms)
            if color_np is None:
                print("[INFO] End of playback or timeout reached.")
                break

            # Sampling
            if frame_idx % max(1, args.every_n) != 0:
                frame_idx += 1
                continue

            t_start = time.perf_counter()

            # Realsense use to deliver RGB8 → convert to BGR for OpenCV/display/YOLO
            if color_np.ndim == 3 and color_np.shape[2] == 3:
                frame_bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
            else:
                # If is not RGB, adapt to BGR
                if color_np.ndim == 2:
                    frame_bgr = cv2.cvtColor(color_np, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = color_np.astype(np.uint8)

            if resize_tuple:
                frame_bgr = cv2.resize(frame_bgr, resize_tuple)

            # YOLO inference
            results = model(frame_bgr, verbose=False)
            detections = results[0].boxes

            # Draw results
            count = draw_detections(frame_bgr, detections, labels, conf_thresh=float(args.thresh))

            # FPS
            t_stop = time.perf_counter()
            inst_fps = float(1.0 / max(1e-6, (t_stop - t_start)))
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(inst_fps)
            avg_fps = float(np.mean(frame_rate_buffer)) if frame_rate_buffer else inst_fps

            # HUD
            cv2.putText(frame_bgr, f'FPS: {avg_fps:0.2f}', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame_bgr, f'Objects: {count}', (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display
            cv2.imshow('RealSense YOLO Detect', frame_bgr)

            # Record
            if args.record:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    h, w = frame_bgr.shape[:2]
                    writer = cv2.VideoWriter(str(Path(args.record)), fourcc, int(args.fps), (w, h))
                writer.write(frame_bgr)
                total_written += 1

            # Keyboard controls
            key = cv2.waitKey(1)
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('s'), ord('S')):
                cv2.waitKey()
            elif key in (ord('p'), ord('P')):
                cv2.imwrite('capture.png', frame_bgr)

            processed += 1
            frame_idx += 1
            if args.max_frames > 0 and processed >= args.max_frames:
                print(f"[INFO] Max frames reached: {args.max_frames} frames processed.")
                break

    finally:
        if writer is not None:
            writer.release()
        pipeline.stop()
        cv2.destroyAllWindows()
        if args.record:
            print(f"[✔] Video saved in: {args.record} (frames: {total_written}, fps: {args.fps})")
        print(f"[OK] Frames processed: {processed} | FPS promedio: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
