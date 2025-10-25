#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a realsense .bag file to MP4 or GIF using color frames.
- Read all the frame with wait_for_frames (playback mode).
- Avoid loading everything into RAM: write incrementally (MP4 with OpenCV; GIF with imageio.get_writer).
- optionals: every-n (sampling), max-frames (max number of frames), timeout (configurable).

Requisites: pyrealsense2, opencv-python, numpy, imageio
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import imageio
from typing import Optional

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 is not instaled. Please install librealsense/pyrealsense2")
    sys.exit(1)


def open_playback(bag_path: Path):
    """Open .bag in playback mode and prepare pipeline/profile."""
    pipeline = rs.pipeline()
    config = rs.config()
    # load from file, just once (no repeat)
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    profile = pipeline.start(config)
    dev = profile.get_device()
    playback = dev.as_playback()

    # Disable real-time and repeat to have full control and more stable
    if hasattr(playback, "set_real_time"):
        playback.set_real_time(False)
    if hasattr(playback, "set_repeat"):
        playback.set_repeat(False)

    return pipeline, profile

# Iter color frames
def iter_color_frames(pipeline: rs.pipeline,
                      align_to_color: bool,
                      timeout_ms: int,
                      every_n: int,
                      max_frames: int):
    """
    Iterator that yields np.ndarray (RGB/BGR depending on what the SDK provides; 
    RealSense color is usually RGB8).     We use align_to_color only when there is depth 
    and you want to align, but here it is irrelevant because we only     export color. 
    It is kept for future compatibility.
   
    """
    align = rs.align(rs.stream.color) if align_to_color else None

    saved = 0
    frame_idx = 0

    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            # End of file or timeout
            break

        if not frames:
            continue

        if align:
            frames = align.process(frames)

        color_frame = frames.get_color_frame()
        if color_frame is None:
            frame_idx += 1
            continue

        if frame_idx % every_n == 0:
            color_np = np.asanyarray(color_frame.get_data())
            yield color_np
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break

        frame_idx += 1

# Convert to MP4
def convert_to_mp4(bag_path: Path,
                   out_path: Path,
                   fps: int,
                   every_n: int,
                   max_frames: int,
                   timeout_ms: int):
    """Incremental writing to MP4 using OpenCV VideoWriter."""
    pipeline, profile = open_playback(bag_path)
    writer = None
    total = 0
    try:
        for color_np in iter_color_frames(pipeline, align_to_color=False,
                                          timeout_ms=timeout_ms,
                                          every_n=every_n,
                                          max_frames=max_frames):
            # Realsense color frames are usually RGB8 and opencv expects BGR8
            if writer is None:
                h, w = color_np.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            total += 1
    finally:
        if writer is not None:
            writer.release()
        pipeline.stop()

    if total == 0:
        # Clean up empty file if no frames were written
        try:
            out_path.unlink(missing_ok=True)  # Python 3.8+: remove if exists
        except TypeError:
            if out_path.exists():
                out_path.unlink()
        print("[⚠] Color frames not written. Does the .bag have a color stream?")
    else:
        print(f"[✔] MP4 saved: {out_path}  (frames: {total}, fps: {fps})")


def convert_to_gif(bag_path: Path,
                   out_path: Path,
                   fps: int,
                   every_n: int,
                   max_frames: int,
                   timeout_ms: int):
    """Incremental writing to GIF using imageio."""
    pipeline, profile = open_playback(bag_path)
    duration = 1.0 / float(fps)
    total = 0

    # imageio waits rgB8; if it is already in RGB, we just ensure dtype and shape
    writer = imageio.get_writer(str(out_path), mode="I", duration=duration, loop=0)

    try:
        for color_np in iter_color_frames(pipeline, align_to_color=False,
                                          timeout_ms=timeout_ms,
                                          every_n=every_n,
                                          max_frames=max_frames):
            # Ensure uint8 and 3 channels
            if color_np.dtype != np.uint8:
                color_np = color_np.astype(np.uint8)
            if color_np.ndim != 3 or color_np.shape[2] != 3:
                # if it is Y8/Y16, convert to RGB (should not happen in color stream)
                color_np = cv2.cvtColor(color_np, cv2.COLOR_GRAY2RGB)
            writer.append_data(color_np)
            total += 1
    finally:
        writer.close()
        pipeline.stop()

    if total == 0:
        try:
            out_path.unlink(missing_ok=True)
        except TypeError:
            if out_path.exists():
                out_path.unlink()
        print("[⚠] Color frames not written. Does the .bag have a color stream?")
    else:
        print(f"[✔] GIF saved: {out_path}  (frames: {total}, fps: {fps})")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a realsense .bag file to MP4 or GIF using color frames."
    )
    parser.add_argument("--video", required=True, help="Path of  .bag file (ex: test.bag)")
    parser.add_argument("--output", required=True, choices=["color"],
                        help="Outout selection (By now just 'color')")
    parser.add_argument("--ext", required=True, choices=["mp4", "gif"],
                        help="Output extension (mp4 o gif)")
    parser.add_argument("--fps", type=int, default=30, help="FPS of the output video/gif")
    parser.add_argument("--every-n", type=int, default=1,
                        help="Save 1 from N frames (sampling)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max number of frames (0 = no limit)")
    parser.add_argument("--timeout-ms", type=int, default=5000,
                        help="Timeout for waiting for frames (ms)")

    args = parser.parse_args()
    bag_path = Path(args.video).expanduser().resolve()
    if not bag_path.exists():
        print(f"[❌] File doesn't exist: {bag_path}")
        sys.exit(1)

    base = bag_path.stem
    out_path = bag_path.with_name(f"{base}_{args.output}.{args.ext}")

    print(f"[⏳] Converting: {bag_path.name}  →  {out_path.name}")
    print(f"[ℹ] FPS: {args.fps} | every-n: {args.every_n} | max-frames: {args.max_frames} | timeout: {args.timeout_ms} ms")

    if args.ext == "mp4":
        convert_to_mp4(bag_path, out_path, fps=args.fps,
                       every_n=max(1, args.every_n),
                       max_frames=max(0, args.max_frames),
                       timeout_ms=max(100, args.timeout_ms))
    else:
        convert_to_gif(bag_path, out_path, fps=args.fps,
                       every_n=max(1, args.every_n),
                       max_frames=max(0, args.max_frames),
                       timeout_ms=max(100, args.timeout_ms))


if __name__ == "__main__":
    main()
