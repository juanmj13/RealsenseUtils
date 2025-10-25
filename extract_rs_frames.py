#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract color and depth frames from an Intel RealSense .bag file. This is useful to extract
images to prepare for training datasets. 
- Color: PNG 8-bit (BGR)
- Depth: PNG 8-bit with thresholding and colormap configurable
Requisites: pyrealsense2, opencv-python, numpy
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Optional  # ✅ Para compatibilidad con Python < 3.10

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 is not instaled. Please instal librealsense/pyrealsense2")
    sys.exit(1)


def _has_colormap_turbo() -> bool:
    return hasattr(cv2, "COLORMAP_TURBO")


def colorize_depth(
    depth_np_u16: np.ndarray,
    depth_scale: Optional[float],
    min_m: Optional[float],
    max_m: Optional[float],
    colormap: str = "jet",
) -> np.ndarray:
    """convert depth u16 to colored depth u8 image for visualization"""
    depth = depth_np_u16.astype(np.float32)

    if depth_scale is not None:
        depth_m = depth * depth_scale
        valid = depth_m[depth_m > 0]
        if min_m is None or max_m is None:
            if valid.size == 0:
                min_v, max_v = 0.0, 1.0
            else:
                p1, p99 = np.percentile(valid, [1, 99])
                min_v = p1 if min_m is None else min_m
                max_v = p99 if max_m is None else max_m
        else:
            min_v, max_v = min_m, max_m
        scaled = np.clip((depth_m - min_v) / (max_v - min_v), 0, 1) * 255.0
    else:
        valid = depth[depth > 0]
        if min_m is None or max_m is None:
            if valid.size == 0:
                min_v, max_v = 0.0, 1000.0
            else:
                p1, p99 = np.percentile(valid, [1, 99])
                min_v = p1 if min_m is None else float(min_m)
                max_v = p99 if max_m is None else float(max_m)
        else:
            min_v, max_v = float(min_m), float(max_m)
        scaled = np.clip((depth - min_v) / (max_v - min_v), 0, 1) * 255.0

    depth_u8 = scaled.astype(np.uint8)
    cm = colormap.lower()

    if cm == "black_to_white":
        colored = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
    elif cm == "white_to_black":
        inv = 255 - depth_u8
        colored = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    elif cm == "turbo" and _has_colormap_turbo():
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    else:
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

    return colored


def extract_frames(
    bag_path: Path,
    out_dir: Path,
    align_to_color: bool = True,
    every_n: int = 1,
    max_frames: int = 0,
    timeout_ms: int = 5000,
    depth_min_m: Optional[float] = None,
    depth_max_m: Optional[float] = None,
    colormap: str = "jet",
):
    video_base = bag_path.stem

    out_color = out_dir / "color"
    out_depth_vis = out_dir / "depth_preview"

    out_color.mkdir(parents=True, exist_ok=True)
    out_depth_vis.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    profile = pipeline.start(config)
    dev = profile.get_device()
    playback = dev.as_playback()
    if hasattr(playback, "set_real_time"):
        playback.set_real_time(False)
    if hasattr(playback, "set_repeat"):
        playback.set_repeat(False)

    depth_scale = None
    try:
        depth_sensor = dev.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
    except Exception:
        pass

    align = rs.align(rs.stream.color) if align_to_color else None

    print(f"[INFO] Extracting from: {bag_path.name}")
    if depth_scale:
        print(f"[INFO] depth_scale: {depth_scale:.8f} m/unit")
    print(f"[INFO] Colormap: {colormap}")
    print(f"[INFO] Save in: {out_dir}")

    saved = 0
    frame_idx = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms)
            except RuntimeError:
                print("[INFO] Timeout reached or end of file.")
                break

            if not frames:
                continue

            if align:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if frame_idx % every_n == 0:
                if color_frame:
                    color_np = np.asanyarray(color_frame.get_data())
                    cv2.imwrite(str(out_color / f"{video_base}_{frame_idx:06d}.png"), color_np)

                if depth_frame:
                    depth_np = np.asanyarray(depth_frame.get_data())
                    depth_color = colorize_depth(depth_np, depth_scale, depth_min_m, depth_max_m, colormap)
                    cv2.imwrite(str(out_depth_vis / f"{video_base}_{frame_idx:06d}.png"), depth_color)

                saved += 1
                if max_frames > 0 and saved >= max_frames:
                    print(f"[INFO] max frames reached: {max_frames} frames saved.")
                    break

            frame_idx += 1

    finally:
        pipeline.stop()

    print(f"[OK] Process completed. Frames saved: {saved}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from Intel RealSense .bag file")
    parser.add_argument("bag", type=str, help="Path to the .bag file. Example: video.bag")
    parser.add_argument("--no-align", action="store_true", help="Don't align depth to color")
    parser.add_argument("--every-n", type=int, default=1, help="Save 1 every N frames (Sampling rate)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max number of frames (0 = No limit)")
    parser.add_argument("--timeout-ms", type=int, default=5000, help="Timeout waiting for frames (ms)")
    parser.add_argument("--min-m", type=float, default=None, help="Min threshold filter in depth (m)")
    parser.add_argument("--max-m", type=float, default=None, help="Max threshold filter in depth (m)")
    parser.add_argument("--colormap", type=str, default="jet",
                        choices=["jet", "black_to_white", "white_to_black"] + (["turbo"] if _has_colormap_turbo() else []),
                        help="colormap for depth visualization")

    args = parser.parse_args()
    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.exists():
        print(f"ERROR: file doesn't exist: {bag_path}")
        sys.exit(1)

    # Save in outputs folder ./outputs/
    out_dir = Path("outputs") / f"{bag_path.stem}_frames"

    extract_frames(
        bag_path=bag_path,
        out_dir=out_dir,
        align_to_color=not args.no_align,
        every_n=max(1, args.every_n),
        max_frames=max(0, args.max_frames),
        timeout_ms=max(100, args.timeout_ms),
        depth_min_m=args.min_m,
        depth_max_m=args.max_m,
        colormap=args.colormap,
    )


if __name__ == "__main__":
    main()
