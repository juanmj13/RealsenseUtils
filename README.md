# RealsenseUtils

A set of **Python utilities** for working with **Intel RealSense \*.bag** files, extract color/depth frames, convert to **MP4/GIF**, and run **YOLO detection** (Ultralytics) and more utilities will be added.

> Tested on Python 3.8–3.11. Requires `librealsense/pyrealsense2` installed on the system.

## Features

- **Extract color and depth frames** (configurable colormap and thresholds).  
- **Convert** a `.bag` file to **MP4** or **GIF** using only the **color stream** (incremental writing).  
- **YOLO detection viewer** for `.bag` files with live display and optional **MP4 recording**.  
- Adjustable **sampling** (`--every-n`), **frame limits** (`--max-frames`), and **timeout** per read.  

## Requirements

### Common
```bash
pip install pyrealsense2 opencv-python numpy
```

### GIF conversion (optional)
```bash
pip install imageio
```

### YOLO detection (optional)
```bash
pip install ultralytics
```

> **Note (RealSense)**: You must have **librealsense** and/or the appropriate **`pyrealsense2`**.

## Scripts and Usage

### 1) `extract_rs_frames.py` — extract color and depth frames
This function is really usefull for extracting color and depth frames from an Intel RealSense .bag file to prepare for training datasets.

**Description**
- Color → 8-bit PNG (BGR as delivered by OpenCV).
- Depth → 8-bit PNG preview with configurable *colormap* (`jet`, `black_to_white`, `white_to_black`, and `turbo` if available in OpenCV).
- Optional alignment of depth to color (enabled by default).

**Usage**

```bash
python extract_rs_frames.py VIDEO.bag \
  --every-n 5 \
  --max-frames 500 \
  --timeout-ms 5000 \
  --min-m 0.2 \
  --max-m 3.0 \
  --colormap jet
# add --no-align if you DO NOT want depth aligned to color
```

**Output**
- `./outputs/VIDEO_frames/color/VIDEO_000000.png`
- `./outputs/VIDEO_frames/depth_preview/VIDEO_000000.png`

**Arguments**
- `bag`: path to `.bag` file
- `--no-align`: skip aligning depth to color
- `--every-n`: save 1 of every *N* frames (sampling)
- `--max-frames`: maximum frames (0 = no limit)
- `--timeout-ms`: timeout waiting for frames (ms)
- `--min-m`, `--max-m`: depth range thresholds (meters)
- `--colormap`: `jet` | `black_to_white` | `white_to_black` | (`turbo` if available)

---

### 2) `convert_bag.py` — `.bag` → **MP4** or **GIF**
Convert a realsense .bag file to MP4 or GIF using color frames.

**Description**
- Reads **only** the **color stream** from the `.bag`.
- Writes incrementally:
  - MP4 using `cv2.VideoWriter` (`mp4v`).
  - GIF using `imageio.get_writer`.
- Never loads the full video into memory.

**Usage**

```bash
# MP4 (30 fps, sampling every 3rd frame)
python convert_bag.py \
  --video VIDEO.bag \
  --output color \
  --ext mp4 \
  --fps 30 \
  --every-n 3 \
  --max-frames 0 \
  --timeout-ms 5000

# GIF (10 fps)
python convert_bag.py \
  --video VIDEO.bag \
  --output color \
  --ext gif \
  --fps 10
```

**Arguments**
- `--video`: input `.bag` file path  
- `--output`: currently only `"color"`  
- `--ext`: `mp4` | `gif`  
- `--fps`: output FPS  
- `--every-n`: save every *N* frames  
- `--max-frames`: limit number of frames (0 = all)  
- `--timeout-ms`: timeout for frame reading  

**Output**
- `VIDEO_color.mp4` or `VIDEO_color.gif` saved next to the `.bag`

---

### 3) `rs_yolo_detect.py` — YOLO viewer and MP4 recorder
YOLO detection viewer for Intel RealSense .bag files.

**Description**
- Loads a **YOLO (Ultralytics)** model and processes the **color stream** from a `.bag`.
- Displays live detection results with FPS and object count overlays.
- Optionally records to MP4 (incremental writing).

**Usage**

```bash
python rs_yolo_detect.py \
  --model runs/detect/train/weights/best.pt \
  --bag   capture.bag \
  --thresh 0.5 \
  --fps 30 \
  --resolution 1280x720 \
  --record out_demo.mp4
```

**Arguments**
- `--model`: YOLO `.pt` model file  
- `--bag`: input `.bag` file  
- `--thresh`: confidence threshold (default 0.5)  
- `--resolution`: display/record size `WxH` (optional resize)  
- `--timeout-ms`: timeout for reading frames  
- `--every-n`: sample 1 out of every N frames  
- `--max-frames`: stop after this many frames (0 = all)  
- `--fps`: FPS for recording (if `--record` is used)  
- `--record`: MP4 file output path (optional)

**Controls**
- `q` → quit  
- `s` → pause  
- `p` → save snapshot `capture.png`

---

## Notes and Best Practices

- **Color BGR/RGB**: RealSense color is typically **RGB8**, while OpenCV uses **BGR**. The scripts handle conversion automatically.  
- **Depth alignment**: Aligning depth to color is useful for dataset generation — disable with `--no-align` if not needed.  
- **Timeouts**: If timeouts occur, verify your `.bag` includes a **color stream** and adjust `--timeout-ms`.  
- **Depth scale**: If the device exposes `depth_scale`, thresholds (`--min-m`, `--max-m`) apply in **meters**.  
- **Performance**: Use `--every-n` to subsample frames for faster previews or inference.  
- **GIFs**: Ideal for short clips; use MP4 for long recordings.

---

## Quick Examples

```bash
# 1) Extract frames every 10 steps (max 300), depth range 0.2–2.0 m
python extract_rs_frames.py my_scene.bag --every-n 10 --max-frames 300 --min-m 0.2 --max-m 2.0

# 2) Create a timelapse MP4 at 60 fps using 1/5 frames
python convert_bag.py --video my_scene.bag --output color --ext mp4 --fps 60 --every-n 5

# 3) Run YOLO detection and record results to 1080p MP4
python rs_yolo_detect.py --model yolov8n.pt --bag my_scene.bag --resolution 1920x1080 --record detections.mp4
```

---

## Troubleshooting

- **“Color frames not written” / No color output** → Ensure the `.bag` includes a **color stream**.  
- **`ImportError: pyrealsense2`** → Install the correct wheel for your Python/OS.  
- **Display freezes** → In *headless* systems, disable display and use MP4/GIF output only.  
- **Low FPS with YOLO** → Reduce resolution (`--resolution`), increase sampling (`--every-n`), or use a smaller model (`yolov8n.pt`).


## Acknowledgements

- Intel **RealSense** (`librealsense`, `pyrealsense2`)  
- **OpenCV**, **NumPy**, **ImageIO**  
- **Ultralytics YOLO**

# Author
- Juan Manuel Jiménez

# Credits
Part of the YOLO detection script (`realsense_yolo_detect.py`) was inspired by work from  
**[Edje Electronics – Train and Deploy YOLO Models](https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models)**.  

Special thanks to Edje Electronics for providing an excellent open-source reference for integrating **Ultralytics YOLO** models in practical Python workflows.
