from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .realsense_io import save_session
from .types import Intrinsics, RGBDFrame


def create_demo_session(session_dir: str | Path) -> Path:
    session_dir = Path(session_dir)
    width, height = 640, 480
    color = np.full((height, width, 3), (238, 235, 226), dtype=np.uint8)

    # Cabinet frame and shelves.
    cv2.rectangle(color, (55, 40), (585, 440), (120, 95, 70), 8)
    for y in [160, 285]:
        cv2.line(color, (58, y), (582, y), (110, 85, 60), 8)
    cv2.rectangle(color, (65, 48), (575, 432), (226, 220, 205), -1)
    for y in [160, 285]:
        cv2.line(color, (65, y), (575, y), (105, 80, 55), 8)

    objects = [
        {"label": "cup", "category": "Tableware", "score": 0.93, "box": [105, 92, 165, 150], "rgb": (70, 160, 230)},
        {"label": "spoon", "category": "Cutlery", "score": 0.88, "box": [245, 112, 335, 138], "rgb": (180, 180, 190)},
        {"label": "cereal", "category": "Breakfast", "score": 0.91, "box": [430, 78, 510, 150], "rgb": (230, 170, 70)},
        {"label": "cola", "category": "Soft Drink", "score": 0.89, "box": [130, 208, 185, 275], "rgb": (210, 45, 45)},
        {"label": "mug", "category": "Tableware", "score": 0.86, "box": [390, 225, 450, 276], "rgb": (70, 180, 120)},
        {"label": "unknown toy", "category": None, "score": 1.0, "box": [240, 340, 330, 410], "rgb": (140, 90, 210)},
    ]
    for item in objects:
        x1, y1, x2, y2 = item["box"]
        cv2.rectangle(color, (x1, y1), (x2, y2), item["rgb"], -1)
        cv2.rectangle(color, (x1, y1), (x2, y2), (35, 35, 35), 2)

    depth_m = np.full((height, width), 1.15, dtype=np.float32)
    depth_m[:, :55] = 0.0
    depth_m[:, 585:] = 0.0
    depth_m[:40, :] = 0.0
    depth_m[440:, :] = 0.0
    depth_m[155:166, 55:585] = 0.92
    depth_m[280:291, 55:585] = 0.98
    for item in objects:
        x1, y1, x2, y2 = item["box"]
        depth_m[y1:y2, x1:x2] = 0.82
    depth_raw = (depth_m / 0.001).astype(np.uint16)

    intr = Intrinsics(width=width, height=height, fx=610.0, fy=610.0, ppx=width / 2.0, ppy=height / 2.0, depth_scale=0.001)
    frame = RGBDFrame(color_rgb=color, depth_raw=depth_raw, intrinsics=intr)
    save_session(frame, session_dir)

    (session_dir / "detections.json").write_text(json.dumps(objects, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_dir
