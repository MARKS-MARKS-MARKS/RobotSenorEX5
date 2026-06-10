from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import Intrinsics, RGBDFrame


class RealSenseCamera:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise RuntimeError("pyrealsense2 is not installed. Run scripts/setup_env.sh first.") from exc

        cam_cfg = self.cfg["camera"]
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, int(cam_cfg["width"]), int(cam_cfg["height"]), rs.format.z16, int(cam_cfg["fps"]))
        config.enable_stream(rs.stream.color, int(cam_cfg["width"]), int(cam_cfg["height"]), rs.format.bgr8, int(cam_cfg["fps"]))
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as exc:
            message = str(exc)
            if "No device connected" in message or "No device" in message:
                raise RuntimeError(
                    "No RealSense device connected. Replug the D435i, check USB 3.x connection, then run "
                    "`bash scripts/check_env.sh` before starting live mode again."
                ) from exc
            if "Device or resource busy" in message or "errno=16" in message:
                raise RuntimeError(
                    "RealSense device is busy. Close other camera users first, for example an existing "
                    "`python -m exp5 --mode live`, realsense-viewer, browser camera tab, or video app."
                ) from exc
            raise
        self.align = rs.align(rs.stream.color)
        sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(sensor.get_depth_scale())

        # Let auto exposure settle before the first saved/processed frame.
        for _ in range(10):
            self.pipeline.wait_for_frames()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def get_frame(self) -> RGBDFrame:
        if self.pipeline is None or self.align is None:
            raise RuntimeError("RealSenseCamera.start() must be called before get_frame().")
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to receive aligned color/depth frames from RealSense.")

        color_bgr = np.asanyarray(color_frame.get_data())
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth_raw = np.asanyarray(depth_frame.get_data()).copy()
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        intrinsics = Intrinsics(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            ppx=float(intr.ppx),
            ppy=float(intr.ppy),
            depth_scale=self.depth_scale,
        )
        return RGBDFrame(color_rgb=color_rgb, depth_raw=depth_raw, intrinsics=intrinsics, timestamp_ms=time.time() * 1000.0)


def save_session(frame: RGBDFrame, session_dir: str | Path) -> Path:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    color_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(session_dir / "color.png"), color_bgr)
    cv2.imwrite(str(session_dir / "depth.png"), frame.depth_raw)
    (session_dir / "intrinsics.json").write_text(
        json.dumps(frame.intrinsics.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (session_dir / "meta.json").write_text(
        json.dumps({"timestamp_ms": frame.timestamp_ms}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return session_dir


def load_session(session_dir: str | Path) -> RGBDFrame:
    session_dir = Path(session_dir)
    color_path = session_dir / "color.png"
    depth_path = session_dir / "depth.png"
    intr_path = session_dir / "intrinsics.json"
    if not color_path.exists() or not depth_path.exists() or not intr_path.exists():
        raise FileNotFoundError(f"Session must contain color.png, depth.png and intrinsics.json: {session_dir}")

    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color_bgr is None or depth_raw is None:
        raise RuntimeError(f"Failed to read RGB-D files from session: {session_dir}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    intrinsics = Intrinsics.from_dict(json.loads(intr_path.read_text(encoding="utf-8")))
    return RGBDFrame(color_rgb=color_rgb, depth_raw=depth_raw, intrinsics=intrinsics)
