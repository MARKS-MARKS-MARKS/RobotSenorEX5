from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import Box, deproject_pixel, valid_depth_mask
from .types import PipelineResult, RGBDFrame


@dataclass
class SceneMap:
    points_m: np.ndarray
    colors_rgb: np.ndarray
    roi_box: Box

    def summary(self) -> dict[str, Any]:
        if self.points_m.size == 0:
            return {
                "point_count": 0,
                "roi_box": list(self.roi_box),
                "bounds_m": None,
            }
        mins = self.points_m.min(axis=0)
        maxs = self.points_m.max(axis=0)
        return {
            "point_count": int(self.points_m.shape[0]),
            "roi_box": list(self.roi_box),
            "bounds_m": {
                "x": [float(mins[0]), float(maxs[0])],
                "y": [float(mins[1]), float(maxs[1])],
                "z": [float(mins[2]), float(maxs[2])],
            },
        }


def build_scene_map(
    frame: RGBDFrame,
    depth_m: np.ndarray,
    roi: Box,
    cfg: dict[str, Any],
    roi_mask: np.ndarray | None = None,
) -> SceneMap | None:
    map_cfg = cfg.get("mapping", {})
    if not bool(map_cfg.get("enabled", True)):
        return None

    x1, y1, x2, y2 = roi
    stride = max(1, int(map_cfg.get("stride_px", 3)))
    ys = np.arange(y1, y2, stride, dtype=np.int32)
    xs = np.arange(x1, x2, stride, dtype=np.int32)
    if ys.size == 0 or xs.size == 0:
        return SceneMap(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), roi)

    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth_m[grid_y, grid_x]
    mask = valid_depth_mask(z, cfg)
    if roi_mask is not None:
        mask &= roi_mask[grid_y, grid_x]
    if not np.any(mask):
        return SceneMap(np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), roi)

    px = grid_x[mask].astype(np.float32)
    py = grid_y[mask].astype(np.float32)
    pz = z[mask].astype(np.float32)
    intr = frame.intrinsics
    points = np.column_stack(
        (
            (px - intr.ppx) * pz / intr.fx,
            (py - intr.ppy) * pz / intr.fy,
            pz,
        )
    ).astype(np.float32)
    colors = frame.color_rgb[grid_y[mask], grid_x[mask]].astype(np.uint8)

    max_points = int(map_cfg.get("max_points", 70000))
    if points.shape[0] > max_points:
        sample_idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int32)
        points = points[sample_idx]
        colors = colors[sample_idx]

    return SceneMap(points, colors, roi)


def export_scene_ply(scene_map: SceneMap, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = scene_map.points_m
    colors = scene_map.colors_rgb
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def draw_scene_map(
    scene_map: SceneMap | None,
    result: PipelineResult,
    cfg: dict[str, Any],
) -> np.ndarray:
    map_cfg = cfg.get("mapping", {})
    width = int(map_cfg.get("render_width", 640))
    height = int(map_cfg.get("render_height", 480))
    canvas = np.full((height, width, 3), (28, 30, 32), dtype=np.uint8)
    cv2.putText(canvas, "RGB-D 3D Map", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)

    if scene_map is None or scene_map.points_m.size == 0:
        cv2.putText(canvas, "No valid depth points", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
        return canvas

    projected = _project_points(scene_map.points_m, width, height)
    if projected is None:
        return canvas
    uv, order, projector = projected
    colors_bgr = scene_map.colors_rgb[order][:, ::-1]
    canvas[uv[:, 1], uv[:, 0]] = colors_bgr

    _draw_axis_hint(canvas)
    _draw_result_markers(canvas, result, projector)
    cv2.putText(
        canvas,
        f"points: {scene_map.points_m.shape[0]}",
        (18, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _project_points(
    points: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, Any] | None:
    pts = points.astype(np.float32)
    x = pts[:, 0]
    y = -pts[:, 1]
    z = pts[:, 2]
    view_x = x + 0.35 * z
    view_y = y + 0.10 * z
    min_x, max_x = float(view_x.min()), float(view_x.max())
    min_y, max_y = float(view_y.min()), float(view_y.max())
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    margin = 42
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)

    def projector(point: tuple[float, float, float]) -> tuple[int, int]:
        px, py, pz = point
        vx = px + 0.35 * pz
        vy = -py + 0.10 * pz
        u = int(round((vx - min_x) * scale + margin))
        v = int(round(height - ((vy - min_y) * scale + margin)))
        return max(0, min(width - 1, u)), max(0, min(height - 1, v))

    u = np.clip(((view_x - min_x) * scale + margin).round().astype(np.int32), 0, width - 1)
    v = np.clip((height - ((view_y - min_y) * scale + margin)).round().astype(np.int32), 0, height - 1)
    order = np.argsort(z)[::-1]
    uv = np.column_stack((u[order], v[order]))
    return uv, order, projector


def _draw_axis_hint(canvas: np.ndarray) -> None:
    base = (canvas.shape[1] - 110, canvas.shape[0] - 52)
    cv2.arrowedLine(canvas, base, (base[0] + 52, base[1]), (80, 220, 80), 2, tipLength=0.25)
    cv2.arrowedLine(canvas, base, (base[0], base[1] - 42), (80, 80, 230), 2, tipLength=0.25)
    cv2.arrowedLine(canvas, base, (base[0] + 35, base[1] + 22), (230, 180, 70), 2, tipLength=0.25)
    cv2.putText(canvas, "X", (base[0] + 58, base[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Y", (base[0] - 5, base[1] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Z", (base[0] + 40, base[1] + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 180, 70), 1, cv2.LINE_AA)


def _draw_result_markers(canvas: np.ndarray, result: PipelineResult, projector: Any) -> None:
    for det in result.objects:
        point = det.contact_center_m or det.center_m
        if point is None:
            continue
        u, v = projector(point)
        cv2.circle(canvas, (u, v), 6, (40, 230, 70), -1)
        cv2.putText(canvas, det.track_id or det.category or det.label, (u + 8, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 230, 70), 1, cv2.LINE_AA)

    for space in result.empty_spaces:
        u, v = projector(space.center_m)
        cv2.circle(canvas, (u, v), 5, (230, 120, 20), 2)
        cv2.putText(canvas, space.track_id or f"L{space.shelf_level}", (u + 7, v + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 120, 20), 1, cv2.LINE_AA)

    rec = result.recommendation
    if rec and rec.empty_space:
        u, v = projector(rec.empty_space.center_m)
        cv2.circle(canvas, (u, v), 10, (0, 220, 255), 2)
