from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exp5.config import load_config
from exp5.geometry import depth_m, detect_shelves, fill_depth_holes, ransac_shelf_plane_debug, resolve_cabinet_roi_with_mask
from exp5.realsense_io import load_session
from exp5.types import Shelf


COLORS = [
    (35, 120, 255),
    (0, 180, 120),
    (255, 170, 40),
    (190, 90, 255),
    (40, 220, 240),
    (255, 90, 90),
]

FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/arialbd.ttf")
_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
    try:
        font = ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = (32, 32, 32),
    bold: bool = False,
) -> None:
    size = max(10, int(round(scale * 30)))
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    rgb_color = (int(color[2]), int(color[1]), int(color[0]))
    draw.text(org, text, font=_font(size, bold), fill=rgb_color)
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _text_size(text: str, scale: float, bold: bool = False) -> tuple[int, int]:
    size = max(10, int(round(scale * 30)))
    font = _font(size, bold)
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])


def _put_label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = (30, 30, 30),
    bg: tuple[int, int, int] = (245, 248, 244),
    bold: bool = False,
) -> None:
    tw, th = _text_size(text, scale, bold)
    x, y = org
    pad_x, pad_y = 4, 3
    cv2.rectangle(img, (x - pad_x, y - pad_y), (x + tw + pad_x, y + th + pad_y), bg, -1)
    cv2.rectangle(img, (x - pad_x, y - pad_y), (x + tw + pad_x, y + th + pad_y), (210, 218, 210), 1)
    _put_text(img, text, org, scale, color, bold)


def _effective_shelf_roi(shelves: list[Shelf], fallback_polygon: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    if not shelves:
        xs = [point[0] for point in fallback_polygon]
        ys = [point[1] for point in fallback_polygon]
        return min(xs), min(ys), max(xs), max(ys)
    x1 = min(shelf.box[0] for shelf in shelves)
    y1 = min(shelf.box[1] for shelf in shelves)
    x2 = max(shelf.box[2] for shelf in shelves)
    y2 = max(shelf.box[3] for shelf in shelves)
    return x1, y1, x2, y2


def _final_internal_boundaries(shelves: list[Shelf], cfg: dict[str, object]) -> list[int]:
    if len(shelves) < 2:
        return []
    margin = int(cfg["geometry"].get("shelf_margin_px", 8))
    rows: list[int] = []
    for upper, lower in zip(shelves, shelves[1:]):
        rows.append(int(round((upper.box[3] + margin + lower.box[1] - margin) * 0.5)))
    return rows


def _select_planes_for_boundaries(debug: dict[str, object], boundary_rows: list[int], max_distance_px: int = 36) -> list[dict[str, object]]:
    planes = debug["planes"]
    if not isinstance(planes, list):
        return []
    selected: list[dict[str, object]] = []
    used: set[int] = set()
    for boundary in boundary_rows:
        best_index = -1
        best_distance = max_distance_px + 1
        for idx, plane in enumerate(planes):
            if idx in used:
                continue
            distance = abs(int(plane["row"]) - boundary)
            if distance < best_distance:
                best_index = idx
                best_distance = distance
        if best_index >= 0 and best_distance <= max_distance_px:
            used.add(best_index)
            selected.append(planes[best_index])
    return selected


def _load_result_geometry(result_path: Path) -> tuple[tuple[int, int, int, int] | None, list[Shelf]]:
    if not result_path.exists():
        return None, []
    data = json.loads(result_path.read_text(encoding="utf-8"))
    roi_box = data.get("map_summary", {}).get("roi_box")
    roi = tuple(int(v) for v in roi_box) if isinstance(roi_box, list) and len(roi_box) == 4 else None
    shelves: list[Shelf] = []
    for item in data.get("shelves", []):
        box = item.get("box")
        if isinstance(box, list) and len(box) == 4:
            shelves.append(
                Shelf(
                    level=int(item.get("level", len(shelves) + 1)),
                    box=tuple(int(v) for v in box),
                    median_depth_m=float(item.get("median_depth_m", 0.0)),
                    plane_hint=str(item.get("plane_hint", "result shelf")),
                )
            )
    return roi, shelves


def _rect_polygon(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _draw_image_panel(
    canvas: np.ndarray,
    color_rgb: np.ndarray,
    sample_polygon: list[tuple[int, int]],
    debug: dict[str, object],
    shelves: list[Shelf],
    selected_planes: list[dict[str, object]],
    boundary_rows: list[int],
    effective_roi: tuple[int, int, int, int],
) -> None:
    x0, y0 = 40, 35
    height, width = color_rgb.shape[:2]
    ex1, ey1, ex2, ey2 = effective_roi
    crop_margin_x = 70
    crop_margin_y = 18
    cx1 = max(0, ex1 - crop_margin_x)
    cy1 = max(0, ey1 - crop_margin_y)
    cx2 = min(width - 1, ex2 + crop_margin_x)
    cy2 = min(height - 1, ey2 + crop_margin_y)

    panel = cv2.cvtColor(color_rgb[cy1:cy2, cx1:cx2], cv2.COLOR_RGB2BGR).copy()

    def shift_point(point: tuple[int, int]) -> tuple[int, int]:
        return int(point[0] - cx1), int(point[1] - cy1)

    shifted_polygon = [shift_point(point) for point in sample_polygon]
    cv2.polylines(panel, [np.asarray(shifted_polygon, dtype=np.int32)], True, (165, 175, 175), 1)

    shifted_roi = (ex1 - cx1, ey1 - cy1, ex2 - cx1, ey2 - cy1)
    rex1, rey1, rex2, rey2 = shifted_roi
    cv2.rectangle(panel, (rex1, rey1), (rex2, rey2), (0, 230, 230), 2)
    _put_text(panel, "effective ROI", (rex1 + 6, max(8, rey1 - 17)), 0.42, (0, 130, 120), bold=True)

    overlay = panel.copy()
    shelf_colors = [(225, 245, 235), (235, 242, 255), (245, 238, 255)]
    for idx, shelf in enumerate(shelves):
        sx1, sy1, sx2, sy2 = shelf.box
        rsx1, rsy1, rsx2, rsy2 = sx1 - cx1, sy1 - cy1, sx2 - cx1, sy2 - cy1
        cv2.rectangle(overlay, (rsx1, rsy1), (rsx2, rsy2), shelf_colors[idx % len(shelf_colors)], -1)
        cv2.rectangle(panel, (rsx1, rsy1), (rsx2, rsy2), (0, 145, 95), 2)
        _put_text(panel, f"Shelf {shelf.level}", (rsx1 + 8, rsy1 + 16), 0.48, (0, 105, 75), bold=True)
    panel = cv2.addWeighted(overlay, 0.22, panel, 0.78, 0)

    rows = debug["rows"]
    cols = debug["cols"]
    if isinstance(rows, np.ndarray) and isinstance(cols, np.ndarray):
        for idx, plane in enumerate(selected_planes):
            color = COLORS[idx % len(COLORS)]
            inliers = plane["inliers"]
            if not isinstance(inliers, np.ndarray):
                continue
            boundary_row = boundary_rows[idx] if idx < len(boundary_rows) else int(plane["row"])
            band_px = 14
            support = inliers[np.abs(rows[inliers] - boundary_row) <= band_px]
            if support.size == 0:
                support = inliers
            for point_idx in support[:: max(1, support.size // 700)]:
                px = int(cols[point_idx] - cx1)
                py = int(rows[point_idx] - cy1)
                if 0 <= px < panel.shape[1] and 0 <= py < panel.shape[0]:
                    cv2.circle(panel, (px, py), 1, color, -1)
            local_boundary = boundary_row - cy1
            cv2.line(panel, (rex1, local_boundary), (rex2, local_boundary), color, 3)
            label_y = max(8, local_boundary - 21)
            _put_text(panel, f"RANSAC support {idx + 1}", (rex1 + 8, label_y), 0.42, color, bold=True)

    canvas[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + panel.shape[1] + 1, y0 + panel.shape[0] + 1), (210, 220, 220), 2)


def _project_xyz(
    points: np.ndarray,
    rect: tuple[int, int, int, int],
    center: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = rect
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    p = points.astype(np.float32).copy()
    p[:, 1] = -p[:, 1]
    p -= center.reshape(1, 3)

    yaw = np.deg2rad(-35.0)
    pitch = np.deg2rad(20.0)
    x1 = np.cos(yaw) * p[:, 0] + np.sin(yaw) * p[:, 2]
    z1 = -np.sin(yaw) * p[:, 0] + np.cos(yaw) * p[:, 2]
    y1 = np.cos(pitch) * p[:, 1] - np.sin(pitch) * z1

    px = x + w * 0.52 + x1 * scale
    py = y + h * 0.53 - y1 * scale
    return px.astype(np.int32), py.astype(np.int32)


def _cloud_projection_params(points: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[np.ndarray, float]:
    if points.shape[0] == 0:
        return np.zeros(3, dtype=np.float32), 1.0
    robust = np.column_stack(
        (
            np.clip(points[:, 0], np.percentile(points[:, 0], 2), np.percentile(points[:, 0], 98)),
            -np.clip(points[:, 1], np.percentile(points[:, 1], 2), np.percentile(points[:, 1], 98)),
            np.clip(points[:, 2], np.percentile(points[:, 2], 2), np.percentile(points[:, 2], 98)),
        )
    ).astype(np.float32)
    center = robust.mean(axis=0)
    span = np.maximum(robust.max(axis=0) - robust.min(axis=0), 1e-3)
    scale = min(rect[2] / max(1e-3, span[0] + span[2] * 0.55), rect[3] / max(1e-3, span[1] + span[2] * 0.35)) * 0.82
    return center, float(scale)


def _draw_cloud_panel(
    canvas: np.ndarray,
    debug: dict[str, object],
    shelves: list[Shelf],
    selected_planes: list[dict[str, object]],
    boundary_rows: list[int],
) -> None:
    rect = (650, 60, 400, 330)
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (245, 248, 247), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 214, 212), 2)
    _put_text(canvas, "3D voxel cloud and shelf planes", (x, y - 20), 0.56, (30, 30, 30), bold=True)

    points = debug["points"]
    rows = debug["rows"]
    cols = debug["cols"]
    if not isinstance(points, np.ndarray) or points.shape[0] == 0:
        _put_text(canvas, "No valid points", (x + 112, y + 170), 0.8, (80, 80, 80))
        return

    center, scale = _cloud_projection_params(points, rect)
    px, py = _project_xyz(points, rect, center, scale)
    for idx in range(0, px.size, max(1, px.size // 2500)):
        cv2.circle(canvas, (int(px[idx]), int(py[idx])), 1, (178, 184, 184), -1)

    shelf_fill_colors = [(170, 218, 190), (170, 202, 240), (207, 185, 238)]
    if isinstance(rows, np.ndarray) and isinstance(cols, np.ndarray):
        for shelf_idx, shelf in enumerate(shelves):
            sx1, sy1, sx2, sy2 = shelf.box
            mask = (cols >= sx1) & (cols <= sx2) & (rows >= sy1) & (rows <= sy2)
            shelf_indices = np.where(mask)[0]
            if shelf_indices.size == 0:
                continue
            color = shelf_fill_colors[shelf_idx % len(shelf_fill_colors)]
            step = max(1, shelf_indices.size // 900)
            for point_idx in shelf_indices[::step]:
                cv2.circle(canvas, (int(px[point_idx]), int(py[point_idx])), 1, color, -1)
            label_point = points[shelf_indices].mean(axis=0, keepdims=True)
            lx, ly = _project_xyz(label_point, rect, center, scale)
            _put_text(canvas, f"L{shelf.level}", (int(lx[0]) + 6, int(ly[0]) - 4), 0.5, (30, 95, 75))

    for idx, plane in enumerate(selected_planes):
        color = COLORS[idx % len(COLORS)]
        inliers = plane["inliers"]
        centroid = plane["centroid"]
        normal = plane["normal"]
        if (
            not isinstance(inliers, np.ndarray)
            or not isinstance(centroid, np.ndarray)
            or not isinstance(normal, np.ndarray)
            or not isinstance(rows, np.ndarray)
        ):
            continue
        boundary_row = boundary_rows[idx] if idx < len(boundary_rows) else int(plane["row"])
        band_px = 14
        support = inliers[np.abs(rows[inliers] - boundary_row) <= band_px]
        if support.size == 0:
            support = inliers
        in_px, in_py = _project_xyz(points[support], rect, center, scale)
        for point_idx in range(0, in_px.size, max(1, in_px.size // 900)):
            cv2.circle(canvas, (int(in_px[point_idx]), int(in_py[point_idx])), 2, color, -1)

        support_centroid = points[support].mean(axis=0).astype(np.float32)
        c_px, c_py = _project_xyz(support_centroid.reshape(1, 3), rect, center, scale)
        normal_end = (support_centroid + normal * 0.18).reshape(1, 3)
        n_px, n_py = _project_xyz(normal_end, rect, center, scale)
        cv2.arrowedLine(canvas, (int(c_px[0]), int(c_py[0])), (int(n_px[0]), int(n_py[0])), color, 2, tipLength=0.25)

    origin = center.copy()
    origin[0] -= 0.33
    origin[1] -= 0.42
    origin[2] -= 0.08
    axes = [
        ("X", np.array([0.22, 0.0, 0.0], dtype=np.float32), (50, 80, 220)),
        ("Y", np.array([0.0, 0.22, 0.0], dtype=np.float32), (40, 150, 70)),
        ("Z", np.array([0.0, 0.0, 0.22], dtype=np.float32), (210, 120, 40)),
    ]
    o_px, o_py = _project_xyz(origin.reshape(1, 3), rect, center, scale)
    for label, delta, color in axes:
        end = origin + delta
        e_px, e_py = _project_xyz(end.reshape(1, 3), rect, center, scale)
        cv2.arrowedLine(canvas, (int(o_px[0]), int(o_py[0])), (int(e_px[0]), int(e_py[0])), color, 2, tipLength=0.25)
        _put_text(canvas, label, (int(e_px[0]) + 3, int(e_py[0]) - 3), 0.44, color)


def _draw_info_panel(
    canvas: np.ndarray,
    shelves: list[Shelf],
    selected_planes: list[dict[str, object]],
    boundary_rows: list[int],
) -> None:
    x, y = 650, 435
    _put_text(canvas, "Selected RANSAC planes for 3 shelves", (x, y), 0.56, (30, 30, 30), bold=True)
    _put_text(canvas, f"Final shelf regions: {len(shelves)}", (x, y + 28), 0.48, (55, 55, 55))
    if not selected_planes:
        _put_text(canvas, "No plane matched final internal boundaries.", (x, y + 62), 0.55, (80, 80, 80))
        return
    for idx, plane in enumerate(selected_planes):
        color = COLORS[idx % len(COLORS)]
        normal = plane["normal"]
        target = boundary_rows[idx] if idx < len(boundary_rows) else int(plane["row"])
        line = f"{idx + 1}. target={target}, row={int(plane['row'])}, inliers={int(plane['inlier_count'])}, ny={float(normal[1]):+.2f}"
        text_y = y + 62 + idx * 31
        cv2.circle(canvas, (x + 9, text_y + 8), 7, color, -1)
        _put_text(canvas, line, (x + 25, text_y), 0.44, (38, 48, 54))


def build_visualization(session: Path, config: Path | None, output: Path, result: Path | None) -> Path:
    cfg = load_config(config)
    frame = load_session(session)
    depth = fill_depth_holes(depth_m(frame.depth_raw, frame.intrinsics), cfg)
    roi, polygon, roi_mask = resolve_cabinet_roi_with_mask(frame.color_rgb, depth, cfg)

    result_roi, result_shelves = _load_result_geometry(result) if result else (None, [])
    if result_roi is not None:
        roi = result_roi
        polygon = _rect_polygon(roi)
        roi_mask = np.zeros(depth.shape[:2], dtype=bool)
        x1, y1, x2, y2 = roi
        roi_mask[y1:y2, x1:x2] = True

    debug = ransac_shelf_plane_debug(depth, frame.intrinsics, cfg, roi, roi_mask)
    shelves = result_shelves or detect_shelves(frame.color_rgb, depth, frame.intrinsics, cfg, roi, roi_mask)
    effective_roi = _effective_shelf_roi(shelves, polygon)
    boundary_rows = _final_internal_boundaries(shelves, cfg)
    selected_planes = _select_planes_for_boundaries(debug, boundary_rows)

    canvas = np.full((670, 1100, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (1100, 670), (252, 252, 252), -1)
    _draw_image_panel(canvas, frame.color_rgb, polygon, debug, shelves, selected_planes, boundary_rows, effective_roi)
    _draw_cloud_panel(canvas, debug, shelves, selected_planes, boundary_rows)
    _draw_info_panel(canvas, shelves, selected_planes, boundary_rows)

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "ransac_planes.png"
    cv2.imwrite(str(out_path), canvas)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize shelf RANSAC plane segmentation.")
    parser.add_argument("--session", type=Path, default=Path("data/sessions/cabinet_01"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/plane_ransac_cabinet01"))
    parser.add_argument("--result", type=Path, default=Path("outputs/cabinet01_pointcloud_shelf/result.json"))
    args = parser.parse_args()
    path = build_visualization(args.session, args.config, args.output, args.result)
    print(path)


if __name__ == "__main__":
    main()
