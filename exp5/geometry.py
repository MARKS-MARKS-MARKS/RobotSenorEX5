from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .types import Detection2D, EmptySpace, Intrinsics, Obstacle, PlacementRecommendation, Shelf


Box = tuple[int, int, int, int]


def depth_m(depth_raw: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    depth = depth_raw.astype(np.float32)
    if depth_raw.dtype != np.float32 and depth_raw.dtype != np.float64:
        depth *= float(intrinsics.depth_scale)
    return depth


def valid_depth_mask(depth: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    cam = cfg["camera"]
    return (depth > float(cam["depth_min_m"])) & (depth < float(cam["depth_max_m"]))


def fill_depth_holes(depth: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    kernel = int(cfg["geometry"].get("depth_hole_fill_kernel", 0))
    if kernel <= 1:
        return depth
    if kernel % 2 == 0:
        kernel += 1
    filled = depth.astype(np.float32).copy()
    valid = valid_depth_mask(filled, cfg)
    if valid.all() or not valid.any():
        return filled
    safe = filled.copy()
    safe[~valid] = float(np.median(safe[valid]))
    blurred = cv2.medianBlur(safe, kernel)
    valid_count = cv2.boxFilter(valid.astype(np.float32), -1, (kernel, kernel), normalize=False)
    local_hole = (~valid) & (valid_count >= (kernel * kernel * 0.45))
    filled[local_hole] = blurred[local_hole]
    return filled


def deproject_pixel(u: float, v: float, z: float, intr: Intrinsics) -> tuple[float, float, float]:
    x = (float(u) - intr.ppx) * z / intr.fx
    y = (float(v) - intr.ppy) * z / intr.fy
    return float(x), float(y), float(z)


def region_median_depth(depth: np.ndarray, box: Box, cfg: dict[str, Any]) -> float | None:
    x1, y1, x2, y2 = box
    crop = depth[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    mask = valid_depth_mask(crop, cfg)
    if crop.size == 0 or mask.mean() < float(cfg["geometry"]["min_valid_depth_ratio"]):
        return None
    return float(np.median(crop[mask]))


def _clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width - 1, int(x2)))
    y2 = max(0, min(height - 1, int(y2)))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def _box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


def _intersection_area(a: Box, b: Box) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _box_from_norm(norm: list[float] | tuple[float, ...], width: int, height: int) -> Box:
    x1, y1, x2, y2 = norm
    return _clamp_box((round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)), width, height)


def _polygon_from_norm(points: list[list[float]] | tuple[tuple[float, float], ...], width: int, height: int) -> list[tuple[int, int]]:
    polygon: list[tuple[int, int]] = []
    for point in points:
        if len(point) != 2:
            continue
        x = max(0, min(width - 1, int(round(float(point[0]) * width))))
        y = max(0, min(height - 1, int(round(float(point[1]) * height))))
        polygon.append((x, y))
    return polygon


def _box_from_polygon(points: list[tuple[int, int]], width: int, height: int) -> Box:
    if len(points) < 3:
        return 0, 0, width - 1, height - 1
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return _clamp_box((min(xs), min(ys), max(xs), max(ys)), width, height)


def roi_mask_from_polygon(shape: tuple[int, int], polygon: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
    return mask > 0


def _auto_roi_polygon_from_color_interior(color_rgb: np.ndarray, cfg: dict[str, Any], fallback: Box) -> list[tuple[int, int]] | None:
    geom = cfg["geometry"]
    if not bool(geom.get("roi_color_auto_enabled", True)):
        return None

    height, width = color_rgb.shape[:2]
    fx1, fy1, fx2, fy2 = fallback
    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    roi_gray = gray[fy1:fy2, fx1:fx2]
    if roi_gray.size == 0:
        return None

    threshold = float(np.percentile(roi_gray, float(geom.get("roi_dark_percentile", 56))))
    threshold = max(float(geom.get("roi_dark_threshold_min", 52)), min(float(geom.get("roi_dark_threshold_max", 82)), threshold))
    dark = (gray <= threshold).astype(np.uint8)

    restricted = np.zeros_like(dark)
    restricted[fy1:fy2, fx1:fx2] = dark[fy1:fy2, fx1:fx2]
    restricted = cv2.morphologyEx(restricted, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8))
    restricted = cv2.morphologyEx(restricted, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(restricted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    fallback_area = max(1, (fx2 - fx1) * (fy2 - fy1))
    min_area = fallback_area * float(geom.get("roi_dark_min_area_ratio", 0.25))
    candidates: list[tuple[float, Box]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.30 or h < height * 0.45:
            continue
        fill = area / float(max(1, w * h))
        center_penalty = abs((x + w * 0.5) - width * 0.55) / width
        score = area * max(0.2, fill) * (1.0 - min(0.8, center_penalty))
        candidates.append((score, (x, y, x + w, y + h)))

    if not candidates:
        return None

    _, box = max(candidates, key=lambda item: item[0])
    x1, y1, x2, y2 = box
    crop = restricted[y1:y2, x1:x2] > 0
    if crop.size:
        col_ratio = crop.mean(axis=0)
        row_ratio = crop.mean(axis=1)
        cols = np.where(col_ratio >= float(geom.get("roi_dark_column_min_ratio", 0.18)))[0]
        rows = np.where(row_ratio >= float(geom.get("roi_dark_row_min_ratio", 0.16)))[0]
        if cols.size:
            x1 += int(cols[0])
            x2 = x1 + int(cols[-1] - cols[0] + 1)
        if rows.size:
            y1 += int(rows[0])
            y2 = y1 + int(rows[-1] - rows[0] + 1)

    margin = int(geom.get("roi_inner_margin_px", 3))
    x1, y1, x2, y2 = _clamp_box((x1 + margin, y1 + margin, x2 - margin, y2 - margin), width, height)
    if (x2 - x1) < width * 0.30 or (y2 - y1) < height * 0.45:
        return None
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _auto_roi_polygon_from_depth(depth: np.ndarray, cfg: dict[str, Any], fallback: Box) -> list[tuple[int, int]] | None:
    x1, y1, x2, y2 = fallback
    height, width = depth.shape[:2]
    valid = valid_depth_mask(depth, cfg).astype(np.uint8)
    restricted = np.zeros_like(valid)
    restricted[y1:y2, x1:x2] = valid[y1:y2, x1:x2]
    kernel = np.ones((15, 15), np.uint8)
    restricted = cv2.morphologyEx(restricted, cv2.MORPH_CLOSE, kernel)
    restricted = cv2.morphologyEx(restricted, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(restricted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    fallback_area = max(1, (x2 - x1) * (y2 - y1))
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < fallback_area * 0.20:
        return None

    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, max(6.0, perimeter * 0.025), True)
    if len(approx) < 3:
        return None
    points = [(int(point[0][0]), int(point[0][1])) for point in approx]
    points = [(max(0, min(width - 1, x)), max(0, min(height - 1, y))) for x, y in points]
    if len(points) > 8:
        rect = cv2.minAreaRect(hull)
        points = [(int(x), int(y)) for x, y in cv2.boxPoints(rect)]
    return points


def resolve_cabinet_roi_polygon(color_rgb: np.ndarray, depth: np.ndarray, cfg: dict[str, Any]) -> list[tuple[int, int]]:
    geom = cfg["geometry"]
    height, width = depth.shape[:2]
    poly_px = geom.get("cabinet_roi_poly_px")
    if poly_px:
        points = [(int(point[0]), int(point[1])) for point in poly_px if len(point) == 2]
        return [(max(0, min(width - 1, x)), max(0, min(height - 1, y))) for x, y in points]

    poly_norm = geom.get("cabinet_roi_poly_norm")
    if poly_norm:
        polygon = _polygon_from_norm(poly_norm, width, height)
        if len(polygon) >= 3:
            return polygon

    box = resolve_cabinet_roi(color_rgb, depth, cfg)
    if str(geom.get("cabinet_roi_mode", "auto")).lower() == "auto":
        auto_poly = _auto_roi_polygon_from_color_interior(color_rgb, cfg, box)
        if auto_poly and len(auto_poly) >= 3:
            return auto_poly
        auto_poly = _auto_roi_polygon_from_depth(depth, cfg, box)
        if auto_poly and len(auto_poly) >= 3:
            return auto_poly

    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def resolve_cabinet_roi_with_mask(color_rgb: np.ndarray, depth: np.ndarray, cfg: dict[str, Any]) -> tuple[Box, list[tuple[int, int]], np.ndarray]:
    height, width = depth.shape[:2]
    polygon = resolve_cabinet_roi_polygon(color_rgb, depth, cfg)
    roi = _box_from_polygon(polygon, width, height)
    mask = roi_mask_from_polygon((height, width), polygon)
    return roi, polygon, mask


def resolve_cabinet_roi(color_rgb: np.ndarray, depth: np.ndarray, cfg: dict[str, Any]) -> Box:
    geom = cfg["geometry"]
    height, width = depth.shape[:2]
    mode = str(geom.get("cabinet_roi_mode", "auto")).lower()
    if mode == "full":
        return 0, 0, width - 1, height - 1

    manual = geom.get("cabinet_roi_px")
    if manual:
        return _clamp_box(tuple(int(v) for v in manual), width, height)

    fallback = _box_from_norm(geom.get("cabinet_roi_norm", [0.06, 0.05, 0.94, 0.98]), width, height)
    if mode != "auto":
        return fallback

    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=90,
        minLineLength=int(min(width, height) * float(geom.get("roi_edge_min_ratio", 0.35))),
        maxLineGap=20,
    )
    if lines is None:
        return fallback

    xs: list[int] = []
    ys: list[int] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx <= 7 and dy >= height * 0.30:
            xs.append(int(round((x1 + x2) * 0.5)))
        elif dy <= 7 and dx >= width * 0.30:
            ys.append(int(round((y1 + y2) * 0.5)))

    fx1, fy1, fx2, fy2 = fallback
    if len(xs) >= 2:
        left_candidates = [x for x in xs if x < width * 0.5]
        right_candidates = [x for x in xs if x > width * 0.5]
        if left_candidates and right_candidates:
            fx1 = max(0, min(left_candidates) - 8)
            fx2 = min(width - 1, max(right_candidates) + 8)
    if len(ys) >= 2:
        top_candidates = [y for y in ys if y < height * 0.5]
        bottom_candidates = [y for y in ys if y > height * 0.5]
        if top_candidates and bottom_candidates:
            fy1 = max(0, min(top_candidates) - 8)
            fy2 = min(height - 1, max(bottom_candidates) + 8)

    if (fx2 - fx1) < width * 0.30 or (fy2 - fy1) < height * 0.30:
        return fallback
    return _clamp_box((fx1, fy1, fx2, fy2), width, height)


def filter_detections_to_roi(
    detections: list[Detection2D],
    roi: Box,
    cfg: dict[str, Any],
    roi_mask: np.ndarray | None = None,
) -> list[Detection2D]:
    if not cfg["geometry"].get("filter_detections_to_roi", True):
        return detections
    rx1, ry1, rx2, ry2 = roi
    min_overlap = float(cfg["geometry"].get("detection_roi_min_overlap", 0.60))
    kept: list[Detection2D] = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        area = max(1, (x2 - x1) * (y2 - y1))
        if roi_mask is not None:
            cx_i = max(0, min(roi_mask.shape[1] - 1, int(round(cx))))
            cy_i = max(0, min(roi_mask.shape[0] - 1, int(round(cy))))
            crop = roi_mask[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            overlap = float(crop.mean()) if crop.size else 0.0
            if roi_mask[cy_i, cx_i] and overlap >= min_overlap:
                kept.append(det)
        else:
            ix1, iy1 = max(x1, rx1), max(y1, ry1)
            ix2, iy2 = min(x2, rx2), min(y2, ry2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2 and inter / area >= min_overlap:
                kept.append(det)
    return kept


def filter_detections_to_single_shelf(
    detections: list[Detection2D],
    shelves: list[Shelf],
    cfg: dict[str, Any],
) -> list[Detection2D]:
    if not cfg["geometry"].get("filter_detections_to_single_shelf", True) or not shelves:
        return detections

    margin = int(cfg["geometry"].get("detection_shelf_margin_px", 4))
    kept: list[Detection2D] = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        cy = (y1 + y2) * 0.5
        matching = []
        for shelf in shelves:
            _, sy1, _, sy2 = shelf.box
            if sy1 <= cy <= sy2 and y1 >= sy1 - margin and y2 <= sy2 + margin:
                matching.append(shelf)
        if len(matching) == 1:
            kept.append(det)
    return kept


def attach_3d_to_detections(detections: list[Detection2D], depth: np.ndarray, intr: Intrinsics, cfg: dict[str, Any]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.box
        z = region_median_depth(depth, det.box, cfg)
        if z is None:
            continue
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        det.center_m = deproject_pixel(cx, cy, z, intr)
        contact_y = min(depth.shape[0] - 1, y2)
        det.contact_center_m = deproject_pixel(cx, contact_y, z, intr)
        width_m = abs((x2 - x1) * z / intr.fx)
        height_m = abs((y2 - y1) * z / intr.fy)
        det.size_m = (float(width_m), float(height_m), 0.0)


def _merge_positions(values: list[int], gap: int) -> list[int]:
    if not values:
        return []
    values = sorted(values)
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(float(np.median(group)))) for group in groups]


def _merge_short_boundary_segments(boundaries: list[int], min_h: int, merge_ratio: float) -> list[int]:
    if len(boundaries) <= 2:
        return boundaries
    min_segment = int(round(min_h * merge_ratio))
    boundaries = sorted(set(boundaries))
    while len(boundaries) > 2:
        spans = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        short = [i for i, span in enumerate(spans) if span < min_segment]
        if not short:
            break
        idx = short[0]
        if idx == 0:
            del boundaries[1]
        elif idx == len(spans) - 1:
            del boundaries[-2]
        else:
            left_span = spans[idx - 1]
            right_span = spans[idx + 1] if idx + 1 < len(spans) else left_span
            if left_span >= right_span:
                del boundaries[idx]
            else:
                del boundaries[idx + 1]
    return boundaries


def _shift_internal_boundaries_down(boundaries: list[int], offset: int, min_h: int) -> list[int]:
    if offset <= 0 or len(boundaries) <= 2:
        return boundaries
    shifted = list(boundaries)
    for idx in range(1, len(boundaries) - 1):
        low = shifted[idx - 1] + min_h
        high = boundaries[idx + 1] - min_h
        target = boundaries[idx] + offset
        if low <= high:
            shifted[idx] = max(low, min(high, target))
        else:
            shifted[idx] = min(boundaries[idx + 1] - 1, target)
    return sorted(set(shifted))


def _configured_shelf_boundaries(geom: dict[str, Any], ry1: int, ry2: int, height: int) -> list[int] | None:
    raw_px = geom.get("shelf_boundaries_px")
    if raw_px:
        boundaries = [max(0, min(height - 1, int(round(float(value))))) for value in raw_px]
        if len(boundaries) >= 2:
            return sorted(set(boundaries))

    raw_norm = geom.get("shelf_boundaries_norm")
    if raw_norm:
        boundaries = []
        for value in raw_norm:
            ratio = max(0.0, min(1.0, float(value)))
            boundaries.append(int(round(ry1 + ratio * (ry2 - ry1))))
        if len(boundaries) >= 2:
            return sorted(set(boundaries))
    return None


def _equal_shelf_boundaries(ry1: int, ry2: int, shelf_count: int) -> list[int]:
    shelf_count = max(1, int(shelf_count))
    return [int(round(value)) for value in np.linspace(ry1, ry2, shelf_count + 1)]


def _refine_equal_boundaries_with_color_edges(
    color_rgb: np.ndarray,
    roi: Box,
    boundaries: list[int],
    cfg: dict[str, Any],
    roi_mask: np.ndarray | None = None,
) -> list[int]:
    geom = cfg["geometry"]
    if not bool(geom.get("shelf_refine_equal_boundaries", True)) or len(boundaries) <= 2:
        return boundaries

    rx1, ry1, rx2, ry2 = roi
    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    band = gray[ry1:ry2, rx1:rx2]
    if band.size == 0:
        return boundaries

    if roi_mask is not None:
        mask_band = roi_mask[ry1:ry2, rx1:rx2]
        weighted = np.where(mask_band, band, np.nan)
        with np.errstate(invalid="ignore"):
            row_mean = np.nanmean(weighted, axis=1)
        if np.any(~np.isfinite(row_mean)):
            fallback = float(np.nanmedian(row_mean[np.isfinite(row_mean)])) if np.any(np.isfinite(row_mean)) else float(np.mean(band))
            row_mean[~np.isfinite(row_mean)] = fallback
    else:
        row_mean = band.mean(axis=1)

    kernel = np.ones(11, dtype=np.float32) / 11.0
    smooth = np.convolve(row_mean.astype(np.float32), kernel, mode="same")
    gradient = np.abs(np.gradient(smooth))
    search = int(geom.get("shelf_refine_search_px", 45))
    min_gradient = float(geom.get("shelf_refine_min_gradient", 1.0))
    refined = [boundaries[0]]
    prev = boundaries[0]

    for idx, boundary in enumerate(boundaries[1:-1], start=1):
        next_boundary = boundaries[idx + 1]
        lo = max(ry1, boundary - search, prev + int(geom.get("shelf_min_height_px", 50)))
        hi = min(ry2, boundary + search, next_boundary - int(geom.get("shelf_min_height_px", 50)))
        if hi <= lo:
            refined.append(boundary)
            prev = boundary
            continue

        local = gradient[lo - ry1 : hi - ry1]
        if local.size == 0:
            refined.append(boundary)
            prev = boundary
            continue
        peak_idx = int(np.argmax(local))
        peak_value = float(local[peak_idx])
        chosen = lo + peak_idx if peak_value >= min_gradient else boundary
        refined.append(chosen)
        prev = chosen

    refined.append(boundaries[-1])
    return sorted(set(refined))


def _sample_voxelized_roi_points(
    depth: np.ndarray,
    intr: Intrinsics,
    cfg: dict[str, Any],
    roi: Box,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom = cfg["geometry"]
    rx1, ry1, rx2, ry2 = roi
    stride = max(1, int(geom.get("shelf_point_sample_stride_px", 2)))
    ys, xs = np.mgrid[ry1:ry2:stride, rx1:rx2:stride]
    if ys.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    valid = valid_depth_mask(depth, cfg) & roi_mask
    sampled_valid = valid[ys, xs]
    if not np.any(sampled_valid):
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    rows = ys[sampled_valid].astype(np.int32)
    cols = xs[sampled_valid].astype(np.int32)
    z = depth[rows, cols].astype(np.float32)
    x = (cols.astype(np.float32) - float(intr.ppx)) * z / float(intr.fx)
    y = (rows.astype(np.float32) - float(intr.ppy)) * z / float(intr.fy)
    points = np.column_stack((x, y, z)).astype(np.float32)

    voxel = max(0.003, float(geom.get("voxel_size_m", 0.01)))
    quantized = np.floor(points / voxel).astype(np.int32)
    _, unique_idx = np.unique(quantized, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return points[unique_idx], rows[unique_idx], cols[unique_idx]


def _row_edge_strength(
    color_rgb: np.ndarray,
    depth: np.ndarray,
    cfg: dict[str, Any],
    roi: Box,
    roi_mask: np.ndarray,
) -> np.ndarray:
    rx1, ry1, rx2, ry2 = roi
    height = depth.shape[0]
    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    band = gray[ry1:ry2, rx1:rx2]
    if band.size == 0:
        return np.zeros(height, dtype=np.float32)

    mask_band = roi_mask[ry1:ry2, rx1:rx2]
    if np.any(mask_band):
        weighted = np.where(mask_band, band, np.nan)
        with np.errstate(invalid="ignore"):
            row_mean = np.nanmean(weighted, axis=1)
        fallback = float(np.nanmedian(row_mean[np.isfinite(row_mean)])) if np.any(np.isfinite(row_mean)) else float(np.mean(band))
        row_mean[~np.isfinite(row_mean)] = fallback
    else:
        row_mean = band.mean(axis=1)

    color_grad = np.abs(np.gradient(np.convolve(row_mean, np.ones(9, dtype=np.float32) / 9.0, mode="same")))
    depth_band = np.where(valid_depth_mask(depth[ry1:ry2, rx1:rx2], cfg) & mask_band, depth[ry1:ry2, rx1:rx2], np.nan)
    depth_grad = np.zeros_like(color_grad)
    rows_with_depth = np.any(np.isfinite(depth_band), axis=1)
    if np.any(rows_with_depth):
        row_depth = np.full(depth_band.shape[0], np.nan, dtype=np.float32)
        row_depth[rows_with_depth] = np.nanmedian(depth_band[rows_with_depth], axis=1)
        fallback_depth = float(np.nanmedian(row_depth[rows_with_depth]))
        row_depth[~np.isfinite(row_depth)] = fallback_depth
        depth_grad = np.abs(np.gradient(np.convolve(row_depth, np.ones(7, dtype=np.float32) / 7.0, mode="same"))) * 120.0

    combined = color_grad + depth_grad
    out = np.zeros(height, dtype=np.float32)
    out[ry1:ry2] = combined.astype(np.float32)
    return out


def _point_cloud_shelf_boundary_candidates(
    color_rgb: np.ndarray,
    depth: np.ndarray,
    intr: Intrinsics,
    cfg: dict[str, Any],
    roi: Box,
    roi_mask: np.ndarray,
) -> list[tuple[int, float]]:
    geom = cfg["geometry"]
    rx1, ry1, rx2, ry2 = roi
    roi_width = max(1, rx2 - rx1)
    points, rows, cols = _sample_voxelized_roi_points(depth, intr, cfg, roi, roi_mask)
    if points.shape[0] == 0:
        return []

    y_bin_m = max(float(geom.get("shelf_plane_y_bin_m", 0.018)), float(geom.get("voxel_size_m", 0.01)))
    qy = np.floor(points[:, 1] / y_bin_m).astype(np.int32)
    bins, inverse, counts = np.unique(qy, return_inverse=True, return_counts=True)
    min_points = int(geom.get("shelf_plane_min_points", 45))
    min_width_ratio = float(geom.get("shelf_plane_min_width_ratio", 0.34))
    min_depth_span = float(geom.get("shelf_plane_min_depth_span_m", 0.02))
    edge_strength = _row_edge_strength(color_rgb, depth, cfg, roi, roi_mask)
    candidates: list[tuple[int, float]] = []

    for bin_index, _bin in enumerate(bins):
        group = inverse == bin_index
        count = int(counts[bin_index])
        if count < min_points:
            continue
        group_rows = rows[group]
        group_cols = cols[group]
        row = int(round(float(np.median(group_rows))))
        if row <= ry1 or row >= ry2:
            continue
        col_span_ratio = (int(group_cols.max()) - int(group_cols.min()) + 1) / float(roi_width)
        if col_span_ratio < min_width_ratio:
            continue
        z_span = float(points[group, 2].max() - points[group, 2].min())
        if z_span < min_depth_span and col_span_ratio < 0.62:
            continue
        x_span = float(points[group, 0].max() - points[group, 0].min())
        edge = float(edge_strength[row]) if 0 <= row < edge_strength.size else 0.0
        score = float(count) * (0.35 + col_span_ratio) * (1.0 + min(1.0, z_span * 4.0)) * (1.0 + min(1.0, x_span * 1.8))
        score *= 1.0 + min(1.0, edge / max(1.0, float(np.percentile(edge_strength, 90)) if edge_strength.size else 1.0))
        candidates.append((row, score))

    merged: list[tuple[int, float]] = []
    gap = max(5, int(geom.get("shelf_plane_merge_gap_px", geom.get("shelf_merge_gap_px", 18))))
    for row in _merge_positions([row for row, _ in candidates], gap):
        near = [(r, s) for r, s in candidates if abs(r - row) <= gap]
        if not near:
            continue
        weighted_row = int(round(sum(r * s for r, s in near) / max(1e-6, sum(s for _, s in near))))
        merged.append((weighted_row, max(s for _, s in near)))
    return sorted(merged, key=lambda item: item[0])


def _select_point_cloud_boundaries(
    color_rgb: np.ndarray,
    depth: np.ndarray,
    intr: Intrinsics,
    cfg: dict[str, Any],
    roi: Box,
    roi_mask: np.ndarray,
    valid_top: int,
    valid_bottom: int,
) -> list[int] | None:
    geom = cfg["geometry"]
    rx1, ry1, rx2, ry2 = roi
    min_h = int(geom["shelf_min_height_px"])
    top = max(ry1, valid_top)
    bottom = min(ry2, valid_bottom)
    if bottom - top < min_h:
        return None

    candidates = [(row, score) for row, score in _point_cloud_shelf_boundary_candidates(color_rgb, depth, intr, cfg, roi, roi_mask) if top + min_h <= row <= bottom - min_h]
    if not candidates:
        return None

    expected_count = int(geom.get("shelf_count", 3))
    infer_count = bool(geom.get("shelf_pointcloud_infer_count", True))
    max_score = max(score for _, score in candidates)
    min_ratio = float(geom.get("shelf_plane_min_score_ratio", 0.20))
    selected: list[int] = []

    if infer_count:
        for row, score in sorted(candidates, key=lambda item: item[1], reverse=True):
            if score < max_score * min_ratio:
                continue
            if all(abs(row - existing) >= min_h for existing in selected):
                selected.append(row)
        selected = sorted(selected)
        inferred_boundaries = _merge_short_boundary_segments([top] + selected + [bottom], min_h, 1.0)
        inferred_count = len(inferred_boundaries) - 1
        min_count = int(geom.get("shelf_pointcloud_min_shelf_count", 2))
        max_count = int(geom.get("shelf_pointcloud_max_shelf_count", 6))
        enough_for_expected = expected_count <= 0 or inferred_count >= expected_count
        if min_count <= inferred_count <= max_count and enough_for_expected:
            return inferred_boundaries

    if not bool(geom.get("shelf_pointcloud_expected_count_fallback", True)):
        return None

    if expected_count <= 0:
        return None
    search = int(geom.get("shelf_refine_search_px", 45))
    edge_strength = _row_edge_strength(color_rgb, depth, cfg, roi, roi_mask)
    base_boundaries = _equal_shelf_boundaries(top, bottom, expected_count)
    base_boundaries = _refine_equal_boundaries_with_color_edges(color_rgb, (rx1, ry1, rx2, ry2), base_boundaries, cfg, roi_mask)
    boundaries = [base_boundaries[0]]
    for expected in base_boundaries[1:-1]:
        nearby = [(row, score) for row, score in candidates if abs(row - expected) <= search]
        if nearby:
            row, _ = max(
                nearby,
                key=lambda item: item[1]
                + float(edge_strength[item[0]]) * 80.0
                - abs(item[0] - expected) * float(geom.get("shelf_pointcloud_expected_distance_penalty", 8.0)),
            )
            boundaries.append(row)
        else:
            boundaries.append(expected)
    boundaries.append(base_boundaries[-1])
    boundaries = _merge_short_boundary_segments(sorted(set(boundaries)), min_h, 1.0)
    return boundaries if len(boundaries) >= 2 else None


def detect_shelves(
    color_rgb: np.ndarray,
    depth: np.ndarray,
    intr: Intrinsics,
    cfg: dict[str, Any],
    roi: Box | None = None,
    roi_mask: np.ndarray | None = None,
) -> list[Shelf]:
    geom = cfg["geometry"]
    height, width = depth.shape[:2]
    rx1, ry1, rx2, ry2 = roi or resolve_cabinet_roi(color_rgb, depth, cfg)
    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 140)

    mask = valid_depth_mask(depth, cfg)
    if roi_mask is None:
        roi_mask = np.zeros_like(mask, dtype=bool)
        roi_mask[ry1:ry2, rx1:rx2] = True
    edges[~roi_mask] = 0
    mask = mask & roi_mask
    roi_width = max(1, rx2 - rx1)
    row_signal = mask[:, rx1:rx2].mean(axis=1)
    valid_rows = np.where(row_signal > 0.15)[0]
    if valid_rows.size == 0:
        valid_top, valid_bottom = 0, height - 1
    else:
        valid_top, valid_bottom = int(valid_rows[0]), int(valid_rows[-1])

    min_h = int(geom["shelf_min_height_px"])
    manual_boundaries = _configured_shelf_boundaries(geom, ry1, ry2, height)
    split_mode = str(geom.get("shelf_split_mode", "equal_count")).lower()
    if manual_boundaries:
        boundaries = manual_boundaries
        plane_hint = "manual configured shelf boundaries"
    elif split_mode in {"point_cloud", "pointcloud", "pc", "voxel", "voxel_plane"}:
        pc_boundaries = _select_point_cloud_boundaries(color_rgb, depth, intr, cfg, (rx1, ry1, rx2, ry2), roi_mask, valid_top, valid_bottom)
        if pc_boundaries:
            boundaries = pc_boundaries
            plane_hint = "voxelized point-cloud horizontal plane bands with RGB-D edge fallback"
        else:
            boundaries = _equal_shelf_boundaries(ry1, ry2, int(geom.get("shelf_count", 3)))
            boundaries = _refine_equal_boundaries_with_color_edges(color_rgb, (rx1, ry1, rx2, ry2), boundaries, cfg, roi_mask)
            plane_hint = "fallback image-depth horizontal band"
    elif split_mode in {"equal", "equal_count", "fixed", "fixed_count"}:
        boundaries = _equal_shelf_boundaries(ry1, ry2, int(geom.get("shelf_count", 3)))
        boundaries = _refine_equal_boundaries_with_color_edges(color_rgb, (rx1, ry1, rx2, ry2), boundaries, cfg, roi_mask)
        plane_hint = "configured equal bands refined by RGB-D edges"
    else:
        min_len = int(roi_width * float(geom["horizontal_line_min_length_ratio"]))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=70, minLineLength=min_len, maxLineGap=20)
        ys: list[int] = []
        if lines is not None:
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = [int(v) for v in line]
                if abs(y2 - y1) <= 5 and abs(x2 - x1) >= min_len:
                    ys.append(int((y1 + y2) / 2))

        row_edges = edges[ry1:ry2, rx1:rx2].mean(axis=1)
        if row_edges.size:
            smooth_kernel = np.ones(9, dtype=np.float32) / 9.0
            row_edges = np.convolve(row_edges, smooth_kernel, mode="same")
            peak_threshold = max(
                float(geom.get("shelf_edge_peak_min_strength", 4.0)),
                float(np.percentile(row_edges, float(geom.get("shelf_edge_peak_percentile", 88)))),
            )
            peak_rows = [ry1 + int(i) for i, value in enumerate(row_edges) if value >= peak_threshold]
            ys.extend(_merge_positions(peak_rows, max(6, int(geom["shelf_merge_gap_px"]) // 2)))

        med = np.where(mask, depth, np.nan)
        row_depth = np.full(height, 1.0, dtype=np.float32)
        rows_with_depth = np.any(np.isfinite(med), axis=1)
        if np.any(rows_with_depth):
            row_depth[rows_with_depth] = np.nanmedian(med[rows_with_depth], axis=1)
            fallback_depth = float(np.nanmedian(row_depth[rows_with_depth]))
            row_depth[~rows_with_depth] = fallback_depth
        grad = np.abs(np.gradient(row_depth))
        if grad.size:
            threshold = max(0.025, float(np.nanpercentile(grad, 88)))
            ys.extend([int(y) for y in np.where(grad > threshold)[0]])

        merged = [y for y in _merge_positions(ys, int(geom["shelf_merge_gap_px"])) if valid_top <= y <= valid_bottom]
        boundaries = [max(valid_top, ry1)] + merged + [min(valid_bottom, ry2)]
        boundaries = _merge_positions(boundaries, int(geom["shelf_merge_gap_px"]))
        boundaries = sorted(set([max(0, min(height - 1, y)) for y in boundaries]))
        boundaries = _merge_short_boundary_segments(
            boundaries,
            min_h,
            float(geom.get("shelf_short_segment_merge_ratio", 1.15)),
        )
        boundaries = _shift_internal_boundaries_down(
            boundaries,
            int(geom.get("shelf_boundary_downshift_px", 0)),
            min_h,
        )
        plane_hint = "image-depth horizontal line/depth discontinuity bands"
    bands: list[tuple[int, int]] = []
    for top, bottom in zip(boundaries, boundaries[1:]):
        if bottom - top >= min_h:
            bands.append((top, bottom))

    if not bands:
        n = 3
        step = max(min_h, (valid_bottom - valid_top + 1) // n)
        bands = []
        top = valid_top
        while top + min_h < valid_bottom:
            bottom = min(valid_bottom, top + step)
            bands.append((top, bottom))
            top = bottom

    shelves: list[Shelf] = []
    margin = int(geom["shelf_margin_px"])
    for idx, (top, bottom) in enumerate(bands, start=1):
        band_mask = mask[top:bottom, :]
        cols = np.where(band_mask.mean(axis=0) > 0.05)[0] if band_mask.size else np.array([])
        if cols.size:
            x1, x2 = int(cols[0]), int(cols[-1])
        else:
            x1, x2 = rx1, rx2
        x1 = max(0, x1 + margin)
        x2 = min(width - 1, x2 - margin)
        y1 = max(0, top + margin)
        y2 = min(height - 1, bottom - margin)
        z = region_median_depth(depth, (x1, y1, x2, y2), cfg)
        shelves.append(Shelf(level=idx, box=(x1, y1, x2, y2), median_depth_m=z or 0.0, plane_hint=plane_hint))
    return shelves


def split_wide_detections_by_depth(
    detections: list[Detection2D],
    depth: np.ndarray,
    cfg: dict[str, Any],
    roi: Box,
) -> list[Detection2D]:
    pp_cfg = cfg["detector"].get("postprocess", {})
    if not bool(pp_cfg.get("split_wide_detections", True)):
        return detections
    rx1, _, rx2, _ = roi
    roi_width = max(1, rx2 - rx1)
    min_width_ratio = float(pp_cfg.get("split_min_width_ratio", 0.28))
    aspect_threshold = float(pp_cfg.get("split_aspect_ratio", 2.4))
    min_area = int(pp_cfg.get("split_min_component_area_px", 180))
    gap = int(pp_cfg.get("split_component_gap_px", 8))
    delta = float(cfg["geometry"].get("foreground_delta_m", 0.08))

    output: list[Detection2D] = []
    for det in detections:
        x1, y1, x2, y2 = det.box
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        is_wide = box_w / roi_width >= min_width_ratio or box_w / box_h >= aspect_threshold
        if not is_wide:
            output.append(det)
            continue

        crop = depth[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if crop.size == 0:
            output.append(det)
            continue
        valid = valid_depth_mask(crop, cfg)
        if valid.mean() < 0.05:
            output.append(det)
            continue

        background_z = float(np.percentile(crop[valid], float(cfg["geometry"].get("foreground_background_percentile", 72))))
        foreground = (valid & (crop < background_z - delta)).astype(np.uint8)
        if gap > 0:
            kernel = np.ones((max(3, gap // 2), max(3, gap // 2)), np.uint8)
            foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)

        components: list[tuple[int, int, int, int, int]] = []
        for comp in range(1, num):
            area = int(stats[comp, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx = int(stats[comp, cv2.CC_STAT_LEFT])
            cy = int(stats[comp, cv2.CC_STAT_TOP])
            cw = int(stats[comp, cv2.CC_STAT_WIDTH])
            ch = int(stats[comp, cv2.CC_STAT_HEIGHT])
            if cw < 8 or ch < 8:
                continue
            components.append((area, x1 + cx, y1 + cy, x1 + cx + cw, y1 + cy + ch))

        if len(components) < 2:
            output.append(det)
            continue

        components.sort(reverse=True)
        for _, cx1, cy1, cx2, cy2 in components[:4]:
            new_det = Detection2D(
                label=det.label,
                category=det.category,
                score=det.score,
                box=_clamp_box((cx1 - gap, cy1 - gap, cx2 + gap, cy2 + gap), depth.shape[1], depth.shape[0]),
                source=det.source,
            )
            output.append(new_det)
    return output


def semantic_unknown_obstacles(
    unknown_detections: list[Detection2D],
    shelves: list[Shelf],
    known_detections: list[Detection2D],
    cfg: dict[str, Any],
) -> list[Obstacle]:
    geom = cfg["geometry"]
    obstacles: list[Obstacle] = []
    min_area = int(geom.get("obstacle_min_area_px", 280))
    min_width = int(geom.get("obstacle_min_width_px", 12))
    min_height = int(geom.get("obstacle_min_height_px", 14))
    max_area_ratio = float(geom.get("semantic_unknown_max_area_ratio", geom.get("obstacle_max_area_ratio", 0.22)))
    max_width_ratio = float(geom.get("semantic_unknown_max_width_ratio", geom.get("obstacle_max_width_ratio", 0.46)))
    max_height_ratio = float(geom.get("obstacle_max_height_ratio", 0.90))
    min_score = float(geom.get("semantic_unknown_min_score", 0.34))

    for det in unknown_detections:
        if det.score < min_score:
            continue
        x1, y1, x2, y2 = det.box
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        area = box_w * box_h
        if area < min_area or box_w < min_width or box_h < min_height:
            continue

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        shelf = next((item for item in shelves if item.box[0] <= cx <= item.box[2] and item.box[1] <= cy <= item.box[3]), None)
        if shelf is None:
            continue

        sx1, sy1, sx2, sy2 = shelf.box
        shelf_w = max(1, sx2 - sx1)
        shelf_h = max(1, sy2 - sy1)
        shelf_area = max(1, shelf_w * shelf_h)
        if area / shelf_area > max_area_ratio:
            continue
        if box_w / shelf_w > max_width_ratio or box_h / shelf_h > max_height_ratio:
            continue

        overlaps_known = False
        for known in known_detections:
            inter = _intersection_area(det.box, known.box)
            if inter / float(min(_box_area(det.box), _box_area(known.box))) >= 0.45:
                overlaps_known = True
                break
        if overlaps_known:
            continue

        center = det.contact_center_m or det.center_m
        obstacles.append(Obstacle(box=det.box, center_m=center, source=f"semantic_unknown:{det.label}"))
    return obstacles


def filter_obstacles_against_detections(
    obstacles: list[Obstacle],
    detections: list[Detection2D],
    cfg: dict[str, Any],
) -> list[Obstacle]:
    if not obstacles or not detections:
        return obstacles

    pp_cfg = cfg.get("detector", {}).get("postprocess", {})
    center_px = float(pp_cfg.get("deduplicate_center_distance_px", 42))
    overlap_threshold = float(cfg.get("geometry", {}).get("obstacle_known_overlap_threshold", 0.45))
    kept: list[Obstacle] = []
    for obs in obstacles:
        ox1, oy1, ox2, oy2 = obs.box
        ocx, ocy = (ox1 + ox2) * 0.5, (oy1 + oy2) * 0.5
        obs_area = _box_area(obs.box)
        duplicate = False
        for det in detections:
            dx1, dy1, dx2, dy2 = det.box
            dcx, dcy = (dx1 + dx2) * 0.5, (dy1 + dy2) * 0.5
            inter = _intersection_area(obs.box, det.box)
            containment = inter / float(min(obs_area, _box_area(det.box)))
            center_distance = ((ocx - dcx) ** 2 + (ocy - dcy) ** 2) ** 0.5
            if containment >= overlap_threshold or center_distance <= center_px:
                duplicate = True
                break
        if not duplicate:
            kept.append(obs)
    return kept


def infer_unknown_obstacles(
    depth: np.ndarray,
    shelves: list[Shelf],
    detections: list[Detection2D],
    cfg: dict[str, Any],
    color_rgb: np.ndarray | None = None,
) -> list[Obstacle]:
    geom = cfg["geometry"]
    known = np.zeros(depth.shape[:2], dtype=np.uint8)
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(known, (x1, y1), (x2, y2), 255, -1)

    obstacles: list[Obstacle] = []
    mask = valid_depth_mask(depth, cfg)
    for shelf in shelves:
        x1, y1, x2, y2 = shelf.box
        band = depth[y1:y2, x1:x2]
        band_mask = mask[y1:y2, x1:x2]
        if band.size == 0 or band_mask.mean() < 0.05:
            continue
        background_z = float(np.percentile(band[band_mask], float(geom.get("foreground_background_percentile", 72))))
        depth_foreground = band_mask & (band < background_z - float(geom["foreground_delta_m"]))
        if color_rgb is not None and bool(geom.get("obstacle_color_foreground_enabled", False)):
            color_band = color_rgb[y1:y2, x1:x2].astype(np.float32)
            valid_pixels = color_band[band_mask] if np.any(band_mask) else color_band.reshape(-1, 3)
            bg_color = np.median(valid_pixels, axis=0)
            color_diff = np.linalg.norm(color_band - bg_color, axis=2)
            color_foreground = color_diff > float(geom.get("obstacle_color_delta", 34))
            foreground = (depth_foreground | (band_mask & color_foreground)).astype(np.uint8)
        else:
            foreground = depth_foreground.astype(np.uint8)
        foreground[known[y1:y2, x1:x2] > 0] = 0
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        shelf_area = max(1, (x2 - x1) * (y2 - y1))
        shelf_width = max(1, x2 - x1)
        shelf_height = max(1, y2 - y1)
        for comp in range(1, num):
            area = int(stats[comp, cv2.CC_STAT_AREA])
            if area < int(geom["obstacle_min_area_px"]):
                continue
            ox = int(stats[comp, cv2.CC_STAT_LEFT]) + x1
            oy = int(stats[comp, cv2.CC_STAT_TOP]) + y1
            ow = int(stats[comp, cv2.CC_STAT_WIDTH])
            oh = int(stats[comp, cv2.CC_STAT_HEIGHT])
            if ow < int(geom.get("obstacle_min_width_px", 0)) or oh < int(geom.get("obstacle_min_height_px", 0)):
                continue
            if area / shelf_area > float(geom.get("obstacle_max_area_ratio", 0.35)):
                continue
            if ow / shelf_width > float(geom.get("obstacle_max_width_ratio", 0.78)):
                continue
            if oh / shelf_height > float(geom.get("obstacle_max_height_ratio", 0.90)):
                continue
            obstacles.append(Obstacle(box=(ox, oy, ox + ow, oy + oh)))
    return obstacles


def _inflate_box(box: tuple[int, int, int, int], amount: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1 - amount), max(0, y1 - amount), min(width - 1, x2 + amount), min(height - 1, y2 + amount)


def find_empty_spaces(
    depth: np.ndarray,
    intr: Intrinsics,
    shelves: list[Shelf],
    detections: list[Detection2D],
    obstacles: list[Obstacle],
    cfg: dict[str, Any],
    roi_mask: np.ndarray | None = None,
) -> list[EmptySpace]:
    geom = cfg["geometry"]
    placement = cfg.get("placement", {})
    item_size = placement.get("active_item_size_m") or placement.get("default_item_size_m") or [0.0, 0.0, 0.0]
    margin = float(placement.get("size_safety_margin_m", 0.0))
    required_width_m = max(float(geom["empty_min_width_m"]), float(item_size[0]) + margin)
    required_height_m = max(float(geom["empty_min_height_m"]), float(item_size[1]) + margin)
    required_depth_m = max(float(geom["empty_min_depth_m"]), float(item_size[2]) + margin)
    height, width = depth.shape[:2]
    occupied = np.zeros((height, width), dtype=np.uint8)
    inflate = int(geom["obstacle_inflate_px"])
    for box in [det.box for det in detections] + [obs.box for obs in obstacles]:
        x1, y1, x2, y2 = _inflate_box(box, inflate, width, height)
        cv2.rectangle(occupied, (x1, y1), (x2, y2), 255, -1)

    spaces: list[EmptySpace] = []
    for shelf in shelves:
        x1, y1, x2, y2 = shelf.box
        z = shelf.median_depth_m or region_median_depth(depth, shelf.box, cfg) or 1.0
        min_w_px = max(18, int(required_width_m * intr.fx / max(0.2, z)))
        min_h_px = max(18, int(required_height_m * intr.fy / max(0.2, z)))
        if x2 <= x1 or y2 <= y1 or (y2 - y1) < min_h_px:
            continue

        shelf_width = max(1, x2 - x1)
        if bool(geom.get("empty_project_occupied_to_full_shelf", True)):
            occupied_cols = np.zeros(shelf_width, dtype=bool)
            min_vertical_overlap = float(geom.get("empty_object_shelf_overlap_min", 0.10))
            for box in [det.box for det in detections] + [obs.box for obs in obstacles]:
                bx1, by1, bx2, by2 = _inflate_box(box, inflate, width, height)
                overlap_y = max(0, min(y2, by2) - max(y1, by1))
                box_h = max(1, by2 - by1)
                center_y = (by1 + by2) * 0.5
                if overlap_y / box_h < min_vertical_overlap and not (y1 <= center_y <= y2):
                    continue
                ox1 = max(x1, bx1) - x1
                ox2 = min(x2, bx2) - x1
                if ox2 > ox1:
                    occupied_cols[max(0, ox1) : min(shelf_width, ox2)] = True
            free_cols = ~occupied_cols
        else:
            contact_h = max(min_h_px, int((y2 - y1) * float(geom.get("empty_contact_band_ratio", 0.45))))
            contact_y1 = max(y1, y2 - contact_h)
            contact_occ = occupied[contact_y1:y2, x1:x2] > 0
            full_occ = occupied[y1:y2, x1:x2] > 0
            occupied_col_ratio = contact_occ.mean(axis=0)
            full_col_ratio = full_occ.mean(axis=0)
            free_cols = (
                (occupied_col_ratio <= float(geom.get("empty_column_occupied_ratio_max", 0.08)))
                & (full_col_ratio <= float(geom.get("empty_column_full_occupied_ratio_max", 0.03)))
            )

        if roi_mask is not None:
            roi_band = roi_mask[y1:y2, x1:x2]
            if roi_band.size:
                roi_col_ratio = roi_band.mean(axis=0)
                free_cols &= roi_col_ratio >= float(geom.get("empty_roi_column_min_ratio", 0.20))

        runs: list[tuple[int, int]] = []
        start = None
        for idx, is_free in enumerate(free_cols.tolist() + [False]):
            if is_free and start is None:
                start = idx
            elif not is_free and start is not None:
                if idx - start >= min_w_px:
                    runs.append((start, idx - 1))
                start = None

        candidates: list[EmptySpace] = []
        for start_x, end_x in runs:
            ex1 = x1 + start_x
            ex2 = x1 + end_x
            ey1 = y1
            ey2 = y2
            cx = (ex1 + ex2) * 0.5
            cy = (ey1 + ey2) * 0.5 if str(geom.get("empty_center_mode", "center")).lower() == "center" else ey2
            center = deproject_pixel(cx, cy, z, intr)
            size_m = (
                float((ex2 - ex1) * z / intr.fx),
                float((ey2 - ey1) * z / intr.fy),
                required_depth_m,
            )
            area = max(1, (ex2 - ex1) * (ey2 - ey1))
            score = float(area / max(1, (x2 - x1) * (y2 - y1)))
            candidates.append(EmptySpace(shelf_level=shelf.level, box=(ex1, ey1, ex2, ey2), center_m=center, size_m=size_m, score=score))

        candidates.sort(key=lambda item: item.score, reverse=True)
        spaces.extend(candidates[: int(geom["empty_max_per_shelf"])])
    return spaces


def _runs_from_free_cols(free_cols: np.ndarray, min_width: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, is_free in enumerate(free_cols.tolist() + [False]):
        if is_free and start is None:
            start = idx
        elif not is_free and start is not None:
            if idx - start >= min_width:
                runs.append((start, idx))
            start = None
    return runs


def _resize_empty_space(space: EmptySpace, new_box: Box, shelf: Shelf, track_suffix: str | None = None) -> EmptySpace:
    ox1, oy1, ox2, oy2 = space.box
    nx1, ny1, nx2, ny2 = new_box
    old_w = max(1, ox2 - ox1)
    old_h = max(1, oy2 - oy1)
    px_to_m_x = space.size_m[0] / old_w
    px_to_m_y = space.size_m[1] / old_h
    old_cx = (ox1 + ox2) * 0.5
    old_cy = (oy1 + oy2) * 0.5
    new_cx = (nx1 + nx2) * 0.5
    new_cy = (ny1 + ny2) * 0.5
    cx, cy, cz = space.center_m
    center = (
        float(cx + (new_cx - old_cx) * px_to_m_x),
        float(cy + (new_cy - old_cy) * px_to_m_y),
        float(cz),
    )
    size = (
        float(max(1, nx2 - nx1) * px_to_m_x),
        float(max(1, ny2 - ny1) * px_to_m_y),
        float(space.size_m[2]),
    )
    shelf_area = max(1, (shelf.box[2] - shelf.box[0]) * (shelf.box[3] - shelf.box[1]))
    score = float(max(1, nx2 - nx1) * max(1, ny2 - ny1) / shelf_area)
    track_id = space.track_id if track_suffix is None or space.track_id is None else f"{space.track_id}{track_suffix}"
    return EmptySpace(
        shelf_level=space.shelf_level,
        box=new_box,
        center_m=center,
        size_m=size,
        score=score,
        track_id=track_id,
    )


def subtract_occupied_from_empty_spaces(
    shelves: list[Shelf],
    empty_spaces: list[EmptySpace],
    detections: list[Detection2D],
    obstacles: list[Obstacle],
    cfg: dict[str, Any],
) -> list[EmptySpace]:
    geom = cfg["geometry"]
    if not bool(geom.get("empty_final_project_occupied", True)):
        return empty_spaces

    shelf_by_level = {shelf.level: shelf for shelf in shelves}
    boxes = [det.box for det in detections] + [obs.box for obs in obstacles]
    if not boxes or not shelves or not empty_spaces:
        return empty_spaces

    inflate = int(geom.get("obstacle_inflate_px", 0))
    min_vertical_overlap = float(geom.get("empty_object_shelf_overlap_min", 0.10))
    min_width = max(1, int(geom.get("empty_final_min_width_px", 24)))
    max_per_shelf = int(geom.get("empty_max_per_shelf", 2))
    output: list[EmptySpace] = []

    for shelf in shelves:
        x1, y1, x2, y2 = shelf.box
        shelf_width = max(1, x2 - x1)
        occupied_cols = np.zeros(shelf_width, dtype=bool)
        for box in boxes:
            bx1, by1, bx2, by2 = box
            bx1 -= inflate
            by1 -= inflate
            bx2 += inflate
            by2 += inflate
            overlap_y = max(0, min(y2, by2) - max(y1, by1))
            box_h = max(1, by2 - by1)
            center_y = (by1 + by2) * 0.5
            if overlap_y / box_h < min_vertical_overlap and not (y1 <= center_y <= y2):
                continue
            ox1 = max(x1, bx1) - x1
            ox2 = min(x2, bx2) - x1
            if ox2 > ox1:
                occupied_cols[max(0, ox1) : min(shelf_width, ox2)] = True

        shelf_candidates: list[EmptySpace] = []
        for space in [item for item in empty_spaces if item.shelf_level == shelf.level]:
            sx1, sy1, sx2, sy2 = space.box
            local_start = max(x1, sx1) - x1
            local_end = min(x2, sx2) - x1
            if local_end <= local_start:
                continue
            free_cols = ~occupied_cols[local_start:local_end]
            runs = _runs_from_free_cols(free_cols, min_width)
            for idx, (run_start, run_end) in enumerate(runs):
                nx1 = x1 + local_start + run_start
                nx2 = x1 + local_start + run_end
                if nx2 <= nx1:
                    continue
                suffix = None if len(runs) == 1 else chr(ord("a") + min(idx, 25))
                shelf_candidates.append(_resize_empty_space(space, (nx1, sy1, nx2, sy2), shelf, suffix))

        shelf_candidates.sort(key=lambda item: item.score, reverse=True)
        output.extend(shelf_candidates[:max_per_shelf])

    return sorted(output, key=lambda item: (item.shelf_level, item.box[0], item.box[1]))


def recommend_placement(
    new_item_category: str,
    detections: list[Detection2D],
    empty_spaces: list[EmptySpace],
    cfg: dict[str, Any],
) -> PlacementRecommendation:
    if not empty_spaces:
        return PlacementRecommendation(new_item_category, None, "No empty space satisfies the configured size/clearance limits.")

    placement = cfg["placement"]
    same_category = [det for det in detections if det.category == new_item_category and (det.contact_center_m or det.center_m) is not None]

    best_space: EmptySpace | None = None
    best_score = -1e9
    best_reason = "Selected largest safe empty space."
    for space in empty_spaces:
        score = float(space.score) * float(placement["free_area_weight"])
        if same_category:
            distances = []
            sx, sy, sz = space.center_m
            same_shelf = False
            for det in same_category:
                dx, dy, dz = det.contact_center_m or det.center_m or (0.0, 0.0, 0.0)
                distances.append(((sx - dx) ** 2 + (sy - dy) ** 2 + (sz - dz) ** 2) ** 0.5)
                _, dy1, _, dy2 = det.box
                det_cy = (dy1 + dy2) * 0.5
                if space.box[1] <= det_cy <= space.box[3]:
                    same_shelf = True
            nearest = min(distances)
            score += float(placement["same_category_bonus"])
            if same_shelf:
                score += float(placement.get("same_category_same_shelf_bonus", 4.0))
            score -= nearest * float(placement["distance_weight"])
            reason = f"Near existing {new_item_category} object; distance {nearest:.3f} m."
            if same_shelf:
                reason += " Same shelf is preferred."
        else:
            reason = "No same-category object found; selected by free area."
        if score > best_score:
            best_score = score
            best_space = space
            best_reason = reason

    return PlacementRecommendation(new_item_category, best_space, best_reason)
