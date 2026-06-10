from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .types import EmptySpace, PipelineResult
from .geometry import resolve_cabinet_roi_polygon


GREEN = (40, 200, 70)
RED = (40, 40, 230)
BLUE = (230, 120, 20)
GOLD = (40, 210, 255)
WHITE = (245, 245, 245)
GRAY = (70, 70, 70)
BLACK = (20, 20, 20)


def _draw_label(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - 7)
    cv2.rectangle(img, (x, y0), (min(img.shape[1] - 1, x + tw + 8), y0 + th + 7), color, -1)
    cv2.putText(img, text, (x + 4, y0 + th + 3), font, scale, WHITE, thickness, cv2.LINE_AA)


def _format_xyz(point: tuple[float, float, float] | None) -> str:
    if point is None:
        return "n/a"
    return f"({point[0]*1000:.0f},{point[1]*1000:.0f},{point[2]*1000:.0f})mm"


def draw_result(color_rgb: np.ndarray, result: PipelineResult, cfg: dict[str, Any]) -> np.ndarray:
    canvas = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    if cfg["geometry"].get("show_cabinet_roi", True):
        # ROI is recomputed from the visual frame only for display; pipeline uses the same resolver.
        fake_depth = np.ones(color_rgb.shape[:2], dtype=np.float32)
        polygon = resolve_cabinet_roi_polygon(color_rgb, fake_depth, cfg)
        if len(polygon) >= 3:
            pts = np.asarray(polygon, dtype=np.int32)
            cv2.polylines(canvas, [pts], True, (0, 210, 210), 1)
            rx1, ry1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            _draw_label(canvas, "Cabinet ROI", (rx1, ry1), (0, 160, 160))

    for shelf in result.shelves:
        x1, y1, x2, y2 = shelf.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), RED, 2)
        _draw_label(canvas, f"Shelf {shelf.level}", (x1, y1), RED)

    for space in result.empty_spaces:
        x1, y1, x2, y2 = space.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BLUE, 2)
        w_cm = space.size_m[0] * 100.0
        h_cm = space.size_m[1] * 100.0
        sid = f"{space.track_id} " if space.track_id else ""
        _draw_label(canvas, f"{sid}Empty L{space.shelf_level} {w_cm:.0f}x{h_cm:.0f}cm s{space.score:.2f}", (x1, y1), BLUE)

    for obs in result.obstacles:
        x1, y1, x2, y2 = obs.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GRAY, 1)
        if obs.source.startswith("semantic_unknown:"):
            _draw_label(canvas, "Unknown obstacle", (x1, y1), GRAY)
        elif obs.source == "depth_cluster":
            _draw_label(canvas, "Depth obstacle", (x1, y1), GRAY)

    for det in result.objects:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GREEN, 2)
        oid = f"{det.track_id} " if det.track_id else ""
        label = f"{oid}{det.category}:{det.score:.2f}" if det.category else f"{oid}{det.label}:{det.score:.2f}"
        _draw_label(canvas, label, (x1, y1), GREEN)

    _draw_recommendation(canvas, result, cfg)

    if bool(cfg.get("output", {}).get("append_info_panel", False)):
        return _append_panel(canvas, result)
    return canvas


def _draw_recommendation(canvas: np.ndarray, result: PipelineResult, cfg: dict[str, Any]) -> None:
    rec = result.recommendation
    if not rec or not rec.empty_space:
        return

    space = rec.empty_space
    sx1, sy1, sx2, sy2 = space.box
    cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), GOLD, 3)

    placement = cfg.get("placement", {})
    item_size = placement.get("active_item_size_m") or placement.get("default_item_size_m") or [0.0, 0.0, 0.0]
    item_w_m = max(0.0, float(item_size[0]))
    item_h_m = max(0.0, float(item_size[1]))
    px_per_m_x = max(1.0, (sx2 - sx1) / max(1e-6, float(space.size_m[0])))
    px_per_m_y = max(1.0, (sy2 - sy1) / max(1e-6, float(space.size_m[1])))
    insert_w = int(max(18, min(sx2 - sx1, item_w_m * px_per_m_x)))
    insert_h = int(max(18, min(sy2 - sy1, item_h_m * px_per_m_y)))
    cx = int((sx1 + sx2) * 0.5)
    cy = int((sy1 + sy2) * 0.5)
    ix1 = max(sx1, cx - insert_w // 2)
    ix2 = min(sx2, ix1 + insert_w)
    ix1 = max(sx1, ix2 - insert_w)
    iy2 = sy2
    iy1 = max(sy1, iy2 - insert_h)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), GOLD, -1)
    cv2.addWeighted(overlay, 0.26, canvas, 0.74, 0.0, canvas)
    cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), GOLD, 3)
    _draw_label(canvas, f"Place {rec.new_item_category} here", (ix1, iy1), GOLD)


def draw_top_view(result: PipelineResult, width: int = 640, height: int = 360) -> np.ndarray:
    view = np.full((height, width, 3), (32, 34, 36), dtype=np.uint8)
    if not result.shelves:
        cv2.putText(view, "No shelves", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)
        return view

    left = 70
    right = width - 24
    top = 32
    gap = 18
    shelf_h = max(48, int((height - top - 24 - gap * (len(result.shelves) - 1)) / len(result.shelves)))

    for idx, shelf in enumerate(result.shelves):
        y1 = top + idx * (shelf_h + gap)
        y2 = min(height - 24, y1 + shelf_h)
        sx1, sy1, sx2, sy2 = shelf.box
        cv2.rectangle(view, (left, y1), (right, y2), (95, 95, 95), 1)
        cv2.putText(view, f"L{shelf.level}", (18, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 1, cv2.LINE_AA)

        def map_x(x: int) -> int:
            return int(left + (x - sx1) / max(1, sx2 - sx1) * (right - left))

        for obs in result.obstacles:
            ox1, oy1, ox2, oy2 = obs.box
            cy = (oy1 + oy2) * 0.5
            if sy1 <= cy <= sy2:
                cv2.rectangle(view, (map_x(ox1), y1 + 8), (map_x(ox2), y2 - 8), GRAY, -1)
                if obs.source.startswith("semantic_unknown:"):
                    cv2.putText(view, "U", (map_x(ox1) + 3, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)
                elif obs.source == "depth_cluster":
                    cv2.putText(view, "D", (map_x(ox1) + 3, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)

        for det in result.objects:
            ox1, oy1, ox2, oy2 = det.box
            cy = (oy1 + oy2) * 0.5
            if sy1 <= cy <= sy2:
                cv2.rectangle(view, (map_x(ox1), y1 + 10), (map_x(ox2), y2 - 10), GREEN, -1)
                cv2.putText(view, det.track_id or det.category or "Obj", (map_x(ox1) + 3, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BLACK, 1, cv2.LINE_AA)

        for space in result.empty_spaces:
            if space.shelf_level == shelf.level:
                ex1, _, ex2, _ = space.box
                is_rec = bool(result.recommendation and result.recommendation.empty_space is space)
                color = GOLD if is_rec else BLUE
                thickness = 3 if is_rec else 2
                cv2.rectangle(view, (map_x(ex1), y1 + 12), (map_x(ex2), y2 - 12), color, thickness)
                label = "PLACE" if is_rec else f"{space.track_id or 'E'} {space.score:.2f}"
                cv2.putText(view, label, (map_x(ex1) + 4, y2 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    rec = result.recommendation
    if rec and rec.empty_space:
        cv2.putText(view, f"Recommended: L{rec.empty_space.shelf_level} {rec.empty_space.track_id or ''}", (left, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return view


def _append_panel(canvas: np.ndarray, result: PipelineResult) -> np.ndarray:
    height = canvas.shape[0]
    panel_w = 380
    panel = np.full((height, panel_w, 3), (35, 35, 35), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def line(text: str, y: int, color: tuple[int, int, int] = WHITE, scale: float = 0.48) -> int:
        cv2.putText(panel, text, (14, y), font, scale, color, 1, cv2.LINE_AA)
        return y + 22

    y = 28
    y = line("Experiment 5 RGB-D Cabinet", y, WHITE, 0.58)
    y = line(f"Device: {result.device}   FPS: {result.fps:.1f}", y)
    y += 8

    counts: dict[str, int] = {}
    for det in result.objects:
        counts[det.category or "Unknown"] = counts.get(det.category or "Unknown", 0) + 1
    y = line("Categories:", y, (210, 210, 210))
    if counts:
        for name, count in counts.items():
            y = line(f"- {name}: {count}", y)
    else:
        y = line("- no semantic object", y)
    y += 8

    y = line("Objects XYZ:", y, (210, 210, 210))
    for idx, det in enumerate(result.objects[:6], start=1):
        y = line(f"{idx}. {det.category}: {_format_xyz(det.contact_center_m)}", y, GREEN, 0.42)
    if len(result.objects) > 6:
        y = line(f"... {len(result.objects) - 6} more", y, GREEN, 0.42)
    y += 8

    y = line("Empty Spaces:", y, (210, 210, 210))
    for idx, space in enumerate(result.empty_spaces[:7], start=1):
        y = line(f"{idx}. L{space.shelf_level}: {_format_xyz(space.center_m)}", y, BLUE, 0.42)
    if len(result.empty_spaces) > 7:
        y = line(f"... {len(result.empty_spaces) - 7} more", y, BLUE, 0.42)
    y += 8

    semantic_unknowns = [obs for obs in result.obstacles if obs.source.startswith("semantic_unknown:")]
    if semantic_unknowns:
        y = line("Unknown Obstacles:", y, (210, 210, 210))
        for idx, obs in enumerate(semantic_unknowns[:4], start=1):
            y = line(f"{idx}. unknown: {_format_xyz(obs.center_m)}", y, GRAY, 0.42)
        if len(semantic_unknowns) > 4:
            y = line(f"... {len(semantic_unknowns) - 4} more", y, GRAY, 0.42)
        y += 8

    rec = result.recommendation
    y = line("Recommendation:", y, (210, 210, 210))
    if rec and rec.empty_space:
        space: EmptySpace = rec.empty_space
        y = line(f"{rec.new_item_category} -> Shelf {space.shelf_level}", y, WHITE, 0.46)
        y = line(_format_xyz(space.center_m), y, WHITE, 0.46)
    else:
        y = line("No valid placement", y, WHITE, 0.46)

    if result.warnings:
        y += 8
        y = line("Warnings:", y, (180, 180, 255), 0.46)
        for warn in result.warnings[:4]:
            y = line(warn[:48], y, (180, 180, 255), 0.38)

    return np.hstack([canvas, panel])


def _draw_top_view_panel(panel: np.ndarray, result: PipelineResult, y: int) -> int:
    if y > panel.shape[0] - 150:
        return y
    mini = draw_top_view(result, width=340, height=120)
    y = min(y, panel.shape[0] - mini.shape[0] - 8)
    panel[y : y + mini.shape[0], 20 : 20 + mini.shape[1]] = mini
    return y + mini.shape[0] + 8
