from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .demo_data import create_demo_session
from .detector import BaseDetector, CachedDetector, build_detector, postprocess_detections
from .geometry import (
    attach_3d_to_detections,
    depth_m,
    detect_shelves,
    filter_detections_to_roi,
    filter_detections_to_single_shelf,
    filter_obstacles_against_detections,
    fill_depth_holes,
    find_empty_spaces,
    infer_unknown_obstacles,
    recommend_placement,
    resolve_cabinet_roi,
    resolve_cabinet_roi_with_mask,
    semantic_unknown_obstacles,
    split_wide_detections_by_depth,
    subtract_occupied_from_empty_spaces,
)
from .mapping import SceneMap, build_scene_map, draw_scene_map, export_scene_ply
from .realsense_io import RealSenseCamera, load_session, save_session
from .runtime_controls import RuntimeControls
from .stabilizer import StaticSceneStabilizer
from .types import PipelineResult, RGBDFrame
from .visualization import draw_result
from .visualization import draw_top_view


@dataclass
class FrameArtifacts:
    result: PipelineResult
    annotated_bgr: np.ndarray
    top_view_bgr: np.ndarray
    map_view_bgr: np.ndarray | None = None
    scene_map: SceneMap | None = None


def process_frame(
    frame: RGBDFrame,
    detector: BaseDetector,
    cfg: dict[str, Any],
    new_item_category: str,
    fps: float = 0.0,
    render: bool = True,
    build_map: bool = True,
    previous_scene_map: SceneMap | None = None,
) -> FrameArtifacts:
    warnings: list[str] = []
    t0 = time.perf_counter()
    depth = fill_depth_holes(depth_m(frame.depth_raw, frame.intrinsics), cfg)
    roi, roi_polygon, roi_mask = resolve_cabinet_roi_with_mask(frame.color_rgb, depth, cfg)
    semantic_detections = filter_detections_to_roi(detector.detect(frame.color_rgb), roi, cfg, roi_mask)
    semantic_detections = split_wide_detections_by_depth(semantic_detections, depth, cfg, roi)
    semantic_detections = filter_detections_to_roi(semantic_detections, roi, cfg, roi_mask)
    semantic_detections = postprocess_detections(semantic_detections, cfg, frame.color_rgb.shape[1], frame.color_rgb.shape[0])
    attach_3d_to_detections(semantic_detections, depth, frame.intrinsics, cfg)
    shelves = detect_shelves(frame.color_rgb, depth, frame.intrinsics, cfg, roi=roi, roi_mask=roi_mask)
    semantic_detections = filter_detections_to_single_shelf(semantic_detections, shelves, cfg)
    detections = [det for det in semantic_detections if det.category is not None]
    unknown_detections = [det for det in semantic_detections if det.category is None]
    semantic_obstacles = semantic_unknown_obstacles(unknown_detections, shelves, detections, cfg)
    accepted_unknown_boxes = {obs.box for obs in semantic_obstacles}
    accepted_unknown_detections = [det for det in unknown_detections if det.box in accepted_unknown_boxes]
    depth_obstacles = infer_unknown_obstacles(depth, shelves, detections + accepted_unknown_detections, cfg, frame.color_rgb)
    obstacles = semantic_obstacles + depth_obstacles
    obstacles = filter_obstacles_against_detections(obstacles, detections, cfg)
    empty_spaces = find_empty_spaces(depth, frame.intrinsics, shelves, detections, obstacles, cfg, roi_mask=roi_mask)
    empty_spaces = subtract_occupied_from_empty_spaces(shelves, empty_spaces, detections, obstacles, cfg)
    recommendation = recommend_placement(new_item_category, detections, empty_spaces, cfg)
    elapsed = max(1e-6, time.perf_counter() - t0)
    result = PipelineResult(
        objects=detections,
        shelves=shelves,
        empty_spaces=empty_spaces,
        obstacles=obstacles,
        recommendation=recommendation,
        fps=fps or (1.0 / elapsed),
        device=detector.device_info.device,
        warnings=warnings,
    )
    if "CUDA requested" in detector.device_info.message:
        result.warnings.append("CUDA unavailable; running on CPU.")

    scene_map = build_scene_map(frame, depth, roi, cfg, roi_mask=roi_mask) if build_map else previous_scene_map
    if scene_map is not None:
        result.map_summary = scene_map.summary()
    if render:
        annotated = draw_result(frame.color_rgb, result, cfg)
        top_view = draw_top_view(result)
        map_view = draw_scene_map(scene_map, result, cfg)
    else:
        annotated = np.zeros((1, 1, 3), dtype=np.uint8)
        top_view = np.zeros((1, 1, 3), dtype=np.uint8)
        map_view = None
    return FrameArtifacts(result, annotated, top_view, map_view, scene_map)


def refresh_visuals(
    frame: RGBDFrame,
    artifacts: FrameArtifacts,
    cfg: dict[str, Any],
    draw_map: bool = True,
    previous_map_view: np.ndarray | None = None,
) -> FrameArtifacts:
    artifacts.annotated_bgr = draw_result(frame.color_rgb, artifacts.result, cfg)
    artifacts.top_view_bgr = draw_top_view(artifacts.result)
    artifacts.map_view_bgr = draw_scene_map(artifacts.scene_map, artifacts.result, cfg) if draw_map else previous_map_view
    return artifacts


def save_outputs(artifacts: FrameArtifacts, output_dir: str | Path, cfg: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "annotated.png"), artifacts.annotated_bgr)
    cv2.imwrite(str(output_dir / "top_view.png"), artifacts.top_view_bgr)
    if artifacts.map_view_bgr is not None:
        cv2.imwrite(str(output_dir / "map_3d.png"), artifacts.map_view_bgr)
    if artifacts.scene_map is not None and bool(cfg.get("mapping", {}).get("save_ply", True)):
        export_scene_ply(artifacts.scene_map, output_dir / "scene_map.ply")
    (output_dir / "result.json").write_text(json.dumps(artifacts.result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def auto_calibrate_roi_from_frame(frame: RGBDFrame, cfg: dict[str, Any], output_dir: str | Path | None = None) -> tuple[int, int, int, int]:
    roi_cfg = deepcopy(cfg)
    roi_cfg["geometry"]["cabinet_roi_mode"] = "auto"
    roi_cfg["geometry"]["cabinet_roi_px"] = None
    roi_cfg["geometry"]["cabinet_roi_poly_px"] = None
    roi_cfg["geometry"]["cabinet_roi_poly_norm"] = None
    depth = fill_depth_holes(depth_m(frame.depth_raw, frame.intrinsics), roi_cfg)
    (x1, y1, x2, y2), polygon, _ = resolve_cabinet_roi_with_mask(frame.color_rgb, depth, roi_cfg)
    h, w = frame.color_rgb.shape[:2]
    roi_norm = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
    poly_norm = _polygon_norm(polygon, w, h)

    cfg["geometry"]["cabinet_roi_mode"] = "poly"
    cfg["geometry"]["cabinet_roi_norm"] = roi_norm
    cfg["geometry"]["cabinet_roi_poly_norm"] = poly_norm
    cfg["geometry"]["cabinet_roi_px"] = None
    cfg["geometry"]["cabinet_roi_poly_px"] = None
    print(f"[ROI] auto calibrated: box_px={(x1, y1, x2, y2)}, box_norm={roi_norm}, poly_norm={poly_norm}")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        preview = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
        cv2.polylines(preview, [np.asarray(polygon, dtype=np.int32)], True, (0, 210, 210), 2)
        cv2.imwrite(str(output_path / "startup_roi.png"), preview)
    return x1, y1, x2, y2


def _auto_calibrate_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("geometry", {}).get("auto_calibrate_on_start", False))


def _polygon_norm(polygon: list[tuple[int, int]], width: int, height: int) -> list[list[float]]:
    return [[round(x / width, 4), round(y / height, 4)] for x, y in polygon]


def _print_summary(result: PipelineResult, output_dir: Path | None = None) -> None:
    print()
    print("=== Experiment 5 Result ===")
    print(f"Device: {result.device}, FPS: {result.fps:.2f}")
    print(
        f"Objects: {len(result.objects)}, Shelves: {len(result.shelves)}, "
        f"Empty spaces: {len(result.empty_spaces)}, Obstacles: {len(result.obstacles)}"
    )
    if result.map_summary:
        print(f"3D map points: {result.map_summary.get('point_count', 0)}")
    for obj in result.objects:
        center = obj.contact_center_m or obj.center_m
        xyz = "n/a" if center is None else f"({center[0]*1000:.1f}, {center[1]*1000:.1f}, {center[2]*1000:.1f}) mm"
        print(f"- {obj.category} / {obj.label}: score={obj.score:.2f}, box={obj.box}, contact_center={xyz}")
    for obs in result.obstacles:
        center = obs.center_m
        xyz = "n/a" if center is None else f"({center[0]*1000:.1f}, {center[1]*1000:.1f}, {center[2]*1000:.1f}) mm"
        print(f"- Obstacle / {obs.source}: box={obs.box}, center={xyz}")
    for space in result.empty_spaces:
        x, y, z = space.center_m
        print(f"- Empty Space L{space.shelf_level}: box={space.box}, center=({x*1000:.1f}, {y*1000:.1f}, {z*1000:.1f}) mm")
    if result.recommendation and result.recommendation.empty_space:
        space = result.recommendation.empty_space
        x, y, z = space.center_m
        print(
            f"Recommendation: place {result.recommendation.new_item_category} on shelf {space.shelf_level} "
            f"at ({x*1000:.1f}, {y*1000:.1f}, {z*1000:.1f}) mm"
        )
        print(f"Reason: {result.recommendation.reason}")
    else:
        print("Recommendation: no valid placement")
    if output_dir:
        print(f"Saved outputs: {output_dir}")


def run_replay(args: Any, cfg: dict[str, Any]) -> None:
    session_dir = Path(args.session)
    frame = load_session(session_dir)
    if _auto_calibrate_enabled(cfg):
        auto_calibrate_roi_from_frame(frame, cfg, args.output)
    detector = build_detector(cfg, args.detector, session_dir=session_dir)
    print(detector.device_info.message)
    artifacts = process_frame(frame, detector, cfg, args.new_item)
    output_dir = save_outputs(artifacts, args.output, cfg)
    _print_summary(artifacts.result, output_dir)
    if args.display:
        _show_artifacts("Experiment 5 Replay", artifacts)
        cv2.waitKey(0)


def run_capture(args: Any, cfg: dict[str, Any]) -> None:
    camera = RealSenseCamera(cfg)
    camera.start()
    try:
        if _auto_calibrate_enabled(cfg):
            calib_frames = int(cfg["geometry"].get("auto_calibration_frames", args.temporal_frames))
            calib_frame = read_temporal_frame(camera, calib_frames, bool(cfg.get("fusion", {}).get("median_color", True)))
            auto_calibrate_roi_from_frame(calib_frame, cfg, args.output)
        frame = read_temporal_frame(camera, args.temporal_frames, bool(cfg.get("fusion", {}).get("median_color", True)))
    finally:
        camera.stop()
    session_dir = save_session(frame, args.session)
    print(f"Saved RGB-D session: {session_dir}")

    if args.process_after_capture:
        detector = build_detector(cfg, args.detector, session_dir=session_dir)
        print(detector.device_info.message)
        artifacts = process_frame(frame, detector, cfg, args.new_item)
        output_dir = save_outputs(artifacts, args.output, cfg)
        _print_summary(artifacts.result, output_dir)


def run_live(args: Any, cfg: dict[str, Any]) -> None:
    camera = RealSenseCamera(cfg)
    camera.start()
    latest_artifacts: FrameArtifacts | None = None
    try:
        output_dir = Path(args.output)
        if _auto_calibrate_enabled(cfg):
            calib_frames = int(cfg["geometry"].get("auto_calibration_frames", args.temporal_frames))
            calib_frame = read_temporal_frame(camera, calib_frames, bool(cfg.get("fusion", {}).get("median_color", True)))
            auto_calibrate_roi_from_frame(calib_frame, cfg, output_dir)

        detector = build_detector(cfg, args.detector)
        detector = CachedDetector(detector, args.detect_every)
        stabilizer = StaticSceneStabilizer(cfg, args.new_item)
        print(detector.device_info.message)
        controls = RuntimeControls(cfg, args, detector) if args.display and bool(cfg.get("ui", {}).get("enabled", True)) else None
        last = time.perf_counter()
        frame_count = 0
        map_update_every = max(1, int(cfg.get("mapping", {}).get("update_every", 5)))
        while True:
            if controls is not None:
                controls.apply(cfg, args, detector, stabilizer)
            frame = read_temporal_frame(camera, args.temporal_frames, bool(cfg.get("fusion", {}).get("median_color", True)))
            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - last)
            last = now
            build_map_now = latest_artifacts is None or frame_count % map_update_every == 0
            artifacts = process_frame(
                frame,
                detector,
                cfg,
                args.new_item,
                fps=fps,
                render=False,
                build_map=build_map_now,
                previous_scene_map=latest_artifacts.scene_map if latest_artifacts else None,
            )
            artifacts.result = stabilizer.update(artifacts.result)
            artifacts = refresh_visuals(
                frame,
                artifacts,
                cfg,
                draw_map=build_map_now or latest_artifacts is None,
                previous_map_view=latest_artifacts.map_view_bgr if latest_artifacts else None,
            )
            latest_artifacts = artifacts
            frame_count += 1

            if args.display:
                _show_artifacts("Experiment 5 Live", artifacts)
                if controls is not None:
                    controls.show(cfg, args)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    auto_calibrate_roi_from_frame(frame, cfg, output_dir)
                    if controls is not None:
                        controls.sync_from_cfg(cfg, args)
                if key == ord("s"):
                    snapshot_dir = output_dir / f"snapshot_{int(time.time())}"
                    saved = save_outputs(artifacts, snapshot_dir, cfg)
                    print(f"[live] saved snapshot: {saved}")
            if args.print_every > 0 and frame_count % args.print_every == 0:
                _print_live_line(artifacts.result)
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()

    if latest_artifacts is not None:
        saved = save_outputs(latest_artifacts, output_dir, cfg)
        _print_summary(latest_artifacts.result, saved)


def run_demo(args: Any, cfg: dict[str, Any]) -> None:
    session_dir = create_demo_session(args.session)
    print(f"Demo session written to: {session_dir}")


def run_calibrate_roi(args: Any, cfg: dict[str, Any]) -> None:
    if args.from_session:
        frame = load_session(args.session)
    else:
        camera = RealSenseCamera(cfg)
        camera.start()
        try:
            frame = read_temporal_frame(camera, args.temporal_frames, bool(cfg.get("fusion", {}).get("median_color", True)))
        finally:
            camera.stop()

    roi_cfg = deepcopy(cfg)
    roi_cfg["geometry"]["cabinet_roi_mode"] = "auto"
    roi_cfg["geometry"]["cabinet_roi_px"] = None
    roi_cfg["geometry"]["cabinet_roi_poly_px"] = None
    roi_cfg["geometry"]["cabinet_roi_poly_norm"] = None
    depth = fill_depth_holes(depth_m(frame.depth_raw, frame.intrinsics), roi_cfg)
    (x1, y1, x2, y2), polygon, _ = resolve_cabinet_roi_with_mask(frame.color_rgb, depth, roi_cfg)
    h, w = frame.color_rgb.shape[:2]
    roi_norm = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
    poly_norm = _polygon_norm(polygon, w, h)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
    cv2.polylines(preview, [np.asarray(polygon, dtype=np.int32)], True, (0, 210, 210), 2)
    cv2.imwrite(str(output_dir / "roi_calibration.png"), preview)

    text = (
        "geometry:\n"
        "  cabinet_roi_mode: poly\n"
        f"  cabinet_roi_norm: {roi_norm}\n"
        f"  cabinet_roi_poly_norm: {poly_norm}\n"
    )
    Path(args.calibration_output).write_text(text, encoding="utf-8")
    print(f"Calibrated ROI box px: {(x1, y1, x2, y2)}")
    print(f"Calibrated ROI box norm: {roi_norm}")
    print(f"Calibrated ROI polygon norm: {poly_norm}")
    print(f"Saved ROI config snippet: {args.calibration_output}")
    print(f"Saved ROI preview: {output_dir / 'roi_calibration.png'}")


def _print_live_line(result: PipelineResult) -> None:
    objects = ", ".join(f"{obj.category}/{obj.label}:{obj.score:.2f}" for obj in result.objects) or "none"
    semantic_unknowns = [obs.source.split(":", 1)[1] for obs in result.obstacles if obs.source.startswith("semantic_unknown:")]
    unknowns = ",".join(semantic_unknowns) if semantic_unknowns else "none"
    print(
        f"[live] device={result.device} fps={result.fps:.1f} objects={objects} "
        f"unknowns={unknowns} empty_spaces={len(result.empty_spaces)}"
    )


def _show_artifacts(prefix: str, artifacts: FrameArtifacts) -> None:
    cv2.imshow(prefix, artifacts.annotated_bgr)
    cv2.imshow(f"{prefix} - Top View", artifacts.top_view_bgr)
    if artifacts.map_view_bgr is not None:
        cv2.imshow(f"{prefix} - 3D Map", artifacts.map_view_bgr)


def read_temporal_frame(camera: RealSenseCamera, frame_count: int, median_color: bool = True) -> RGBDFrame:
    frame_count = max(1, int(frame_count))
    if frame_count == 1:
        return camera.get_frame()
    frames = [camera.get_frame() for _ in range(frame_count)]
    depth_stack = np.stack([frame.depth_raw.astype(np.float32) for frame in frames], axis=0)
    fused_depth = np.median(depth_stack, axis=0).astype(frames[-1].depth_raw.dtype)
    if median_color:
        color_stack = np.stack([frame.color_rgb.astype(np.float32) for frame in frames], axis=0)
        color_rgb = np.median(color_stack, axis=0).astype(np.uint8)
    else:
        color_rgb = frames[-1].color_rgb
    return RGBDFrame(
        color_rgb=color_rgb,
        depth_raw=fused_depth,
        intrinsics=frames[-1].intrinsics,
        timestamp_ms=frames[-1].timestamp_ms,
    )
