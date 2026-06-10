from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, normalize_category_name
from .pipeline import run_calibrate_roi, run_capture, run_demo, run_live, run_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 5: RGB-D cabinet object recognition and empty-space localization.")
    parser.add_argument("--mode", choices=["live", "capture", "replay", "demo", "calibrate_roi"], required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--session", default="data/sessions/demo")
    parser.add_argument("--output", default="outputs/latest")
    parser.add_argument("--new-item", default=None, help="Category or item prompt of the new item to place, e.g. Tableware or cup.")
    parser.add_argument("--item-size", default=None, help="New item physical size in meters: width,height,depth.")
    parser.add_argument("--detector", choices=["grounding_dino", "replay", "none"], default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="Live mode frame limit; 0 means until q/ESC.")
    parser.add_argument("--temporal-frames", type=int, default=None, help="Median-fuse this many RealSense frames.")
    parser.add_argument("--no-color-fusion", action="store_true", help="Use the latest RGB frame instead of median-fusing RGB frames.")
    parser.add_argument("--roi", default=None, help="Cabinet ROI in pixels: x1,y1,x2,y2. Use this to exclude background.")
    parser.add_argument("--roi-norm", default=None, help="Cabinet ROI normalized: x1,y1,x2,y2 in 0..1.")
    parser.add_argument("--roi-poly", default=None, help="Cabinet polygon ROI in pixels: x1,y1;x2,y2;x3,y3;...")
    parser.add_argument("--roi-poly-norm", default=None, help="Cabinet polygon ROI normalized: x1,y1;x2,y2;x3,y3;...")
    parser.add_argument("--no-auto-roi", action="store_true", help="Disable startup ROI auto calibration.")
    parser.add_argument("--print-every", type=int, default=30, help="Live mode terminal summary interval in frames; 0 disables.")
    parser.add_argument("--detect-every", type=int, default=5, help="Live mode semantic detection interval; cached detections are reused between updates.")
    parser.add_argument("--no-stability", action="store_true", help="Disable static-scene temporal smoothing/hysteresis.")
    parser.add_argument("--no-controls", action="store_true", help="Disable the live OpenCV parameter control panel.")
    parser.add_argument("--from-session", action="store_true", help="For calibrate_roi, estimate ROI from --session instead of live camera.")
    parser.add_argument("--calibration-output", default="configs/calibrated_roi.yaml")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--process-after-capture", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(Path(args.config) if args.config else None)
    if args.roi:
        cfg["geometry"]["cabinet_roi_px"] = [int(v.strip()) for v in args.roi.split(",")]
        cfg["geometry"]["cabinet_roi_poly_px"] = None
        cfg["geometry"]["cabinet_roi_poly_norm"] = None
        cfg["geometry"]["cabinet_roi_mode"] = "manual"
        cfg["geometry"]["auto_calibrate_on_start"] = False
    if args.roi_norm:
        cfg["geometry"]["cabinet_roi_norm"] = [float(v.strip()) for v in args.roi_norm.split(",")]
        cfg["geometry"]["cabinet_roi_poly_px"] = None
        cfg["geometry"]["cabinet_roi_poly_norm"] = None
        cfg["geometry"]["cabinet_roi_mode"] = "norm"
        cfg["geometry"]["auto_calibrate_on_start"] = False
    if args.roi_poly:
        cfg["geometry"]["cabinet_roi_poly_px"] = [
            [int(v.strip()) for v in pair.split(",")]
            for pair in args.roi_poly.split(";")
            if pair.strip()
        ]
        cfg["geometry"]["cabinet_roi_px"] = None
        cfg["geometry"]["cabinet_roi_mode"] = "poly"
        cfg["geometry"]["auto_calibrate_on_start"] = False
    if args.roi_poly_norm:
        cfg["geometry"]["cabinet_roi_poly_norm"] = [
            [float(v.strip()) for v in pair.split(",")]
            for pair in args.roi_poly_norm.split(";")
            if pair.strip()
        ]
        cfg["geometry"]["cabinet_roi_px"] = None
        cfg["geometry"]["cabinet_roi_mode"] = "poly"
        cfg["geometry"]["auto_calibrate_on_start"] = False
    if args.no_auto_roi:
        cfg["geometry"]["auto_calibrate_on_start"] = False
    if args.no_controls:
        cfg["ui"]["enabled"] = False
    if args.no_stability:
        cfg["stability"]["enabled"] = False
    if args.no_color_fusion:
        cfg["fusion"]["median_color"] = False
    if args.temporal_frames is None:
        args.temporal_frames = int(cfg.get("fusion", {}).get("temporal_frames", 1))
    if args.item_size:
        cfg["placement"]["active_item_size_m"] = [float(v.strip()) for v in args.item_size.split(",")]
    if args.new_item is None:
        args.new_item = cfg["placement"]["default_new_item_category"]
    args.new_item = normalize_category_name(args.new_item, cfg)
    args.display = (not args.no_display) and bool(cfg["output"].get("display", True))
    if args.no_display:
        args.display = False

    if args.mode == "demo":
        run_demo(args, cfg)
    elif args.mode == "calibrate_roi":
        run_calibrate_roi(args, cfg)
    elif args.mode == "replay":
        run_replay(args, cfg)
    elif args.mode == "capture":
        run_capture(args, cfg)
    elif args.mode == "live":
        run_live(args, cfg)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
