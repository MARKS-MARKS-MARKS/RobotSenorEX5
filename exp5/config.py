from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "camera": {
        "width": 640,
        "height": 480,
        "fps": 30,
        "depth_min_m": 0.25,
        "depth_max_m": 2.20,
        "align_to_color": True,
    },
    "fusion": {
        "temporal_frames": 2,
        "median_color": True,
    },
    "detector": {
        "type": "grounding_dino",
        "model_id": "IDEA-Research/grounding-dino-tiny",
        "local_files_only": "auto",
        "prefer_gpu": True,
        "box_threshold": 0.30,
        "text_threshold": 0.25,
        "half_precision": True,
        "allow_fallback": True,
        "prompts": {
            "Cutlery": ["fork", "knife", "spoon", "chopsticks", "table knife", "silverware"],
            "Tableware": ["cup", "mug", "plate", "dish", "glass", "bowl"],
            "Breakfast": [
                "cereal",
                "cereal box",
                "breakfast cereal",
                "milk",
                "milk carton",
                "bread",
                "oatmeal",
                "yogurt",
                "breakfast food",
            ],
            "Soft Drink": ["cola", "ice tea", "iced tea", "soft drink", "soda", "soda can", "cola can", "beverage bottle", "juice"],
        },
        "unknown_prompts": [
            "object",
            "item",
            "package",
            "container",
            "box",
            "food package",
            "fruit",
            "snack",
            "toy",
            "book",
        ],
        "unknown_threshold": 0.28,
        "low_conf_known_as_unknown": True,
        "category_thresholds": {
            "Cutlery": 0.40,
            "Tableware": 0.30,
            "Breakfast": 0.33,
            "Soft Drink": 0.30,
        },
        "postprocess": {
            "deduplicate_center_distance_px": 42,
            "deduplicate_center_distance_ratio": 0.45,
            "duplicate_prefer_tighter_area_ratio": 1.8,
            "duplicate_iou_threshold": 0.25,
            "duplicate_containment_threshold": 0.60,
            "duplicate_center_distance_ratio": 0.28,
            "duplicate_center_min_iou": 0.05,
            "unknown_known_iou_threshold": 0.45,
            "unknown_known_containment_threshold": 0.75,
            "unknown_known_area_ratio_max": 2.2,
            "split_wide_detections": True,
            "split_aspect_ratio": 2.4,
            "split_min_width_ratio": 0.28,
            "split_min_component_area_px": 180,
            "split_component_gap_px": 8,
        },
    },
    "mapping": {
        "enabled": True,
        "stride_px": 5,
        "update_every": 5,
        "max_points": 35000,
        "save_ply": True,
        "render_width": 520,
        "render_height": 390,
    },
    "geometry": {
        "auto_calibrate_on_start": True,
        "auto_calibration_frames": 5,
        "cabinet_roi_mode": "norm",
        "cabinet_roi_px": None,
        "cabinet_roi_norm": [0.156, 0.05, 0.93, 0.98],
        "cabinet_roi_poly_px": None,
        "cabinet_roi_poly_norm": None,
        "show_cabinet_roi": True,
        "filter_detections_to_roi": True,
        "detection_roi_min_overlap": 0.60,
        "filter_detections_to_single_shelf": True,
        "detection_shelf_margin_px": 4,
        "roi_edge_min_ratio": 0.35,
        "voxel_size_m": 0.01,
        "min_valid_depth_ratio": 0.10,
        "shelf_split_mode": "equal_count",
        "shelf_count": 3,
        "shelf_boundaries_px": None,
        "shelf_boundaries_norm": None,
        "horizontal_line_min_length_ratio": 0.25,
        "shelf_min_height_px": 50,
        "shelf_merge_gap_px": 18,
        "shelf_margin_px": 8,
        "shelf_edge_peak_percentile": 88,
        "shelf_edge_peak_min_strength": 4.0,
        "shelf_short_segment_merge_ratio": 1.15,
        "shelf_boundary_downshift_px": 12,
        "foreground_delta_m": 0.08,
        "foreground_background_percentile": 72,
        "depth_hole_fill_kernel": 5,
        "obstacle_min_area_px": 280,
        "obstacle_min_width_px": 12,
        "obstacle_min_height_px": 14,
        "obstacle_color_delta": 34,
        "obstacle_max_area_ratio": 0.35,
        "obstacle_max_width_ratio": 0.78,
        "obstacle_max_height_ratio": 0.90,
        "obstacle_known_overlap_threshold": 0.45,
        "obstacle_inflate_px": 12,
        "empty_project_occupied_to_full_shelf": True,
        "empty_final_project_occupied": True,
        "empty_final_min_width_px": 24,
        "empty_object_shelf_overlap_min": 0.10,
        "empty_roi_column_min_ratio": 0.20,
        "empty_center_mode": "center",
        "empty_min_width_m": 0.14,
        "empty_min_height_m": 0.12,
        "empty_min_depth_m": 0.12,
        "empty_max_per_shelf": 2,
        "empty_contact_band_ratio": 0.45,
        "empty_column_occupied_ratio_max": 0.08,
        "empty_column_full_occupied_ratio_max": 0.03,
        "empty_free_ratio": 0.72,
    },
    "placement": {
        "default_new_item_category": "Tableware",
        "default_item_size_m": [0.14, 0.12, 0.12],
        "size_safety_margin_m": 0.02,
        "same_category_bonus": 2.0,
        "free_area_weight": 1.0,
        "distance_weight": 0.4,
    },
    "stability": {
        "enabled": True,
        "object_min_hits": 1,
        "empty_min_hits": 1,
        "shelf_min_hits": 1,
        "max_misses": 12,
        "object_iou_threshold": 0.20,
        "object_center_distance_px": 90,
        "empty_iou_threshold": 0.22,
        "empty_center_distance_px": 120,
        "box_smoothing_alpha": 0.45,
    },
    "ui": {
        "enabled": True,
        "control_window": "Exp5 Controls",
    },
    "output": {
        "dir": "outputs/latest",
        "save_visualization": True,
        "save_top_view": True,
        "save_json": True,
        "display": True,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if not path:
        return cfg

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML config files. Run scripts/setup_env.sh first.") from exc

    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(cfg, user_cfg)


def flatten_prompts(cfg: dict[str, Any]) -> tuple[list[str], dict[str, str | None]]:
    prompts = cfg["detector"]["prompts"]
    labels: list[str] = []
    label_to_category: dict[str, str | None] = {}
    for category, names in prompts.items():
        for name in names:
            clean = str(name).strip().lower()
            labels.append(clean)
            label_to_category[clean] = category
    for name in cfg["detector"].get("unknown_prompts", []):
        clean = str(name).strip().lower()
        if clean and clean not in label_to_category:
            labels.append(clean)
            label_to_category[clean] = None
    return labels, label_to_category


def normalize_category_name(value: str, cfg: dict[str, Any]) -> str:
    text = value.strip()
    if not text:
        return str(cfg["placement"]["default_new_item_category"])
    categories = {str(name).lower(): str(name) for name in cfg["detector"]["prompts"].keys()}
    if text.lower() in categories:
        return categories[text.lower()]
    _, label_to_category = flatten_prompts(cfg)
    mapped = label_to_category.get(text.lower())
    return mapped if mapped is not None else text
