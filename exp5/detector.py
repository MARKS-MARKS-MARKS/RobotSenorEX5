from __future__ import annotations

import contextlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import flatten_prompts
from .device import DeviceInfo, choose_device
from .types import Detection2D


def normalize_proxy_environment() -> None:
    """Hugging Face/httpx accepts socks5/socks5h, not the generic socks scheme."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(name)
        if value and value.startswith("socks://"):
            os.environ[name] = "socks5h://" + value[len("socks://") :]


def _clip_box(box: list[float] | tuple[float, ...], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _is_duplicate_detection(a: Detection2D, b: Detection2D, iou_threshold: float, containment_threshold: float) -> bool:
    iou = _iou(a.box, b.box)
    if iou >= iou_threshold:
        return True
    inter = _intersection_area(a.box, b.box)
    containment = inter / float(min(_box_area(a.box), _box_area(b.box)))
    return containment >= containment_threshold


def _center_distance_ratio(a: Detection2D, b: Detection2D) -> float:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    distance = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    diag = min(
        max(1.0, ((ax2 - ax1) ** 2 + (ay2 - ay1) ** 2) ** 0.5),
        max(1.0, ((bx2 - bx1) ** 2 + (by2 - by1) ** 2) ** 0.5),
    )
    return distance / diag


def _center_distance_px(a: Detection2D, b: Detection2D) -> float:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def _is_duplicate_by_center(a: Detection2D, b: Detection2D, cfg: dict[str, Any]) -> bool:
    pp_cfg = cfg["detector"].get("postprocess", {})
    max_px = float(pp_cfg.get("deduplicate_center_distance_px", 42))
    max_ratio = float(pp_cfg.get("deduplicate_center_distance_ratio", 0.45))
    return _center_distance_px(a, b) <= max_px or _center_distance_ratio(a, b) <= max_ratio


def _better_duplicate_representative(candidate: Detection2D, current: Detection2D, cfg: dict[str, Any]) -> bool:
    pp_cfg = cfg["detector"].get("postprocess", {})
    cand_area = _box_area(candidate.box)
    curr_area = _box_area(current.box)
    tight_ratio = float(pp_cfg.get("duplicate_prefer_tighter_area_ratio", 1.8))

    if current.category is None and candidate.category is not None:
        return True
    if current.category is not None and candidate.category is None:
        return False
    if curr_area >= cand_area * tight_ratio:
        return True
    if candidate.score >= current.score + 0.08:
        return True
    return candidate.score >= current.score - 0.03 and cand_area < curr_area


def _is_duplicate_by_geometry(a: Detection2D, b: Detection2D, cfg: dict[str, Any]) -> bool:
    pp_cfg = cfg["detector"].get("postprocess", {})
    iou = _iou(a.box, b.box)
    if (a.category is None) != (b.category is None):
        inter = _intersection_area(a.box, b.box)
        area_a = _box_area(a.box)
        area_b = _box_area(b.box)
        containment = inter / float(min(area_a, area_b))
        area_ratio = max(area_a, area_b) / float(min(area_a, area_b))
        return (
            iou >= float(pp_cfg.get("unknown_known_iou_threshold", 0.45))
            or (
                containment >= float(pp_cfg.get("unknown_known_containment_threshold", 0.75))
                and area_ratio <= float(pp_cfg.get("unknown_known_area_ratio_max", 2.2))
            )
        )
    if _is_duplicate_detection(
        a,
        b,
        float(pp_cfg.get("duplicate_iou_threshold", 0.25)),
        float(pp_cfg.get("duplicate_containment_threshold", 0.60)),
    ):
        return True
    return (
        iou >= float(pp_cfg.get("duplicate_center_min_iou", 0.05))
        and _center_distance_ratio(a, b) <= float(pp_cfg.get("duplicate_center_distance_ratio", 0.28))
    )


def _merge_detection_group(group: list[Detection2D]) -> Detection2D:
    category_scores: dict[str, float] = {}
    for det in group:
        if det.category:
            category_scores[det.category] = category_scores.get(det.category, 0.0) + det.score
    if category_scores:
        best_category = max(category_scores.items(), key=lambda item: item[1])[0]
        label_candidates = [det for det in group if det.category == best_category]
    else:
        best_category = None
        label_candidates = group

    best = max(label_candidates, key=lambda det: (det.score, _box_area(det.box)))
    x1 = min(det.box[0] for det in group)
    y1 = min(det.box[1] for det in group)
    x2 = max(det.box[2] for det in group)
    y2 = max(det.box[3] for det in group)
    return Detection2D(
        label=best.label,
        category=best_category,
        score=best.score,
        box=(x1, y1, x2, y2),
        source=best.source,
    )


def postprocess_detections(
    detections: list[Detection2D],
    cfg: dict[str, Any],
    width: int,
    height: int,
) -> list[Detection2D]:
    del width, height
    kept: list[Detection2D] = []
    ordered = sorted(
        detections,
        key=lambda item: (item.category is not None, item.score, -_box_area(item.box)),
        reverse=True,
    )

    for det in ordered:
        duplicate_idx = next((idx for idx, old in enumerate(kept) if _is_duplicate_by_center(det, old, cfg)), None)
        if duplicate_idx is None:
            kept.append(det)
            continue
        if _better_duplicate_representative(det, kept[duplicate_idx], cfg):
            track_id = kept[duplicate_idx].track_id
            kept[duplicate_idx] = det
            kept[duplicate_idx].track_id = track_id

    return sorted(kept, key=lambda item: item.score, reverse=True)


def normalize_label(label: Any, label_to_category: dict[str, str | None]) -> tuple[str, str | None]:
    text = str(label).lower().strip()
    text = text.replace(".", " ").replace(",", " ")
    for candidate, category in label_to_category.items():
        if candidate in text:
            return candidate, category
    return text or "unknown", None


class BaseDetector:
    device_info: DeviceInfo

    def detect(self, color_rgb: np.ndarray) -> list[Detection2D]:
        raise NotImplementedError


class CachedDetector(BaseDetector):
    def __init__(self, inner: BaseDetector, interval: int) -> None:
        self.inner = inner
        self.interval = max(1, int(interval))
        self.device_info = inner.device_info
        self.frame_index = 0
        self.last_detections: list[Detection2D] = []

    def detect(self, color_rgb: np.ndarray) -> list[Detection2D]:
        if self.frame_index == 0 or self.frame_index % self.interval == 0:
            self.last_detections = self.inner.detect(color_rgb)
        self.frame_index += 1
        return self.last_detections


class NullDetector(BaseDetector):
    def __init__(self, reason: str = "detector disabled") -> None:
        self.reason = reason
        self.device_info = DeviceInfo("cpu", False, f"Using device: cpu ({reason})")

    def detect(self, color_rgb: np.ndarray) -> list[Detection2D]:
        print(f"[WARN] No semantic detections: {self.reason}")
        return []


class ReplayDetector(BaseDetector):
    def __init__(self, session_dir: str | Path, cfg: dict[str, Any]) -> None:
        self.session_dir = Path(session_dir)
        self.device_info = DeviceInfo("cpu", False, "Using device: cpu (replay detector)")
        _, self.label_to_category = flatten_prompts(cfg)

    def detect(self, color_rgb: np.ndarray) -> list[Detection2D]:
        path = self.session_dir / "detections.json"
        if not path.exists():
            print(f"[WARN] Replay detector did not find {path}")
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        height, width = color_rgb.shape[:2]
        detections: list[Detection2D] = []
        for item in data:
            label, category = normalize_label(item.get("label", "unknown"), self.label_to_category)
            detections.append(
                Detection2D(
                    label=label,
                    category=item.get("category") or category,
                    score=float(item.get("score", 1.0)),
                    box=_clip_box(item["box"], width, height),
                    source="replay",
                )
            )
        return detections


class GroundingDinoDetector(BaseDetector):
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.labels, self.label_to_category = flatten_prompts(cfg)
        self.text_prompt = ". ".join(self.labels) + "."
        det_cfg = cfg["detector"]
        self.box_threshold = float(det_cfg["box_threshold"])
        self.text_threshold = float(det_cfg["text_threshold"])
        self.half_precision = bool(det_cfg.get("half_precision", True))
        self.unknown_threshold = float(det_cfg.get("unknown_threshold", self.box_threshold))
        self.low_conf_known_as_unknown = bool(det_cfg.get("low_conf_known_as_unknown", True))
        self.category_thresholds = {
            str(category): float(threshold)
            for category, threshold in det_cfg.get("category_thresholds", {}).items()
        }
        self.device_info = choose_device(bool(det_cfg.get("prefer_gpu", True)))
        print(self.device_info.message)

        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except Exception as exc:
            raise RuntimeError(f"Grounding DINO dependencies are missing: {exc}") from exc

        self.torch = torch
        model_id = det_cfg["model_id"]
        normalize_proxy_environment()
        local_pref = det_cfg.get("local_files_only", "auto")
        local_first = local_pref == "auto" or bool(local_pref)
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_first)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=local_first)
        except Exception:
            if local_pref != "auto":
                raise
            print("[WARN] Local Grounding DINO cache not ready; retrying online download.")
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=False)
        self.model.to(self.device_info.device)
        self.model.eval()
        print(f"Loaded Grounding DINO model: {model_id}")

    def _move_inputs(self, inputs: Any) -> Any:
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(self.device_info.device)
            else:
                moved[key] = value
        return moved

    def detect(self, color_rgb: np.ndarray) -> list[Detection2D]:
        from PIL import Image

        image = Image.fromarray(color_rgb)
        height, width = color_rgb.shape[:2]
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt")
        inputs = self._move_inputs(inputs)

        use_amp = self.device_info.device == "cuda" and self.half_precision
        autocast = (
            self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )
        with self.torch.no_grad(), autocast:
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([[height, width]], device=self.device_info.device)
        post_threshold = min([self.box_threshold, self.unknown_threshold] + list(self.category_thresholds.values()))
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            post_process = self.processor.post_process_grounded_object_detection
            signature = inspect.signature(post_process)
            kwargs: dict[str, Any] = {
                "outputs": outputs,
                "target_sizes": target_sizes,
            }
            if "input_ids" in signature.parameters:
                kwargs["input_ids"] = inputs.get("input_ids")
            if "box_threshold" in signature.parameters:
                kwargs["box_threshold"] = post_threshold
            elif "threshold" in signature.parameters:
                kwargs["threshold"] = post_threshold
            if "text_threshold" in signature.parameters:
                kwargs["text_threshold"] = self.text_threshold
            if "text_labels" in signature.parameters:
                kwargs["text_labels"] = [self.labels]
            results = post_process(**kwargs)[0]
        else:
            results = self.processor.post_process_object_detection(
                outputs,
                threshold=post_threshold,
                target_sizes=target_sizes,
            )[0]

        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results["text_labels"] if "text_labels" in results else results.get("labels", [])
        detections: list[Detection2D] = []
        for box, score, label in zip(boxes, scores, labels):
            label_text, category = normalize_label(label, self.label_to_category)
            score_value = float(score.detach().cpu().item() if hasattr(score, "detach") else score)
            if category is None:
                if score_value < self.unknown_threshold:
                    continue
                output_category = None
                source = "grounding_dino_unknown"
            else:
                known_threshold = self.category_thresholds.get(category, self.box_threshold)
                if score_value >= known_threshold:
                    output_category = category
                    source = "grounding_dino"
                elif self.low_conf_known_as_unknown and score_value >= self.unknown_threshold:
                    output_category = None
                    source = "grounding_dino_low_conf_known"
                else:
                    continue
            detections.append(
                Detection2D(
                    label=label_text,
                    category=output_category,
                    score=score_value,
                    box=_clip_box(box.detach().cpu().tolist() if hasattr(box, "detach") else box, width, height),
                    source=source,
                )
            )
        return postprocess_detections(detections, self.cfg, width, height)


def build_detector(cfg: dict[str, Any], detector_type: str | None = None, session_dir: str | Path | None = None) -> BaseDetector:
    detector_type = detector_type or cfg["detector"].get("type", "grounding_dino")
    if detector_type == "none":
        return NullDetector("semantic detector disabled by CLI")
    if detector_type == "replay":
        if not session_dir:
            return NullDetector("replay detector requires --session")
        return ReplayDetector(session_dir, cfg)
    if detector_type != "grounding_dino":
        return NullDetector(f"unknown detector type: {detector_type}")

    try:
        return GroundingDinoDetector(cfg)
    except Exception as exc:
        if cfg["detector"].get("allow_fallback", True):
            return NullDetector(f"Grounding DINO unavailable: {exc}")
        raise
