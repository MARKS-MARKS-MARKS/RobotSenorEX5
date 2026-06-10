from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .geometry import (
    filter_detections_to_single_shelf,
    filter_obstacles_against_detections,
    recommend_placement,
    subtract_occupied_from_empty_spaces,
)
from .types import Detection2D, EmptySpace, PipelineResult, Shelf


T = TypeVar("T")
Box = tuple[int, int, int, int]


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _smooth_box(old: Box, new: Box, alpha: float) -> Box:
    return tuple(int(round(alpha * n + (1.0 - alpha) * o)) for o, n in zip(old, new))  # type: ignore[return-value]


def _center_distance(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


@dataclass
class _Track(Generic[T]):
    item: T
    hits: int = 1
    misses: int = 0


class StaticSceneStabilizer:
    def __init__(self, cfg: dict, new_item_category: str) -> None:
        self.cfg = cfg
        self.new_item_category = new_item_category
        st = cfg.get("stability", {})
        self.enabled = bool(st.get("enabled", True))
        self.max_misses = int(st.get("max_misses", 8))
        self.object_min_hits = int(st.get("object_min_hits", 2))
        self.empty_min_hits = int(st.get("empty_min_hits", 2))
        self.shelf_min_hits = int(st.get("shelf_min_hits", 1))
        self.object_iou = float(st.get("object_iou_threshold", 0.35))
        self.empty_iou = float(st.get("empty_iou_threshold", 0.30))
        self.object_center_px = float(st.get("object_center_distance_px", 90))
        self.empty_center_px = float(st.get("empty_center_distance_px", 120))
        self.alpha = float(st.get("box_smoothing_alpha", 0.65))
        self.object_tracks: list[_Track[Detection2D]] = []
        self.empty_tracks: list[_Track[EmptySpace]] = []
        self.shelf_tracks: list[_Track[Shelf]] = []
        self.next_object_id = 1
        self.next_empty_id = 1

    def update(self, result: PipelineResult) -> PipelineResult:
        if not self.enabled:
            return result
        self._refresh_config()
        self.object_tracks = self._update_objects(self.object_tracks, result.objects)
        self.empty_tracks = self._update_empty_spaces(self.empty_tracks, result.empty_spaces)
        self.shelf_tracks = self._update_shelves(self.shelf_tracks, result.shelves)

        result.objects = sorted(
            [deepcopy(track.item) for track in self.object_tracks if track.hits >= self.object_min_hits],
            key=lambda item: (item.category or "", item.label, item.box[1], item.box[0]),
        )
        result.empty_spaces = sorted(
            [deepcopy(track.item) for track in self.empty_tracks if track.hits >= self.empty_min_hits],
            key=lambda item: (item.shelf_level, item.box[0], item.box[1]),
        )
        result.shelves = sorted(
            [deepcopy(track.item) for track in self.shelf_tracks if track.hits >= self.shelf_min_hits],
            key=lambda item: item.level,
        )
        result.objects = filter_detections_to_single_shelf(result.objects, result.shelves, self.cfg)
        result.obstacles = filter_obstacles_against_detections(result.obstacles, result.objects, self.cfg)
        result.empty_spaces = subtract_occupied_from_empty_spaces(
            result.shelves,
            result.empty_spaces,
            result.objects,
            result.obstacles,
            self.cfg,
        )
        result.recommendation = recommend_placement(self.new_item_category, result.objects, result.empty_spaces, self.cfg)
        return result

    def _update_objects(self, tracks: list[_Track[Detection2D]], items: list[Detection2D]) -> list[_Track[Detection2D]]:
        def compatible(track: _Track[Detection2D], item: Detection2D) -> bool:
            same_label_family = track.item.category == item.category
            overlap_match = _iou(track.item.box, item.box) >= self.object_iou
            center_match = _center_distance(track.item.box, item.box) <= self.object_center_px
            return (same_label_family and (overlap_match or center_match)) or _iou(track.item.box, item.box) >= 0.45

        def merge(old: Detection2D, new: Detection2D) -> Detection2D:
            item = deepcopy(new)
            item.box = _smooth_box(old.box, new.box, self.alpha)
            item.score = max(old.score * (1.0 - self.alpha) + new.score * self.alpha, new.score)
            if old.category and old.score >= new.score * 0.92:
                item.category = old.category
                item.label = old.label
            item.track_id = old.track_id
            return item

        updated = self._update_tracks(tracks, items, compatible, merge)
        for track in updated:
            if track.item.track_id is None:
                track.item.track_id = f"Obj{self.next_object_id}"
                self.next_object_id += 1
        return updated

    def _update_empty_spaces(self, tracks: list[_Track[EmptySpace]], items: list[EmptySpace]) -> list[_Track[EmptySpace]]:
        def compatible(track: _Track[EmptySpace], item: EmptySpace) -> bool:
            return track.item.shelf_level == item.shelf_level and (
                _iou(track.item.box, item.box) >= self.empty_iou
                or _center_distance(track.item.box, item.box) <= self.empty_center_px
            )

        def merge(old: EmptySpace, new: EmptySpace) -> EmptySpace:
            item = deepcopy(new)
            item.box = _smooth_box(old.box, new.box, self.alpha)
            ox, oy, oz = old.center_m
            nx, ny, nz = new.center_m
            item.center_m = (
                self.alpha * nx + (1.0 - self.alpha) * ox,
                self.alpha * ny + (1.0 - self.alpha) * oy,
                self.alpha * nz + (1.0 - self.alpha) * oz,
            )
            item.track_id = old.track_id
            return item

        updated = self._update_tracks(tracks, items, compatible, merge)
        for track in updated:
            if track.item.track_id is None:
                track.item.track_id = f"E{self.next_empty_id}"
                self.next_empty_id += 1
        return updated

    def _update_shelves(self, tracks: list[_Track[Shelf]], items: list[Shelf]) -> list[_Track[Shelf]]:
        by_level = {item.level: item for item in items}
        updated: list[_Track[Shelf]] = []
        seen: set[int] = set()
        for track in tracks:
            new = by_level.get(track.item.level)
            if new is None:
                track.misses += 1
                if track.misses <= self.max_misses:
                    updated.append(track)
                continue
            item = deepcopy(new)
            item.box = _smooth_box(track.item.box, new.box, self.alpha)
            item.median_depth_m = self.alpha * new.median_depth_m + (1.0 - self.alpha) * track.item.median_depth_m
            updated.append(_Track(item=item, hits=track.hits + 1, misses=0))
            seen.add(new.level)
        for item in items:
            if item.level not in seen:
                updated.append(_Track(item=deepcopy(item)))
        return updated

    def _update_tracks(
        self,
        tracks: list[_Track[T]],
        items: list[T],
        compatible: Callable[[_Track[T], T], bool],
        merge: Callable[[T, T], T],
    ) -> list[_Track[T]]:
        matched_items: set[int] = set()
        updated: list[_Track[T]] = []
        for track in tracks:
            best_idx = -1
            best_score = 0.0
            for idx, item in enumerate(items):
                if idx in matched_items or not compatible(track, item):
                    continue
                score = _iou(getattr(track.item, "box"), getattr(item, "box"))
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0:
                matched_items.add(best_idx)
                updated.append(_Track(item=merge(track.item, items[best_idx]), hits=track.hits + 1, misses=0))
            else:
                track.misses += 1
                if track.misses <= self.max_misses:
                    updated.append(track)
        for idx, item in enumerate(items):
            if idx not in matched_items:
                updated.append(_Track(item=deepcopy(item)))
        return updated

    def _refresh_config(self) -> None:
        st = self.cfg.get("stability", {})
        self.max_misses = int(st.get("max_misses", self.max_misses))
        self.object_min_hits = int(st.get("object_min_hits", self.object_min_hits))
        self.empty_min_hits = int(st.get("empty_min_hits", self.empty_min_hits))
        self.shelf_min_hits = int(st.get("shelf_min_hits", self.shelf_min_hits))
        self.object_iou = float(st.get("object_iou_threshold", self.object_iou))
        self.empty_iou = float(st.get("empty_iou_threshold", self.empty_iou))
        self.object_center_px = float(st.get("object_center_distance_px", self.object_center_px))
        self.empty_center_px = float(st.get("empty_center_distance_px", self.empty_center_px))
        self.alpha = float(st.get("box_smoothing_alpha", self.alpha))
