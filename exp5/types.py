from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


Box = tuple[int, int, int, int]
Point3D = tuple[float, float, float]


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    depth_scale: float = 0.001

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Intrinsics":
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            ppx=float(data["ppx"]),
            ppy=float(data["ppy"]),
            depth_scale=float(data.get("depth_scale", 0.001)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RGBDFrame:
    color_rgb: Any
    depth_raw: Any
    intrinsics: Intrinsics
    timestamp_ms: float = 0.0


@dataclass
class Detection2D:
    label: str
    category: str | None
    score: float
    box: Box
    source: str = "grounding_dino"
    track_id: str | None = None
    center_m: Point3D | None = None
    contact_center_m: Point3D | None = None
    size_m: Point3D | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["box"] = list(self.box)
        return data


@dataclass
class Shelf:
    level: int
    box: Box
    median_depth_m: float
    plane_hint: str = "image-depth horizontal band"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["box"] = list(self.box)
        return data


@dataclass
class EmptySpace:
    shelf_level: int
    box: Box
    center_m: Point3D
    size_m: Point3D
    score: float
    track_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["box"] = list(self.box)
        return data


@dataclass
class Obstacle:
    box: Box
    center_m: Point3D | None = None
    source: str = "depth_cluster"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["box"] = list(self.box)
        return data


@dataclass
class PlacementRecommendation:
    new_item_category: str
    empty_space: EmptySpace | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_item_category": self.new_item_category,
            "empty_space": self.empty_space.to_dict() if self.empty_space else None,
            "reason": self.reason,
        }


@dataclass
class PipelineResult:
    objects: list[Detection2D] = field(default_factory=list)
    shelves: list[Shelf] = field(default_factory=list)
    empty_spaces: list[EmptySpace] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    recommendation: PlacementRecommendation | None = None
    map_summary: dict[str, Any] = field(default_factory=dict)
    fps: float = 0.0
    device: str = "cpu"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "fps": self.fps,
            "objects": [obj.to_dict() for obj in self.objects],
            "shelves": [shelf.to_dict() for shelf in self.shelves],
            "empty_spaces": [space.to_dict() for space in self.empty_spaces],
            "obstacles": [obs.to_dict() for obs in self.obstacles],
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "map_summary": self.map_summary,
            "warnings": self.warnings,
        }
