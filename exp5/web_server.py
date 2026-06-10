from __future__ import annotations

import json
import threading
import time
import uuid
from io import BytesIO
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file, send_from_directory
import cv2
import numpy as np

from .config import load_config, normalize_category_name
from .detector import BaseDetector, build_detector
from .geometry import recommend_placement
from .pipeline import FrameArtifacts, auto_calibrate_roi_from_frame, process_frame, refresh_visuals, save_outputs
from .realsense_io import load_session


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "web_static"
SESSION_ROOT = ROOT_DIR / "data" / "sessions"
WEB_OUTPUT_ROOT = ROOT_DIR / "outputs" / "web"
ASSET_NAMES = {"annotated.png", "top_view.png", "map_3d.png", "scene_map.ply", "result.json"}
SESSION_ASSET_NAMES = {"color.png", "depth.png", "intrinsics.json", "meta.json", "detections.json"}


@dataclass
class RunState:
    id: str
    session: str
    detector: str
    new_item: str
    item_size: list[float] | None
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    output_dir: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "session": self.session,
            "detector": self.detector,
            "new_item": self.new_item,
            "item_size": self.item_size,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_dir": self.output_dir,
            "result": self.result,
            "error": self.error,
        }
        if self.status == "done":
            payload["assets"] = {
                name: f"/api/assets/{self.id}/{name}"
                for name in ("annotated.png", "top_view.png", "map_3d.png", "result.json")
                if (WEB_OUTPUT_ROOT / self.id / name).exists()
            }
        return payload


class WebRuntime:
    def __init__(self, config_path: str | Path = "configs/default.yaml") -> None:
        self.config_path = Path(config_path)
        self.runs: dict[str, RunState] = {}
        self.artifacts: dict[str, FrameArtifacts] = {}
        self.frames: dict[str, Any] = {}
        self.configs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.detector_lock = threading.Lock()
        self.grounding_detector: BaseDetector | None = None

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        if not SESSION_ROOT.exists():
            return sessions
        for session_dir in sorted(path for path in SESSION_ROOT.iterdir() if path.is_dir()):
            color_path = session_dir / "color.png"
            depth_path = session_dir / "depth.png"
            intrinsics_path = session_dir / "intrinsics.json"
            if not color_path.exists() or not depth_path.exists() or not intrinsics_path.exists():
                continue
            meta = _read_json(session_dir / "meta.json", {})
            detections = _read_json(session_dir / "detections.json", None)
            sessions.append(
                {
                    "name": session_dir.name,
                    "path": str(session_dir.relative_to(ROOT_DIR)),
                    "has_replay_detections": detections is not None,
                    "detection_count": len(detections) if isinstance(detections, list) else 0,
                    "timestamp_ms": meta.get("timestamp_ms") if isinstance(meta, dict) else None,
                    "assets": {
                        "color": f"/api/session-assets/{session_dir.name}/color.png",
                        "depth": f"/api/session-assets/{session_dir.name}/depth.png",
                    },
                }
            )
        return sessions

    def create_run(self, payload: dict[str, Any]) -> RunState:
        cfg = load_config(self.config_path)
        session_name = _safe_name(str(payload.get("session") or "demo"))
        detector = str(payload.get("detector") or "grounding_dino")
        if detector not in {"grounding_dino", "replay", "none"}:
            raise ValueError("detector must be one of: grounding_dino, replay, none")
        session_dir = SESSION_ROOT / session_name
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_name}")

        new_item = normalize_category_name(str(payload.get("new_item") or cfg["placement"]["default_new_item_category"]), cfg)
        item_size = _parse_item_size(payload.get("item_size"))
        run_id = uuid.uuid4().hex[:12]
        state = RunState(
            id=run_id,
            session=session_name,
            detector=detector,
            new_item=new_item,
            item_size=item_size,
        )
        with self.lock:
            self.runs[run_id] = state

        thread = threading.Thread(target=self._run_pipeline, args=(state, payload), daemon=True)
        thread.start()
        return state

    def get_run(self, run_id: str) -> RunState | None:
        with self.lock:
            return self.runs.get(run_id)

    def _run_pipeline(self, state: RunState, payload: dict[str, Any]) -> None:
        self._update_state(state.id, status="running", started_at=time.time())
        try:
            cfg = load_config(self.config_path)
            cfg["output"]["display"] = False
            cfg["placement"]["active_item_size_m"] = cfg["placement"].get("candidate_item_size_m", [0.02, 0.02, 0.02])
            cfg["geometry"]["semantic_unknown_obstacles_enabled"] = bool(payload.get("use_unknown_obstacles", False))
            frame = load_session(SESSION_ROOT / state.session)
            output_dir = WEB_OUTPUT_ROOT / state.id
            roi_norm = _parse_roi_norm(payload.get("roi_norm"))
            if roi_norm is not None:
                cfg["geometry"]["cabinet_roi_norm"] = roi_norm
                cfg["geometry"]["cabinet_roi_px"] = None
                cfg["geometry"]["cabinet_roi_poly_px"] = None
                cfg["geometry"]["cabinet_roi_poly_norm"] = None
                cfg["geometry"]["cabinet_roi_mode"] = "norm"
                cfg["geometry"]["auto_calibrate_on_start"] = False
            auto_roi = bool(payload.get("auto_roi", True)) and roi_norm is None
            if auto_roi:
                auto_calibrate_roi_from_frame(frame, cfg, output_dir)
            detector = self._build_detector(cfg, state.detector, SESSION_ROOT / state.session)
            artifacts = process_frame(frame, detector, cfg, state.new_item, render=True, build_map=True)
            artifacts.result.recommendation = None
            refresh_visuals(frame, artifacts, cfg)
            save_outputs(artifacts, output_dir, cfg)
            with self.lock:
                self.artifacts[state.id] = artifacts
                self.frames[state.id] = frame
                self.configs[state.id] = deepcopy(cfg)
            self._update_state(
                state.id,
                status="done",
                finished_at=time.time(),
                output_dir=str(output_dir.relative_to(ROOT_DIR)),
                result=_summarize_result(artifacts.result.to_dict()),
            )
        except Exception as exc:
            self._update_state(state.id, status="error", finished_at=time.time(), error=str(exc))

    def _build_detector(self, cfg: dict[str, Any], detector_type: str, session_dir: Path) -> BaseDetector:
        if detector_type != "grounding_dino":
            return build_detector(cfg, detector_type, session_dir=session_dir)
        with self.detector_lock:
            if self.grounding_detector is None:
                self.grounding_detector = build_detector(cfg, detector_type, session_dir=session_dir)
                if "Grounding DINO unavailable" in self.grounding_detector.device_info.message:
                    message = self.grounding_detector.device_info.message.replace("Using device: cpu (", "").rstrip(")")
                    self.grounding_detector = None
                    raise RuntimeError(message)
            return self.grounding_detector

    def _update_state(self, run_id: str, **updates: Any) -> None:
        with self.lock:
            state = self.runs[run_id]
            for key, value in updates.items():
                setattr(state, key, value)

    def update_placement(self, run_id: str, payload: dict[str, Any]) -> RunState:
        with self.lock:
            stored_cfg = self.configs.get(run_id)
        cfg = deepcopy(stored_cfg) if stored_cfg is not None else load_config(self.config_path)
        cfg["output"]["display"] = False
        item_size = _parse_item_size(payload.get("item_size"))
        if item_size:
            cfg["placement"]["active_item_size_m"] = item_size
        new_item = normalize_category_name(str(payload.get("new_item") or cfg["placement"]["default_new_item_category"]), cfg)

        with self.lock:
            state = self.runs.get(run_id)
            base_artifacts = self.artifacts.get(run_id)
            frame = self.frames.get(run_id)
        if state is None or base_artifacts is None or frame is None:
            raise FileNotFoundError("Run result not found. Run detection first.")
        if state.status != "done":
            raise RuntimeError("Detection is not complete yet.")

        artifacts = deepcopy(base_artifacts)
        filtered_spaces = _filter_spaces_for_item(artifacts.result.empty_spaces, cfg)
        artifacts.result.recommendation = recommend_placement(new_item, artifacts.result.objects, filtered_spaces, cfg)
        if artifacts.result.recommendation and artifacts.result.recommendation.empty_space:
            artifacts.result.recommendation.reason += (
                f" {len(filtered_spaces)}/{len(artifacts.result.empty_spaces)} detected spaces fit the new item size."
            )
        refresh_visuals(frame, artifacts, cfg)
        output_dir = WEB_OUTPUT_ROOT / run_id
        save_outputs(artifacts, output_dir, cfg)
        result = _summarize_result(artifacts.result.to_dict())

        with self.lock:
            self.artifacts[run_id] = artifacts
            state.new_item = new_item
            state.item_size = item_size
            state.result = result
            state.output_dir = str(output_dir.relative_to(ROOT_DIR))
            state.finished_at = time.time()
        return state


def create_app(config_path: str | Path = "configs/default.yaml") -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
    runtime = WebRuntime(config_path)

    @app.get("/")
    def index() -> Any:
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"ok": True, "sessions": len(runtime.list_sessions())})

    @app.get("/api/sessions")
    def sessions() -> Any:
        return jsonify({"sessions": runtime.list_sessions()})

    @app.post("/api/runs")
    def create_run() -> Any:
        try:
            state = runtime.create_run(request.get_json(silent=True) or {})
            return jsonify(state.to_dict()), 202
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/runs/<run_id>")
    def get_run(run_id: str) -> Any:
        state = runtime.get_run(_safe_name(run_id))
        if state is None:
            return jsonify({"error": "run not found"}), 404
        return jsonify(state.to_dict())

    @app.post("/api/runs/<run_id>/placement")
    def update_placement(run_id: str) -> Any:
        try:
            state = runtime.update_placement(_safe_name(run_id), request.get_json(silent=True) or {})
            return jsonify(state.to_dict())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/assets/<run_id>/<filename>")
    def output_asset(run_id: str, filename: str) -> Any:
        safe_run_id = _safe_name(run_id)
        if filename not in ASSET_NAMES:
            return jsonify({"error": "asset not allowed"}), 404
        path = WEB_OUTPUT_ROOT / safe_run_id / filename
        if not path.exists():
            return jsonify({"error": "asset not found"}), 404
        return send_file(path)

    @app.get("/api/session-assets/<session>/<filename>")
    def session_asset(session: str, filename: str) -> Any:
        safe_session = _safe_name(session)
        if filename not in SESSION_ASSET_NAMES:
            return jsonify({"error": "asset not allowed"}), 404
        path = SESSION_ROOT / safe_session / filename
        if not path.exists():
            return jsonify({"error": "asset not found"}), 404
        if filename == "depth.png":
            preview = _depth_preview_png(path)
            return send_file(preview, mimetype="image/png")
        return send_file(path)

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Offline web UI for Experiment 5.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _parse_item_size(value: Any) -> list[float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        parts = value
    else:
        raise ValueError("item_size must be width,height,depth")
    if len(parts) != 3:
        raise ValueError("item_size must contain 3 values: width,height,depth")
    return [float(part) for part in parts]


def _parse_roi_norm(value: Any) -> list[float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        parts = value
    else:
        raise ValueError("roi_norm must be x1,y1,x2,y2")
    if len(parts) != 4:
        raise ValueError("roi_norm must contain 4 values: x1,y1,x2,y2")
    x1, y1, x2, y2 = [float(part) for part in parts]
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
        raise ValueError("roi_norm is too small")
    return [x1, y1, x2, y2]


def _filter_spaces_for_item(empty_spaces: list[Any], cfg: dict[str, Any]) -> list[Any]:
    placement = cfg.get("placement", {})
    item_size = placement.get("active_item_size_m") or placement.get("default_item_size_m") or [0.0, 0.0, 0.0]
    margin = float(placement.get("size_safety_margin_m", 0.0))
    required_width = float(item_size[0]) + margin
    required_height = float(item_size[1]) + margin
    return [
        space
        for space in empty_spaces
        if float(space.size_m[0]) >= required_width and float(space.size_m[1]) >= required_height
    ]


def _depth_preview_png(path: Path) -> BytesIO:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    valid = depth[depth > 0]
    if valid.size:
        lo = float(np.percentile(valid, 2))
        hi = float(np.percentile(valid, 98))
        if hi <= lo:
            hi = lo + 1.0
        normalized = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    else:
        normalized = np.zeros_like(depth, dtype=np.float32)
    preview = (normalized * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".png", preview)
    if not ok:
        raise RuntimeError("Failed to encode depth preview")
    return BytesIO(encoded.tobytes())


def _safe_name(value: str) -> str:
    name = Path(value).name
    if not name or name in {".", ".."}:
        raise ValueError("invalid name")
    return name


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    summary = deepcopy(result)
    recommendation = summary.get("recommendation") or {}
    empty_space = recommendation.get("empty_space") or None
    summary["counts"] = {
        "objects": len(summary.get("objects", [])),
        "shelves": len(summary.get("shelves", [])),
        "empty_spaces": len(summary.get("empty_spaces", [])),
        "obstacles": len(summary.get("obstacles", [])),
    }
    summary["recommendation_summary"] = {
        "new_item_category": recommendation.get("new_item_category"),
        "shelf_level": empty_space.get("shelf_level") if isinstance(empty_space, dict) else None,
        "center_m": empty_space.get("center_m") if isinstance(empty_space, dict) else None,
        "reason": recommendation.get("reason"),
    }
    return summary


if __name__ == "__main__":
    main()
