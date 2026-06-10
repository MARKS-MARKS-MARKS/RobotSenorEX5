from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import normalize_category_name
from .detector import BaseDetector, CachedDetector


FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

CATEGORY_CN = {
    "Cutlery": "刀叉勺类",
    "Tableware": "杯盘碗类",
    "Breakfast": "早餐食品",
    "Soft Drink": "软饮料",
}


class RuntimeControls:
    def __init__(self, cfg: dict[str, Any], args: Any, detector: BaseDetector) -> None:
        self.cfg = cfg
        self.args = args
        self.detector = detector
        # OpenCV/Qt trackbars need an ASCII window handle on some Linux builds.
        # Chinese is rendered inside the image panel instead of used as the OS window name.
        self.window = str(cfg.get("ui", {}).get("control_window", "Exp5 Controls")) or "Exp5 Controls"
        self.window = self.window.encode("ascii", "ignore").decode("ascii").strip() or "Exp5 Controls"
        self.categories = list(cfg["detector"]["prompts"].keys())
        self.fonts = self._load_fonts()
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 900, 840)
        cv2.imshow(self.window, np.full((80, 420, 3), (32, 34, 36), dtype=np.uint8))
        cv2.waitKey(1)
        self._create_trackbars(cfg, args)

    def _create_trackbars(self, cfg: dict[str, Any], args: Any) -> None:
        placement = cfg["placement"]
        item_size = placement.get("active_item_size_m") or placement.get("default_item_size_m") or [0.14, 0.12, 0.12]
        roi = cfg["geometry"].get("cabinet_roi_norm", [0.0, 0.0, 1.0, 1.0])
        cat_idx = max(0, self.categories.index(args.new_item) if args.new_item in self.categories else 0)
        self._trackbar("new item", cat_idx, max(0, len(self.categories) - 1))
        self._trackbar("item W cm", int(round(float(item_size[0]) * 100)), 80)
        self._trackbar("item H cm", int(round(float(item_size[1]) * 100)), 80)
        self._trackbar("item D cm", int(round(float(item_size[2]) * 100)), 80)
        self._trackbar("box th %", int(round(float(cfg["detector"]["box_threshold"]) * 100)), 95)
        self._trackbar("text th %", int(round(float(cfg["detector"]["text_threshold"]) * 100)), 95)
        thresholds = cfg["detector"].get("category_thresholds", {})
        self._trackbar("Cutlery th %", int(round(float(thresholds.get("Cutlery", cfg["detector"]["box_threshold"])) * 100)), 95)
        self._trackbar("Tableware th %", int(round(float(thresholds.get("Tableware", cfg["detector"]["box_threshold"])) * 100)), 95)
        self._trackbar("Breakfast th %", int(round(float(thresholds.get("Breakfast", cfg["detector"]["box_threshold"])) * 100)), 95)
        self._trackbar("Drink th %", int(round(float(thresholds.get("Soft Drink", cfg["detector"]["box_threshold"])) * 100)), 95)
        self._trackbar("Unknown th %", int(round(float(cfg["detector"].get("unknown_threshold", cfg["detector"]["box_threshold"])) * 100)), 95)
        self._trackbar("detect every", max(1, int(args.detect_every)), 20)
        self._trackbar("temporal fr", max(1, int(args.temporal_frames)), 9)
        self._trackbar("ROI x1 %", int(round(float(roi[0]) * 100)), 100)
        self._trackbar("ROI y1 %", int(round(float(roi[1]) * 100)), 100)
        self._trackbar("ROI x2 %", int(round(float(roi[2]) * 100)), 100)
        self._trackbar("ROI y2 %", int(round(float(roi[3]) * 100)), 100)
        self._trackbar("empty W cm", int(round(float(cfg["geometry"]["empty_min_width_m"]) * 100)), 80)
        self._trackbar("empty H cm", int(round(float(cfg["geometry"]["empty_min_height_m"]) * 100)), 80)
        self._trackbar("max misses", int(cfg["stability"]["max_misses"]), 30)
        self._trackbar("smooth %", int(round(float(cfg["stability"]["box_smoothing_alpha"]) * 100)), 95)

    def _trackbar(self, name: str, value: int, maximum: int) -> None:
        cv2.createTrackbar(name, self.window, max(0, min(maximum, value)), max(1, maximum), lambda _value: None)

    def apply(self, cfg: dict[str, Any], args: Any, detector: BaseDetector, stabilizer: Any) -> None:
        category = self.categories[min(self._pos("new item"), len(self.categories) - 1)]
        args.new_item = normalize_category_name(category, cfg)
        stabilizer.new_item_category = args.new_item
        cfg["placement"]["active_item_size_m"] = [
            max(1, self._pos("item W cm")) / 100.0,
            max(1, self._pos("item H cm")) / 100.0,
            max(1, self._pos("item D cm")) / 100.0,
        ]

        box_threshold = max(1, self._pos("box th %")) / 100.0
        text_threshold = max(1, self._pos("text th %")) / 100.0
        cfg["detector"]["box_threshold"] = box_threshold
        cfg["detector"]["text_threshold"] = text_threshold
        category_thresholds = {
            "Cutlery": max(1, self._pos("Cutlery th %")) / 100.0,
            "Tableware": max(1, self._pos("Tableware th %")) / 100.0,
            "Breakfast": max(1, self._pos("Breakfast th %")) / 100.0,
            "Soft Drink": max(1, self._pos("Drink th %")) / 100.0,
        }
        unknown_threshold = max(1, self._pos("Unknown th %")) / 100.0
        cfg["detector"]["category_thresholds"] = category_thresholds
        cfg["detector"]["unknown_threshold"] = unknown_threshold
        self._apply_detector_thresholds(detector, box_threshold, text_threshold, category_thresholds, unknown_threshold)

        args.detect_every = max(1, self._pos("detect every"))
        if isinstance(detector, CachedDetector):
            detector.interval = args.detect_every
        args.temporal_frames = max(1, self._pos("temporal fr"))

        x1 = self._pos("ROI x1 %") / 100.0
        y1 = self._pos("ROI y1 %") / 100.0
        x2 = self._pos("ROI x2 %") / 100.0
        y2 = self._pos("ROI y2 %") / 100.0
        x2 = max(x1 + 0.05, min(1.0, x2))
        y2 = max(y1 + 0.05, min(1.0, y2))
        cfg["geometry"]["cabinet_roi_mode"] = "norm"
        cfg["geometry"]["cabinet_roi_norm"] = [x1, y1, x2, y2]
        cfg["geometry"]["cabinet_roi_px"] = None
        cfg["geometry"]["cabinet_roi_poly_px"] = None
        cfg["geometry"]["cabinet_roi_poly_norm"] = None

        cfg["geometry"]["empty_min_width_m"] = max(1, self._pos("empty W cm")) / 100.0
        cfg["geometry"]["empty_min_height_m"] = max(1, self._pos("empty H cm")) / 100.0
        cfg["stability"]["max_misses"] = max(0, self._pos("max misses"))
        cfg["stability"]["box_smoothing_alpha"] = max(1, self._pos("smooth %")) / 100.0

    def sync_from_cfg(self, cfg: dict[str, Any], args: Any) -> None:
        roi = cfg["geometry"].get("cabinet_roi_norm", [0.0, 0.0, 1.0, 1.0])
        self._set("ROI x1 %", int(round(float(roi[0]) * 100)))
        self._set("ROI y1 %", int(round(float(roi[1]) * 100)))
        self._set("ROI x2 %", int(round(float(roi[2]) * 100)))
        self._set("ROI y2 %", int(round(float(roi[3]) * 100)))
        if args.new_item in self.categories:
            self._set("new item", self.categories.index(args.new_item))

    def show(self, cfg: dict[str, Any], args: Any) -> None:
        panel = np.full((820, 900, 3), (32, 34, 36), dtype=np.uint8)
        self._section(panel, (18, 18, 864, 88), "实验五实时参数控制", [
            "滑条实时生效；推荐位置会随“新增物体”和尺寸变化自动更新。",
            "快捷键：q/ESC 退出，r 重新标定 ROI，s 保存当前快照。",
        ], accent=(55, 170, 240))

        size = cfg["placement"].get("active_item_size_m", cfg["placement"]["default_item_size_m"])
        th = cfg["detector"].get("category_thresholds", {})
        roi = cfg["geometry"].get("cabinet_roi_norm", [0, 0, 1, 1])

        left_x, right_x = 18, 458
        card_w = 424
        self._section(panel, (left_x, 124, card_w, 112), "新增物体与放置", [
            f"new item：{self._category_text(args.new_item)}",
            f"item W/H/D cm：{size[0]*100:.0f} / {size[1]*100:.0f} / {size[2]*100:.0f}",
            "这里设置准备放入柜子的新物体，用于推荐空位。",
        ], accent=(40, 205, 120))

        self._section(panel, (left_x, 248, card_w, 190), "检测阈值", [
            f"box th：{cfg['detector']['box_threshold']:.2f}，text th：{cfg['detector']['text_threshold']:.2f}",
            f"Cutlery th：{th.get('Cutlery', 0):.2f}  刀叉勺类",
            f"Tableware th：{th.get('Tableware', 0):.2f}  杯盘碗类",
            f"Breakfast th：{th.get('Breakfast', 0):.2f}  早餐食品",
            f"Drink th：{th.get('Soft Drink', 0):.2f}  软饮料",
            f"Unknown th：{cfg['detector'].get('unknown_threshold', 0):.2f}  未知障碍",
            "调高可减少误检；调低可减少漏检。",
        ], accent=(245, 190, 65))

        self._section(panel, (left_x, 450, card_w, 120), "运行节奏", [
            f"detect every：每 {args.detect_every} 帧更新一次 Grounding DINO",
            f"temporal fr：{args.temporal_frames} 帧 RGB-D 中值融合",
            "检测间隔越小越实时；融合帧数越大越稳但更慢。",
        ], accent=(180, 140, 245))

        self._section(panel, (right_x, 124, card_w, 178), "柜体 ROI", [
            f"ROI x1/y1/x2/y2：{roi[0]:.2f}, {roi[1]:.2f}, {roi[2]:.2f}, {roi[3]:.2f}",
            "四个 ROI 滑条是归一化百分比，范围 0-100。",
            "启动时会自动标定 ROI；按 r 可用当前画面重新标定。",
            "若背景被算为空位，收紧 ROI 边界。",
        ], accent=(70, 190, 240))

        self._section(panel, (right_x, 314, card_w, 118), "空位判定", [
            f"empty W/H cm：{cfg['geometry']['empty_min_width_m']*100:.0f} / {cfg['geometry']['empty_min_height_m']*100:.0f}",
            "这是候选空位的最小宽高。",
            "新增物体尺寸也会参与空位过滤。",
        ], accent=(60, 135, 255))

        self._section(panel, (right_x, 444, card_w, 150), "稳定与防抖", [
            f"max misses：{cfg['stability']['max_misses']} 帧短时漏检保持",
            f"smooth：{cfg['stability']['box_smoothing_alpha']:.2f} 框坐标平滑系数",
            "max misses 越大越不易丢物体；smooth 越大越跟随当前帧。",
            "静态柜子建议 max misses 8-15，smooth 0.35-0.60。",
        ], accent=(255, 115, 105))

        self._section(panel, (18, 610, 864, 192), "滑条名称对照", [
            "new item=新增类别；item W/H/D cm=新增物体尺寸；box/text th=全局检测阈值。",
            "Cutlery/Tableware/Breakfast/Drink/Unknown th=类别和未知障碍阈值；detect every=检测间隔。",
            "temporal fr=多帧融合；ROI x1/y1/x2/y2=柜体范围；empty W/H=最小空位。",
            "max misses=漏检保持；smooth=框平滑。所有数值改动会在下一帧生效。",
        ], accent=(150, 210, 110))

        cv2.imshow(self.window, panel)

    def _pos(self, name: str) -> int:
        return cv2.getTrackbarPos(name, self.window)

    def _set(self, name: str, value: int) -> None:
        cv2.setTrackbarPos(name, self.window, value)

    def _apply_detector_thresholds(
        self,
        detector: BaseDetector,
        box_threshold: float,
        text_threshold: float,
        category_thresholds: dict[str, float],
        unknown_threshold: float,
    ) -> None:
        current: Any = detector
        while isinstance(current, CachedDetector):
            current = current.inner
        if hasattr(current, "box_threshold"):
            current.box_threshold = box_threshold
        if hasattr(current, "text_threshold"):
            current.text_threshold = text_threshold
        if hasattr(current, "category_thresholds"):
            current.category_thresholds = dict(category_thresholds)
        if hasattr(current, "unknown_threshold"):
            current.unknown_threshold = unknown_threshold
        if hasattr(current, "low_conf_known_as_unknown"):
            current.low_conf_known_as_unknown = bool(self.cfg["detector"].get("low_conf_known_as_unknown", True))

    def _load_fonts(self) -> dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
        font_path = next((path for path in FONT_CANDIDATES if Path(path).exists()), None)
        if font_path is None:
            return {
                "heading": ImageFont.load_default(),
                "body": ImageFont.load_default(),
            }
        return {
            "heading": ImageFont.truetype(font_path, 20),
            "body": ImageFont.truetype(font_path, 15),
        }

    def _section(
        self,
        panel: np.ndarray,
        rect: tuple[int, int, int, int],
        title: str,
        lines: list[str],
        accent: tuple[int, int, int],
    ) -> None:
        x, y, w, h = rect
        cv2.rectangle(panel, (x, y), (x + w, y + h), (44, 47, 50), -1)
        cv2.rectangle(panel, (x, y), (x + w, y + h), (78, 82, 86), 1)
        cv2.rectangle(panel, (x, y), (x + 5, y + h), accent, -1)
        self._text(panel, title, (x + 16, y + 10), (245, 245, 245), "heading")
        text_y = y + 42
        for line in lines:
            for wrapped in self._wrap_text(line, w - 36, "body"):
                if text_y > y + h - 18:
                    return
                self._text(panel, wrapped, (x + 18, text_y), (214, 218, 220), "body")
                text_y += 21

    def _text(
        self,
        panel: np.ndarray,
        text: str,
        xy: tuple[int, int],
        color_bgr: tuple[int, int, int],
        font_key: str = "body",
    ) -> None:
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text(xy, text, font=self.fonts[font_key], fill=color_rgb)
        panel[:, :, :] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def _category_text(self, category: str) -> str:
        return f"{category}（{CATEGORY_CN.get(category, category)}）"

    def _wrap_text(self, text: str, max_width: int, font_key: str) -> list[str]:
        font = self.fonts[font_key]
        lines: list[str] = []
        current = ""
        for char in text:
            candidate = current + char
            bbox = font.getbbox(candidate)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
        return lines
