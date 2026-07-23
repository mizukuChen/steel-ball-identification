"""从原始照片中提取钢珠模板（带透明通道的PNG）。"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BallCircle:
    cx: int
    cy: int
    r: int


def detect_ball(image: np.ndarray) -> BallCircle | None:
    """在单球、浅色背景的照片中定位钢珠，返回圆心和半径（像素）。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min_dim,
        param1=100,
        param2=40,
        minRadius=int(min_dim * 0.02),
        maxRadius=int(min_dim * 0.15),
    )

    if circles is None:
        return None

    # HoughCircles 按累加器强度排序，取第一个即最强的圆
    cx, cy, r = circles[0][0]
    return BallCircle(cx=int(round(cx)), cy=int(round(cy)), r=int(round(r)))


def extract_template(
    image: np.ndarray,
    circle: BallCircle,
    padding_ratio: float = 0.3,
    feather_px: int = 3,
) -> np.ndarray:
    """按检测到的圆裁剪钢珠区域，生成带羽化边缘透明通道的 RGBA 图像。"""
    pad = int(circle.r * (1 + padding_ratio))
    x0 = max(circle.cx - pad, 0)
    y0 = max(circle.cy - pad, 0)
    x1 = min(circle.cx + pad, image.shape[1])
    y1 = min(circle.cy + pad, image.shape[0])

    crop = image[y0:y1, x0:x1]
    local_cx = circle.cx - x0
    local_cy = circle.cy - y0

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (local_cx, local_cy), circle.r, 255, thickness=-1)

    if feather_px > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_px)

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba


def process_photo(image_path: Path) -> tuple[np.ndarray, BallCircle] | None:
    """读取一张原始照片，检测并提取钢珠模板。检测失败返回 None。"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    circle = detect_ball(image)
    if circle is None:
        return None

    template = extract_template(image, circle)
    return template, circle


def draw_debug(image: np.ndarray, circle: BallCircle) -> np.ndarray:
    """在原图上画出检测到的圆，用于人工核验。"""
    debug = image.copy()
    cv2.circle(debug, (circle.cx, circle.cy), circle.r, (0, 255, 0), 2)
    cv2.circle(debug, (circle.cx, circle.cy), 2, (0, 0, 255), 3)
    return debug
