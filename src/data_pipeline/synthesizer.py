"""将钢珠模板贴合到背景上，生成带YOLO标注的合成训练数据。"""
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_yolo(self, img_w: int, img_h: int) -> str:
        cx = (self.x1 + self.x2) / 2 / img_w
        cy = (self.y1 + self.y2) / 2 / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def load_templates(template_dir: Path) -> list[np.ndarray]:
    templates = []
    for path in sorted(template_dir.glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            templates.append(img)
    return templates


def load_backgrounds(bg_dirs: list[Path]) -> list[np.ndarray]:
    backgrounds = []
    for bg_dir in bg_dirs:
        for path in sorted(bg_dir.glob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(path))
            if img is not None:
                backgrounds.append(img)
    return backgrounds


def random_crop_resize_bg(bg: np.ndarray, size: int) -> np.ndarray:
    h, w = bg.shape[:2]
    crop_size = min(h, w)
    y0 = random.randint(0, h - crop_size) if h > crop_size else 0
    x0 = random.randint(0, w - crop_size) if w > crop_size else 0
    crop = bg[y0 : y0 + crop_size, x0 : x0 + crop_size]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def transform_ball(template: np.ndarray, target_radius: int) -> np.ndarray:
    """随机旋转 + 缩放到目标半径。"""
    h, w = template.shape[:2]
    scale = (target_radius * 2) / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    angle = random.uniform(0, 360)
    center = (new_w / 2, new_h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    diag = int(np.ceil(np.hypot(new_w, new_h)))
    rot_mat[0, 2] += (diag - new_w) / 2
    rot_mat[1, 2] += (diag - new_h) / 2

    rotated = cv2.warpAffine(
        resized, rot_mat, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0)
    )
    return rotated


def match_lighting(ball_rgba: np.ndarray, bg_patch: np.ndarray) -> np.ndarray:
    """按背景局部亮度/色调，调整钢珠的亮度和色温。"""
    alpha = ball_rgba[:, :, 3]
    fg_mask = alpha > 10
    if not fg_mask.any():
        return ball_rgba

    ball_bgr = ball_rgba[:, :, :3].astype(np.float32)
    bg_bgr = bg_patch.astype(np.float32)

    for c in range(3):
        ball_mean = ball_bgr[:, :, c][fg_mask].mean()
        bg_mean = bg_bgr[:, :, c].mean()
        if ball_mean < 1:
            continue
        ratio = np.clip(bg_mean / ball_mean, 0.7, 1.4)
        ball_bgr[:, :, c] *= ratio

    brightness_jitter = random.uniform(0.9, 1.1)
    ball_bgr *= brightness_jitter

    result = ball_rgba.copy()
    result[:, :, :3] = np.clip(ball_bgr, 0, 255).astype(np.uint8)
    return result


def place_ball(
    canvas: np.ndarray,
    ball_rgba: np.ndarray,
    cx: int,
    cy: int,
    light_angle_deg: float,
) -> BBox | None:
    """在画布上合成阴影 + 钢珠，返回外接框；越界过多时返回 None。"""
    bh, bw = ball_rgba.shape[:2]
    ch, cw = canvas.shape[:2]

    x0, y0 = cx - bw // 2, cy - bh // 2
    x1, y1 = x0 + bw, y0 + bh

    vis_x0, vis_y0 = max(x0, 0), max(y0, 0)
    vis_x1, vis_y1 = min(x1, cw), min(y1, ch)
    if vis_x1 - vis_x0 < bw * 0.3 or vis_y1 - vis_y0 < bh * 0.3:
        return None

    radius = min(bh, bw) / 2
    shadow_dx = radius * 0.6 * np.cos(np.radians(light_angle_deg + 180))
    shadow_dy = radius * 0.35 * np.sin(np.radians(light_angle_deg + 180))
    shadow_center = (int(cx + shadow_dx), int(cy + radius * 0.5 + shadow_dy))
    shadow_axes = (int(radius * 1.1), int(radius * 0.5))

    shadow_layer = np.zeros((ch, cw), dtype=np.uint8)
    cv2.ellipse(shadow_layer, shadow_center, shadow_axes, 0, 0, 360, 255, -1)
    shadow_layer = cv2.GaussianBlur(shadow_layer, (0, 0), sigmaX=radius * 0.25)
    shadow_strength = random.uniform(0.15, 0.4)
    shadow_alpha = (shadow_layer.astype(np.float32) / 255.0) * shadow_strength
    for c in range(3):
        canvas[:, :, c] = (
            canvas[:, :, c].astype(np.float32) * (1 - shadow_alpha)
        ).astype(np.uint8)

    src_x0, src_y0 = vis_x0 - x0, vis_y0 - y0
    src_x1, src_y1 = src_x0 + (vis_x1 - vis_x0), src_y0 + (vis_y1 - vis_y0)

    ball_crop = ball_rgba[src_y0:src_y1, src_x0:src_x1]
    alpha = (ball_crop[:, :, 3:4].astype(np.float32)) / 255.0
    fg = ball_crop[:, :, :3].astype(np.float32)
    bg_region = canvas[vis_y0:vis_y1, vis_x0:vis_x1].astype(np.float32)

    blended = fg * alpha + bg_region * (1 - alpha)
    canvas[vis_y0:vis_y1, vis_x0:vis_x1] = blended.astype(np.uint8)

    alpha_mask = ball_crop[:, :, 3] > 20
    ys, xs = np.where(alpha_mask)
    if len(xs) == 0:
        return None
    return BBox(
        x1=float(vis_x0 + xs.min()),
        y1=float(vis_y0 + ys.min()),
        x2=float(vis_x0 + xs.max()),
        y2=float(vis_y0 + ys.max()),
    )


def generate_sample(
    templates: list[np.ndarray],
    backgrounds: list[np.ndarray],
    canvas_size: int = 640,
    min_balls: int = 1,
    max_balls: int = 8,
) -> tuple[np.ndarray, list[BBox]]:
    bg = random.choice(backgrounds)
    canvas = random_crop_resize_bg(bg, canvas_size)

    num_balls = random.randint(min_balls, max_balls)
    light_angle = random.uniform(0, 360)

    placed_circles: list[tuple[int, int, int]] = []
    bboxes: list[BBox] = []

    for _ in range(num_balls):
        template = random.choice(templates)
        target_radius = random.randint(int(canvas_size * 0.02), int(canvas_size * 0.09))

        cx = cy = None
        for _try in range(20):
            cand_x = random.randint(0, canvas_size)
            cand_y = random.randint(0, canvas_size)
            ok = True
            for px, py, pr in placed_circles:
                min_dist = (target_radius + pr) * 0.55
                if np.hypot(cand_x - px, cand_y - py) < min_dist:
                    ok = False
                    break
            if ok:
                cx, cy = cand_x, cand_y
                break
        if cx is None:
            cx, cy = random.randint(0, canvas_size), random.randint(0, canvas_size)

        ball = transform_ball(template, target_radius)

        y0, y1 = max(cy - target_radius, 0), min(cy + target_radius, canvas_size)
        x0, x1 = max(cx - target_radius, 0), min(cx + target_radius, canvas_size)
        if y1 <= y0 or x1 <= x0:
            continue
        bg_patch = canvas[y0:y1, x0:x1]
        ball = match_lighting(ball, bg_patch)

        bbox = place_ball(canvas, ball, cx, cy, light_angle)
        if bbox is not None:
            bboxes.append(bbox)
            placed_circles.append((cx, cy, target_radius))

    return canvas, bboxes
