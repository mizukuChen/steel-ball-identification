"""生成合成训练数据集（图片 + YOLO标注），并划分 train/val。"""
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2

from data_pipeline.synthesizer import generate_sample, load_backgrounds, load_templates

TEMPLATE_DIR = PROJECT_ROOT / "data" / "templates"
BG_DIRS = [
    PROJECT_ROOT / "data" / "backgrounds" / "downloaded",
    PROJECT_ROOT / "data" / "backgrounds" / "generated",
]
OUT_ROOT = PROJECT_ROOT / "data" / "synthetic"

CANVAS_SIZE = 640
VAL_RATIO = 0.15


def main(num_images: int = 2000) -> None:
    templates = load_templates(TEMPLATE_DIR)
    backgrounds = load_backgrounds(BG_DIRS)
    print(f"模板数: {len(templates)}, 背景数: {len(backgrounds)}")

    if not templates or not backgrounds:
        print("模板或背景为空，终止")
        return

    for split in ("train", "val"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_val = int(num_images * VAL_RATIO)

    for i in range(num_images):
        split = "val" if i < n_val else "train"
        canvas, bboxes = generate_sample(templates, backgrounds, canvas_size=CANVAS_SIZE)

        img_path = OUT_ROOT / "images" / split / f"synth_{i:05d}.jpg"
        label_path = OUT_ROOT / "labels" / split / f"synth_{i:05d}.txt"

        cv2.imwrite(str(img_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = [b.to_yolo(CANVAS_SIZE, CANVAS_SIZE) for b in bboxes]
        label_path.write_text("\n".join(lines), encoding="utf-8")

        if (i + 1) % 200 == 0:
            print(f"进度: {i + 1}/{num_images}")

    yaml_content = f"""path: {OUT_ROOT.as_posix()}
train: images/train
val: images/val
names:
  0: steel_ball
"""
    (OUT_ROOT / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"完成：共生成 {num_images} 张（train={num_images - n_val}, val={n_val}）")


if __name__ == "__main__":
    random.seed(42)
    main()
