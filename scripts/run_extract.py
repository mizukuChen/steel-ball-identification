"""从 data/raw_photos 中提取钢珠模板，输出到 data/templates。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2

from data_pipeline.extractor import draw_debug, process_photo

RAW_DIR = PROJECT_ROOT / "data" / "raw_photos"
TEMPLATE_DIR = PROJECT_ROOT / "data" / "templates"
DEBUG_DIR = TEMPLATE_DIR / "_debug"


def main() -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    photo_paths = sorted(RAW_DIR.glob("*.jpg"))
    print(f"找到 {len(photo_paths)} 张原始照片")

    ok_count = 0
    for path in photo_paths:
        result = process_photo(path)
        if result is None:
            print(f"[失败] {path.name} 未检测到钢珠")
            continue

        template, circle = result
        out_path = TEMPLATE_DIR / f"{path.stem}.png"
        cv2.imwrite(str(out_path), template)

        image = cv2.imread(str(path))
        debug = draw_debug(image, circle)
        debug_path = DEBUG_DIR / f"{path.stem}_debug.jpg"
        cv2.imwrite(str(debug_path), debug)

        print(f"[成功] {path.name} -> r={circle.r}px  中心=({circle.cx},{circle.cy})")
        ok_count += 1

    print(f"完成：{ok_count}/{len(photo_paths)} 张成功提取")


if __name__ == "__main__":
    main()
