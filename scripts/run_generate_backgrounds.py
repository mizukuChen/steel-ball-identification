"""批量生成程序化噪音/纹理背景，输出到 data/backgrounds/generated。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2

from data_pipeline.background_generator import generate_random_background

OUT_DIR = PROJECT_ROOT / "data" / "backgrounds" / "generated"


def main(count: int = 40) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = generate_random_background()
        cv2.imwrite(str(OUT_DIR / f"gen_{i:03d}.png"), img)
    print(f"生成完成：{count} 张 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
