"""用训练好的YOLO权重对单张图片做推理，输出钢珠的位置/大小/数量并可视化。"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import cv2
from ultralytics import YOLO

DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "runs" / "steelball_test" / "weights" / "best.pt"


def main(image_path: str, weights: str, conf: float) -> None:
    model = YOLO(weights)
    results = model(image_path, conf=conf, verbose=False)
    r = results[0]

    img = cv2.imread(image_path)
    boxes = r.boxes

    print(f"图片: {image_path}")
    print(f"置信度阈值: {conf}")
    print(f"检测到钢珠数量: {len(boxes)}")
    print("-" * 60)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        radius = (w + h) / 4  # 等效半径
        c = float(box.conf[0])
        print(
            f"  #{i + 1:2d}  中心=({cx:6.1f},{cy:6.1f})  "
            f"半径≈{radius:5.1f}px  置信度={c:.2f}"
        )
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            img, f"{c:.2f}", (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )

    out_path = Path(image_path).with_name(Path(image_path).stem + "_pred.jpg")
    cv2.imwrite(str(out_path), img)
    print("-" * 60)
    print(f"可视化已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="待检测图片路径")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = parser.parse_args()
    main(args.image, args.weights, args.conf)
