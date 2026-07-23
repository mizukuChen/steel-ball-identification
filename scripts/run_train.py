"""YOLO 训练脚本（合成数据集）。"""
import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "synthetic" / "data.yaml"


def main(epochs: int, imgsz: int, model_size: str, batch: int) -> None:
    model = YOLO(f"yolo11{model_size}.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(PROJECT_ROOT / "models" / "runs"),
        name="steelball_test",
        exist_ok=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="n", help="n/s/m/l/x")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    main(args.epochs, args.imgsz, args.model, args.batch)
