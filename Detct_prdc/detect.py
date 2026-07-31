"""使用项目内 MBE-Net 权重执行本地批量检测。"""

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "MBE-Net" / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="MBE-Net SAR 舰船检测")
    parser.add_argument("source", type=Path, help="待检测图片、视频或目录")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="本地 .pt 权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图片尺寸")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "runs" / "detect",
        help="检测结果目录",
    )
    args = parser.parse_args()

    model = YOLO(str(args.model))
    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        project=str(args.output),
        name="exp",
        save=True,
    )


if __name__ == "__main__":
    main()
