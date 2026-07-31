# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.9"

import os

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"

from ultralytics.models.yolo.model import YOLO

__all__ = ("__version__", "YOLO")
