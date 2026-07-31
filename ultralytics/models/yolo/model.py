# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.model import Model
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.nn.tasks import DetectionModel


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model, task="detect", verbose=False):
        """Initialize the inference-only detection model used by SAR-Sense."""
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "predictor": DetectionPredictor,
            },
        }
