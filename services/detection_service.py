import base64
import logging
import os
import tempfile
from io import BytesIO

from PIL import Image

from api.dependencies import get_yolo_model

logger = logging.getLogger(__name__)


def detect_ships_from_bytes(content: bytes, filename: str | None) -> dict:
    """Run blocking PIL/YOLO detection work."""
    temp_dir = tempfile.gettempdir()
    # basename 防路径穿越：客户端可能传 "../../evil.png" 之类文件名
    safe_name = os.path.basename(filename or "upload.png")
    temp_path = os.path.join(temp_dir, f"sar_detect_{safe_name}")
    with open(temp_path, "wb") as f:
        f.write(content)
    logger.info(f"File saved to: {temp_path}")

    logger.info("Loading image...")
    img = Image.open(BytesIO(content))
    logger.info("Loading YOLO model...")
    model = get_yolo_model()
    logger.info("Running detection...")
    results = model.predict(source=img, imgsz=640)
    logger.info(f"Detection complete, found {len(results[0].boxes)} objects")

    ship_count = len(results[0].boxes)

    res_plotted = results[0].plot()
    result_image = Image.fromarray(res_plotted[:, :, ::-1])

    buffered = BytesIO()
    result_image.save(buffered, format="PNG")
    result_base64 = base64.b64encode(buffered.getvalue()).decode()

    orig_buffered = BytesIO()
    img.save(orig_buffered, format="PNG")
    orig_base64 = base64.b64encode(orig_buffered.getvalue()).decode()

    detections = []
    for box in results[0].boxes:
        detections.append({
            "confidence": float(box.conf[0]),
            "class": int(box.cls[0]),
            "bbox": box.xyxy[0].tolist(),
        })

    return {
        "success": True,
        "ship_count": ship_count,
        "original_image": orig_base64,
        "result_image": result_base64,
        "detections": detections,
        "temp_path": temp_path,
        "message": f"检测完成，共发现 {ship_count} 个目标",
    }
