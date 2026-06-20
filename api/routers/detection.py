import base64
import logging
import os
import tempfile
import traceback
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from api.dependencies import get_yolo_model

logger = logging.getLogger(__name__)
router = APIRouter(tags=["detection"])


@router.post("/detect")
async def detect_ships(image: UploadFile = File(...)):
    """SAR ship detection endpoint"""
    try:
        logger.info(f"Detection request received: {image.filename}")

        content = await image.read()

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"sar_detect_{image.filename}")
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
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0]),
                'bbox': box.xyxy[0].tolist()
            })

        return {
            'success': True,
            'ship_count': ship_count,
            'original_image': orig_base64,
            'result_image': result_base64,
            'detections': detections,
            'temp_path': temp_path,
            'message': f'检测完成，共发现 {ship_count} 个目标'
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
