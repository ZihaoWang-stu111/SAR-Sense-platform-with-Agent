import os

from utils.logger_handler import logger


def extract_image_text(file_path: str, max_chars: int = 3000) -> str:
    """使用 RapidOCR 提取图片文字，未识别到内容或识别失败时返回空字符串。"""
    try:
        from rapidocr_onnxruntime import RapidOCR

        result, _ = RapidOCR()(file_path)
        if not result:
            return ""

        lines = [
            str(text).strip()
            for _, text, _ in result
            if text and str(text).strip()
        ]
        content = "\n".join(lines)
        if len(content) > max_chars:
            content = (
                content[:max_chars]
                + f"\n\n... (OCR共识别{len(lines)}段文字，已截断前{max_chars}字符)"
            )
        return content
    except Exception as exc:
        logger.warning(
            f"图片OCR识别失败: {os.path.basename(file_path)} - {exc}",
            exc_info=True,
        )
        return ""
