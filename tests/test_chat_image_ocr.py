import io
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from api.routers import files


class FakeUpload:
    def __init__(self, content: bytes):
        self.filename = "screenshot.png"
        self.content_type = "image/png"
        self._content = content

    async def read(self):
        return self._content


def make_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "white").save(output, format="PNG")
    return output.getvalue()


class ChatImageOCRTest(unittest.IsolatedAsyncioTestCase):
    async def test_image_upload_returns_ocr_content_and_detection_upload_id(self):
        with (
            patch.object(files, "rate_limit", AsyncMock()),
            patch.object(files, "save_upload", return_value="img_test"),
            patch.object(files, "get_upload_path", return_value="stored.png"),
            patch.object(files, "extract_image_text") as extract_image_text,
            patch.object(
                files,
                "run_in_threadpool",
                AsyncMock(return_value="表格标题 舰船检测结果"),
            ) as run_in_threadpool,
        ):
            response = await files.extract_file(
                file=FakeUpload(make_png()),
                user={"id": 7},
            )

        self.assertEqual(response["upload_id"], "img_test")
        self.assertEqual(response["content"], "表格标题 舰船检测结果")
        run_in_threadpool.assert_awaited_once_with(
            extract_image_text,
            "stored.png",
        )

    def test_frontend_sends_image_ocr_and_upload_id_together(self):
        source = Path("js/chat.js").read_text(encoding="utf-8")
        start = source.index("let fullMessage = message;")
        end = source.index("const userMessage", start)
        attachment_block = source[start:end]

        self.assertIn("if (attachmentUploadId)", attachment_block)
        self.assertIn("OCR识别结果", attachment_block)
        self.assertIn("${attachmentContent}", attachment_block)
        self.assertIn("${attachmentUploadId}", attachment_block)
        self.assertNotIn("用户上传了SAR图像", attachment_block)


if __name__ == "__main__":
    unittest.main()
