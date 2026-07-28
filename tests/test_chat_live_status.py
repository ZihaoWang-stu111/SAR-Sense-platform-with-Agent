import unittest
from pathlib import Path


class ChatLiveStatusFrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("js/chat.js").read_text(encoding="utf-8")

    def test_foreground_and_background_streams_handle_status(self):
        self.assertEqual(self.source.count("eventType === 'status'"), 2)
        self.assertIn("loadingStatus: '正在等待处理...'", self.source)

    def test_thought_steps_map_to_friendly_status(self):
        self.assertIn("function getLoadingStatusForStep(step)", self.source)
        self.assertIn("rag_summarize: '正在检索知识库...'", self.source)
        self.assertIn("web_search: '正在搜索网络...'", self.source)
        self.assertIn("detect_ships: '正在检测图像...'", self.source)
        self.assertIn("return '正在整理结果...'", self.source)
        self.assertIn("return '正在生成回答...'", self.source)

    def test_first_typewriter_character_hides_status(self):
        typewriter_start = self.source.index("async function processTypewriterQueue()")
        typewriter_end = self.source.index(
            "async function appendDetectImages",
            typewriter_start,
        )
        typewriter = self.source[typewriter_start:typewriter_end]
        self.assertIn("msg.loadingStatus = null", typewriter)


if __name__ == "__main__":
    unittest.main()
