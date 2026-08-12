import json
import re
import subprocess
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

    def test_loading_status_has_accessible_markup_and_reduced_motion(self):
        css = Path("css/style_v2.css").read_text(encoding="utf-8")
        self.assertIn("function renderAssistantLoadingStatus(status)", self.source)
        self.assertIn(
            'class="message-status assistant-loading-status"',
            self.source,
        )
        self.assertIn('role="status"', self.source)
        self.assertIn('aria-live="polite"', self.source)
        self.assertIn(".assistant-loading-status", css)
        self.assertIn("@media (prefers-reduced-motion: reduce)", css)

    def test_only_structured_rag_sources_are_folded(self):
        self.assertIn("function hasStructuredCitationSource(html, match)", self.source)
        self.assertIn(
            ".filter(match => hasStructuredCitationSource(html, match))",
            self.source,
        )

        pattern_match = re.search(
            r"const citationSourceLinePattern\s*=\s*(/(?:\\.|[^/\r\n])+/[a-z]*);",
            self.source,
        )
        self.assertIsNotNone(pattern_match)
        cases = [
            ("[1] paper.pdf | chunk_id=abc | page=7 | score=0.9876", True),
            ("[1] paper.pdf | chunk_id=abc | score=0.9876", True),
            ("[1] 普通正文", False),
            ("[1] paper.pdf | score=0.9876", False),
            ("[1] paper.pdf | chunk_id=abc", False),
        ]
        script = (
            f"const pattern = {pattern_match.group(1)};"
            f"const cases = {json.dumps(cases, ensure_ascii=True)};"
            "const failures = cases.filter(([line, expected]) => "
            "pattern.test(line) !== expected);"
            "if (failures.length) {"
            "console.error(JSON.stringify(failures));"
            "process.exit(1);"
            "}"
        )
        result = subprocess.run(
            ["node", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
