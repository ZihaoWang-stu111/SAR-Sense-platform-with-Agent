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

    def test_evidence_drawer_markup_and_safe_rendering(self):
        html = Path("templates/chat.html").read_text(encoding="utf-8")
        css = Path("css/style_v2.css").read_text(encoding="utf-8")

        for element_id in (
            "evidenceDrawer",
            "evidenceDrawerClose",
            "evidenceDrawerContent",
            "evidenceDrawerDownload",
        ):
            self.assertIn(f'id="{element_id}"', html)

        self.assertIn("function openEvidenceDrawer", self.source)
        self.assertIn("function renderEvidenceContent(text)", self.source)
        self.assertIn("processTables(escapeHtml(text))", self.source)
        self.assertIn(
            "content.innerHTML = renderEvidenceContent(data.content || '')",
            self.source,
        )
        self.assertIn("/api/knowledge/evidence/", self.source)
        self.assertIn(".evidence-drawer", css)
        self.assertIn(".evidence-drawer-content table", css)
        self.assertIn(".evidence-drawer::backdrop", css)
        self.assertIn("@media (max-width: 640px)", css)

    def test_citations_hide_internal_ids_and_use_one_delegated_handler(self):
        self.assertIn('data-parent-id="${encodeURIComponent', self.source)
        self.assertIn('<button type="button" class="citation-source-item"', self.source)
        self.assertNotIn('<span class="citation-meta">chunk:', self.source)
        self.assertEqual(self.source.count("initCitationClickHandlers(chatMessages)"), 1)
        self.assertNotIn("initCitationClickHandlers(contentDiv)", self.source)

    def test_message_frame_only_wraps_message_content(self):
        theme = Path("css/home_renovation_v2.css").read_text(encoding="utf-8")

        self.assertIn(".message.assistant .message-content", theme)
        self.assertIn(".message.user .message-content", theme)
        self.assertNotRegex(theme, r"\.message\.assistant\s*\{")
        self.assertNotRegex(theme, r"\.message\.user\s*\{")

    def test_long_message_content_cannot_expand_chat_column(self):
        css = Path("css/style_v2.css").read_text(encoding="utf-8")

        for selector in (".chat-main", ".message", ".message-content"):
            self.assertRegex(
                css,
                rf"{re.escape(selector)}\s*\{{[^}}]*min-width:\s*0",
            )
        self.assertRegex(
            css,
            r"\.message-content\s*\{[^}]*overflow-wrap:\s*anywhere",
        )

    def test_chat_styles_have_matching_cache_busters(self):
        html = Path("templates/chat.html").read_text(encoding="utf-8")

        self.assertIn("css/style_v2.css?v=20260814-6", html)
        self.assertIn("css/home_renovation_v2.css?v=20260814-6", html)

    def test_chat_uses_wide_layout_without_squeezing_tables(self):
        css = Path("css/style_v2.css").read_text(encoding="utf-8")

        self.assertRegex(
            css,
            r"\.chat\s*>\s*\.container\s*\{[^}]*max-width:\s*none",
        )
        self.assertRegex(
            css,
            r"\.chat-layout\s*\{[^}]*max-width:\s*none",
        )
        self.assertRegex(
            css,
            r"\.message\.assistant\s*\{[^}]*max-width:\s*100%",
        )
        self.assertRegex(
            css,
            r"\.message-content\s+\.table-wrapper\s+table\s*\{[^}]*"
            r"width:\s*100%",
        )
        self.assertRegex(
            css,
            r"\.message-content\s+th\s*\{[^}]*"
            r"white-space:\s*nowrap",
        )
        self.assertRegex(
            css,
            r"\.message-content\s+td\s*\{[^}]*"
            r"white-space:\s*normal[^}]*overflow-wrap:\s*break-word",
        )

    def test_long_chat_scrolls_inside_message_panel(self):
        css = Path("css/style_v2.css").read_text(encoding="utf-8")

        self.assertRegex(
            css,
            r"\.chat-main\s*\{[^}]*height:\s*640px",
        )
        self.assertRegex(
            css,
            r"\.chat-messages\s*\{[^}]*min-height:\s*0",
        )
        self.assertRegex(
            css,
            r"\.chat-messages\s*\{[^}]*overscroll-behavior-y:\s*contain",
        )

    def test_input_bar_uses_same_bottom_inset_as_sidebar(self):
        css = Path("css/style_v2.css").read_text(encoding="utf-8")
        html = Path("templates/chat.html").read_text(encoding="utf-8")

        self.assertRegex(
            css,
            r"\.chat-input-area\s*\{[^}]*"
            r"padding:\s*var\(--space-md\)\s+var\(--space-lg\)\s*;",
        )
        self.assertNotIn('class="chat-hint"', html)

    def test_chat_section_has_compact_bottom_spacing(self):
        css = Path("css/style_v2.css").read_text(encoding="utf-8")

        self.assertRegex(
            css,
            r"\.chat\s*\{[^}]*padding-bottom:\s*var\(--space-md\)",
        )


if __name__ == "__main__":
    unittest.main()
