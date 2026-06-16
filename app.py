import time
import os
import html
import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from agent.react_agent import ReactAgent, _thought_chains
from rag.vector_store import get_vector_store_service
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from agent.metrics_collector import AgentMetrics
from utils.conversation_manager import ConversationManager
from agent.tools import agent_tools


def inject_custom_css():
    st.markdown("""
    <style>
    /* ========== 全局变量 ========== */
    :root {
        --primary: #38bdf8;
        --primary-deep: #0284c7;
        --primary-light: #e0f2fe;
        --success: #0d9488;
        --success-light: #e6fffa;
        --warning: #f59e0b;
        --warning-light: #fffbeb;
        --danger: #ef4444;
        --danger-light: #fef2f2;
        --bg: #f0f9ff;
        --card-bg: rgba(255,255,255,0.55);
        --card-border: rgba(255,255,255,0.35);
        --text: #0c4a6e;
        --text-secondary: #64748b;
        --border: rgba(56,189,248,0.15);
        --radius: 18px;
        --radius-sm: 12px;
        --shadow-sm: 0 2px 8px rgba(56,189,248,0.08);
        --shadow: 0 4px 16px rgba(56,189,248,0.10);
        --shadow-lg: 0 8px 32px rgba(56,189,248,0.15);
        --glass-blur: blur(18px);
        --transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ========== 动画关键帧 ========== */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInLeft {
        from { opacity: 0; transform: translateX(-24px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes fadeInRight {
        from { opacity: 0; transform: translateX(24px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 8px rgba(56,189,248,0.3); }
        50% { box-shadow: 0 0 20px rgba(56,189,248,0.6); }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* ========== 全局背景 + 动态渐变 ========== */
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 30%, #ffffff 60%, #e0f2fe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 0%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ========== 侧边栏 ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e0f2fe 0%, #bae6fd 50%, #7dd3fc 100%);
        border-right: 1px solid rgba(56,189,248,0.15);
    }
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 30% 20%, rgba(255,255,255,0.3), transparent 60%);
        pointer-events: none;
    }
    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem;
        position: relative;
        z-index: 1;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio p {
        color: #0c4a6e !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(56,189,248,0.2);
        margin: 1rem 0;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.35rem;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        padding: 0.7rem 1rem;
        border-radius: 10px;
        transition: var(--transition);
        border: 1px solid transparent;
        font-size: 0.95rem;
        font-weight: 500;
        color: #0369a1 !important;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        background: rgba(56,189,248,0.1);
        border-color: rgba(56,189,248,0.2);
        color: #0c4a6e !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] {
        background: rgba(56,189,248,0.15);
        border-color: rgba(56,189,248,0.3);
        color: #0c4a6e !important;
        box-shadow: 0 0 20px rgba(56,189,248,0.1);
        border-left: 3px solid #0284c7;
        animation: pulseGlow 3s ease-in-out infinite;
    }

    /* ========== 侧栏按钮（会话列表） ========== */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.5) !important;
        color: #0369a1 !important;
        border: 1px solid rgba(56,189,248,0.2) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(8px);
        transition: var(--transition) !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.7) !important;
        border-color: rgba(56,189,248,0.4) !important;
        box-shadow: 0 0 16px rgba(56,189,248,0.2);
        transform: translateY(-1px);
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #38bdf8, #0284c7) !important;
        border-color: rgba(56,189,248,0.3) !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] .stButton > button[aria-label="删除此对话"],
    [data-testid="stSidebar"] .stButton > button[title="删除此对话"] {
        background: rgba(239,68,68,0.08) !important;
        color: #ef4444 !important;
        border: 1px solid rgba(239,68,68,0.2) !important;
        font-size: 0.75rem !important;
        padding: 0.2rem 0.5rem !important;
        min-width: 2rem !important;
    }
    [data-testid="stSidebar"] .stButton > button[aria-label="删除此对话"]:hover,
    [data-testid="stSidebar"] .stButton > button[title="删除此对话"]:hover {
        background: rgba(239,68,68,0.15) !important;
        border-color: rgba(239,68,68,0.4) !important;
        box-shadow: 0 0 12px rgba(239,68,68,0.2);
    }

    /* ========== 标题 ========== */
    h1 {
        font-weight: 800 !important;
        font-size: 2rem !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(135deg, #0c4a6e, #0284c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem !important;
    }
    h3 {
        font-weight: 700 !important;
        color: #0369a1 !important;
        font-size: 1.15rem !important;
    }

    /* ========== 毛玻璃卡片 ========== */
    .custom-card {
        background: var(--card-bg);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--card-border);
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        transition: var(--transition);
        animation: fadeInUp 0.5s ease-out both;
    }
    .custom-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
        border-color: rgba(56,189,248,0.3);
    }
    .custom-card-accent {
        border-left: 4px solid var(--primary);
    }

    /* ========== 指标卡片 ========== */
    .metric-card {
        background: rgba(255,255,255,0.5);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--card-border);
        border-radius: var(--radius);
        padding: 1.25rem 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        animation: fadeInUp 0.5s ease-out both;
        transition: var(--transition);
    }
    .metric-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    .metric-card .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 500;
        margin-top: 0.25rem;
    }

    /* ========== 信息提示条 ========== */
    [data-testid="stAlert"] {
        border-radius: var(--radius-sm) !important;
        border: none !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: var(--shadow-sm);
        padding: 1rem 1.25rem !important;
        animation: fadeInUp 0.4s ease-out both;
    }
    [data-testid="stAlert"][kind="info"] {
        background: linear-gradient(135deg, rgba(224,242,254,0.8), rgba(186,230,253,0.6)) !important;
        border-left: 4px solid #38bdf8 !important;
    }
    [data-testid="stAlert"][kind="success"] {
        background: linear-gradient(135deg, rgba(204,251,241,0.8), rgba(167,243,208,0.6)) !important;
        border-left: 4px solid #2dd4bf !important;
    }
    [data-testid="stAlert"][kind="warning"] {
        background: linear-gradient(135deg, rgba(254,243,199,0.8), rgba(253,230,138,0.6)) !important;
        border-left: 4px solid #fbbf24 !important;
    }

    /* ========== 文件上传区域 ========== */
    [data-testid="stFileUploader"] {
        border-radius: var(--radius) !important;
        transition: var(--transition);
    }
    [data-testid="stFileUploader"] section {
        border: 2px dashed rgba(56,189,248,0.3) !important;
        border-radius: var(--radius) !important;
        background: rgba(255,255,255,0.4) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 2rem !important;
        transition: var(--transition);
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #38bdf8 !important;
        background: rgba(224,242,254,0.5) !important;
        box-shadow: 0 0 0 4px rgba(56,189,248,0.1);
    }
    [data-testid="stFileUploader"] section p {
        color: #0369a1 !important;
        font-size: 0.95rem;
    }

    /* ========== 图片容器 ========== */
    [data-testid="stImage"] {
        border-radius: var(--radius-sm) !important;
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    [data-testid="stImage"] img {
        border-radius: var(--radius-sm) !important;
    }

    /* ========== 按钮 ========== */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: var(--transition) !important;
        border: 1px solid rgba(56,189,248,0.2) !important;
        background: rgba(255,255,255,0.6) !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        box-shadow: var(--shadow-sm);
        color: #0284c7 !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 24px rgba(56,189,248,0.3), var(--shadow-lg);
        border-color: rgba(56,189,248,0.4) !important;
        background: rgba(224,242,254,0.7) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #38bdf8, #0284c7) !important;
        color: #fff !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(56,189,248,0.3);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 24px rgba(56,189,248,0.5);
        transform: translateY(-2px);
    }

    /* ========== 聊天界面 ========== */
    [data-testid="stChatMessage"] {
        border-radius: var(--radius) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: var(--shadow-sm);
        margin-bottom: 0.75rem !important;
        border: 1px solid rgba(56,189,248,0.1);
        animation: fadeInUp 0.4s ease-out both;
    }
    [data-testid="stChatMessage"][data-testid="stChatMessage"] {
        padding: 1rem 1.25rem !important;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
        filter: drop-shadow(0 2px 4px rgba(56,189,248,0.2));
    }

    /* ========== 聊天输入框 ========== */
    [data-testid="stChatInput"] {
        position: relative;
        padding-top: 1.5rem !important;
    }
    [data-testid="stChatInput"] > div {
        border-radius: 20px !important;
        background: rgba(255,255,255,0.5) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(56,189,248,0.2) !important;
        box-shadow: 0 4px 20px rgba(56,189,248,0.08), inset 0 1px 0 rgba(255,255,255,0.6) !important;
        transition: var(--transition);
        overflow: hidden;
    }
    [data-testid="stChatInput"] > div:hover {
        border-color: rgba(56,189,248,0.4) !important;
        box-shadow: 0 6px 28px rgba(56,189,248,0.15), inset 0 1px 0 rgba(255,255,255,0.6) !important;
    }
    [data-testid="stChatInput"]:focus-within > div {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.12), 0 6px 28px rgba(56,189,248,0.15) !important;
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 16px !important;
        border: none !important;
        background: transparent !important;
        color: #0c4a6e !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(56,189,248,0.5) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] button[kind="header"] {
        background: linear-gradient(135deg, #38bdf8, #0284c7) !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(56,189,248,0.3) !important;
        transition: var(--transition) !important;
    }
    [data-testid="stChatInput"] button[kind="header"]:hover {
        box-shadow: 0 4px 16px rgba(56,189,248,0.5) !important;
        transform: scale(1.05);
    }

    /* ========== 分割线 ========== */
    hr {
        border-color: rgba(56,189,248,0.15) !important;
        margin: 1.25rem 0 !important;
    }

    /* ========== Spinner ========== */
    [data-testid="stSpinner"] {
        color: #38bdf8 !important;
    }

    /* ========== Tabs ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 14px;
        padding: 0.3rem;
        border: 1px solid rgba(56,189,248,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        transition: var(--transition);
        color: #0369a1 !important;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(56,189,248,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56,189,248,0.2), rgba(125,211,252,0.15)) !important;
        color: #0284c7 !important;
        box-shadow: 0 2px 8px rgba(56,189,248,0.15);
    }

    /* ========== DataFrame ========== */
    .stDataFrame {
        border-radius: var(--radius) !important;
        overflow: hidden;
        border: 1px solid rgba(56,189,248,0.1);
        box-shadow: var(--shadow-sm);
    }

    /* ========== 响应式 ========== */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.5rem !important;
        }
        .main .block-container {
            padding: 1rem;
        }
    }

    /* ========== 思维链可视化 ========== */
    .thought-chain-details {
        margin-top: 0.75rem;
        border: 1px solid rgba(56,189,248,0.2) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.35) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 0.5rem 1rem !important;
        transition: var(--transition);
        max-width: 100%;
    }
    .thought-chain-details[open] {
        background: rgba(255,255,255,0.5) !important;
        border-color: rgba(56,189,248,0.35) !important;
        box-shadow: 0 4px 16px rgba(56,189,248,0.1) !important;
    }
    .thought-chain-details summary {
        font-weight: 600 !important;
        color: #0369a1 !important;
        cursor: pointer;
        font-size: 0.88rem;
        padding: 0.35rem 0;
        user-select: none;
        list-style: none;
    }
    .thought-chain-details summary::-webkit-details-marker { display: none; }
    .thought-chain-details summary::before {
        content: '▸ ';
        color: #38bdf8;
        font-weight: 700;
    }
    .thought-chain-details[open] summary::before {
        content: '▾ ';
    }
    .thought-chain-container {
        padding: 0.5rem 0 0.25rem;
    }
    .thought-step {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        position: relative;
        margin-bottom: 0.6rem;
    }
    .step-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
        margin-top: 0.55rem;
        z-index: 1;
    }
    .step-line {
        position: absolute;
        left: 4.5px;
        top: 20px;
        bottom: -4px;
        width: 2px;
        background: rgba(56,189,248,0.15);
        border-radius: 1px;
    }
    .step-card {
        flex: 1;
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: var(--transition);
        min-width: 0;
    }
    .step-card:hover {
        box-shadow: 0 2px 10px rgba(56,189,248,0.12);
    }
    .step-header {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .step-icon {
        font-size: 0.88rem;
    }
    .step-label {
        font-weight: 700;
        font-size: 0.82rem;
    }
    .step-time {
        color: #94a3b8;
        font-size: 0.72rem;
        margin-left: auto;
        font-weight: 400;
    }
    .step-body {
        font-size: 0.78rem;
        color: #475569;
        line-height: 1.45;
        max-height: 100px;
        overflow-y: auto;
        word-break: break-word;
    }
    .arg-key {
        color: #0284c7;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .arg-val {
        color: #475569;
        font-size: 0.78rem;
    }

    /* 溯源 UI 样式 */
    .message-with-citations {
        margin-top: 8px;
    }
    .citation-panel {
        margin-top: 12px;
        border-top: 1px solid rgba(100,116,139,0.2);
        padding-top: 8px;
    }
    .citation-panel summary {
        cursor: pointer;
        font-size: 0.85rem;
        color: #64748b;
        user-select: none;
        padding: 4px 0;
        list-style: none;
    }
    .citation-panel summary:hover {
        color: #475569;
    }
    .citation-panel summary::-webkit-details-marker {
        display: none;
    }
    .citation-list {
        margin-top: 8px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .citation-group {
        margin-bottom: 12px;
    }
    .citation-group:last-child {
        margin-bottom: 0;
    }
    .citation-group-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #6366f1;
        margin-bottom: 6px;
        padding-left: 4px;
        border-left: 3px solid #6366f1;
    }
    .citation-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        background: rgba(100,116,139,0.05);
    }
    .citation-item:hover {
        background: rgba(100,116,139,0.1);
    }
    .citation-badge {
        background: #38bdf8;
        color: #1e293b;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .citation-name {
        color: #1e293b;
        font-weight: 500;
    }
    .citation-meta {
        color: #64748b;
        margin-left: auto;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)


def _render_thought_chain_html(steps):
    if not steps or len(steps) <= 1:
        return ""

    # 统计每个工具名的调用次数，为多次调用添加序号
    tool_call_counts = {}
    tool_call_numbers = {}  # tool_call_id -> 显示序号

    for step in steps:
        if step["step_type"] == "tool_call" and step.get("tool_name") and step.get("tool_call_id"):
            name = step["tool_name"]
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
            tool_call_numbers[step["tool_call_id"]] = tool_call_counts[name]

    step_count = len(steps)
    step_config = {
        "thinking": {"icon": "🤔", "label": "思考", "color": "#38bdf8", "bg": "rgba(56,189,248,0.08)"},
        "tool_call": {"icon": "🔧", "color": "#22c55e", "bg": "rgba(34,197,94,0.08)"},
        "tool_result": {"icon": "👁️", "color": "#f59e0b", "bg": "rgba(245,158,11,0.08)"},
        "final_answer": {"icon": "💡", "label": "生成回答", "color": "#0d9488", "bg": "rgba(13,148,136,0.08)"},
    }

    html_parts = [f"<details class='thought-chain-details'><summary>🧠 推理过程 ({step_count}步) · 点击展开</summary>"]
    html_parts.append("<div class='thought-chain-container'>")

    for i, step in enumerate(steps):
        cfg = step_config.get(step["step_type"], {"icon": "●", "label": step["step_type"], "color": "#64748b", "bg": "rgba(100,116,139,0.08)"})
        icon = cfg["icon"]
        color = cfg["color"]
        bg = cfg["bg"]

        # 生成标签（带序号）
        if step["step_type"] == "tool_call" and step.get("tool_name"):
            tool_name = step["tool_name"]
            count = tool_call_counts.get(tool_name, 1)
            num = tool_call_numbers.get(step.get("tool_call_id"), 1)
            suffix = f" #{num}" if count > 1 else ""
            label = f"调用 {tool_name}{suffix}"
        elif step["step_type"] == "tool_result" and step.get("tool_name"):
            tool_name = step["tool_name"]
            count = tool_call_counts.get(tool_name, 1)
            num = tool_call_numbers.get(step.get("tool_call_id"), "")
            suffix = f" #{num}" if count > 1 and num else ""
            label = f"观察结果 ({tool_name}{suffix})"
        else:
            label = cfg.get("label", step["step_type"])

        content = step.get("content", "")
        tool_args = step.get("tool_args", {})
        if tool_args and step["step_type"] == "tool_call":
            args_parts = []
            for k, v in tool_args.items():
                args_parts.append(f"<span class='arg-key'>{k}</span>=<span class='arg-val'>{str(v)[:60]}</span>")
            content = " ".join(args_parts)

        timestamp = step.get("timestamp", "")
        is_last = (i == len(steps) - 1)
        line_html = "" if is_last else "<div class='step-line'></div>"

        html_parts.append(
            f"<div class='thought-step'>"
            f"<div class='step-dot' style='background:{color};box-shadow:0 0 6px {color};'></div>"
            f"{line_html}"
            f"<div class='step-card' style='background:{bg};border-left:3px solid {color};'>"
            f"<div class='step-header'><span class='step-icon'>{icon}</span> <span class='step-label' style='color:{color};'>{label}</span> <span class='step-time'>{timestamp}</span></div>"
            f"<div class='step-body'>{content}</div>"
            f"</div></div>"
        )

    html_parts.append("</div></details>")
    return "".join(html_parts)


def _render_citations_html(content):
    """解析内容中的多个"参考来源"并渲染溯源 UI"""
    if not content or "参考来源" not in content:
        return content

    import re
    matches = list(re.finditer(r'参考来源[：:]', content))

    if len(matches) == 0:
        return content

    # 单次 RAG
    if len(matches) == 1:
        return _render_single_rag_citation(content, matches[0])

    # 多次 RAG
    return _render_multiple_rag_citations(content, matches)


def _render_single_rag_citation(content, match):
    """处理单次 RAG 的参考来源"""
    split_idx = match.end()
    body = content[:match.start()].strip()
    source_block = content[split_idx:].strip()

    import re
    sources = []
    for line in source_block.split('\n'):
        line = line.strip()
        m = re.match(r'\[(\d+)\]\s*([^|]+)(?:\s*\|\s*chunk_id=(\S+))?(?:\s*\|\s*score=(\S+))?', line)
        if m:
            sources.append({
                'index': m.group(1),
                'filename': m.group(2).strip(),
                'chunk_id': m.group(3) or '-',
                'score': m.group(4) or '-'
            })

    if not sources:
        return content

    # 生成溯源 HTML
    source_items = ''.join([
        f"<div class='citation-item'>"
        f"<span class='citation-badge'>[{s['index']}]</span>"
        f"<span class='citation-name'>📄 {s['filename']}</span>"
        f"<span class='citation-meta'>chunk: {s['chunk_id']} · score: {s['score']}</span>"
        f"</div>"
        for s in sources
    ])

    citation_html = f"""
    <details class='citation-panel'>
        <summary>📚 参考来源（{len(sources)}篇）</summary>
        <div class='citation-list'>{source_items}</div>
    </details>
    """

    return f"<div class='message-with-citations'>{body}{citation_html}</div>"


def _render_multiple_rag_citations(content, matches):
    """处理多次 RAG 的参考来源"""
    import re

    body_parts = [content[:matches[0].start()].strip()]
    source_groups = []

    for i, match in enumerate(matches):
        source_start = match.end()
        next_section_start = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section = content[source_start:next_section_start]
        lines = [l.strip() for l in section.split('\n') if l.strip()]

        sources = []
        j = 0

        # 解析来源列表行
        while j < len(lines):
            m = re.match(r'\[(\d+)\]\s*([^|]+)(?:\s*\|\s*chunk_id=(\S+))?(?:\s*\|\s*score=(\S+))?', lines[j])
            if m:
                sources.append({
                    'index': m.group(1),
                    'filename': m.group(2).strip(),
                    'chunk_id': m.group(3) or '-',
                    'score': m.group(4) or '-'
                })
                j += 1
            else:
                break

        if sources:
            source_groups.append({'group_index': i + 1, 'sources': sources})

        # 收集后续正文
        if j < len(lines):
            remaining = '\n'.join(lines[j:])
            if remaining.strip():
                body_parts.append(remaining)

    if not source_groups:
        return content

    # 合并正文
    body = '\n\n'.join([p for p in body_parts if p and p.strip()])

    # 生成分组溯源 HTML
    group_htmls = []
    for group in source_groups:
        items = ''.join([
            f"<div class='citation-item'>"
            f"<span class='citation-badge'>[{s['index']}]</span>"
            f"<span class='citation-name'>📄 {s['filename']}</span>"
            f"<span class='citation-meta'>chunk: {s['chunk_id']} · score: {s['score']}</span>"
            f"</div>"
            for s in group['sources']
        ])
        group_htmls.append(
            f"<div class='citation-group'>"
            f"<div class='citation-group-title'>🔍 第{group['group_index']}次检索</div>"
            f"{items}"
            f"</div>"
        )

    total_sources = sum(len(g['sources']) for g in source_groups)
    citation_html = f"""
    <details class='citation-panel'>
        <summary>📚 参考来源（{total_sources}篇，来自{len(source_groups)}次检索）</summary>
        <div class='citation-list'>{''.join(group_htmls)}</div>
    </details>
    """

    return f"<div class='message-with-citations'>{body}{citation_html}</div>"


# ================= 1. 全局状态初始化 =================
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "conv_manager" not in st.session_state:
    st.session_state["conv_manager"] = ConversationManager()

if "current_conv_id" not in st.session_state:
    st.session_state["current_conv_id"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource
def load_yolo_model():
    return YOLO('Detct_prdc/MBE-Net/weights/best.pt')


# ================= 2. 页面配置 =================
st.set_page_config(
    page_title="SAR舰船检测 · 智能平台",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# ================= 3. 左侧导航栏 =================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size:2.8rem; margin-bottom:0.3rem;">🛰️</div>
        <div style="font-size:1.3rem; font-weight:700; color:#0c4a6e; letter-spacing:0.03em;">SAR-Sense</div>
        <div style="font-size:0.78rem; color:#0369a1; margin-top:0.15rem;">SAR Ship Detection</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if "_page_switch" in st.session_state and st.session_state["_page_switch"]:
        st.session_state["nav_radio"] = st.session_state["_page_switch"]
        st.session_state["_page_switch"] = None

    page = st.radio(
        "功能导航",
        ["🛰️ 产品测试 (SAR检测)", "🤖 智能体问答", "📚 知识库管理", "📊 可观测性"],
        label_visibility="collapsed",
        key="nav_radio"
    )
    st.divider()

    if page == "🤖 智能体问答":
        conv_mgr = st.session_state["conv_manager"]

        if st.button("➕ 新建对话", use_container_width=True):
            new_id = conv_mgr.create_conversation()
            st.session_state["current_conv_id"] = new_id
            st.session_state["messages"] = []
            st.rerun()

        conv_list = conv_mgr.list_conversations()
        if conv_list:
            st.markdown("<div style='font-size:0.82rem; color:#94a3b8; margin-bottom:0.3rem;'>💬 历史对话</div>", unsafe_allow_html=True)
            for conv in conv_list:
                col_c, col_d = st.columns([5, 1])
                with col_c:
                    is_active = conv["id"] == st.session_state["current_conv_id"]
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(f"{'▸ ' if is_active else ''}{conv['title']}", key=f"conv_{conv['id']}", type=btn_type, use_container_width=True):
                        conv_data = conv_mgr.load_conversation(conv["id"])
                        st.session_state["current_conv_id"] = conv["id"]
                        st.session_state["messages"] = conv_data["messages"]
                        st.rerun()
                with col_d:
                    if st.button("✕", key=f"del_{conv['id']}", help="删除此对话"):
                        conv_mgr.delete_conversation(conv["id"])
                        if st.session_state["current_conv_id"] == conv["id"]:
                            st.session_state["current_conv_id"] = None
                            st.session_state["messages"] = []
                        st.rerun()

        st.divider()

    st.markdown("""
    <div style="font-size:0.75rem; color:#475569; line-height:1.6; margin-top:1rem;">
        <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.3rem;">
            <span style="display:inline-block; width:7px; height:7px; background:#22c55e; border-radius:50%; box-shadow:0 0 6px #22c55e;"></span>
            系统运行中
        </div>
        <div>v2.0 · Powered by YOLO</div>
    </div>
    """, unsafe_allow_html=True)

# ================= 4. 核心功能分发 =================

if page == "🛰️ 产品测试 (SAR检测)":
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
        <div style="font-size:2.2rem;">🛰️</div>
        <div>
            <div style="font-size:1.8rem; font-weight:800; color:#1e293b; letter-spacing:-0.02em;">SAR 舰船检测工作台</div>
            <div style="font-size:0.9rem; color:#64748b; margin-top:0.1rem;">Synthetic Aperture Radar · Ship Detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown("""
        <div class="custom-card" style="text-align:center;">
            <div style="font-size:1.8rem;">🧠</div>
            <div style="font-weight:700; color:#1e293b; margin:0.4rem 0 0.2rem;">YOLO 视觉引擎</div>
            <div style="font-size:0.82rem; color:#64748b;">深度学习目标检测</div>
        </div>
        """, unsafe_allow_html=True)
    with col_info2:
        st.markdown("""
        <div class="custom-card" style="text-align:center;">
            <div style="font-size:1.8rem;">⚡</div>
            <div style="font-weight:700; color:#1e293b; margin:0.4rem 0 0.2rem;">实时推演</div>
            <div style="font-size:0.82rem; color:#64748b;">毫秒级响应速度</div>
        </div>
        """, unsafe_allow_html=True)
    with col_info3:
        st.markdown("""
        <div class="custom-card" style="text-align:center;">
            <div style="font-size:1.8rem;">🎯</div>
            <div style="font-weight:700; color:#1e293b; margin:0.4rem 0 0.2rem;">高精度识别</div>
            <div style="font-size:0.82rem; color:#64748b;">多尺度舰船定位</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.info("📤 请上传待检测的 SAR 图像，YOLO 视觉引擎将自动识别舰船目标。")
    uploaded_file = st.file_uploader("拖拽或点击上传图像文件", type=["jpg", "jpeg", "png", "tif"])

    if uploaded_file is not None:
        model = load_yolo_model()
        image = Image.open(uploaded_file)

        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, f"sar_detect_{uploaded_file.name}")
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("### 📊 检测结果")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<p style="font-weight:600; color:#475569; margin-bottom:0.5rem;">📷 原始输入图像</p>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown('<p style="font-weight:600; color:#475569; margin-bottom:0.5rem;">🔍 YOLO 检测结果</p>', unsafe_allow_html=True)
            with st.spinner("YOLO 引擎正在进行推演..."):
                results = model.predict(source=image, imgsz=640)
                res_plotted = results[0].plot()
                st.image(res_plotted, channels="BGR", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ship_count = len(results[0].boxes)
        metric_col1, metric_col2, metric_col3 = st.columns([1, 1, 2])
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{ship_count}</div>
                <div class="metric-label">🚢 检测目标数</div>
            </div>
            """, unsafe_allow_html=True)
        with metric_col2:
            status_icon = "✅" if ship_count > 0 else "⚠️"
            status_text = "检测成功" if ship_count > 0 else "无目标"
            status_color = "#059669" if ship_count > 0 else "#d97706"
            st.markdown(f"""
            <div class="metric-card" style="background:linear-gradient(135deg, {'#f0fdf4, #ecfdf5' if ship_count > 0 else '#fffbeb, #fef3c7'}); border-color:{'#a7f3d0' if ship_count > 0 else '#fcd34d'};">
                <div class="metric-value" style="color:{status_color};">{status_icon}</div>
                <div class="metric-label">{status_text}</div>
            </div>
            """, unsafe_allow_html=True)
        with metric_col3:
            if ship_count > 0:
                st.success(f"✅ 检测完成！共发现 **{ship_count}** 处舰船目标。")
            else:
                st.warning("👀 未检测到明显的舰船目标。")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🤖 发送给AI助手深度分析", use_container_width=True, type="primary"):
            st.session_state["pending_detection"] = temp_image_path
            st.session_state["_page_switch"] = "🤖 智能体问答"
            st.rerun()


elif page == "🤖 智能体问答":
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
        <div style="font-size:2.2rem;">🤖</div>
        <div>
            <div style="font-size:1.8rem; font-weight:800; color:#1e293b; letter-spacing:-0.02em;">SAR舰船检测 智能助手</div>
            <div style="font-size:0.9rem; color:#64748b; margin-top:0.1rem;">AI-Powered Intelligent Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        st.markdown("""
        <div class="custom-card" style="text-align:center; padding:0.8rem;">
            <div style="font-size:0.82rem; color:#64748b;">💬 对话轮次</div>
            <div style="font-size:1.4rem; font-weight:700; color:#1e293b;">{}</div>
        </div>
        """.format(len(st.session_state["messages"]) // 2), unsafe_allow_html=True)
    with col_status2:
        st.markdown("""
        <div class="custom-card" style="text-align:center; padding:0.8rem;">
            <div style="font-size:0.82rem; color:#64748b;">� 模型状态</div>
            <div style="font-size:1.4rem; font-weight:700; color:#059669;">就绪</div>
        </div>
        """, unsafe_allow_html=True)
    with col_status3:
        st.markdown("""
        <div class="custom-card" style="text-align:center; padding:0.8rem;">
            <div style="font-size:0.82rem; color:#64748b;">⚡ 响应模式</div>
            <div style="font-size:1.4rem; font-weight:700; color:#2563eb;">流式</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def _capture(generator, cache_list):
        for chunk in generator:
            cache_list.append(chunk)
            for char in chunk:
                time.sleep(0.01)
                yield char

    # --- 渲染已有消息 ---
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "assistant":
                with st.chat_message(msg["role"]):
                    # 渲染消息内容（应用溯源解析）
                    content_html = _render_citations_html(msg["content"])
                    st.markdown(content_html, unsafe_allow_html=True)

                    # 渲染思考链
                    if msg.get("thought_steps") and len(msg["thought_steps"]) > 1:
                        st.markdown(_render_thought_chain_html(msg["thought_steps"]), unsafe_allow_html=True)
            else:
                st.chat_message(msg["role"]).write(msg["content"])

    if st.session_state.get("_pending_viz"):
        viz_path = st.session_state["_pending_viz"]
        if os.path.exists(viz_path):
            st.image(viz_path, caption="🔍 舰船检测结果可视化", use_container_width=True)
        st.session_state["_pending_viz"] = None

    # --- 流式输出占位区（在输入框上方） ---
    stream_placeholder = st.empty()

    # --- 输入区域 ---
    col_input, col_attach = st.columns([1, 0.05], gap="small")

    with col_attach:
        with st.popover("📎", use_container_width=True):
            uploaded_attachment = st.file_uploader(
                "上传附件（自动识别内容）",
                type=["txt", "md", "pdf", "docx", "png", "jpg", "jpeg", "csv", "json", "log", "py"],
                key="chat_attachment",
                label_visibility="collapsed"
            )
    with col_input:
        prompt = st.chat_input("💬 在此输入您的问题...")

    if uploaded_attachment is not None:
        if "_last_attach_name" not in st.session_state:
            st.session_state["_last_attach_name"] = None

        if uploaded_attachment.name != st.session_state["_last_attach_name"]:
            stub_name = uploaded_attachment.name
            import tempfile
            temp_dir = tempfile.gettempdir()
            attach_path = os.path.join(temp_dir, stub_name)
            with open(attach_path, "wb") as f:
                f.write(uploaded_attachment.getbuffer())

            with st.spinner(f"📄 正在处理附件「{stub_name}」..."):
                from agent.tools.agent_tools import extract_file_content
                extracted = extract_file_content.invoke({"file_path": attach_path})
                st.session_state["_pending_attach_prompt"] = (
                    f"[用户上传了附件「{stub_name}」，文件路径：{attach_path}，内容如下]\n\n{extracted}"
                )
            st.session_state["_last_attach_name"] = stub_name
            st.rerun()

    effective_prompt = None
    if "_pending_attach_prompt" in st.session_state and st.session_state["_pending_attach_prompt"]:
        if prompt:
            effective_prompt = prompt + "\n\n" + st.session_state["_pending_attach_prompt"]
            st.session_state["_pending_attach_prompt"] = None
        else:
            st.caption(f"📎 附件已就绪，请输入问题后发送")
    elif prompt:
        effective_prompt = prompt

    # --- 检测自动触发 ---
    if "pending_detection" in st.session_state and st.session_state["pending_detection"]:
        image_path = st.session_state["pending_detection"]
        st.session_state["pending_detection"] = None
        auto_prompt = f"请对以下SAR图像进行舰船检测并深度分析：{image_path}"

        conv_mgr = st.session_state["conv_manager"]
        if st.session_state["current_conv_id"] is None:
            new_id = conv_mgr.create_conversation(auto_prompt)
            st.session_state["current_conv_id"] = new_id

        st.session_state["messages"].append({"role": "user", "content": auto_prompt})
        conv_mgr.append_message(st.session_state["current_conv_id"], "user", auto_prompt)

        # 对话记忆压缩：摘要 + 最近 N 轮
        chat_pack = conv_mgr.build_chat_pack(st.session_state["current_conv_id"])

        response_messages = []
        with st.spinner("🤖 AI助手正在检测并分析SAR图像..."):
            res_stream = st.session_state["agent"].execute_stream(
                chat_pack,
                conversation_id=st.session_state["current_conv_id"]
            )

            with stream_placeholder.container():
                st.chat_message("user").write(auto_prompt)
                st.chat_message("assistant").write_stream(_capture(res_stream, response_messages))

        assistant_content = "".join(response_messages)  # 拼接所有 chunks
        if agent_tools._last_viz_path:
            st.session_state["_pending_viz"] = agent_tools._last_viz_path
            agent_tools._last_viz_path = None
        conv_id = st.session_state["current_conv_id"]
        thought_snapshot = list(_thought_chains.get(conv_id, {}).get("steps", []))
        assistant_msg = {"role": "assistant", "content": assistant_content}
        if thought_snapshot:
            assistant_msg["thought_steps"] = thought_snapshot
        st.session_state["messages"].append(assistant_msg)
        conv_mgr.append_message(
            st.session_state["current_conv_id"], "assistant", assistant_content,
            thought_steps=thought_snapshot if thought_snapshot else None
        )
        st.rerun()

    # --- 普通对话 ---
    if effective_prompt:
        conv_mgr = st.session_state["conv_manager"]
        display_prompt = effective_prompt
        if len(display_prompt) > 500:
            display_prompt = effective_prompt[:500] + f"\n\n... (附加了附件内容，共{len(effective_prompt)}字符)"

        if st.session_state["current_conv_id"] is None:
            new_id = conv_mgr.create_conversation(effective_prompt)
            st.session_state["current_conv_id"] = new_id

        st.session_state["messages"].append({"role": "user", "content": effective_prompt})
        conv_mgr.append_message(st.session_state["current_conv_id"], "user", effective_prompt)

        # 对话记忆压缩：摘要 + 最近 N 轮
        chat_pack = conv_mgr.build_chat_pack(st.session_state["current_conv_id"])

        response_messages = []
        with st.spinner("智能客服思考中..."):
            res_stream = st.session_state["agent"].execute_stream(
                chat_pack,
                conversation_id=st.session_state["current_conv_id"]
            )

            with stream_placeholder.container():
                st.chat_message("user").write(display_prompt)
                st.chat_message("assistant").write_stream(_capture(res_stream, response_messages))

        assistant_content = "".join(response_messages)  # 拼接所有 chunks
        if agent_tools._last_viz_path:
            st.session_state["_pending_viz"] = agent_tools._last_viz_path
            agent_tools._last_viz_path = None
        conv_id = st.session_state["current_conv_id"]
        thought_snapshot = list(_thought_chains.get(conv_id, {}).get("steps", []))
        assistant_msg = {"role": "assistant", "content": assistant_content}
        if thought_snapshot:
            assistant_msg["thought_steps"] = thought_snapshot
        st.session_state["messages"].append(assistant_msg)
        conv_mgr.append_message(
            st.session_state["current_conv_id"], "assistant", assistant_content,
            thought_steps=thought_snapshot if thought_snapshot else None
        )
        st.rerun()

elif page == "📚 知识库管理":
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
        <div style="font-size:2.2rem;">📚</div>
        <div>
            <div style="font-size:1.8rem; font-weight:800; color:#1e293b; letter-spacing:-0.02em;">知识库管理</div>
            <div style="font-size:0.9rem; color:#64748b; margin-top:0.1rem;">Knowledge Base · Vector Store</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    data_dir = get_abs_path(chroma_conf["data_path"])
    os.makedirs(data_dir, exist_ok=True)

    tab1, tab2 = st.tabs(["📤 上传文件", "📋 文件列表"])

    with tab1:
        st.markdown("### 📤 上传知识库文件")
        st.info("支持 TXT / PDF 格式文件，上传后点击「入库」按钮即可自动分块并写入向量数据库。")

        uploaded_files = st.file_uploader(
            "拖拽或点击上传文件",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="kb_uploader"
        )

        if uploaded_files:
            saved_count = 0
            already_saved = st.session_state.setdefault("_saved_files", set())
            for uploaded_file in uploaded_files:
                safe_name = os.path.basename(uploaded_file.name)
                save_path = os.path.join(data_dir, safe_name)
                if safe_name in already_saved or os.path.exists(save_path):
                    continue
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
                already_saved.add(safe_name)
                st.success(f"✅ 已保存：`{safe_name}`")

            if saved_count > 0:
                st.info(f"共保存 {saved_count} 个文件到 `data/` 目录，请点击下方按钮入库。")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 一键入库", type="primary", use_container_width=True):
            with st.spinner("正在加载向量库服务并处理文件..."):
                try:
                    vs = get_vector_store_service()
                    new_count, updated_count, skipped_count, removed_count = vs.load_document()
                    st.session_state["_saved_files"] = set()
                    if new_count > 0 or updated_count > 0:
                        msg = f"✅ 知识库入库完成！新增 {new_count}、更新 {updated_count} 个文件。"
                        if removed_count > 0:
                            msg += f" 已清理 {removed_count} 个已删除文件的残留数据。"
                        st.success(msg)
                    elif skipped_count > 0:
                        st.info(f"ℹ️ 共扫描文件均无变更，无需重复处理。")
                    else:
                        st.info("ℹ️ 未检测到新文件，请先上传文件。")
                except Exception as e:
                    st.error(f"❌ 入库失败：{str(e)}")

    with tab2:
        st.markdown("### 📋 已入库文档")

        try:
            vs = get_vector_store_service()
            manifest = vs.manifest
        except Exception as e:
            manifest = {}
            st.error(f"❌ 读取 manifest 失败：{str(e)}")

        docs = sorted(
            manifest.items(),
            key=lambda item: item[1].get("ingested_at", ""),
            reverse=True
        )
        total_chunks = sum(entry.get("chunk_count", 0) for _, entry in docs)

        if docs:
            col_status1, col_status2, col_status3 = st.columns(3)
            with col_status1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(docs)}</div>
                    <div class="metric-label">📄 已入库文档</div>
                </div>
                """, unsafe_allow_html=True)
            with col_status2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_chunks}</div>
                    <div class="metric-label">🧩 向量切片</div>
                </div>
                """, unsafe_allow_html=True)
            with col_status3:
                st.markdown("""
                <div class="metric-card" style="background:linear-gradient(135deg, #eff6ff, #dbeafe); border-color:#93c5fd;">
                    <div class="metric-value" style="color:#2563eb;">ChromaDB</div>
                    <div class="metric-label">🗄️ 向量引擎</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for idx, (fname, entry) in enumerate(docs, 1):
                doc_id = entry.get("doc_id", "-")
                file_hash = entry.get("file_hash", "")
                hash_short = f"{file_hash[:8]}...{file_hash[-6:]}" if len(file_hash) > 16 else (file_hash or "-")
                chunk_count = entry.get("chunk_count", 0)
                chunk_method = entry.get("chunk_method", "-")
                file_type = entry.get("file_type", "-")
                status = entry.get("status", "active")
                ingested_at = entry.get("ingested_at", "-").replace("T", " ")
                icon = "📕" if file_type == "pdf" else "📄"
                safe_name = html.escape(fname)
                safe_doc_id = html.escape(str(doc_id))
                safe_hash = html.escape(hash_short)

                st.markdown(f"""
                <div class="custom-card" style="padding:1rem 1.1rem; margin-bottom:0.55rem;">
                    <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:1rem;">
                        <div style="display:flex; align-items:flex-start; gap:0.7rem; min-width:0;">
                            <span style="font-size:1.4rem; line-height:1.2;">{icon}</span>
                            <div style="min-width:0;">
                                <div style="font-weight:700; color:#1e293b; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{safe_name}</div>
                                <div style="font-size:0.78rem; color:#64748b; margin-top:0.22rem;">
                                    doc_id: <code>{safe_doc_id}</code> · {file_type} · {chunk_method}
                                </div>
                            </div>
                        </div>
                        <span style="font-size:0.72rem; color:#0d9488; background:#ecfdf5; border:1px solid #99f6e4; border-radius:999px; padding:0.18rem 0.55rem;">{status}</span>
                    </div>
                    <div style="display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:0.5rem; margin-top:0.75rem;">
                        <div style="font-size:0.76rem; color:#64748b;">chunks<br><b style="color:#0f172a;">{chunk_count}</b></div>
                        <div style="font-size:0.76rem; color:#64748b;">hash<br><b style="color:#0f172a;">{safe_hash}</b></div>
                        <div style="font-size:0.76rem; color:#64748b;">indexed<br><b style="color:#0f172a;">{ingested_at}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                delete_col, spacer_col = st.columns([1, 5])
                with delete_col:
                    if st.button("删除", key=f"kb_delete_{doc_id}", help="删除 Chroma 向量和 data/ 下的原始文件"):
                        deleted_chunks = vs.delete_document_by_doc_id(doc_id, delete_file=True)
                        st.success(f"已删除 {fname}，清理 {deleted_chunks} 个 chunk。")
                        st.rerun()
                with spacer_col:
                    st.caption("删除后不会在下一次入库时被自动扫回。")
        else:
            st.info("📭 knowledge manifest 暂无已入库文档，请切换到「上传文件」标签页添加。")

elif page == "📊 可观测性":
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem;">
        <div style="font-size:2.2rem;">📊</div>
        <div>
            <div style="font-size:1.8rem; font-weight:800; color:#1e293b; letter-spacing:-0.02em;">Agent 可观测性面板</div>
            <div style="font-size:0.9rem; color:#64748b; margin-top:0.1rem;">Observability · Metrics · Tool Track</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    metrics = AgentMetrics()

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(f"""
        <div class="metric-card" style="background:linear-gradient(135deg, #eff6ff, #dbeafe); border-color:#93c5fd; text-align:center;">
            <div class="metric-value" style="color:#2563eb;">{metrics.conversation_rounds}</div>
            <div class="metric-label">💬 对话轮次</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="metric-card" style="background:linear-gradient(135deg, #fef3c7, #fef9c3); border-color:#fcd34d; text-align:center;">
            <div class="metric-value" style="color:#d97706;">{metrics.total_tool_calls}</div>
            <div class="metric-label">🔧 工具调用次数</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        st.markdown(f"""
        <div class="metric-card" style="background:linear-gradient(135deg, #f0fdf4, #ecfdf5); border-color:#a7f3d0; text-align:center;">
            <div class="metric-value" style="color:#059669;">{metrics.overall_success_rate}%</div>
            <div class="metric-label">✅ 工具调用成功率</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s4:
        st.markdown(f"""
        <div class="metric-card" style="background:linear-gradient(135deg, #fdf2f8, #fce7f3); border-color:#f9a8d4; text-align:center;">
            <div class="metric-value" style="color:#be185d;">{metrics.avg_tool_calls_per_round}</div>
            <div class="metric-label">⚡ 每轮平均调用</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_r1, col_r2 = st.columns([1, 3])
    with col_r1:
        if st.button("🔄 重置指标", use_container_width=True):
            metrics.reset()
            st.success("指标已重置")
            st.rerun()

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📈 工具调用分布", "📋 调用时间线", "📊 工具详情"])

    with tab1:
        st.markdown("### 📈 各工具调用频次")
        tool_stats = metrics.get_tool_stats()
        if tool_stats:
            chart_data = pd.DataFrame([
                {"工具": s["tool_name"], "调用次数": s["total"], "成功": s["success"], "失败": s["fail"]}
                for s in tool_stats
            ])

            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            matplotlib.rcParams['axes.unicode_minus'] = False

            fig, ax = plt.subplots(figsize=(10, 4.5))
            names = chart_data["工具"].tolist()
            x = range(len(names))
            bar_w = 0.35
            ax.bar([i - bar_w / 2 for i in x], chart_data["成功"], bar_w, color="#059669", label="成功")
            ax.bar([i + bar_w / 2 for i in x], chart_data["失败"], bar_w, color="#dc2626", label="失败")
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9, rotation=30, ha="center")
            ax.set_ylabel("调用次数", fontsize=11)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("📭 暂无工具调用记录，请先在智能体问答页面发起对话。")

    with tab2:
        st.markdown("### 📋 工具调用时间线")
        records = metrics.get_recent_records(50)
        if records:
            for r in records:
                status_color = "#059669" if r["success"] else "#dc2626"
                status_icon = "✅" if r["success"] else "❌"
                st.markdown(f"""
                <div class="custom-card" style="display:flex; align-items:center; justify-content:space-between; padding:0.6rem 1rem; margin-bottom:0.3rem;">
                    <div style="display:flex; align-items:center; gap:0.6rem;">
                        <span style="color:{status_color}; font-weight:bold;">{status_icon}</span>
                        <span style="font-weight:600; color:#1e293b;">{r["tool_name"]}</span>
                    </div>
                    <div style="display:flex; gap:1.2rem; font-size:0.82rem; color:#64748b;">
                        <span>⏱ {r["duration_ms"]}ms</span>
                        <span>🕐 {r["timestamp"]}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📭 暂无调用记录。")

    with tab3:
        st.markdown("### 📊 各工具详细统计")
        tool_stats = metrics.get_tool_stats()
        if tool_stats:
            detail_data = pd.DataFrame([
                {
                    "工具名称": s["tool_name"],
                    "调用次数": s["total"],
                    "成功": s["success"],
                    "失败": s["fail"],
                    "成功率": f"{s['success_rate']}%",
                    "平均耗时(ms)": s["avg_duration_ms"],
                }
                for s in tool_stats
            ])
            st.dataframe(detail_data, use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔁 循环调用检测")
            all_records = metrics.tool_call_records
            if len(all_records) >= 3:
                repeat_warnings = []
                for i in range(len(all_records) - 2):
                    if all_records[i]["tool_name"] == all_records[i + 1]["tool_name"] == all_records[i + 2]["tool_name"]:
                        repeat_warnings.append({
                            "工具": all_records[i]["tool_name"],
                            "连续次数": "≥3",
                            "首次时间": all_records[i]["timestamp"],
                            "末次时间": all_records[i + 2]["timestamp"],
                            "状态": "⚠️ 疑似循环调用",
                        })
                if repeat_warnings:
                    st.warning(f"检测到 {len(repeat_warnings)} 次可能的循环调用，请关注以下工具")
                    st.dataframe(
                        pd.DataFrame(repeat_warnings),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success("✅ 未检测到循环调用，所有工具调用正常。")
            else:
                st.info("📭 调用次数不足，无法检测循环调用（至少需要 3 次）。")
        else:
            st.info("📭 暂无工具调用记录，请先在智能体问答页面发起对话。")
