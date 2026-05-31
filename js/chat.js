// SAR-Sense Chat Page JavaScript

const API_BASE = '';

const state = {
  currentConversationId: null,
  messages: [],
  isStreaming: false,
  attachedFile: null
};

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initScrollReveal();
  initParticles();
  initMobileMenu();
  initChat();
  loadConversations();
  highlightNav('chat');
});

function highlightNav(page) {
  document.querySelectorAll('.navbar-nav a').forEach(a => {
    if (a.getAttribute('href') === `${page}.html`) {
      a.classList.add('active');
    }
  });
}

// ==================== Navbar ====================
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  window.addEventListener('scroll', () => {
    if (window.pageYOffset > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });
}

// ==================== Scroll Reveal ====================
function initScrollReveal() {
  const revealElements = document.querySelectorAll('.reveal');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
        const children = entry.target.querySelectorAll('.stagger');
        children.forEach((child, index) => {
          child.style.animationDelay = `${index * 0.1}s`;
          child.classList.add('animate-slide-up');
        });
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
  revealElements.forEach(el => observer.observe(el));
}

// ==================== Particles ====================
function initParticles() {
  const particlesContainer = document.querySelector('.particles');
  if (!particlesContainer) return;
  for (let i = 0; i < 50; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    particle.style.left = `${Math.random() * 100}%`;
    particle.style.top = `${Math.random() * 100}%`;
    const size = Math.random() * 3 + 1;
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    particle.style.animation = `float ${Math.random() * 3 + 2}s ease-in-out infinite`;
    particle.style.animationDelay = `${Math.random() * 2}s`;
    particle.style.opacity = Math.random() * 0.5 + 0.1;
    particlesContainer.appendChild(particle);
  }
}

// ==================== Mobile Menu ====================
function initMobileMenu() {
  const menuBtn = document.querySelector('.mobile-menu-btn');
  const nav = document.querySelector('.navbar-nav');
  if (!menuBtn || !nav) return;
  menuBtn.addEventListener('click', () => {
    nav.classList.toggle('active');
    menuBtn.textContent = nav.classList.contains('active') ? '✕' : '☰';
  });
  nav.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      nav.classList.remove('active');
      menuBtn.textContent = '☰';
    });
  });
}

// ==================== Conversation Management ====================
async function loadConversations() {
  try {
    const response = await fetch(`${API_BASE}/api/conversations`);
    const data = await response.json();
    if (data.success) {
      renderConversationList(data.conversations);
    }
  } catch (error) {
    console.error('Failed to load conversations:', error);
  }
}

function renderConversationList(conversations) {
  const list = document.getElementById('conversationList');
  if (!list) return;
  list.innerHTML = '';

  if (conversations.length === 0) {
    list.innerHTML = '<div style="text-align: center; color: var(--text-tertiary); padding: var(--space-lg); font-size: 0.875rem;">暂无对话</div>';
    return;
  }

  conversations.forEach(conv => {
    const item = document.createElement('div');
    item.className = `conversation-item ${conv.id === state.currentConversationId ? 'active' : ''}`;
    item.innerHTML = `
      <span class="conversation-title">${escapeHtml(conv.title)}</span>
      <button class="conversation-delete" data-id="${conv.id}" title="删除">✕</button>
    `;
    item.addEventListener('click', (e) => {
      if (!e.target.classList.contains('conversation-delete')) {
        loadConversation(conv.id);
      }
    });
    item.querySelector('.conversation-delete').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(conv.id);
    });
    list.appendChild(item);
  });
}

async function loadConversation(convId) {
  try {
    const response = await fetch(`${API_BASE}/api/conversations/${convId}`);
    const data = await response.json();
    if (data.success) {
      state.currentConversationId = convId;
      state.messages = data.conversation.messages || [];
      renderMessages();
      loadConversations();
    }
  } catch (error) {
    console.error('Failed to load conversation:', error);
  }
}

async function createConversation(firstMessage) {
  try {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: firstMessage })
    });
    const data = await response.json();
    if (data.success) {
      state.currentConversationId = data.conversation_id;
      state.messages = [];
      loadConversations();
      return data.conversation_id;
    }
  } catch (error) {
    console.error('Failed to create conversation:', error);
  }
  return null;
}

async function deleteConversation(convId) {
  try {
    await fetch(`${API_BASE}/api/conversations/${convId}`, { method: 'DELETE' });
    if (state.currentConversationId === convId) {
      state.currentConversationId = null;
      state.messages = [];
      renderMessages();
    }
    loadConversations();
  } catch (error) {
    console.error('Failed to delete conversation:', error);
  }
}

async function appendMessageToConversation(role, content, thoughtSteps = null) {
  if (!state.currentConversationId) return;
  try {
    await fetch(`${API_BASE}/api/conversations/${state.currentConversationId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role, content, thought_steps: thoughtSteps })
    });
  } catch (error) {
    console.error('Failed to append message:', error);
  }
}

// ==================== Chat Feature ====================
function initChat() {
  const chatMessages = document.getElementById('chatMessages');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const newConvBtn = document.getElementById('newConvBtn');
  const attachBtn = document.getElementById('attachBtn');
  const fileInput = document.getElementById('fileInput');
  const removeAttachment = document.getElementById('removeAttachment');

  if (!chatMessages || !chatInput) return;

  if (newConvBtn) {
    newConvBtn.addEventListener('click', () => {
      state.currentConversationId = null;
      state.messages = [];
      state.attachedFile = null;
      renderMessages();
      loadConversations();
      updateAttachmentIndicator();
    });
  }

  if (attachBtn && fileInput) {
    attachBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async (e) => {
      if (e.target.files.length > 0) {
        state.attachedFile = { name: e.target.files[0].name, file: e.target.files[0] };
        updateAttachmentIndicator();
      }
    });
  }

  if (removeAttachment) {
    removeAttachment.addEventListener('click', () => {
      state.attachedFile = null;
      updateAttachmentIndicator();
    });
  }

  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
  });

  sendBtn.addEventListener('click', () => sendMessage());
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

function updateAttachmentIndicator() {
  const indicator = document.getElementById('attachmentIndicator');
  const nameSpan = document.getElementById('attachmentName');
  if (state.attachedFile) {
    indicator.style.display = 'flex';
    nameSpan.textContent = state.attachedFile.name;
  } else {
    indicator.style.display = 'none';
  }
}

async function sendMessage() {
  const chatInput = document.getElementById('chatInput');
  let message = chatInput.value.trim();
  if (!message && !state.attachedFile) return;
  if (state.isStreaming) return;

  if (!state.currentConversationId) {
    const preview = message.substring(0, 20) || '附件分析';
    await createConversation(preview);
  }

  let attachmentContent = '';
  let attachmentName = '';
  let attachmentPath = '';
  if (state.attachedFile) {
    attachmentName = state.attachedFile.name;
    const result = await extractFileContent(state.attachedFile.file);
    attachmentContent = result.content;
    attachmentPath = result.filePath;
    state.attachedFile = null;
    updateAttachmentIndicator();
  }

  let fullMessage = message;
  if (attachmentContent) {
    fullMessage = `[用户上传了附件「${attachmentName}」，文件路径：${attachmentPath}，内容如下]\n\n${attachmentContent}\n\n${message || '请分析'}`;
  }

  const userMessage = { role: 'user', content: message || '请分析附件内容' };
  state.messages.push(userMessage);
  renderMessages();
  // 用户消息由后端 /api/chat/stream 统一写入对话记录（纯文本），此处不再重复追加

  chatInput.value = '';
  chatInput.style.height = 'auto';
  await sendMessageStreaming(fullMessage, message || '请分析附件内容');
}

async function extractFileContent(file) {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const response = await fetch(`${API_BASE}/api/extract-file`, {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    if (data.success) return { content: data.content, filePath: data.file_path };
    return { content: '', filePath: '' };
  } catch (error) {
    console.error('Failed to extract file:', error);
    return { content: '', filePath: '' };
  }
}

async function sendMessageStreaming(message, displayMessage) {
  state.isStreaming = true;
  const assistantMessage = { role: 'assistant', content: '', thoughtSteps: [], pendingChunks: [], isTyping: false };
  state.messages.push(assistantMessage);
  renderMessages();

  const chatMessages = document.getElementById('chatMessages');
  chatMessages.scrollTop = chatMessages.scrollHeight;
  startTypewriterEffect();

  try {
    const messagesHistory = state.messages.slice(0, -2).map(m => ({ role: m.role, content: m.content }));
    const response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        display_message: displayMessage,
        messages: messagesHistory,
        conversation_id: state.currentConversationId
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'chunk') {
              assistantMessage.pendingChunks.push(data.content);
            } else if (data.type === 'thought_step') {
              assistantMessage.thoughtSteps.push(data.step);
              updateThoughtChainRealtime(assistantMessage.thoughtSteps);
            } else if (data.type === 'done') {
              assistantMessage.streamDone = true;
            } else if (data.type === 'error') {
              assistantMessage.pendingChunks.push(`\n\n[错误: ${data.message}]`);
            }
          } catch (e) {}
        }
      }
    }

    while (assistantMessage.pendingChunks.length > 0 || assistantMessage.isTyping) {
      await new Promise(resolve => setTimeout(resolve, 50));
    }

    updateLastMessage(true);
    appendMessageToConversation('assistant', assistantMessage.content, assistantMessage.thoughtSteps);
  } catch (error) {
    console.error('Streaming error:', error);
    assistantMessage.content += '\n\n[连接错误，请重试]';
    updateLastMessage(true);
  }
  state.isStreaming = false;
}

let isTypewriterRunning = false;

function startTypewriterEffect() {
  if (isTypewriterRunning) return;
  isTypewriterRunning = true;
  processTypewriterQueue();
}

async function processTypewriterQueue() {
  while (state.isStreaming || state.messages[state.messages.length - 1]?.pendingChunks?.length > 0) {
    const assistantMessage = state.messages[state.messages.length - 1];
    if (!assistantMessage) break;

    if (assistantMessage.pendingChunks.length > 0) {
      const chunk = assistantMessage.pendingChunks.shift();
      assistantMessage.isTyping = true;
      for (let i = 0; i < chunk.length; i++) {
        assistantMessage.content += chunk[i];
        updateLastMessage(false);
        await new Promise(resolve => setTimeout(resolve, 15));
      }
      assistantMessage.isTyping = false;
    } else {
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
  isTypewriterRunning = false;
}

function updateLastMessage(isFinal = false) {
  const chatMessages = document.getElementById('chatMessages');
  const lastMessage = chatMessages.lastElementChild;
  if (!lastMessage || !lastMessage.classList.contains('assistant')) return;

  const contentDiv = lastMessage.querySelector('.message-content');
  const assistantMessage = state.messages[state.messages.length - 1];
  const existingThoughtChain = contentDiv.querySelector('.thought-chain');

  let html = renderMarkdown(assistantMessage.content);
  if (!isFinal && state.isStreaming) {
    html += '<span class="streaming-cursor"></span>';
  }

  if (existingThoughtChain) {
    contentDiv.innerHTML = html;
    contentDiv.appendChild(existingThoughtChain);
  } else {
    contentDiv.innerHTML = html;
    if (isFinal && assistantMessage.thoughtSteps && assistantMessage.thoughtSteps.length > 1) {
      const thoughtChainHtml = renderThoughtChain(assistantMessage.thoughtSteps);
      contentDiv.innerHTML += thoughtChainHtml;
      const thoughtChain = contentDiv.querySelector('.thought-chain');
      if (thoughtChain) {
        const header = thoughtChain.querySelector('.thought-chain-header');
        header.addEventListener('click', () => {
          thoughtChain.classList.toggle('expanded');
          const toggle = thoughtChain.querySelector('.thought-chain-toggle');
          toggle.textContent = thoughtChain.classList.contains('expanded') ? '点击收起 ▴' : '点击展开 ▾';
        });
      }
    }
  }
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateThoughtChainRealtime(thoughtSteps) {
  const chatMessages = document.getElementById('chatMessages');
  const lastMessage = chatMessages.lastElementChild;
  if (!lastMessage || !lastMessage.classList.contains('assistant')) return;

  const contentDiv = lastMessage.querySelector('.message-content');
  let thoughtChainContainer = contentDiv.querySelector('.thought-chain');

  if (!thoughtChainContainer) {
    const container = document.createElement('div');
    container.className = 'thought-chain';
    container.innerHTML = `
      <div class="thought-chain-header">
        <div class="thought-chain-title">
          <span>🧠</span>
          <span>推理过程</span>
        </div>
        <div class="thought-chain-toggle">点击展开 ▾</div>
      </div>
      <div class="thought-chain-body">
        <div class="thought-timeline"></div>
      </div>
    `;
    contentDiv.appendChild(container);
    const header = container.querySelector('.thought-chain-header');
    header.addEventListener('click', () => {
      container.classList.toggle('expanded');
      const toggle = container.querySelector('.thought-chain-toggle');
      toggle.textContent = container.classList.contains('expanded') ? '点击收起 ▴' : '点击展开 ▾';
    });
    thoughtChainContainer = container;
  }

  const timeline = thoughtChainContainer.querySelector('.thought-timeline');
  const stepConfig = {
    thinking: { icon: '💭', label: '思考' },
    tool_call: { icon: '🔧', label: '工具调用' },
    tool_result: { icon: '👁️', label: '观察结果' },
    final_answer: { icon: '💡', label: '生成回答' }
  };

  let html = '';
  thoughtSteps.forEach(step => {
    const config = stepConfig[step.step_type] || stepConfig.thinking;
    let label = config.label;
    if (step.step_type === 'tool_call' && step.tool_name) label = `调用 ${step.tool_name}`;
    else if (step.step_type === 'tool_result' && step.tool_name) label = `观察结果 (${step.tool_name})`;

    let bodyHtml = escapeHtml(step.content || '');
    if (step.tool_args && Object.keys(step.tool_args).length > 0) {
      bodyHtml += '<div class="thought-args">';
      Object.entries(step.tool_args).forEach(([key, value]) => {
        const valStr = typeof value === 'string' ? value.substring(0, 60) : JSON.stringify(value).substring(0, 60);
        bodyHtml += `<span class="thought-arg"><span class="thought-arg-key">${escapeHtml(key)}</span>=<span class="thought-arg-val">${escapeHtml(valStr)}</span></span>`;
      });
      bodyHtml += '</div>';
    }

    html += `
      <div class="thought-step ${step.step_type}">
        <div class="thought-step-card">
          <div class="thought-step-header">
            <div class="thought-step-label">
              <span class="icon">${config.icon}</span>
              <span>${escapeHtml(label)}</span>
            </div>
            <span class="thought-step-time">${step.timestamp || ''}</span>
          </div>
          <div class="thought-step-body">${bodyHtml}</div>
        </div>
      </div>
    `;
  });

  timeline.innerHTML = html;
  const titleSpan = thoughtChainContainer.querySelector('.thought-chain-title span:last-child');
  if (titleSpan) titleSpan.textContent = `推理过程 (${thoughtSteps.length}步)`;
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderMessages() {
  const chatMessages = document.getElementById('chatMessages');
  if (!chatMessages) return;

  if (state.messages.length === 0) {
    chatMessages.innerHTML = `
      <div class="message assistant">
        <div class="message-avatar">🤖</div>
        <div class="message-content">
          <p>你好！我是 SAR-Sense 智能助手。我可以帮你：</p>
          <ul>
            <li>分析 SAR 图像检测结果</li>
            <li>查询海洋环境数据</li>
            <li>检索知识库文档</li>
            <li>生成专业分析报告</li>
          </ul>
          <p>请问有什么可以帮你的？</p>
        </div>
      </div>
    `;
    return;
  }

  chatMessages.innerHTML = '';
  state.messages.forEach((msg, index) => {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${msg.role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = msg.role === 'user' ? '👤' : '🤖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    let html = renderMarkdown(msg.content);
    if (msg.role === 'assistant' && index === state.messages.length - 1 && state.isStreaming) {
      html += '<span class="streaming-cursor"></span>';
    }
    contentDiv.innerHTML = html;

    if (msg.role === 'assistant' && msg.thought_steps && msg.thought_steps.length > 1) {
      const thoughtChainHtml = renderThoughtChain(msg.thought_steps);
      contentDiv.innerHTML += thoughtChainHtml;
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
  });

  chatMessages.querySelectorAll('.thought-chain').forEach(chain => {
    const header = chain.querySelector('.thought-chain-header');
    header.addEventListener('click', () => {
      chain.classList.toggle('expanded');
      const toggle = chain.querySelector('.thought-chain-toggle');
      toggle.textContent = chain.classList.contains('expanded') ? '点击收起 ▴' : '点击展开 ▾';
    });
  });

  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ==================== Markdown Rendering ====================
function renderMarkdown(text) {
  if (!text) return '';
  const parts = [];
  let remaining = text;
  let inCodeBlock = false;
  let codeBlockContent = '';
  let codeBlockLang = '';

  while (remaining.length > 0) {
    if (!inCodeBlock) {
      const codeStart = remaining.indexOf('```');
      if (codeStart === -1) {
        parts.push({ type: 'text', content: remaining });
        break;
      }
      if (codeStart > 0) parts.push({ type: 'text', content: remaining.substring(0, codeStart) });
      const langEnd = remaining.indexOf('\n', codeStart + 3);
      if (langEnd !== -1) {
        codeBlockLang = remaining.substring(codeStart + 3, langEnd).trim();
        remaining = remaining.substring(langEnd + 1);
      } else {
        remaining = remaining.substring(codeStart + 3);
      }
      inCodeBlock = true;
      codeBlockContent = '';
    } else {
      const codeEnd = remaining.indexOf('```');
      if (codeEnd === -1) {
        codeBlockContent += remaining;
        parts.push({ type: 'code', content: codeBlockContent, lang: codeBlockLang });
        remaining = '';
        break;
      }
      codeBlockContent += remaining.substring(0, codeEnd);
      parts.push({ type: 'code', content: codeBlockContent, lang: codeBlockLang });
      remaining = remaining.substring(codeEnd + 3);
      inCodeBlock = false;
      codeBlockLang = '';
    }
  }
  if (inCodeBlock && codeBlockContent) {
    parts.push({ type: 'code', content: codeBlockContent, lang: codeBlockLang });
  }

  let html = '';
  for (const part of parts) {
    if (part.type === 'code') {
      const escaped = escapeHtml(part.content);
      const langLabel = part.lang ? `<div class="code-lang">${escapeHtml(part.lang)}</div>` : '';
      html += `<div class="code-block">${langLabel}<pre><code>${escaped}</code></pre></div>`;
    } else {
      html += renderInlineMarkdown(part.content);
    }
  }
  return html;
}

function renderInlineMarkdown(text) {
  if (!text) return '';
  text = text.replace(/\r\n/g, '\n');
  text = text.replace(/\r/g, '\n');
  text = text.replace(/\n{2,}/g, '\n');

  let html = escapeHtml(text);
  html = processTables(html);
  html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
  html = html.replace(/~~(.+?)~~/g, '<del>$1</del>');
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="md-link">$1</a>');
  html = html.replace(/^######\s+(.+)$/gm, '<h6>$1</h6>');
  html = html.replace(/^#####\s+(.+)$/gm, '<h5>$1</h5>');
  html = html.replace(/^####\s+(.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
  html = html.replace(/^---+$/gm, '<hr class="md-hr">');
  html = html.replace(/^&gt;\s+(.+)$/gm, '<blockquote class="md-blockquote">$1</blockquote>');

  const lines = html.split('\n');
  let inList = false;
  let listType = '';
  const processedLines = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const ulMatch = line.match(/^[\-\*]\s+(.*)/);
    const olMatch = line.match(/^\d+\.\s+(.*)/);

    if (ulMatch) {
      if (!inList || listType !== 'ul') {
        if (inList) processedLines.push(listType === 'ul' ? '</ul>' : '</ol>');
        processedLines.push('<ul>');
        inList = true;
        listType = 'ul';
      }
      processedLines.push(`<li>${ulMatch[1]}</li>`);
    } else if (olMatch) {
      if (!inList || listType !== 'ol') {
        if (inList) processedLines.push(listType === 'ul' ? '</ul>' : '</ol>');
        processedLines.push('<ol>');
        inList = true;
        listType = 'ol';
      }
      processedLines.push(`<li>${olMatch[1]}</li>`);
    } else {
      if (inList) {
        processedLines.push(listType === 'ul' ? '</ul>' : '</ol>');
        inList = false;
        listType = '';
      }
      processedLines.push(line);
    }
  }
  if (inList) processedLines.push(listType === 'ul' ? '</ul>' : '</ol>');

  html = processedLines.join('\n');
  html = html.replace(/\n/g, '<br>');
  html = html.replace(/(<br\s*\/?>\s*){2,}/gi, '<br>');
  html = html.replace(/(<br>\s*){2,}/g, '<br>');
  html = html.replace(/(<\/h[1-6]>)<br>/g, '$1');
  html = html.replace(/(<\/ul>)<br>/g, '$1');
  html = html.replace(/(<\/ol>)<br>/g, '$1');
  html = html.replace(/(<\/blockquote>)<br>/g, '$1');
  html = html.replace(/(<hr[^>]*>)<br>/g, '$1');
  html = html.replace(/(<div class="code-block">)/g, '<br>$1');
  html = html.replace(/(<\/table>)<br>/g, '$1');
  return html;
}

function processTables(html) {
  const lines = html.split('\n');
  const result = [];
  let i = 0;
  while (i < lines.length) {
    if (lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
      if (i + 1 < lines.length && /^\|[\s\-:|]+\|$/.test(lines[i + 1].trim())) {
        const tableLines = [lines[i], lines[i + 1]];
        i += 2;
        while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
          tableLines.push(lines[i]);
          i++;
        }
        result.push(buildTableHtml(tableLines));
        continue;
      }
    }
    result.push(lines[i]);
    i++;
  }
  return result.join('\n');
}

function buildTableHtml(tableLines) {
  const headerCells = parseTableRow(tableLines[0]);
  const dataRows = [];
  for (let i = 2; i < tableLines.length; i++) {
    dataRows.push(parseTableRow(tableLines[i]));
  }
  let html = '<div class="table-wrapper"><table>';
  html += '<thead><tr>';
  headerCells.forEach(cell => html += `<th>${cell.trim()}</th>`);
  html += '</tr></thead><tbody>';
  dataRows.forEach(row => {
    html += '<tr>';
    row.forEach(cell => html += `<td>${cell.trim()}</td>`);
    html += '</tr>';
  });
  html += '</tbody></table></div>';
  return html;
}

function parseTableRow(line) {
  const cells = line.split('|');
  cells.shift();
  cells.pop();
  return cells;
}

function renderThoughtChain(steps) {
  const stepConfig = {
    thinking: { icon: '💭', label: '思考', color: '#3b82f6' },
    tool_call: { icon: '🔧', label: '工具调用', color: '#22c55e' },
    tool_result: { icon: '👁️', label: '观察结果', color: '#f59e0b' },
    final_answer: { icon: '💡', label: '生成回答', color: '#06b6d4' }
  };

  let html = `
    <div class="thought-chain">
      <div class="thought-chain-header">
        <div class="thought-chain-title">
          <span>🧠</span>
          <span>推理过程 (${steps.length}步)</span>
        </div>
        <div class="thought-chain-toggle">点击展开 ▾</div>
      </div>
      <div class="thought-chain-body">
        <div class="thought-timeline">
  `;

  steps.forEach(step => {
    const config = stepConfig[step.step_type] || stepConfig.thinking;
    let label = config.label;
    if (step.step_type === 'tool_call' && step.tool_name) label = `调用 ${step.tool_name}`;
    else if (step.step_type === 'tool_result' && step.tool_name) label = `观察结果 (${step.tool_name})`;

    let bodyHtml = escapeHtml(step.content || '');
    if (step.tool_args && Object.keys(step.tool_args).length > 0) {
      bodyHtml += '<div class="thought-args">';
      Object.entries(step.tool_args).forEach(([key, value]) => {
        const valStr = typeof value === 'string' ? value.substring(0, 60) : JSON.stringify(value).substring(0, 60);
        bodyHtml += `<span class="thought-arg"><span class="thought-arg-key">${escapeHtml(key)}</span>=<span class="thought-arg-val">${escapeHtml(valStr)}</span></span>`;
      });
      bodyHtml += '</div>';
    }

    html += `
      <div class="thought-step ${step.step_type}">
        <div class="thought-step-card">
          <div class="thought-step-header">
            <div class="thought-step-label">
              <span class="icon">${config.icon}</span>
              <span>${escapeHtml(label)}</span>
            </div>
            <span class="thought-step-time">${step.timestamp || ''}</span>
          </div>
          <div class="thought-step-body">${bodyHtml}</div>
        </div>
      </div>
    `;
  });

  html += `</div></div></div>`;
  return html;
}

// ==================== Utility ====================
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
