// SAR-Sense Chat Page JavaScript

const API_BASE = '';

const state = {
  currentConversationId: null,
  messages: [],
  attachedFile: null,

  // 多Reader管理（支持后台流式输出）
  activeReaders: new Map(),      // conversationId -> { reader, abortController, startTime }
  streamingStatus: new Map(),    // conversationId -> { isStreaming, progress }
  backgroundMessages: new Map(),  // conversationId -> messages[]

  pollingTimer: null,             // 切回后等 assistant 落库的轮询定时器（整页跳转场景兜底）
  pollingConversationId: null     // 正在轮询的会话；切到别的会话立即停
};

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initScrollReveal();
  initParticles();
  initMobileMenu();
  initChat();
  loadConversations();
  highlightNav('chat');

  // 切到其他页面（知识库/检测/指标）= 整页跳转，SSE 与 state 全销毁；
  // 回来时按 localStorage 记住的上次会话自动恢复，能看到已落库的 assistant 回答
  // （后端 generate 已解耦，切走时 agent 后台跑完，回答已存库）
  const lastId = localStorage.getItem('lastConversationId');
  if (lastId) {
    loadConversation(lastId);
  }
});

function highlightNav(page) {
  document.querySelectorAll('.navbar-nav a').forEach(a => {
    const href = a.getAttribute('href');
    if (href && href.startsWith(`${page}.html`)) {
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
    const response = await apiFetch(`${API_BASE}/api/conversations`);
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

    // 添加生成状态图标
    let statusIcon = '';
    if (state.streamingStatus.has(conv.id)) {
      const status = state.streamingStatus.get(conv.id);
      if (status.isStreaming) {
        statusIcon = '<span class="streaming-indicator" title="正在生成">⏳</span>';
      }
    }

    item.innerHTML = `
      ${statusIcon}
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
    stopPollingForAssistant();  // 切会话即停旧轮询
    const previousConvId = state.currentConversationId;

    // 不中断SSE，只是转入后台
    if (previousConvId && state.activeReaders.has(previousConvId)) {
      console.log(`[Background] 会话 ${previousConvId} 转入后台继续生成`);

      // 将当前正在打字的消息的 pendingChunks 立即累积到 content
      const lastMessage = state.messages[state.messages.length - 1];
      if (lastMessage && lastMessage.role === 'assistant') {
        // 让打字机快速打完当前 chunk（跳过 15ms/char 延迟），避免阻塞切换
        // 不在这里累积 pendingChunks——切回时的 backlog-flush 会处理，避免与打字机当前 chunk 乱序
        lastMessage.fastForward = true;
        // 注册到后台消息，供切回该会话时 syncBackgroundMessages 恢复（否则回复丢失）
        state.backgroundMessages.set(previousConvId, [lastMessage]);
        console.log(`[Background] 会话 ${previousConvId} 转后台，内容长度: ${lastMessage.content.length}`);
      }
      // 不调用cancel，让SSE连接继续运行
    }

    const response = await apiFetch(`${API_BASE}/api/conversations/${convId}`);
    const data = await response.json();
    if (data.success) {
      state.currentConversationId = convId;
      state.messages = data.conversation.messages || [];
      localStorage.setItem('lastConversationId', convId);  // 记住：切走整页跳转后回来恢复

      // 如果新会话也在后台streaming，同步状态
      if (state.activeReaders.has(convId)) {
        syncBackgroundMessages(convId);
      }

      // 切回的会话仍在流式：先把后台积累的 pendingChunks 一次性累积到 content
      // （避免逐字 15ms 打字机慢慢追 backlog，那样切回来要等很久才追上），再 render
      if (state.streamingStatus.has(convId)) {
        const lastMsg = state.messages[state.messages.length - 1];
        if (lastMsg && lastMsg.role === 'assistant' && lastMsg.pendingChunks) {
          while (lastMsg.pendingChunks.length > 0) {
            lastMsg.content += lastMsg.pendingChunks.shift();
          }
        }
        // 切走时可能设了 fastForward 但没被消费（chunk 间），切回来重置，让新 chunk 正常逐字打
        if (lastMsg) lastMsg.fastForward = false;
      }

      renderMessages();
      // 恢复打字机，续打后续到达的新 chunk（之前切走时 break 了，isTypewriterRunning=false）
      if (state.streamingStatus.has(convId)) {
        startTypewriterEffect();
      }
      loadConversations();

      // 切回整页跳转场景兜底：最后一条是 user（assistant 还没落库，agent 后台在跑）
      // 且无前台 SSE（前台 SSE 在收就不重复拉）-> 轮询等 assistant 冒出来
      const lastMsg = state.messages[state.messages.length - 1];
      if (lastMsg && lastMsg.role === 'user' && !state.streamingStatus.has(convId)) {
        startPollingForAssistant(convId);
      }
    }
  } catch (error) {
    console.error('Failed to load conversation:', error);
  }
}

// 切回整页跳转场景的兜底：agent 在后台跑、assistant 还没落库时，轮询直到出现
function stopPollingForAssistant() {
  if (state.pollingTimer) {
    clearTimeout(state.pollingTimer);
    state.pollingTimer = null;
  }
  state.pollingConversationId = null;
}

function startPollingForAssistant(convId) {
  stopPollingForAssistant();  // 先清旧的，避免叠加
  state.pollingConversationId = convId;
  const deadline = Date.now() + 5 * 60 * 1000;  // 最多轮询 5 分钟（agent 通常 <1 分钟，留余量）

  const poll = async () => {
    // 切到别的会话/新建/删除 -> pollingConversationId 变，立即停
    if (state.pollingConversationId !== convId) return;
    // 用户在轮询期间发了新消息（前台 SSE 起来）-> 停轮询，交给前台 SSE
    if (state.streamingStatus.has(convId)) { stopPollingForAssistant(); return; }
    if (Date.now() > deadline) { stopPollingForAssistant(); return; }

    try {
      const resp = await apiFetch(`${API_BASE}/api/conversations/${convId}`);
      const data = await resp.json();
      if (data.success) {
        const msgs = data.conversation.messages || [];
        const last = msgs[msgs.length - 1];
        if (last && last.role === 'assistant') {
          // assistant 已落库，恢复并渲染
          state.messages = msgs;
          renderMessages();
          loadConversations();  // 侧边栏 updated_at 也刷新
          stopPollingForAssistant();
          console.log(`[Poll] 会话 ${convId} 的 assistant 已落库，停止轮询`);
          return;
        }
      }
    } catch (e) {
      console.warn(`[Poll] 拉取会话 ${convId} 失败，重试:`, e);
    }
    state.pollingTimer = setTimeout(poll, 2000);
  };
  state.pollingTimer = setTimeout(poll, 2000);
}


async function createConversation(firstMessage) {
  try {
    const response = await apiFetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: firstMessage })
    });
    const data = await response.json();
    if (data.success) {
      state.currentConversationId = data.conversation_id;
      localStorage.setItem('lastConversationId', data.conversation_id);  // 新会话也记，切走整页跳转后能恢复
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
    await apiFetch(`${API_BASE}/api/conversations/${convId}`, { method: 'DELETE' });
    if (state.currentConversationId === convId) {
      state.currentConversationId = null;
      state.messages = [];
      localStorage.removeItem('lastConversationId');  // 当前会话被删：清除记忆
      stopPollingForAssistant();  // 当前会话被删：停轮询
      renderMessages();
    }
    loadConversations();
  } catch (error) {
    console.error('Failed to delete conversation:', error);
  }
}


// 同步后台消息到当前会话
function syncBackgroundMessages(conversationId) {
  const bgMessages = state.backgroundMessages.get(conversationId);
  if (bgMessages && bgMessages.length > 0) {
    const lastMsg = bgMessages[bgMessages.length - 1];

    // 检查是否已在messages中
    const existingIndex = state.messages.findIndex(m =>
      m.role === 'assistant' && !m.streamDone
    );

    if (existingIndex >= 0) {
      state.messages[existingIndex] = lastMsg;
    } else {
      state.messages.push(lastMsg);
    }
  }
}

// 后台完成通知
function notifyBackgroundCompletion(conversationId) {
  console.log(`[Background] 会话 ${conversationId} 完成`);

  // 页面标题提示
  if (document.hidden) {
    document.title = '(1) 回答完成 - SAR-Sense';
  }

  // 侧边栏高亮
  setTimeout(() => {
    const item = document.querySelector(`.conversation-item[onclick*="${conversationId}"]`);
    if (item) {
      item.classList.add('completed-pulse');
      setTimeout(() => item.classList.remove('completed-pulse'), 3000);
    }
  }, 100);

  // 刷新侧边栏
  loadConversations();
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
  initCitationClickHandlers(chatMessages);
  initEvidenceDrawer();

  if (newConvBtn) {
    newConvBtn.addEventListener('click', async () => {
      const previousConvId = state.currentConversationId;

      // 如果之前的会话正在streaming，转入后台
      if (previousConvId && state.activeReaders.has(previousConvId)) {
        console.log(`[Background] 会话 ${previousConvId} 转入后台继续生成`);

        // 将当前正在打字的消息的 pendingChunks 立即累积到 content
        const lastMessage = state.messages[state.messages.length - 1];
        if (lastMessage && lastMessage.role === 'assistant') {
          // 等待打字机完成当前 chunk
          while (lastMessage.isTyping) {
            await new Promise(resolve => setTimeout(resolve, 20));
          }
          // 累积所有剩余的 pendingChunks
          if (lastMessage.pendingChunks) {
            while (lastMessage.pendingChunks.length > 0) {
              lastMessage.content += lastMessage.pendingChunks.shift();
            }
          }
          console.log(`[Background] 已累积当前消息内容，长度: ${lastMessage.content.length}`);
        }
        // 不中断，让它继续运行
      }

      // 清空状态
      state.currentConversationId = null;
      state.messages = [];
      state.attachedFile = null;
      localStorage.removeItem('lastConversationId');  // 新对话：清除上次会话记忆
      stopPollingForAssistant();  // 新对话：停掉切回轮询
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

  // 超时清理机制
  setInterval(() => {
    const now = Date.now();
    const TIMEOUT = 5 * 60 * 1000;  // 5分钟超时

    state.activeReaders.forEach((info, convId) => {
      if (now - info.startTime > TIMEOUT) {
        console.warn(`[Timeout] 会话 ${convId} 超时，强制中断`);
        info.abortController.abort();
      }
    });
  }, 30000);  // 每30秒检查一次

  // 页面可见性变化时恢复标题
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      document.title = 'SAR-Sense';
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

  // 检查当前会话是否正在流式输出
  const currentConvId = state.currentConversationId;
  if (state.streamingStatus.has(currentConvId)) return;

  // 并发限制：最多3个会话同时生成
  const MAX_CONCURRENT = 3;
  if (state.activeReaders.size >= MAX_CONCURRENT) {
    alert('同时最多3个会话生成，请等待完成后再试');
    return;
  }

  if (!state.currentConversationId) {
    const preview = message.substring(0, 20) || '附件分析';
    await createConversation(preview);
  }

  let attachmentContent = '';
  let attachmentName = '';
  let attachmentUploadId = '';
  if (state.attachedFile) {
    attachmentName = state.attachedFile.name;
    const result = await extractFileContent(state.attachedFile.file);
    attachmentContent = result.content;
    attachmentUploadId = result.uploadId;
    state.attachedFile = null;
    updateAttachmentIndicator();
  }

  let fullMessage = message;
  if (attachmentUploadId) {
    // 图片同时携带 OCR 文本和 upload_id：文字问题直接回答，明确要求检测时再调用 detect_ships。
    const ocrSection = attachmentContent
      ? `OCR识别结果：\n${attachmentContent}`
      : 'OCR未识别到可用文字。';
    const imageRequest = message || (attachmentContent
      ? '请分析图片中识别出的文字内容'
      : '图片中未识别到文字，请询问我是否需要进行SAR舰船检测');
    fullMessage = `[用户上传了图片「${attachmentName}」]\n\n${ocrSection}\n\n图片上传标识：${attachmentUploadId}。仅当用户明确要求舰船检测时，才使用 detect_ships 工具（传入 upload_id）。\n\n${imageRequest}`;
  } else if (attachmentContent) {
    // 文档：已提取文本，直接给 Agent 分析（不传服务端路径）
    fullMessage = `[用户上传了附件「${attachmentName}」，内容如下]\n\n${attachmentContent}\n\n${message || '请分析'}`;
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
    const response = await apiFetch(`${API_BASE}/api/extract-file`, {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    if (data.success) return { content: data.content || '', uploadId: data.upload_id || '' };
    return { content: '', uploadId: '' };
  } catch (error) {
    console.error('Failed to extract file:', error);
    return { content: '', uploadId: '' };
  }
}

const TOOL_LOADING_STATUS = Object.freeze({
  rag_summarize: '正在检索知识库...',
  web_search: '正在搜索网络...',
  detect_ships: '正在检测图像...',
  extract_file_content: '正在解析文件...',
  delegate_research: '正在进行深度研究...'
});

function getLoadingStatusForStep(step) {
  if (!step) return null;
  if (step.step_type === 'tool_call') {
    return TOOL_LOADING_STATUS[step.tool_name] || '正在调用工具...';
  }
  if (step.step_type === 'tool_result') {
    return '正在整理结果...';
  }
  if (step.step_type === 'final_answer') {
    return '正在生成回答...';
  }
  if (step.step_type === 'thinking') {
    return '正在思考...';
  }
  return null;
}

// 后台流式处理
async function processStreamInBackground(reader, conversationId) {
  const decoder = new TextDecoder();
  let buffer = '';

  // 创建后台消息对象
  let assistantMessage = {
    role: 'assistant',
    content: '',
    rag_results: [],
    thoughtSteps: [],
    streamDone: false,
    pendingChunks: [],
    isTyping: false,
    loadingStatus: '正在等待处理...'
  };

  // 缓存到后台消息中
  if (!state.backgroundMessages.has(conversationId)) {
    state.backgroundMessages.set(conversationId, []);
  }
  state.backgroundMessages.get(conversationId).push(assistantMessage);

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      // SSE 事件以空行分隔（兼容 \n\n 和 \r\n\r\n），按事件块解析
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop();

      for (const evtBlock of events) {
        let eventType = 'message';
        let dataStr = '';
        for (const line of evtBlock.split(/\r?\n/)) {
          const trimmed = line.trim();
          if (trimmed.startsWith('event:')) {
            eventType = trimmed.slice(6).trim();
          } else if (trimmed.startsWith('data:')) {
            dataStr = trimmed.slice(5).trim();
          }
        }
        try {
          const data = dataStr ? JSON.parse(dataStr) : {};

          if (eventType === 'status') {
            assistantMessage.loadingStatus = data.content || '正在思考...';
          } else if (eventType === 'chunk') {
            assistantMessage.loadingStatus = null;
            assistantMessage.content += data.content;

            // 更新进度
            const status = state.streamingStatus.get(conversationId);
            if (status) {
              status.progress = assistantMessage.content.length;
            }
          } else if (eventType === 'rag_result') {
            assistantMessage.loadingStatus = '正在生成回答...';
            if (!assistantMessage.rag_results) assistantMessage.rag_results = [];
            assistantMessage.rag_results.push(data.content);
          } else if (eventType === 'thought_step') {
            assistantMessage.thoughtSteps.push(data.step);
            const nextStatus = getLoadingStatusForStep(data.step);
            if (nextStatus) assistantMessage.loadingStatus = nextStatus;
          } else if (eventType === 'done') {
            assistantMessage.streamDone = true;
            assistantMessage.loadingStatus = null;

            // assistant 由后端 generate() finally 存库，前端不再重复存（避免切页面 streamDone 没触发导致丢回答）
            // 通知用户
            notifyBackgroundCompletion(conversationId);
          } else if (eventType === 'error') {
            assistantMessage.loadingStatus = null;
            assistantMessage.content += `\n\n[错误: ${data.message}]`;
          }
        } catch (e) {
          console.error('[Background] Parse error:', e);
        }
      }
    }
  } catch (error) {
    if (error.name !== 'AbortError') {
      console.error(`[Background] 会话 ${conversationId} 错误:`, error);
      assistantMessage.loadingStatus = null;
      assistantMessage.content += '\n\n[⚠️ 连接中断]';
    }
  } finally {
    state.backgroundMessages.delete(conversationId);
  }
}

async function sendMessageStreaming(message, displayMessage) {
  const conversationId = state.currentConversationId;
  const abortController = new AbortController();

  // 注册到活跃readers
  state.activeReaders.set(conversationId, {
    reader: null,
    abortController,
    startTime: Date.now()
  });

  state.streamingStatus.set(conversationId, {
    isStreaming: true,
    progress: 0
  });

  const assistantMessage = {
    role: 'assistant',
    content: '',
    rag_results: [],
    thoughtSteps: [],
    pendingChunks: [],
    isTyping: false,
    loadingStatus: '正在等待处理...'
  };
  state.messages.push(assistantMessage);
  renderMessages();

  const chatMessages = document.getElementById('chatMessages');
  chatMessages.scrollTop = chatMessages.scrollHeight;
  startTypewriterEffect();

  try {
    const messagesHistory = state.messages.slice(0, -2).map(m => ({ role: m.role, content: m.content }));
    const response = await apiFetch(`${API_BASE}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        display_message: displayMessage,
        messages: messagesHistory,
        conversation_id: conversationId
      }),
      signal: abortController.signal
    });

    const reader = response.body.getReader();
    state.activeReaders.get(conversationId).reader = reader;

    // 前台处理：实时更新UI（始终用前台模式，因为发送时就是当前会话）
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE 事件以空行分隔（兼容 \n\n 和 \r\n\r\n），按事件块解析
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop();

      for (const evtBlock of events) {
        let eventType = 'message';
        let dataStr = '';
        for (const line of evtBlock.split(/\r?\n/)) {
          const trimmed = line.trim();
          if (trimmed.startsWith('event:')) {
            eventType = trimmed.slice(6).trim();
          } else if (trimmed.startsWith('data:')) {
            dataStr = trimmed.slice(5).trim();
          }
        }
        try {
          const data = dataStr ? JSON.parse(dataStr) : {};
          if (eventType === 'status') {
            assistantMessage.loadingStatus = data.content || '正在思考...';
            updateLastMessage(false);
          } else if (eventType === 'chunk') {
            assistantMessage.pendingChunks.push(data.content);
          } else if (eventType === 'rag_result') {
            assistantMessage.loadingStatus = '正在生成回答...';
            if (!assistantMessage.rag_results) assistantMessage.rag_results = [];
            assistantMessage.rag_results.push(data.content);
            updateLastMessage(false);
          } else if (eventType === 'thought_step') {
            assistantMessage.thoughtSteps.push(data.step);
            const nextStatus = getLoadingStatusForStep(data.step);
            if (nextStatus) assistantMessage.loadingStatus = nextStatus;
            updateLastMessage(false);
            updateThoughtChainRealtime(assistantMessage.thoughtSteps);
          } else if (eventType === 'done') {
            assistantMessage.streamDone = true;
            if (assistantMessage.pendingChunks.length === 0 && !assistantMessage.isTyping) {
              assistantMessage.loadingStatus = null;
            }
          } else if (eventType === 'error') {
            assistantMessage.loadingStatus = null;
            assistantMessage.pendingChunks.push(`\n\n[错误: ${data.message}]`);
          }
        } catch (e) {
          console.error('[Stream] Parse error:', e, evtBlock);
        }
      }
    }

    // 等待打字机效果完成（仅当前会话）
    const isForeground = conversationId === state.currentConversationId;
    console.log(`[Stream End] 会话 ${conversationId}，当前会话: ${state.currentConversationId}，模式: ${isForeground ? '前台' : '后台'}`);

    if (isForeground) {
      // 前台模式：等待打字机完成
      while (assistantMessage.pendingChunks.length > 0 || assistantMessage.isTyping) {
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      updateLastMessage(true);
      console.log(`[前台模式] 会话 ${conversationId} 完成，内容长度: ${assistantMessage.content.length}`);
    } else {
      // 后台模式：直接累积所有内容，不等待打字机
      console.log(`[后台模式] 会话 ${conversationId} 切换到后台，pendingChunks: ${assistantMessage.pendingChunks.length}`);
      while (assistantMessage.pendingChunks.length > 0) {
        assistantMessage.content += assistantMessage.pendingChunks.shift();
      }
      console.log(`[后台模式] 会话 ${conversationId} 完成，内容长度: ${assistantMessage.content.length}`);
      // 通知用户后台会话已完成
      notifyBackgroundCompletion(conversationId);
    }

    console.log(`[会话完成] 会话 ${conversationId}，内容长度: ${assistantMessage.content.length}（assistant 由后端存库）`);
  } catch (error) {
    if (error.name === 'AbortError') {
      console.log('[Stream] 用户取消');
    } else {
      console.error('Streaming error:', error);
      assistantMessage.loadingStatus = null;
      assistantMessage.content += '\n\n[连接错误，请重试]';
      updateLastMessage(true);
    }
  } finally {
    state.activeReaders.delete(conversationId);
    state.streamingStatus.delete(conversationId);
  }
}
let isTypewriterRunning = false;

function startTypewriterEffect() {
  if (isTypewriterRunning) return;
  isTypewriterRunning = true;
  processTypewriterQueue();
}

async function processTypewriterQueue() {
  const currentConvId = state.currentConversationId;
  while (state.streamingStatus.has(currentConvId)) {
    // 安全检查：确保当前会话没有切换
    if (currentConvId !== state.currentConversationId) {
      console.log('[Typewriter] 会话已切换，停止打字机效果');
      break;
    }

    // 处理所有未完成的 assistant 消息
    let hasWork = false;
    for (let i = state.messages.length - 1; i >= 0; i--) {
      const msg = state.messages[i];
      if (msg.role === 'assistant' && msg.pendingChunks && msg.pendingChunks.length > 0) {
        const chunk = msg.pendingChunks.shift();
        if (chunk && msg.loadingStatus) {
          msg.loadingStatus = null;
        }
        msg.isTyping = true;
        for (let j = 0; j < chunk.length; j++) {
          msg.content += chunk[j];
          if (msg.fastForward) {
            // 会话切走，剩余字符一次性追加（跳过 15ms/char），避免阻塞切换 + 与后续 chunk 乱序
            msg.content += chunk.slice(j + 1);
            msg.fastForward = false;
            msg.isTyping = false;
            break;
          }
          if (i === state.messages.length - 1) {
            updateLastMessage(false);
          }
          await new Promise(resolve => setTimeout(resolve, 15));
        }
        msg.isTyping = false;
        hasWork = true;
        break; // 一次处理一个 chunk
      }
    }

    if (!hasWork) {
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
  isTypewriterRunning = false;
}

// 把 detect_ships 工具产生的检测图渲染成回答下方卡片。
// detect_image step 只带 upload_id（短标识），图字节由 /api/image/{upload_id} 按需拉取——
// 不走 SSE 管（避免 base64 撑爆流），与附件上传同走 upload_store 机制。
async function appendDetectImages(contentDiv, thoughtSteps) {
  const imgSteps = (thoughtSteps || []).filter(s => s && s.step_type === 'detect_image' && s.upload_id);
  for (const step of imgSteps) {
    // 防重复：updateLastMessage 会被多次调用，同一 upload_id 的卡片已存在就跳过
    if (contentDiv.querySelector(`.detect-result-card[data-upload-id="${step.upload_id}"]`)) continue;
    try {
      const resp = await apiFetch(`${API_BASE}/api/image/${step.upload_id}`);
      if (!resp.ok) continue;
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const card = document.createElement('div');
      card.className = 'detect-result-card';
      card.dataset.uploadId = step.upload_id;
      card.innerHTML = `
        <div class="detect-result-title">🔍 SAR 舰船检测结果图</div>
        <img src="${url}" alt="SAR舰船检测结果">
      `;
      contentDiv.appendChild(card);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (e) {
      console.warn('拉取检测结果图失败:', e);
    }
  }
}

function buildAssistantDisplayContent(message) {
  const ragResults = Array.isArray(message?.rag_results) ? message.rag_results.filter(Boolean) : [];
  const content = message?.content || '';
  if (!ragResults.length) return content;
  return `${ragResults.join('\n\n')}\n\n${content}`.trim();
}

function renderAssistantLoadingStatus(status) {
  if (!status) return '';
  return `
    <div class="message-status assistant-loading-status" role="status" aria-live="polite">
      <span class="spinner" aria-hidden="true"></span>
      <span>${escapeHtml(status)}</span>
    </div>
  `;
}

function renderAssistantDisplayHtml(message, isStreaming) {
  const ragResults = Array.isArray(message?.rag_results) ? message.rag_results.filter(Boolean) : [];
  const hasAnswer = Boolean((message?.content || '').trim());
  const showLoading = Boolean(isStreaming && message?.loadingStatus && !hasAnswer);
  let html = renderMarkdown(buildAssistantDisplayContent(message));
  html = renderWithCitations(
    html,
    isStreaming && !showLoading,
    ragResults.length > 0 && !hasAnswer
  );
  const loadingHtml = showLoading
    ? renderAssistantLoadingStatus(message.loadingStatus)
    : '';
  return `${loadingHtml}${html}`;
}

function updateLastMessage(isFinal = false) {
  const chatMessages = document.getElementById('chatMessages');
  const lastMessage = chatMessages.lastElementChild;
  if (!lastMessage || !lastMessage.classList.contains('assistant')) return;

  const contentDiv = lastMessage.querySelector('.message-content');
  const assistantMessage = state.messages[state.messages.length - 1];
  const existingThoughtChain = contentDiv.querySelector('.thought-chain');

  const currentConvId = state.currentConversationId;
  // 流式期也调 renderWithCitations：把 RAG answer 折叠进 🔎 检索材料，避免大段占屏把
  // 最终回答挤到底部。streamingCursor 由 renderWithCitations 插到 mainBody 末尾。
  const isStreaming = !isFinal && state.streamingStatus.has(currentConvId);
  const html = renderAssistantDisplayHtml(assistantMessage, isStreaming);

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
  if (isFinal) {
    appendDetectImages(contentDiv, assistantMessage.thoughtSteps);
  }
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateThoughtChainRealtime(thoughtSteps) {
  // detect_image 步骤不进思维链，由 appendDetectImages 渲染成回答下方卡片
  thoughtSteps = (thoughtSteps || []).filter(s => s && s.step_type !== 'detect_image' && s.agent_name !== 'sar-researcher');
  // 防御检查：确保当前会话正在流式输出
  const currentConvId = state.currentConversationId;
  if (!state.streamingStatus.has(currentConvId)) return;

  // 防御检查：确保有消息存在
  if (state.messages.length === 0) return;

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

  // 统计每个工具名的调用次数，为多次调用添加序号
  const toolCallCounts = {};
  const toolCallNumbers = new Map();  // tool_call_id -> 显示序号

  thoughtSteps.forEach(step => {
    if (step.step_type === 'tool_call' && step.tool_name && step.tool_call_id) {
      const name = step.tool_name;
      toolCallCounts[name] = (toolCallCounts[name] || 0) + 1;
      toolCallNumbers.set(step.tool_call_id, toolCallCounts[name]);
    }
  });

  let html = '';
  thoughtSteps.forEach(step => {
    const config = stepConfig[step.step_type] || stepConfig.thinking;
    let label = config.label;

    if (step.step_type === 'tool_call' && step.tool_name) {
      const count = toolCallCounts[step.tool_name] || 1;
      const num = toolCallNumbers.get(step.tool_call_id) || 1;
      const suffix = count > 1 ? ` #${num}` : '';
      label = `调用 ${step.tool_name}${suffix}`;
    }
    else if (step.step_type === 'tool_result' && step.tool_name) {
      const toolName = step.tool_name;
      const count = toolCallCounts[toolName] || 1;
      const num = toolCallNumbers.get(step.tool_call_id) || '';
      const suffix = count > 1 && num ? ` #${num}` : '';
      label = `观察结果 (${toolName}${suffix})`;
    }

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
    const currentConvId = state.currentConversationId;
    const isStreamingMsg = msg.role === 'assistant' && index === state.messages.length - 1 && state.streamingStatus.has(currentConvId);
    if (msg.role === 'assistant') {
      html = renderAssistantDisplayHtml(msg, isStreamingMsg);
    }
    contentDiv.innerHTML = html;

    if (msg.role === 'assistant' && msg.thought_steps && msg.thought_steps.length > 1) {
      const thoughtChainHtml = renderThoughtChain(msg.thought_steps);
      contentDiv.innerHTML += thoughtChainHtml;
    }
    if (msg.role === 'assistant') {
      appendDetectImages(contentDiv, msg.thought_steps);
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

// ==================== Citation Rendering ====================
const citationSourceLinePattern = /^\[(\d+)\]\s*([^|]+)\s*\|\s*chunk_id=(\S+)(?:\s*\|\s*page=(\S+))?\s*\|\s*score=(\S+)$/;

function hasStructuredCitationSource(html, match) {
  const sourceStart = match.index + match[0].length;
  const firstSourceLine = html.substring(sourceStart)
    .split(/<br\s*\/?>/i)
    .map(line => line.trim())
    .find(Boolean);
  const sourceMatch = firstSourceLine?.match(citationSourceLinePattern);
  return Boolean(sourceMatch?.[3] && sourceMatch?.[5]);
}

function citationDataAttributes(source) {
  return `data-parent-id="${encodeURIComponent(source.parentId)}" ` +
    `data-filename="${encodeURIComponent(source.filename)}" ` +
    `data-page="${encodeURIComponent(source.page || '')}" ` +
    `data-score="${encodeURIComponent(source.score || '')}"`;
}

function formatCitationScore(value) {
  const score = Number(value);
  if (Number.isFinite(score) && score >= 0 && score <= 1) {
    return `${(score * 100).toFixed(1)}%`;
  }
  return value && value !== '-' ? String(value) : '未知';
}

function decorateCitationBody(text, sources, displayIndexes = null) {
  let decorated = text || '';
  decorated = decorated.replace(/\[(\d+)\]/g, (raw, number) => {
    const index = parseInt(number);
    const source = sources.find(item => item.index === index);
    if (!source) return raw;
    const displayIndex = displayIndexes?.get(index) || index;
    return `<button type="button" class="citation-ref" ${citationDataAttributes(source)} ` +
      `aria-label="查看第 ${displayIndex} 条引用证据">[${displayIndex}]</button>`;
  });
  return decorated.replace(/《(.+?)》/g, '<span class="citation-filename">📄 $1</span>');
}

function renderCitationSourceItem(source, displayIndex) {
  const pageText = source.page ? `第 ${escapeHtml(source.page)} 页 · ` : '';
  return `<button type="button" class="citation-source-item" ${citationDataAttributes(source)}>
    <span class="citation-badge">[${displayIndex}]</span>
    <span class="citation-name">📄 ${source.filename}</span>
    <span class="citation-meta">${pageText}相关度 ${escapeHtml(formatCitationScore(source.score))}</span>
    <span class="citation-view">查看证据</span>
  </button>`;
}

function renderWithCitations(html, streamingCursor = false, forceFoldToolBody = false) {
  // 只处理 assistant 消息中有引用标记的文本
  // streamingCursor: 流式期传 true，把打字机光标插到 mainBody 末尾（content 打字机处）
  const cursor = streamingCursor ? '<span class="streaming-cursor"></span>' : '';
  if (!html || !html.includes('参考来源')) return html + cursor;

  try {
    // 容错中文冒号、英文冒号、全角冒号，允许前后空格
    const matches = [...html.matchAll(/参考来源\s*[：:︰]\s*/g)]
      .filter(match => hasStructuredCitationSource(html, match));
    if (matches.length === 0) return html + cursor;

    // 调试：打印原始 HTML 和匹配到的"参考来源"数量
    console.log('[renderWithCitations] 匹配到', matches.length, '个"参考来源"');
    console.log('[renderWithCitations] HTML 长度:', html.length);

    // 单次 RAG：保持原有逻辑
    if (matches.length === 1) {
      return renderSingleRagCitation(html, matches[0], cursor, forceFoldToolBody);
    }

    // 多次 RAG：新逻辑
    return renderMultipleRagCitations(html, matches, cursor, forceFoldToolBody);
  } catch (e) {
    console.error('[renderWithCitations] 解析失败，显示原始内容', e);
    return html + cursor; // 优雅降级：出错时返回原始 HTML
  }
}

function renderSingleRagCitation(html, match, cursor = '', forceFoldToolBody = false) {
  // 现有的单次 RAG 逻辑
  const splitIdx = match.index + match[0].length;
  let body = html.substring(0, match.index).trim();
  const sourceBlock = html.substring(splitIdx).trim();
  const sourceLines = sourceBlock.split(/<br\s*\/?>/i).map(l => l.trim()).filter(l => l);

  // 解析来源列表
  const sources = [];
  let consumedSourceLines = 0;
  for (const line of sourceLines) {
    const m = line.match(citationSourceLinePattern);
    if (m) {
      sources.push({
        index: parseInt(m[1]),
        filename: (m[2] || '').trim(),
        parentId: m[3] || '-',
        page: m[4] && m[4] !== '-' ? m[4] : null,
        score: m[5] || '-'
      });
      consumedSourceLines += 1;
    } else {
      break;
    }
  }
  if (sources.length === 0) return html;

  const displayIndexByOriginal = new Map(
    sources.map((source, idx) => [source.index, idx + 1])
  );

  const trailingBody = sourceLines.slice(consumedSourceLines).join('<br>').trim();

  const shouldFoldToolBody = Boolean(trailingBody) || forceFoldToolBody;
  const mainBody = decorateCitationBody(
    trailingBody || (shouldFoldToolBody ? '' : body),
    sources,
    displayIndexByOriginal
  ) + cursor;
  const toolBody = shouldFoldToolBody
    ? decorateCitationBody(body, sources, displayIndexByOriginal)
    : '';

  const sourceItems = sources.map(source => renderCitationSourceItem(
    source,
    displayIndexByOriginal.get(source.index) || source.index
  )).join('');

  const sourcePanel = `<details class="citation-sources">
    <summary>📚 参考来源（${sources.length}篇）</summary>
    <div class="citation-source-list">${sourceItems}</div>
  </details>`;

  const toolPanel = toolBody ? `<details class="citation-sources rag-tool-output">
    <summary>🔎 检索材料</summary>
    <div class="citation-source-list">${toolBody}</div>
  </details>` : '';

  return `<div class="message-with-citations">${mainBody}${toolPanel}${sourcePanel}</div>`;
}

function renderMultipleRagCitations(html, matches, cursor = '', forceFoldToolBody = false) {
  console.log('[renderMultipleRagCitations] 开始解析，参考来源数量:', matches.length);

  let sourceGroups = [];  // 按 RAG 调用分组的来源
  let pendingToolBody = html.substring(0, matches[0].index).trim();
  let finalBody = '';

  for (let i = 0; i < matches.length; i++) {
    const sourceStart = matches[i].index + matches[i][0].length;
    const nextSectionStart = (i + 1 < matches.length) ? matches[i + 1].index : html.length;
    const section = html.substring(sourceStart, nextSectionStart);

    console.log(`[renderMultipleRagCitations] 第${i+1}个"参考来源"区域长度:`, section.length);

    const lines = section.split(/<br\s*\/?>/i).map(l => l.trim()).filter(l => l);

    const sources = [];
    let j = 0;

    // 解析来源列表行
    while (j < lines.length) {
      const m = lines[j].match(citationSourceLinePattern);
      if (m) {
        sources.push({
          index: parseInt(m[1]),
          filename: m[2].trim(),
          parentId: m[3] || '-',
          page: m[4] && m[4] !== '-' ? m[4] : null,
          score: m[5] || '-'
        });
        j++;
      } else {
        break;  // 遇到非来源行，停止解析
      }
    }

    if (sources.length > 0) {
      sourceGroups.push({
        groupIndex: i + 1,
        sources,
        toolBody: pendingToolBody
      });
      console.log(`[renderMultipleRagCitations] 第${i+1}组解析到 ${sources.length} 个来源`);
    }

    const remainingText = lines.slice(j).join('<br>').trim();
    if (remainingText) {
      console.log(`[renderMultipleRagCitations] 第${i+1}组后续正文长度:`, remainingText.length);
    }

    if (i + 1 < matches.length) {
      pendingToolBody = remainingText;
    } else {
      finalBody = remainingText;
    }
  }

  // 如果没有解析到任何来源，回退到原始 HTML
  if (sourceGroups.length === 0) return html;

  const toolBodies = sourceGroups.map(g => g.toolBody).filter(text => text && text.trim());
  const shouldFoldTools = Boolean(finalBody) || forceFoldToolBody;
  const mainBodyRaw = shouldFoldTools ? finalBody : toolBodies.join('<br><br>');
  const bodyWithHighlight = decorateCitationBody(mainBodyRaw, []) + cursor;

  console.log('[renderMultipleRagCitations] 最终正文长度:', mainBodyRaw.length);

  // 生成分组来源面板
  const groupHtmls = sourceGroups.map(group => {
    const items = group.sources
      .map(source => renderCitationSourceItem(source, source.index))
      .join('');

    return `
      <div class="citation-group">
        <div class="citation-group-title">🔍 第${group.groupIndex}次检索</div>
        ${items}
      </div>
    `;
  }).join('');

  const totalSources = sourceGroups.reduce((sum, g) => sum + g.sources.length, 0);
  const sourcePanel = `<details class="citation-sources">
    <summary>📚 参考来源（${totalSources}篇，来自${sourceGroups.length}次检索）</summary>
    <div class="citation-source-list">${groupHtmls}</div>
  </details>`;

  const toolGroupHtmls = shouldFoldTools ? sourceGroups
    .filter(group => group.toolBody && group.toolBody.trim())
    .map(group => `
      <div class="citation-group">
        <div class="citation-group-title">🔍 第${group.groupIndex}次检索</div>
        <div class="rag-tool-body">${decorateCitationBody(group.toolBody, group.sources)}</div>
      </div>
    `).join('') : '';

  const toolPanel = toolGroupHtmls ? `<details class="citation-sources rag-tool-output">
    <summary>🔎 检索材料（${toolBodies.length}次）</summary>
    <div class="citation-source-list">${toolGroupHtmls}</div>
  </details>` : '';

  return `<div class="message-with-citations">${bodyWithHighlight}${toolPanel}${sourcePanel}</div>`;
}

function initCitationClickHandlers(container) {
  container.addEventListener('click', (event) => {
    const trigger = event.target.closest('.citation-ref, .citation-source-item');
    if (!trigger || !container.contains(trigger) || !trigger.dataset.parentId) return;
    event.preventDefault();
    openEvidenceDrawer({
      parentId: decodeURIComponent(trigger.dataset.parentId),
      filename: decodeURIComponent(trigger.dataset.filename || ''),
      page: decodeURIComponent(trigger.dataset.page || ''),
      score: decodeURIComponent(trigger.dataset.score || ''),
      trigger,
    });
  });
}

let evidenceRequestSerial = 0;
let evidenceLastTrigger = null;

function renderEvidenceContent(text) {
  return processTables(escapeHtml(text))
    .replace(/\n/g, '<br>')
    .replace(/(<\/table>)<br>/g, '$1');
}

function initEvidenceDrawer() {
  const drawer = document.getElementById('evidenceDrawer');
  document.getElementById('evidenceDrawerClose')?.addEventListener('click', closeEvidenceDrawer);
  document.getElementById('evidenceDrawerDownload')?.addEventListener('click', downloadEvidenceFile);
  drawer?.addEventListener('click', (event) => {
    const rect = drawer.getBoundingClientRect();
    const inside = event.clientX >= rect.left && event.clientX <= rect.right
      && event.clientY >= rect.top && event.clientY <= rect.bottom;
    if (!inside) drawer.close();
  });
  drawer?.addEventListener('close', () => {
    evidenceRequestSerial += 1;
    evidenceLastTrigger?.focus();
    evidenceLastTrigger = null;
  });
}

async function openEvidenceDrawer({ parentId, filename, page, score, trigger }) {
  const drawer = document.getElementById('evidenceDrawer');
  const body = document.getElementById('evidenceDrawerBody');
  const title = document.getElementById('evidenceDrawerTitle');
  const meta = document.getElementById('evidenceDrawerMeta');
  const status = document.getElementById('evidenceDrawerStatus');
  const content = document.getElementById('evidenceDrawerContent');
  const download = document.getElementById('evidenceDrawerDownload');
  if (!drawer || !parentId) return;

  const requestSerial = ++evidenceRequestSerial;
  evidenceLastTrigger = trigger || null;
  if (!drawer.open) drawer.showModal();
  body.setAttribute('aria-busy', 'true');
  title.textContent = filename || '证据详情';
  meta.textContent = [
    page ? `第 ${page} 页` : '',
    score ? `相关度 ${formatCitationScore(score)}` : '',
  ].filter(Boolean).join(' · ');
  status.textContent = '正在读取证据...';
  content.textContent = '';
  download.hidden = true;
  download.dataset.url = '';
  download.dataset.filename = '';

  try {
    const response = await apiFetch(
      `${API_BASE}/api/knowledge/evidence/${encodeURIComponent(parentId)}`
    );
    if (!response.ok) throw new Error('evidence unavailable');
    const data = await response.json();
    if (requestSerial !== evidenceRequestSerial) return;

    title.textContent = data.filename || filename || '证据详情';
    meta.textContent = [
      data.page !== null && data.page !== undefined ? `第 ${data.page} 页` : '',
      score ? `相关度 ${formatCitationScore(score)}` : '',
    ].filter(Boolean).join(' · ');
    status.textContent = '';
    content.innerHTML = renderEvidenceContent(data.content || '');
    download.hidden = !data.download_url;
    download.dataset.url = data.download_url || '';
    download.dataset.filename = data.filename || 'document';
  } catch (error) {
    if (requestSerial !== evidenceRequestSerial) return;
    status.textContent = '证据不存在或当前无权访问';
    content.textContent = '';
  } finally {
    if (requestSerial === evidenceRequestSerial) {
      body.setAttribute('aria-busy', 'false');
    }
  }
}

function closeEvidenceDrawer() {
  const drawer = document.getElementById('evidenceDrawer');
  if (drawer?.open) drawer.close();
}

async function downloadEvidenceFile(event) {
  const button = event.currentTarget;
  if (!button.dataset.url) return;
  const response = await apiFetch(button.dataset.url);
  if (!response.ok) return;
  const objectUrl = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = button.dataset.filename || 'document';
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}

function renderThoughtChain(steps) {
  // detect_image 步骤不进思维链，由 appendDetectImages 单独渲染成回答下方卡片
  steps = (steps || []).filter(s => s && s.step_type !== 'detect_image' && s.agent_name !== 'sar-researcher');
  if (steps.length === 0) return '';
  const stepConfig = {
    thinking: { icon: '💭', label: '思考', color: '#3b82f6' },
    tool_call: { icon: '🔧', label: '工具调用', color: '#22c55e' },
    tool_result: { icon: '👁️', label: '观察结果', color: '#f59e0b' },
    final_answer: { icon: '💡', label: '生成回答', color: '#06b6d4' }
  };

  // 统计每个工具名的调用次数，为多次调用添加序号
  const toolCallCounts = {};
  const toolCallNumbers = new Map();  // tool_call_id -> 显示序号

  steps.forEach(step => {
    if (step.step_type === 'tool_call' && step.tool_name && step.tool_call_id) {
      const name = step.tool_name;
      toolCallCounts[name] = (toolCallCounts[name] || 0) + 1;
      toolCallNumbers.set(step.tool_call_id, toolCallCounts[name]);
    }
  });

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

    if (step.step_type === 'tool_call' && step.tool_name) {
      const count = toolCallCounts[step.tool_name] || 1;
      const num = toolCallNumbers.get(step.tool_call_id) || 1;
      const suffix = count > 1 ? ` #${num}` : '';
      label = `调用 ${step.tool_name}${suffix}`;
    }
    else if (step.step_type === 'tool_result' && step.tool_name) {
      const toolName = step.tool_name;
      const count = toolCallCounts[toolName] || 1;
      const num = toolCallNumbers.get(step.tool_call_id) || '';
      const suffix = count > 1 && num ? ` #${num}` : '';
      label = `观察结果 (${toolName}${suffix})`;
    }

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
