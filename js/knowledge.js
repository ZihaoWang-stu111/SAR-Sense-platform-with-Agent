// SAR-Sense Knowledge Base Page JavaScript

const API_BASE = '';

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initScrollReveal();
  initParticles();
  initMobileMenu();
  initKnowledge();
  highlightNav('knowledge');
});

function highlightNav(page) {
  document.querySelectorAll('.navbar-nav a').forEach(a => {
    if (a.getAttribute('href') === `${page}.html`) {
      a.classList.add('active');
    }
  });
}

function initNavbar() {
  const navbar = document.querySelector('.navbar');
  window.addEventListener('scroll', () => {
    if (window.pageYOffset > 50) navbar.classList.add('scrolled');
    else navbar.classList.remove('scrolled');
  });
}

function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
        entry.target.querySelectorAll('.stagger').forEach((child, i) => {
          child.style.animationDelay = `${i * 0.1}s`;
          child.classList.add('animate-slide-up');
        });
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

function initParticles() {
  const c = document.querySelector('.particles');
  if (!c) return;
  for (let i = 0; i < 50; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.left = `${Math.random() * 100}%`;
    p.style.top = `${Math.random() * 100}%`;
    const s = Math.random() * 3 + 1;
    p.style.width = `${s}px`;
    p.style.height = `${s}px`;
    p.style.animation = `float ${Math.random() * 3 + 2}s ease-in-out infinite`;
    p.style.animationDelay = `${Math.random() * 2}s`;
    p.style.opacity = Math.random() * 0.5 + 0.1;
    c.appendChild(p);
  }
}

function initMobileMenu() {
  const btn = document.querySelector('.mobile-menu-btn');
  const nav = document.querySelector('.navbar-nav');
  if (!btn || !nav) return;
  btn.addEventListener('click', () => {
    nav.classList.toggle('active');
    btn.textContent = nav.classList.contains('active') ? '×' : '☰';
  });
  nav.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      nav.classList.remove('active');
      btn.textContent = '☰';
    });
  });
}

function initKnowledge() {
  initTabs();
  initUpload();
  loadFileList();
}

function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });
}

function initUpload() {
  const uploadArea = document.getElementById('kbUploadArea');
  const fileInput = document.getElementById('kbFileInput');
  const uploadBtn = document.getElementById('kbUploadBtn');
  const ingestBtn = document.getElementById('ingestBtn');

  if (!uploadArea || !fileInput || !uploadBtn || !ingestBtn) return;

  let selectedFiles = [];

  uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  uploadArea.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      selectedFiles = Array.from(e.target.files);
      showSelectedFiles(selectedFiles);
    }
  });

  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });

  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
  });

  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      selectedFiles = Array.from(e.dataTransfer.files);
      showSelectedFiles(selectedFiles);
    }
  });

  ingestBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
      showStatus('请先选择文件', 'warning');
      return;
    }
    await uploadAndIngest(selectedFiles);
    selectedFiles = [];
  });

  function showSelectedFiles(files) {
    const names = files.map(f => f.name).join('、');
    document.getElementById('selectedFiles').textContent = names;
    document.getElementById('selectedFilesArea').style.display = 'flex';
    ingestBtn.disabled = false;
  }
}

async function uploadAndIngest(files) {
  const ingestBtn = document.getElementById('ingestBtn');
  const btnText = ingestBtn.querySelector('.btn-text');
  const btnSpinner = ingestBtn.querySelector('.btn-spinner');

  ingestBtn.disabled = true;
  btnText.textContent = '入库中...';
  btnSpinner.style.display = 'inline-block';

  try {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));

    showStatus('正在上传并写入向量库，请稍候...', 'info');

    const response = await apiFetch(`${API_BASE}/api/knowledge/upload`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (response.ok && data.success) {
      showStatus(data.message || '入库成功', 'success');
      document.getElementById('selectedFilesArea').style.display = 'none';
      document.getElementById('kbFileInput').value = '';
      await loadFileList();
    } else {
      showStatus('入库失败: ' + (data.detail || data.error || '未知错误'), 'error');
    }
  } catch (error) {
    showStatus('上传失败: ' + error.message, 'error');
  } finally {
    ingestBtn.disabled = false;
    btnText.textContent = '一键入库';
    btnSpinner.style.display = 'none';
  }
}

async function loadFileList() {
  showListStatus('正在读取 manifest...', 'info');
  try {
    const response = await apiFetch(`${API_BASE}/api/knowledge/files`);
    const data = await response.json();

    if (response.ok && data.success) {
      renderFileList(data.files || []);
      updateStats(data);
      showListStatus('', 'info', false);
    } else {
      showListStatus('文件列表读取失败: ' + (data.detail || data.error || '未知错误'), 'error');
    }
  } catch (error) {
    console.error('Failed to load file list:', error);
    showListStatus('文件列表读取失败: ' + error.message, 'error');
  }
}

function renderFileList(files) {
  const listEl = document.getElementById('fileList');
  if (!listEl) return;

  if (files.length === 0) {
    listEl.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📥</div>
        <p>知识库暂无已入库文档</p>
        <p class="empty-hint">切换到「上传文件」标签页添加 TXT 或 PDF</p>
      </div>
    `;
    return;
  }

  listEl.innerHTML = files.map(f => {
    const name = f.name || '未命名文档';
    const icon = (f.file_type || '').toLowerCase() === 'pdf' || name.toLowerCase().endsWith('.pdf') ? '📄' : '📝';
    const docId = f.doc_id || '';
    const hash = f.file_hash || '';
    return `
      <div class="file-item kb-file-card">
        <div class="file-main">
          <div class="file-info">
            <span class="file-icon">${icon}</span>
            <div class="file-title-block">
              <span class="file-name" title="${escapeAttr(name)}">${escapeHtml(name)}</span>
              <span class="file-subtitle">${escapeHtml(f.file_type || 'unknown')} · ${escapeHtml(f.chunk_method || 'unknown')} chunking</span>
            </div>
          </div>
          <span class="status-pill status-${escapeAttr(f.status || 'active')}">${escapeHtml(statusText(f.status))}</span>
        </div>
        <div class="file-meta-grid">
          <div class="file-meta-item">
            <span class="meta-label">doc_id</span>
            <span class="meta-value">${escapeHtml(docId)}</span>
          </div>
          <div class="file-meta-item">
            <span class="meta-label">chunks</span>
            <span class="meta-value">${Number(f.chunk_count || 0)}</span>
          </div>
          <div class="file-meta-item">
            <span class="meta-label">hash</span>
            <span class="meta-value">${escapeHtml(shortHash(hash))}</span>
          </div>
          <div class="file-meta-item">
            <span class="meta-label">indexed</span>
            <span class="meta-value">${escapeHtml(formatDate(f.ingested_at))}</span>
          </div>
        </div>
        <div class="file-actions">
          ${(typeof isAdmin === 'function' && isAdmin()) ? `
          <button class="btn btn-secondary file-delete" data-doc-id="${escapeAttr(docId)}" data-name="${escapeAttr(name)}" title="删除文档及向量">
            删除
          </button>` : ''}
        </div>
      </div>
    `;
  }).join('');

  bindDeleteButtons();
}

function bindDeleteButtons() {
  document.querySelectorAll('.file-delete').forEach(btn => {
    btn.addEventListener('click', async () => {
      const docId = btn.dataset.docId;
      const name = btn.dataset.name;
      await deleteKnowledgeFile(docId, name, btn);
    });
  });
}

async function deleteKnowledgeFile(docId, name, button) {
  if (!docId) {
    showListStatus('缺少 doc_id，无法删除该文档', 'error');
    return;
  }

  const confirmed = window.confirm(`确认删除「${name}」？这会清理 Chroma 向量，并删除 data/ 下的原始文件。`);
  if (!confirmed) return;

  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = '删除中...';
  showListStatus(`正在删除 ${name}...`, 'info');

  try {
    const response = await apiFetch(`${API_BASE}/api/knowledge/files/${encodeURIComponent(docId)}?delete_file=true`, {
      method: 'DELETE'
    });
    const data = await response.json();

    if (response.ok && data.success) {
      showListStatus(data.message || '删除成功', 'success');
      await loadFileList();
    } else {
      showListStatus('删除失败: ' + (data.detail || data.error || '未知错误'), 'error');
    }
  } catch (error) {
    showListStatus('删除失败: ' + error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

function updateStats(data) {
  const files = data.files || data || [];
  const countEl = document.getElementById('fileCount');
  const chunkEl = document.getElementById('chunkCount');

  if (countEl) countEl.textContent = data.total_files ?? files.length;
  if (chunkEl) {
    const totalChunks = data.total_chunks ?? files.reduce((sum, f) => sum + Number(f.chunk_count || 0), 0);
    chunkEl.textContent = totalChunks;
  }
}

function statusText(status) {
  if (status === 'active') return 'active';
  if (status === 'deleted') return 'deleted';
  return status || 'active';
}

function shortHash(hash) {
  if (!hash) return '-';
  return hash.length > 16 ? `${hash.slice(0, 8)}...${hash.slice(-6)}` : hash;
}

function formatDate(value) {
  if (!value) return '-';
  return value.replace('T', ' ');
}

function showStatus(message, type) {
  const el = document.getElementById('uploadStatus');
  if (!el) return;
  el.textContent = message;
  el.className = `status-message status-${type}`;
  el.style.display = 'block';
  if (type === 'success') {
    setTimeout(() => { el.style.display = 'none'; }, 5000);
  }
}

function showListStatus(message, type, visible = true) {
  const el = document.getElementById('fileListStatus');
  if (!el) return;
  el.textContent = message;
  el.className = `status-message status-${type}`;
  el.style.display = visible && message ? 'block' : 'none';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = String(text);
  return div.innerHTML;
}

function escapeAttr(text) {
  return escapeHtml(text).replace(/"/g, '&quot;');
}
