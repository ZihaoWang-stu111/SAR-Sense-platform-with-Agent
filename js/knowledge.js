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

// ==================== Navbar ====================
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  window.addEventListener('scroll', () => {
    if (window.pageYOffset > 50) navbar.classList.add('scrolled');
    else navbar.classList.remove('scrolled');
  });
}

// ==================== Scroll Reveal ====================
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

// ==================== Particles ====================
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

// ==================== Mobile Menu ====================
function initMobileMenu() {
  const btn = document.querySelector('.mobile-menu-btn');
  const nav = document.querySelector('.navbar-nav');
  if (!btn || !nav) return;
  btn.addEventListener('click', () => {
    nav.classList.toggle('active');
    btn.textContent = nav.classList.contains('active') ? '✕' : '☰';
  });
  nav.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => { nav.classList.remove('active'); btn.textContent = '☰'; });
  });
}

// ==================== Knowledge Base ====================
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
  const uploadStatus = document.getElementById('uploadStatus');

  if (!uploadArea) return;

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

    showStatus('正在上传并入库，请稍候...', 'info');

    const response = await fetch(`${API_BASE}/api/knowledge/upload`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (data.success) {
      showStatus(data.message || '入库成功！', 'success');
      document.getElementById('selectedFilesArea').style.display = 'none';
      document.getElementById('kbFileInput').value = '';
      loadFileList();
    } else {
      showStatus('入库失败: ' + (data.error || '未知错误'), 'error');
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
  try {
    const response = await fetch(`${API_BASE}/api/knowledge/files`);
    const data = await response.json();

    if (data.success) {
      renderFileList(data.files);
      updateStats(data.files);
    }
  } catch (error) {
    console.error('Failed to load file list:', error);
  }
}

function renderFileList(files) {
  const listEl = document.getElementById('fileList');
  if (!listEl) return;

  if (files.length === 0) {
    listEl.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">📭</div>
        <p>知识库暂无文件</p>
        <p class="empty-hint">切换到「上传文件」标签页添加</p>
      </div>
    `;
    return;
  }

  listEl.innerHTML = files.map(f => {
    const icon = f.name.endsWith('.pdf') ? '📕' : '📄';
    const size = formatFileSize(f.size);
    return `
      <div class="file-item">
        <div class="file-info">
          <span class="file-icon">${icon}</span>
          <span class="file-name">${escapeHtml(f.name)}</span>
        </div>
        <span class="file-size">${size}</span>
      </div>
    `;
  }).join('');
}

function updateStats(files) {
  const countEl = document.getElementById('fileCount');
  if (countEl) countEl.textContent = files.length;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
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

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
