// SAR-Sense Metrics Page JavaScript

const API_BASE = '';

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initScrollReveal();
  initParticles();
  initMobileMenu();
  initMetrics();
  highlightNav('metrics');
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

// ==================== Metrics ====================
function initMetrics() {
  initTabs();
  loadMetrics();

  const resetBtn = document.getElementById('resetBtn');
  if (resetBtn) {
    resetBtn.addEventListener('click', resetMetrics);
  }

  // Auto-refresh every 10s
  setInterval(loadMetrics, 10000);
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

async function loadMetrics() {
  try {
    const response = await fetch(`${API_BASE}/api/metrics`);
    const data = await response.json();

    if (data.success) {
      renderMetricsCards(data.metrics);
      renderToolDistribution(data.metrics.tool_stats);
      renderTimeline(data.metrics.recent_records);
      renderToolDetails(data.metrics.tool_stats, data.metrics.recent_records);
    }
  } catch (error) {
    console.error('Failed to load metrics:', error);
  }
}

async function resetMetrics() {
  try {
    await fetch(`${API_BASE}/api/metrics/reset`, { method: 'POST' });
    loadMetrics();
  } catch (error) {
    console.error('Failed to reset metrics:', error);
  }
}

function renderMetricsCards(metrics) {
  document.getElementById('metricRounds').textContent = metrics.conversation_rounds;
  document.getElementById('metricToolCalls').textContent = metrics.total_tool_calls;
  document.getElementById('metricSuccessRate').textContent = metrics.overall_success_rate + '%';
  document.getElementById('metricAvgCalls').textContent = metrics.avg_tool_calls_per_round;
}

function renderToolDistribution(toolStats) {
  const canvas = document.getElementById('toolChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const container = canvas.parentElement;
  canvas.width = container.offsetWidth;
  canvas.height = container.offsetHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!toolStats || toolStats.length === 0) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.font = '14px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('暂无工具调用记录', canvas.width / 2, canvas.height / 2);
    return;
  }

  const padding = { top: 40, right: 30, bottom: 60, left: 50 };
  const chartWidth = canvas.width - padding.left - padding.right;
  const chartHeight = canvas.height - padding.top - padding.bottom;

  const maxVal = Math.max(...toolStats.map(s => s.total), 1);
  const barWidth = Math.min(40, (chartWidth / toolStats.length) * 0.6);
  const gap = (chartWidth - barWidth * toolStats.length) / (toolStats.length + 1);

  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(canvas.width - padding.right, y);
    ctx.stroke();
  }

  // Y-axis labels
  ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.font = '11px -apple-system, sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + (chartHeight / 5) * i;
    const val = Math.round(maxVal * (1 - i / 5));
    ctx.fillText(val, padding.left - 8, y + 4);
  }

  // Bars
  toolStats.forEach((stat, i) => {
    const x = padding.left + gap + i * (barWidth + gap);
    const successHeight = (stat.success / maxVal) * chartHeight;
    const failHeight = (stat.fail / maxVal) * chartHeight;

    // Success bar (green)
    const successGrad = ctx.createLinearGradient(0, padding.top + chartHeight - successHeight, 0, padding.top + chartHeight);
    successGrad.addColorStop(0, '#22c55e');
    successGrad.addColorStop(1, '#16a34a');
    ctx.fillStyle = successGrad;
    ctx.fillRect(x, padding.top + chartHeight - successHeight, barWidth, successHeight);

    // Fail bar (red, stacked on top)
    if (stat.fail > 0) {
      const failGrad = ctx.createLinearGradient(0, padding.top + chartHeight - successHeight - failHeight, 0, padding.top + chartHeight - successHeight);
      failGrad.addColorStop(0, '#ef4444');
      failGrad.addColorStop(1, '#dc2626');
      ctx.fillStyle = failGrad;
      ctx.fillRect(x, padding.top + chartHeight - successHeight - failHeight, barWidth, failHeight);
    }

    // X-axis label
    ctx.save();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '10px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.translate(x + barWidth / 2, canvas.height - 10);
    ctx.rotate(-Math.PI / 6);
    const displayName = stat.tool_name.length > 10 ? stat.tool_name.substring(0, 10) + '...' : stat.tool_name;
    ctx.fillText(displayName, 0, 0);
    ctx.restore();
  });

  // Legend
  ctx.fillStyle = '#22c55e';
  ctx.beginPath();
  ctx.arc(canvas.width - padding.right - 80, 15, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.font = '11px -apple-system, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('成功', canvas.width - padding.right - 72, 19);

  ctx.fillStyle = '#ef4444';
  ctx.beginPath();
  ctx.arc(canvas.width - padding.right - 30, 15, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.fillText('失败', canvas.width - padding.right - 22, 19);
}

function renderTimeline(records) {
  const el = document.getElementById('timelineList');
  if (!el) return;

  if (!records || records.length === 0) {
    el.innerHTML = '<div class="empty-state"><div class="empty-icon">📭</div><p>暂无调用记录</p></div>';
    return;
  }

  el.innerHTML = records.map(r => {
    const statusColor = r.success ? '#22c55e' : '#ef4444';
    const statusIcon = r.success ? '✅' : '❌';
    return `
      <div class="timeline-item">
        <div class="timeline-left">
          <span class="timeline-status" style="color: ${statusColor}">${statusIcon}</span>
          <span class="timeline-tool">${escapeHtml(r.tool_name)}</span>
        </div>
        <div class="timeline-right">
          <span class="timeline-duration">⏱ ${r.duration_ms}ms</span>
          <span class="timeline-time">${r.timestamp}</span>
        </div>
      </div>
    `;
  }).join('');
}

function renderToolDetails(toolStats, records) {
  const tableEl = document.getElementById('toolDetailsTable');
  if (!tableEl) return;

  if (!toolStats || toolStats.length === 0) {
    tableEl.innerHTML = '<div class="empty-state"><div class="empty-icon">📭</div><p>暂无统计数据</p></div>';
    return;
  }

  let html = `
    <table class="details-table">
      <thead>
        <tr>
          <th>工具名称</th>
          <th>调用次数</th>
          <th>成功</th>
          <th>失败</th>
          <th>成功率</th>
          <th>平均耗时</th>
        </tr>
      </thead>
      <tbody>
  `;

  toolStats.forEach(s => {
    html += `
      <tr>
        <td>${escapeHtml(s.tool_name)}</td>
        <td>${s.total}</td>
        <td class="text-success">${s.success}</td>
        <td class="text-error">${s.fail}</td>
        <td>${s.success_rate}%</td>
        <td>${s.avg_duration_ms}ms</td>
      </tr>
    `;
  });

  html += '</tbody></table>';
  tableEl.innerHTML = html;

  // Loop detection
  const loopEl = document.getElementById('loopDetection');
  if (!loopEl || !records || records.length < 3) {
    if (loopEl) loopEl.innerHTML = '';
    return;
  }

  const warnings = [];
  for (let i = 0; i < records.length - 2; i++) {
    if (records[i].tool_name === records[i + 1].tool_name && records[i + 1].tool_name === records[i + 2].tool_name) {
      warnings.push({
        tool: records[i].tool_name,
        startTime: records[i].timestamp,
        endTime: records[i + 2].timestamp
      });
    }
  }

  if (warnings.length > 0) {
    loopEl.innerHTML = `
      <div class="loop-warning">
        <div class="loop-warning-header">
          <span>⚠️ 循环调用检测</span>
          <span class="loop-count">${warnings.length} 次疑似循环</span>
        </div>
        <div class="loop-list">
          ${warnings.map(w => `
            <div class="loop-item">
              <span class="loop-tool">${escapeHtml(w.tool)}</span>
              <span class="loop-time">${w.startTime} ~ ${w.endTime}</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  } else {
    loopEl.innerHTML = '<div class="loop-ok">✅ 未检测到循环调用，所有工具调用正常。</div>';
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
