// SAR-Sense Main JavaScript — Premium interactions
// Apple-style: spring tilt + magnetic CTA + smooth reveal + counter

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initScrollReveal();
  initParticles();
  initChart();
  initCounters();
  initMobileMenu();
  initTrustBars();
  initHeroParallax();        // 新增：hero mockup 跟随鼠标 3D 倾斜
  initMagneticCTAs();         // 新增:核心 CTA 按钮磁吸
  initSmoothScroll();         // 新增：锚链接平滑滚动
});

// ==================== Navbar ====================
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  if (!navbar) return;

  const onScroll = () => {
    if (window.pageYOffset > 16) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  };
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });
}

// ==================== Scroll Reveal ====================
function initScrollReveal() {
  const revealElements = document.querySelectorAll('.reveal');
  if (!revealElements.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.12,
    rootMargin: '0px 0px -60px 0px'
  });

  revealElements.forEach(el => observer.observe(el));
}

// ==================== Particles Background ====================
function initParticles() {
  const particlesContainer = document.querySelector('.particles');
  if (!particlesContainer) return;

  const isMobile = window.matchMedia('(max-width: 768px)').matches;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reducedMotion) return;

  const particleCount = isMobile ? 24 : 48;
  const frag = document.createDocumentFragment();
  for (let i = 0; i < particleCount; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    particle.style.left = `${Math.random() * 100}%`;
    particle.style.top = `${Math.random() * 100}%`;
    const size = Math.random() * 2.5 + 0.6;
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    particle.style.animation = `float ${Math.random() * 4 + 5}s ease-in-out infinite`;
    particle.style.animationDelay = `${Math.random() * 3}s`;
    particle.style.opacity = (Math.random() * 0.45 + 0.10).toFixed(2);
    frag.appendChild(particle);
  }
  particlesContainer.appendChild(frag);
}

// ==================== Chart Initialization ====================
function initChart() {
  const canvas = document.getElementById('detectionChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const container = canvas.parentElement;
  const dataPoints = 24;
  const detectionData = generateSampleData(dataPoints, 10, 50);
  const accuracyData = generateSampleData(dataPoints, 85, 99);

  // Apple-system color tokens (sync with CSS)
  const COLOR_DETECT  = '#0a84ff';
  const COLOR_DETECT_FILL = 'rgba(10, 132, 255, 0.10)';
  const COLOR_ACCURACY = '#bf5af2';
  const COLOR_ACCURACY_FILL = 'rgba(191, 90, 242, 0.08)';

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = container.offsetWidth;
    const h = container.offsetHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);
    drawChart(w, h);
  }

  function drawChart(width, height) {
    const padding = { top: 40, right: 60, bottom: 40, left: 50 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    drawLine(detectionData, 0, 60, padding, chartWidth, chartHeight, COLOR_DETECT, COLOR_DETECT_FILL);
    drawLine(accuracyData, 80, 100, padding, chartWidth, chartHeight, COLOR_ACCURACY, COLOR_ACCURACY_FILL);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.40)';
    ctx.font = '11px Inter, -apple-system, sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i < dataPoints; i += 4) {
      const x = padding.left + (chartWidth / (dataPoints - 1)) * i;
      ctx.fillText(`${String(i).padStart(2, '0')}:00`, x, height - 10);
    }
    drawLegend(width - padding.right, 18);
  }

  function drawLine(data, minVal, maxVal, padding, chartWidth, chartHeight, strokeColor, fillColor) {
    const points = data.map((val, i) => ({
      x: padding.left + (chartWidth / (dataPoints - 1)) * i,
      y: padding.top + chartHeight - ((val - minVal) / (maxVal - minVal)) * chartHeight
    }));

    // Filled area (smooth)
    ctx.beginPath();
    ctx.moveTo(points[0].x, padding.top + chartHeight);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(points[points.length - 1].x, padding.top + chartHeight);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();

    // Smoothed line via quadratic curve through midpoints
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const xc = (points[i - 1].x + points[i].x) / 2;
      const yc = (points[i - 1].y + points[i].y) / 2;
      ctx.quadraticCurveTo(points[i - 1].x, points[i - 1].y, xc, yc);
    }
    ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.shadowColor = strokeColor;
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  function drawLegend(x, y) {
    ctx.fillStyle = COLOR_DETECT;
    ctx.beginPath();
    ctx.arc(x - 120, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.72)';
    ctx.font = '500 12px Inter, -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('检测数量', x - 112, y + 4);

    ctx.fillStyle = COLOR_ACCURACY;
    ctx.beginPath();
    ctx.arc(x - 30, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.72)';
    ctx.fillText('准确率', x - 22, y + 4);
  }

  function generateSampleData(count, min, max) {
    return Array.from({ length: count }, () =>
      Math.floor(Math.random() * (max - min + 1)) + min
    );
  }

  resizeCanvas();
  window.addEventListener('resize', debounce(resizeCanvas, 200));
}

// ==================== Counter Animation ====================
function initCounters() {
  const counters = document.querySelectorAll('.trust-value');
  if (!counters.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const counter = entry.target;
        const target = counter.getAttribute('data-target');
        const suffix = counter.getAttribute('data-suffix') || '';
        const prefix = counter.getAttribute('data-prefix') || '';
        animateCounter(counter, target, prefix, suffix);
        observer.unobserve(counter);
      }
    });
  }, { threshold: 0.45 });

  counters.forEach(counter => observer.observe(counter));
}

function animateCounter(element, target, prefix, suffix) {
  const duration = 1800;
  const start = 0;
  const end = parseFloat(target);
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // ease-out-expo — Apple smooth landing
    const eased = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
    const current = start + (end - start) * eased;

    if (String(target).includes('.')) {
      element.textContent = prefix + current.toFixed(1) + suffix;
    } else {
      element.textContent = prefix + Math.floor(current).toLocaleString() + suffix;
    }
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ==================== Trust progress bars ====================
function initTrustBars() {
  const bars = document.querySelectorAll('.trust-bar-fill');
  if (!bars.length) return;
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const bar = entry.target;
        const w = bar.getAttribute('data-width') || '0%';
        bar.style.width = '0%';
        requestAnimationFrame(() => {
          // 给一帧延迟让初始 0% 渲染，触发 transition
          setTimeout(() => { bar.style.width = w; }, 60);
        });
        observer.unobserve(bar);
      }
    });
  }, { threshold: 0.45 });
  bars.forEach(bar => observer.observe(bar));
}

// ==================== Hero 3D Parallax — 跟随鼠标的微倾斜 ====================
function initHeroParallax() {
  const mockup = document.querySelector('.hero-mockup');
  const visual = document.querySelector('.hero-visual');
  if (!mockup || !visual) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  if (window.matchMedia('(max-width: 1024px)').matches) return;

  const MAX_TILT = 6;     // 度
  let rafId = null;

  function onMove(e) {
    const rect = visual.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = (e.clientX - cx) / (rect.width / 2);   // -1 ~ 1
    const dy = (e.clientY - cy) / (rect.height / 2);
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(() => {
      const rotY = (-8) + dx * MAX_TILT;        // 基础 -8° (CSS 默认) + 偏移
      const rotX = 4 + (-dy) * MAX_TILT;        // 基础 4° + 偏移
      mockup.style.transform = `rotateY(${rotY}deg) rotateX(${rotX}deg)`;
    });
  }

  function onLeave() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(() => {
      mockup.style.transform = '';   // 回到 CSS 默认值（带 transition 缓动）
    });
  }

  // CSS 加上 transition 让 mockup 跟随更柔和
  mockup.style.transition = 'transform 0.4s cubic-bezier(0.32, 0.72, 0, 1)';

  window.addEventListener('mousemove', onMove, { passive: true });
  document.addEventListener('mouseleave', onLeave);
}

// ==================== Magnetic CTAs — 主按钮在悬停半径内吸附鼠标 ====================
function initMagneticCTAs() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  if (window.matchMedia('(max-width: 768px)').matches) return;

  const magnets = document.querySelectorAll('.hero-actions .btn-primary, .cta-actions .btn-primary');
  const STRENGTH = 0.28;     // 0~1，越大跟随越强

  magnets.forEach(btn => {
    btn.style.transition = 'transform 0.4s cubic-bezier(0.32, 0.72, 0, 1)';

    btn.addEventListener('mousemove', (e) => {
      const rect = btn.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;
      btn.style.transform = `translate(${x * STRENGTH}px, ${y * STRENGTH}px) scale(1.02)`;
    });

    btn.addEventListener('mouseleave', () => {
      btn.style.transform = '';
    });
  });
}

// ==================== Smooth in-page scroll ====================
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', (e) => {
      const id = link.getAttribute('href');
      if (id.length <= 1) return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      const navOffset = 80;
      const top = target.getBoundingClientRect().top + window.pageYOffset - navOffset;
      window.scrollTo({ top, behavior: 'smooth' });
    });
  });
}

// ==================== Mobile Menu ====================
function initMobileMenu() {
  const menuBtn = document.querySelector('.mobile-menu-btn');
  const nav = document.querySelector('.navbar-nav');
  if (!menuBtn || !nav) return;

  menuBtn.addEventListener('click', () => {
    nav.classList.toggle('active');
  });

  nav.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      nav.classList.remove('active');
    });
  });
}

// ==================== Utility ====================
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
