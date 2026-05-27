// SAR-Sense Main JavaScript (Landing Page)

// ==================== DOM Ready ====================
document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initScrollReveal();
  initParticles();
  initChart();
  initCounters();
  initMobileMenu();
  highlightNav('index');
});

function highlightNav(page) {
  document.querySelectorAll('.navbar-nav a').forEach(a => {
    const href = a.getAttribute('href');
    if (href === `${page}.html` || (page === 'index' && href === 'index.html')) {
      a.classList.add('active');
    }
  });
}

// ==================== Navbar ====================
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  let lastScroll = 0;

  window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    if (currentScroll > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
    lastScroll = currentScroll;
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
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  });
  revealElements.forEach(el => observer.observe(el));
}

// ==================== Particles Background ====================
function initParticles() {
  const particlesContainer = document.querySelector('.particles');
  if (!particlesContainer) return;
  const particleCount = 50;
  for (let i = 0; i < particleCount; i++) {
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

// ==================== Chart Initialization ====================
function initChart() {
  const canvas = document.getElementById('detectionChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const container = canvas.parentElement;
  const dataPoints = 24;
  const detectionData = generateSampleData(dataPoints, 10, 50);
  const accuracyData = generateSampleData(dataPoints, 85, 99);

  function resizeCanvas() {
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
    drawChart();
  }

  function drawChart() {
    const width = canvas.width;
    const height = canvas.height;
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

    drawLine(ctx, detectionData, 0, 60, padding, chartWidth, chartHeight, '#00f0ff', 'rgba(0, 240, 255, 0.1)');
    drawLine(ctx, accuracyData, 80, 100, padding, chartWidth, chartHeight, '#7b61ff', 'rgba(123, 97, 255, 0.1)');

    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i < dataPoints; i += 4) {
      const x = padding.left + (chartWidth / (dataPoints - 1)) * i;
      ctx.fillText(`${i}:00`, x, height - 10);
    }
    drawLegend(ctx, width - padding.right, 15);
  }

  function drawLine(ctx, data, minVal, maxVal, padding, chartWidth, chartHeight, strokeColor, fillColor) {
    const points = data.map((val, i) => ({
      x: padding.left + (chartWidth / (dataPoints - 1)) * i,
      y: padding.top + chartHeight - ((val - minVal) / (maxVal - minVal)) * chartHeight
    }));

    ctx.beginPath();
    ctx.moveTo(points[0].x, padding.top + chartHeight);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(points[points.length - 1].x, padding.top + chartHeight);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.fill();

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
    ctx.stroke();
  }

  function drawLegend(ctx, x, y) {
    ctx.fillStyle = '#00f0ff';
    ctx.beginPath();
    ctx.arc(x - 120, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '12px -apple-system, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('检测数量', x - 112, y + 4);

    ctx.fillStyle = '#7b61ff';
    ctx.beginPath();
    ctx.arc(x - 30, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.fillText('准确率', x - 22, y + 4);
  }

  function generateSampleData(count, min, max) {
    return Array.from({ length: count }, () =>
      Math.floor(Math.random() * (max - min + 1)) + min
    );
  }

  resizeCanvas();
  window.addEventListener('resize', debounce(resizeCanvas, 250));
}

// ==================== Counter Animation ====================
function initCounters() {
  const counters = document.querySelectorAll('.trust-value');
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
  }, { threshold: 0.5 });
  counters.forEach(counter => observer.observe(counter));
}

function animateCounter(element, target, prefix, suffix) {
  const duration = 2000;
  const start = 0;
  const end = parseFloat(target);
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = start + (end - start) * eased;

    if (target.includes('.')) {
      element.textContent = prefix + current.toFixed(1) + suffix;
    } else {
      element.textContent = prefix + Math.floor(current) + suffix;
    }

    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
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
