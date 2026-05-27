// SAR-Sense Detection Page JavaScript

const API_BASE = '';

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initScrollReveal();
  initParticles();
  initMobileMenu();
  initDetection();
  highlightNav('detection');
});

function highlightNav(page) {
  document.querySelectorAll('.navbar-nav a').forEach(a => {
    if (a.getAttribute('href') === `${page}.html`) {
      a.classList.add('active');
    }
  });
}

function initDetection() {
  const uploadArea = document.getElementById('uploadArea');
  const imageInput = document.getElementById('imageInput');
  const uploadBtn = document.getElementById('uploadBtn');
  const filename = document.getElementById('filename');
  const detectionResults = document.getElementById('detectionResults');
  const detectionLoading = document.getElementById('detectionLoading');
  const detectBtn = document.getElementById('detectBtn');
  const resetBtn = document.getElementById('resetBtn');

  if (!uploadArea || !imageInput) return;

  let selectedFile = null;

  uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    imageInput.click();
  });

  uploadArea.addEventListener('click', () => {
    imageInput.click();
  });

  imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      selectedFile = e.target.files[0];
      filename.textContent = selectedFile.name;
      showDetectionResults();
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
      selectedFile = e.dataTransfer.files[0];
      filename.textContent = selectedFile.name;
      showDetectionResults();
    }
  });

  detectBtn.addEventListener('click', () => {
    if (selectedFile) runDetection(selectedFile);
  });

  resetBtn.addEventListener('click', () => resetDetection());

  function showDetectionResults() {
    uploadArea.style.display = 'none';
    detectionResults.style.display = 'block';
    const reader = new FileReader();
    reader.onload = (e) => {
      document.getElementById('originalImage').src = e.target.result;
    };
    reader.readAsDataURL(selectedFile);
  }

  function resetDetection() {
    selectedFile = null;
    imageInput.value = '';
    filename.textContent = '';
    uploadArea.style.display = 'block';
    detectionResults.style.display = 'none';
    detectionLoading.style.display = 'none';
    document.getElementById('originalImage').src = '';
    document.getElementById('resultImage').src = '';
    document.getElementById('shipCount').textContent = '0';
    document.getElementById('detectionStatus').textContent = '-';
    document.getElementById('detectionTime').textContent = '-';
  }

  async function runDetection(file) {
    detectionLoading.style.display = 'block';
    detectionResults.style.display = 'none';

    const formData = new FormData();
    formData.append('image', file);
    const startTime = Date.now();

    try {
      const response = await fetch(`${API_BASE}/api/detect`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (data.success) {
        const elapsed = Date.now() - startTime;
        document.getElementById('resultImage').src = `data:image/png;base64,${data.result_image}`;
        document.getElementById('shipCount').textContent = data.ship_count;
        document.getElementById('detectionStatus').textContent = data.ship_count > 0 ? '检测成功' : '无目标';
        document.getElementById('detectionStatus').style.color = data.ship_count > 0 ? '#28c840' : '#f59e0b';
        document.getElementById('detectionTime').textContent = `${elapsed}ms`;

        detectionLoading.style.display = 'none';
        detectionResults.style.display = 'block';
      } else {
        throw new Error(data.error || 'Detection failed');
      }
    } catch (error) {
      console.error('Detection error:', error);
      detectionLoading.style.display = 'none';
      detectionResults.style.display = 'block';
      document.getElementById('detectionStatus').textContent = '检测失败';
      document.getElementById('detectionStatus').style.color = '#ff5f57';
      alert('检测失败: ' + error.message);
    }
  }
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
