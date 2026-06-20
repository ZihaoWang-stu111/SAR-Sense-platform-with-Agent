// SAR-Sense — Luxury Layer JavaScript
// Apple Liquid Glass 之上的奢华交互层
//
// 提供：
//   1. Custom cursor (dot + delayed ring)
//   2. Cursor spotlight — 鼠标位置带动的全局聚光
//   3. Scroll-tied parallax — Hero/Story 元素随滚动慢速位移
//   4. Card tilt — 大型玻璃面板鼠标 3D 倾斜
//   5. Luxury reveal — 章节标题更慢更优雅的入场
//
// 全部尊重 prefers-reduced-motion / hover: none

(function () {
  'use strict';

  const PREFERS_REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const IS_TOUCH = !window.matchMedia('(hover: hover) and (pointer: fine)').matches;

  let cursorLayerInitialized = false;
  let pageEffectsInitialized = false;

  function initCursorLayer() {
    if (cursorLayerInitialized || !document.body) return;
    cursorLayerInitialized = true;

    if (!IS_TOUCH && !PREFERS_REDUCED) {
      initCustomCursor();
      initCursorSpotlight();
    }
  }

  function initPageEffects() {
    if (pageEffectsInitialized) return;
    pageEffectsInitialized = true;

    if (!IS_TOUCH && !PREFERS_REDUCED) {
      initCardTilt();
    }
    if (!PREFERS_REDUCED) {
      initParallax();
    }
    initLuxReveal();
  }

  function initLuxuryLayer() {
    initCursorLayer();
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initPageEffects, { once: true });
    } else {
      initPageEffects();
    }
  }

  if (document.body) {
    initLuxuryLayer();
  } else {
    document.addEventListener('DOMContentLoaded', initLuxuryLayer, { once: true });
  }

  // ==================== Custom cursor ====================
  function initCustomCursor() {
    const dot = document.querySelector('.lux-cursor-dot') || document.createElement('div');
    dot.className = 'lux-cursor-dot';
    if (!dot.parentElement) document.body.appendChild(dot);

    const ring = document.querySelector('.lux-cursor-ring') || document.createElement('div');
    ring.className = 'lux-cursor-ring';
    if (!ring.parentElement) document.body.appendChild(ring);

    document.body.classList.add('lux-cursor-on');

    let dotX = window.innerWidth / 2;
    let dotY = window.innerHeight / 2;
    let ringX = dotX;
    let ringY = dotY;
    let mouseX = dotX;
    let mouseY = dotY;
    let visible = false;
    let isPointerDown = false;

    document.addEventListener('mousemove', (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
      if (!visible) {
        dot.style.opacity = '1';
        ring.style.opacity = '1';
        visible = true;
      }
    });

    document.addEventListener('mouseleave', () => {
      dot.style.opacity = '0';
      ring.style.opacity = '0';
      visible = false;
      isPointerDown = false;
      dot.classList.remove('is-down');
      ring.classList.remove('is-down');
    });
    document.addEventListener('mouseenter', () => {
      dot.style.opacity = '1';
      ring.style.opacity = '1';
      visible = true;
    });

    // hover detection — interactive elements
    const interactiveSelector = 'a, button, [role="button"], .btn, .feature-card, .security-card, .trust-card, .tech-card, .m-card, .preview-tab, .tab-btn';
    const textSelector = 'input[type="text"], input[type="email"], input[type="search"], textarea, [contenteditable="true"]';

    document.addEventListener('mouseover', (e) => {
      if (e.target.closest(textSelector)) {
        ring.classList.add('is-text');
        ring.classList.remove('is-hover');
      } else if (e.target.closest(interactiveSelector)) {
        ring.classList.add('is-hover');
        ring.classList.remove('is-text');
      } else {
        ring.classList.remove('is-hover');
        ring.classList.remove('is-text');
      }
    });

    function tick() {
      // Dot follows immediately
      dotX += (mouseX - dotX) * 0.6;
      dotY += (mouseY - dotY) * 0.6;
      // Ring lags behind for elegance
      ringX += (mouseX - ringX) * 0.15;
      ringY += (mouseY - ringY) * 0.15;

      const pressScale = isPointerDown ? ' scale(0.82)' : '';
      dot.style.transform  = `translate3d(${dotX}px, ${dotY}px, 0) translate(-50%, -50%)`;
      ring.style.transform = `translate3d(${ringX}px, ${ringY}px, 0) translate(-50%, -50%)${pressScale}`;

      requestAnimationFrame(tick);
    }
    tick();

    function setPointerDown(value, event) {
      if (value && event) {
        mouseX = event.clientX;
        mouseY = event.clientY;
        dotX = mouseX;
        dotY = mouseY;
        ringX = mouseX;
        ringY = mouseY;
      }
      isPointerDown = value;
      dot.classList.toggle('is-down', value);
      ring.classList.toggle('is-down', value);
      if (value) {
        dot.style.opacity = '1';
        ring.style.opacity = '1';
        visible = true;
      }
    }

    const pointerStateOptions = { capture: true, passive: true };
    window.addEventListener('pointerdown', (event) => setPointerDown(true, event), pointerStateOptions);
    window.addEventListener('pointerup', () => setPointerDown(false), pointerStateOptions);
    window.addEventListener('pointercancel', () => setPointerDown(false), pointerStateOptions);
    window.addEventListener('mousedown', (event) => setPointerDown(true, event), pointerStateOptions);
    window.addEventListener('mouseup', () => setPointerDown(false), pointerStateOptions);
    window.addEventListener('blur', () => setPointerDown(false));
  }

  // ==================== Cursor spotlight ====================
  function initCursorSpotlight() {
    const spotlight = document.createElement('div');
    spotlight.className = 'lux-spotlight';
    document.body.appendChild(spotlight);

    let lastX = 50, lastY = 50;
    let pendingFrame = false;

    document.addEventListener('mousemove', (e) => {
      const x = (e.clientX / window.innerWidth) * 100;
      const y = (e.clientY / window.innerHeight) * 100;
      lastX = x;
      lastY = y;
      if (!pendingFrame) {
        pendingFrame = true;
        requestAnimationFrame(() => {
          spotlight.style.setProperty('--mx', lastX + '%');
          spotlight.style.setProperty('--my', lastY + '%');
          pendingFrame = false;
        });
      }
    }, { passive: true });
  }

  // ==================== Card tilt ====================
  function initCardTilt() {
    const tiltSelector = '.lux-stage-card.front, .lux-story-visual, .lux-cta-card';
    const cards = document.querySelectorAll(tiltSelector);
    const MAX_TILT = 4;       // 度

    cards.forEach(card => {
      let raf = null;
      const parent = card.classList.contains('lux-stage-card') ? card.parentElement : card;

      function onMove(e) {
        const rect = parent.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        const dx = (e.clientX - cx) / (rect.width / 2);   // -1 ~ 1
        const dy = (e.clientY - cy) / (rect.height / 2);
        if (raf) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          const rx = -dy * MAX_TILT;
          const ry = dx * MAX_TILT;
          card.style.transform = `perspective(1400px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg)`;
        });
      }

      function onLeave() {
        if (raf) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
          card.style.transform = '';
        });
      }

      parent.addEventListener('mousemove', onMove);
      parent.addEventListener('mouseleave', onLeave);
    });
  }

  // ==================== Scroll-tied Parallax ====================
  function initParallax() {
    const items = document.querySelectorAll('[data-lux-parallax]');
    if (!items.length) return;

    const targets = [...items].map(el => ({
      el,
      speed: parseFloat(el.dataset.luxParallax) || 0.15,
    }));

    let pending = false;

    function tick() {
      const y = window.scrollY;
      targets.forEach(({ el, speed }) => {
        const rect = el.getBoundingClientRect();
        // 当 element 在视口附近时才计算
        if (rect.bottom < -200 || rect.top > window.innerHeight + 200) return;
        const offset = (rect.top - window.innerHeight / 2) * speed;
        el.style.transform = `translate3d(0, ${offset.toFixed(2)}px, 0)`;
      });
      pending = false;
    }

    function onScroll() {
      if (!pending) {
        pending = true;
        requestAnimationFrame(tick);
      }
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    tick();
  }

  // ==================== Luxury reveal ====================
  // 比标准 reveal 慢一档；用于章节大字
  function initLuxReveal() {
    const items = document.querySelectorAll('.lux-reveal');
    if (!items.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('active');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.18, rootMargin: '0px 0px -8% 0px' });

    items.forEach(el => observer.observe(el));
  }

})();
