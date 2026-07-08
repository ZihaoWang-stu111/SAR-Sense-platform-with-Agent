const TOKEN_KEY = "sar_sense_token";
const USER_KEY = "sar_sense_user";

function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setToken(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  const value = typeof user === "object" ? JSON.stringify(user || {}) : JSON.stringify({ username: user || "", role: "" });
  localStorage.setItem(USER_KEY, value);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

function getCurrentUser() {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return {};
  try {
    const user = JSON.parse(raw);
    return typeof user === "object" && user ? user : { username: String(user || "") };
  } catch {
    return { username: raw };
  }
}

function isLoggedIn() {
  return !!getToken();
}

function getUsername() {
  return getCurrentUser().username || "";
}

function getRole() {
  return getCurrentUser().role || "";
}

function isAdmin() {
  return getRole() === "admin";
}

async function refreshCurrentUser() {
  const token = getToken();
  if (!token) return {};
  const res = await fetch("/api/auth/me", {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (res.status === 401) {
    clearToken();
    window.location.href = "login.html";
    return {};
  }
  if (!res.ok) return getCurrentUser();
  const data = await res.json();
  if (data && data.user) {
    setToken(token, data.user);
    return data.user;
  }
  return getCurrentUser();
}

async function apiFetch(url, options = {}) {
  const token = getToken();
  options.headers = {
    ...(options.headers || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const res = await fetch(url, options);
  if (res.status === 401) {
    clearToken();
    window.location.href = "login.html";
    throw new Error("Unauthorized");
  }
  return res;
}

function requireAuth() {
  const isPublic = document.body.hasAttribute("data-public");
  if (!isPublic && !isLoggedIn()) {
    window.location.href = "login.html";
    return false;
  }
  return true;
}

function renderAuthUI() {
  const box = document.querySelector(".navbar-actions");
  if (!box) return;

  if (isLoggedIn()) {
    const user = getCurrentUser();
    const span = document.createElement("span");
    span.className = "auth-user";
    span.style.cssText = "color: var(--text-secondary, #64748b); font-size: 0.875rem; margin-right: 12px; display:inline-flex; align-items:center;";
    span.textContent = `${user.username || ""}${user.role ? ` · ${user.role}` : ""}`;

    const btn = document.createElement("a");
    btn.href = "#";
    btn.className = "btn btn-secondary btn-sm";
    btn.textContent = "退出";
    btn.style.marginRight = "8px";
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      clearToken();
      window.location.href = "login.html";
    });
    box.insertBefore(btn, box.firstChild);
    box.insertBefore(span, btn);
  } else {
    const btn = document.createElement("a");
    btn.href = "login.html";
    btn.className = "btn btn-secondary btn-sm";
    btn.textContent = "登录 / 注册";
    btn.style.marginRight = "8px";
    box.insertBefore(btn, box.firstChild);
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  if (requireAuth()) {
    await refreshCurrentUser();
  }
  renderAuthUI();

  // ==================== 全局 Davies 时钟与 Hamburger 初始化 ====================
  initDaviesNavbarInteractions();
});

function initDaviesNavbarInteractions() {
  // 1. 24h CUP 实时时钟
  const clockEl = document.getElementById("davies-clock");
  if (clockEl) {
    const timeFormatter = new Intl.DateTimeFormat('en-GB', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });

    function updateClock() {
      const now = new Date();
      clockEl.textContent = "CUP " + timeFormatter.format(now);
    }

    updateClock();
    setInterval(updateClock, 1000);
  }

  // 2. 移动端 Hamburger Toggle
  const mobileBtn = document.querySelector(".mobile-toggle-btn");
  const mobilePanel = document.getElementById("mobile-nav-panel");
  if (mobileBtn && mobilePanel) {
    mobileBtn.addEventListener("click", () => {
      const isOpen = mobilePanel.classList.toggle("open");
      mobileBtn.querySelector(".btn-text").textContent = isOpen ? "Close" : "Menu";
    });
  }
}
