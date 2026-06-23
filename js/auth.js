// 认证工具：token 存取、apiFetch（自动带 Authorization 头）、页面守卫、navbar UI。
// 必须在页面 JS 之前引入。
const TOKEN_KEY = "sar_sense_token";
const USER_KEY = "sar_sense_user";

function getToken() { return localStorage.getItem(TOKEN_KEY); }
function setToken(token, username) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, username || "");
}
function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}
function isLoggedIn() { return !!getToken(); }
function getUsername() { return localStorage.getItem(USER_KEY) || ""; }
function isAdmin() { return getUsername() === "admin"; }

// 封装 fetch：注入 Authorization 头，401 清 token 跳登录
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
    throw new Error("未授权，跳转登录");
  }
  return res;
}

// 页面守卫：非 public 页面无 token 跳登录（login.html 用 <body data-public> 豁免）
function requireAuth() {
  const isPublic = document.body.hasAttribute("data-public");
  if (!isPublic && !isLoggedIn()) {
    window.location.href = "login.html";
    return false;
  }
  return true;
}

// 向 .navbar-actions 注入登录态 UI
function renderAuthUI() {
  const box = document.querySelector(".navbar-actions");
  if (!box) return;
  if (isLoggedIn()) {
    const user = getUsername();
    const span = document.createElement("span");
    span.className = "auth-user";
    span.style.cssText = "color: var(--text-secondary, #64748b); font-size: 0.875rem; margin-right: 12px; display:inline-flex; align-items:center;";
    span.textContent = `👤 ${user}`;
    const btn = document.createElement("a");
    btn.href = "#";
    btn.className = "btn btn-secondary btn-sm";
    btn.textContent = "登出";
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

document.addEventListener("DOMContentLoaded", () => {
  if (requireAuth()) {
    renderAuthUI();
  }
});
