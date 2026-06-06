/**
 * EvalSystem Dashboard — Shared JavaScript
 * Handles API calls, dark mode, navigation highlight, and shared utilities.
 */

// ============ Dark Mode ============
function initDarkMode() {
  const saved = localStorage.getItem('evalsystem-dark') === 'true';
  if (saved) {
    document.documentElement.classList.add('dark');
  }
  // Expose toggle globally
  window.toggleDark = function () {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('evalsystem-dark', isDark);
  };
}

// ============ Toast Notifications ============
function showToast(message, type = 'info') {
  const colors = {
    info: 'bg-blue-500',
    success: 'bg-green-500',
    error: 'bg-red-500',
    warning: 'bg-yellow-500 text-yellow-900',
  };
  const el = document.createElement('div');
  el.className = `toast fixed top-4 right-4 z-50 ${colors[type] || colors.info} text-white px-5 py-3 rounded-xl shadow-2xl text-sm font-medium`;
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ============ API Helpers ============
async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url} ${r.status}`);
  return r.json();
}

async function apiPost(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || `POST ${url} ${r.status}`);
  return data;
}

// ============ Formatting Utilities ============
function formatTime(isoStr) {
  const d = new Date(isoStr);
  return d.toLocaleString('zh-CN', { hour12: false });
}

function formatDuration(seconds) {
  if (seconds == null) return '-';
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function scoreColor(score) {
  if (score >= 85) return 'text-green-600 dark:text-green-400';
  if (score >= 70) return 'text-blue-600 dark:text-blue-400';
  if (score >= 55) return 'text-yellow-600 dark:text-yellow-400';
  return 'text-red-600 dark:text-red-400';
}

function scoreBg(score) {
  if (score >= 85) return 'bg-green-500';
  if (score >= 70) return 'bg-blue-500';
  if (score >= 55) return 'bg-yellow-500';
  return 'bg-red-500';
}

function confidenceColor(c) {
  if (c >= 0.9) return 'text-green-600 dark:text-green-400';
  if (c >= 0.7) return 'text-blue-600 dark:text-blue-400';
  return 'text-yellow-600 dark:text-yellow-400';
}

// ============ Persona Chip Logic ============
window.PERSONA_META = {
  cooperative:     { label: '配合型', color: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200', desc: '积极回应，按流程配合' },
  hesitant:        { label: '犹豫型', color: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200', desc: '语气犹豫，需要确认' },
  resistant:       { label: '抗拒型', color: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200', desc: '不耐烦，拒绝配合' },
  off_topic:       { label: '跑题型', color: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200', desc: '话题发散，偏离主线' },
  contradictory:   { label: '矛盾型', color: 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200', desc: '前后矛盾，改变主意' },
  busy:            { label: '忙碌型', color: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200', desc: '赶时间，要求简短' },
  confused:        { label: '困惑型', color: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200', desc: '听不懂，需要解释' },
  impatient:       { label: '急躁型', color: 'bg-rose-100 text-rose-800 dark:bg-rose-900 dark:text-rose-200', desc: '催促施压，语气强硬' },
  boundary:        { label: '边界型', color: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-200', desc: '试探边界，提出异常请求' },
  red_team_l1:     { label: '红队 L1', color: 'bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200', desc: '简单对抗测试' },
  red_team_l2:     { label: '红队 L2', color: 'bg-gray-300 text-gray-900 dark:bg-gray-600 dark:text-gray-100', desc: '中级对抗测试' },
  red_team_l3:     { label: '红队 L3', color: 'bg-gray-800 text-white dark:bg-black dark:text-gray-100', desc: '高级对抗测试' },
};

function personaLabel(p) { return PERSONA_META[p]?.label || p; }

// ============ Report Helpers ============
function reportKindBadge(kind) {
  const m = {
    report:       'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    comparison:   'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    calibration:  'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
  };
  const labels = { report: '评测', comparison: '对比', calibration: '校准' };
  return `<span class="text-xs font-medium px-2.5 py-0.5 rounded-full ${m[kind] || 'bg-gray-100 text-gray-800'}">${labels[kind] || kind}</span>`;
}

// ============ Init Nav Highlight ============
function initNav() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(a => {
    if (a.getAttribute('href') === path) {
      a.classList.add('nav-active');
    }
  });
}

// ============ DOM Ready ============
document.addEventListener('DOMContentLoaded', () => {
  initDarkMode();
  initNav();
});
