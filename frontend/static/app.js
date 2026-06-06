/**
 * EvalSystem Frontend — SPA with hash-based routing
 * Config: window.EVALSYSTEM_API_BASE — set before this script loads
 */

const API = (window.EVALSYSTEM_API_BASE || '').replace(/\/+$/, '');

// ============ API Helpers ============
async function apiGet(url) {
  const r = await fetch(API + url);
  if (!r.ok) throw new Error(`GET ${url} ${r.status}`);
  return r.json();
}
async function apiPost(url, body) {
  const r = await fetch(API + url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || `POST ${url} ${r.status}`);
  return data;
}

// ============ Dark Mode ============
function initDarkMode() {
  const saved = localStorage.getItem('evalsystem-dark') === 'true';
  if (saved) document.documentElement.classList.add('dark');
  window.toggleDark = function () {
    const isDark = document.documentElement.classList.toggle('dark');
    localStorage.setItem('evalsystem-dark', isDark);
  };
}

// ============ Router ============
let currentRoute = 'dashboard';

function navigate(hash) {
  const route = hash.replace('#', '') || 'dashboard';
  currentRoute = route;
  document.querySelectorAll('.page-section').forEach(el => el.classList.add('hidden'));
  const target = document.getElementById('page-' + route);
  if (target) target.classList.remove('hidden');
  document.querySelectorAll('.nav-link').forEach(a => {
    const href = a.getAttribute('href');
    a.classList.toggle('nav-active', href === '#' + route);
  });
  // Scroll to top
  window.scrollTo(0, 0);
}

window.addEventListener('hashchange', () => navigate(window.location.hash));
window.addEventListener('DOMContentLoaded', () => navigate(window.location.hash || '#dashboard'));

// ============ Toast ============
function showToast(message, type = 'info') {
  const colors = { info: 'bg-blue-500', success: 'bg-green-500', error: 'bg-red-500' };
  const el = document.createElement('div');
  el.className = `toast fixed top-4 right-4 z-50 ${colors[type] || 'bg-blue-500'} text-white px-5 py-3 rounded-xl shadow-2xl text-sm font-medium`;
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ============ Persona Metadata ============
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

// ============ Formatting ============
function formatTime(isoStr) {
  const d = new Date(isoStr);
  return d.toLocaleString('zh-CN', { hour12: false });
}
function scoreColor(s) {
  if (s == null) return '';
  if (s >= 85) return 'text-green-600 dark:text-green-400';
  if (s >= 70) return 'text-blue-600 dark:text-blue-400';
  if (s >= 55) return 'text-yellow-600 dark:text-yellow-400';
  return 'text-red-600 dark:text-red-400';
}
function scoreBg(s) {
  if (s >= 85) return 'bg-green-500';
  if (s >= 70) return 'bg-blue-500';
  if (s >= 55) return 'bg-yellow-500';
  return 'bg-red-500';
}
function reportKindBadge(kind) {
  const labels = { report: '评测', comparison: '对比', calibration: '校准' };
  const colors = {
    report: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
    comparison: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
    calibration: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
  };
  return `<span class="text-xs font-medium px-2.5 py-0.5 rounded-full ${colors[kind] || 'bg-gray-100'}">${labels[kind] || kind}</span>`;
}
function confidenceColor(c) {
  if (c >= 0.9) return 'text-green-600 dark:text-green-400';
  if (c >= 0.7) return 'text-blue-600 dark:text-blue-400';
  return 'text-yellow-600 dark:text-yellow-400';
}

// ============ Init ============
document.addEventListener('DOMContentLoaded', () => {
  initDarkMode();
  // Initialize dynamic content
  if (document.getElementById('dashboard-page')) initDashboard();
  if (document.getElementById('eval-page')) initEval();
  if (document.getElementById('reports-page')) initReports();
  if (document.getElementById('jobs-page')) initJobs();
});

// ============ Dashboard ============
async function initDashboard() {
  try {
    const reports = await apiGet('/api/reports');
    document.getElementById('kpi-reports').textContent = reports.length;
  } catch (e) {}
  try {
    const jobs = await apiGet('/api/jobs');
    document.getElementById('kpi-jobs').textContent = jobs.length;
  } catch (e) {}
  try {
    const reports = await apiGet('/api/reports');
    renderReportList(reports.slice(0, 6), 'recent-reports');
  } catch (e) {}
}

// ============ Eval Form ============
async function initEval() {
  try {
    const data = await apiGet('/api/instructions');
    const sel = document.getElementById('instr-select');
    data.instructions.forEach(i => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = i.split('/').pop();
      sel.appendChild(opt);
    });
  } catch (e) {}
}

const evalForm = {
  instruction_path: '',
  sessions: 1,
  self_consistency: 1,
  concurrency: 4,
  seed: 42,
  branch_test: false,
  generate_pdf: false,
  selectedPersonas: ['cooperative'],
  running: false,
  preset: 'mini',
};
evalForm.togglePersona = function (key) {
  const idx = this.selectedPersonas.indexOf(key);
  if (idx >= 0) this.selectedPersonas.splice(idx, 1);
  else this.selectedPersonas.push(key);
};
evalForm.setPreset = function (mode) {
  this.preset = mode;
  if (mode === 'mini') {
    this.selectedPersonas = ['cooperative', 'busy', 'red_team_l1'];
    this.sessions = 1;
    this.self_consistency = 1;
    this.branch_test = false;
  } else if (mode === 'full') {
    this.selectedPersonas = Object.keys(window.PERSONA_META);
    this.sessions = 3;
    this.self_consistency = 3;
    this.branch_test = true;
    this.concurrency = 4;
  }
};
evalForm.startEval = async function () {
  if (!this.instruction_path || this.selectedPersonas.length === 0 || this.running) return;
  this.running = true;
  try {
    const data = await apiPost('/api/eval/start', {
      instruction_path: this.instruction_path,
      personas: this.selectedPersonas,
      sessions: parseInt(this.sessions) || 1,
      branch_test: this.branch_test,
      self_consistency: parseInt(this.self_consistency) || 1,
      concurrency: parseInt(this.concurrency) || 4,
      seed: parseInt(this.seed) || null,
      generate_pdf: this.generate_pdf,
    });
    showToast('评测已启动! Job ID: ' + data.job_id, 'success');
  } catch (e) {
    showToast('启动失败: ' + e.message, 'error');
  } finally {
    this.running = false;
  }
};

// ============ Reports ============
let reportsCache = [];

async function initReports() {
  try {
    reportsCache = await apiGet('/api/reports');
    renderReportList(reportsCache, 'report-list');
  } catch (e) {}
}
function filterReports(kind) {
  const filtered = kind ? reportsCache.filter(r => r.kind === kind) : reportsCache;
  renderReportList(filtered, 'report-list');
}

// ============ Jobs ============
async function initJobs() {
  await refreshJobs();
  setInterval(refreshJobs, 3000);
}
async function refreshJobs() {
  try {
    const jobs = await apiGet('/api/jobs');
    renderJobList(jobs);
    // Update counters
    document.getElementById('jobs-total').textContent = jobs.length;
    document.getElementById('jobs-running').textContent = jobs.filter(j => j.status === 'running').length;
    document.getElementById('jobs-finished').textContent = jobs.filter(j => j.status === 'finished').length;
    document.getElementById('jobs-failed').textContent = jobs.filter(j => j.status === 'failed').length;
  } catch (e) {}
}
async function loadLog(jobId) {
  try {
    const data = await apiGet(`/api/jobs/${jobId}/log?tail=200`);
    const el = document.getElementById('log-' + jobId);
    if (el) {
      el.textContent = data.lines.join('\n') || '(无日志)';
      el.scrollTop = el.scrollHeight;
    }
  } catch (e) {
    const el = document.getElementById('log-' + jobId);
    if (el) el.textContent = '(加载失败)';
  }
}
async function autoRefreshLog(jobId) {
  await loadLog(jobId);
  setTimeout(() => {
    document.querySelectorAll('.job-card').forEach(card => {
      if (card.dataset.jobId === jobId && !card.classList.contains('hidden')) {
        autoRefreshLog(jobId);
      }
    });
  }, 2500);
}

// ============ Calibration ============
const calForm = { sc: 3, running: false, results: null, resultFiles: null };
calForm.runCalibration = async function () {
  this.running = true;
  this.results = null;
  this.resultFiles = null;
  try {
    const data = await apiPost('/api/calibration/start', { self_consistency: parseInt(this.sc) || 1 });
    showToast('校准已启动! Job ID: ' + data.job_id, 'success');
    const poll = setInterval(async () => {
      try {
        const allJobs = await apiGet('/api/jobs');
        const job = allJobs.find(j => j.id === data.job_id);
        if (job && job.status === 'finished') {
          clearInterval(poll);
          calForm.running = false;
          calForm.resultFiles = job.report_files;
          if (job.report_files?.json) {
            const r = await fetch(API + '/files/' + job.report_files.json);
            calForm.results = await r.json();
            renderCalibrationResults(calForm.results, calForm.resultFiles);
          }
          showToast('校准完成!', 'success');
        } else if (job && job.status === 'failed') {
          clearInterval(poll);
          calForm.running = false;
          showToast('校准失败: ' + (job.error || '未知错误'), 'error');
        }
      } catch (e) {}
    }, 2000);
  } catch (e) {
    showToast('启动失败: ' + e.message, 'error');
    this.running = false;
  }
};

// ============ Render Helpers ============
function renderReportList(reports, containerId) {
  const div = document.getElementById(containerId);
  if (!div) return;
  if (!reports.length) {
    div.innerHTML = '<div class="text-center py-10 text-gray-400 dark:text-gray-500 text-sm">尚无报告</div>';
    return;
  }
  div.innerHTML = reports.map(r => {
    const s = r.summary || {};
    const links = Object.entries(r.files).map(([ext, fname]) =>
      `<a href="${API}/files/${fname}" target="_blank" class="px-2.5 py-1 rounded text-xs font-medium ${ext === 'html' ? 'bg-primary-500 text-white hover:bg-primary-600' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'} transition-colors">${ext.toUpperCase()}</a>`
    ).join('');
    return `<div class="flex items-start gap-4 p-4 border-b border-gray-100 dark:border-gray-800 last:border-0 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
      ${s.overall_score != null ? `<div class="flex-shrink-0 w-14 h-14 rounded-full flex items-center justify-center text-base font-bold" style="background:conic-gradient(${scoreBg(s.overall_score)} ${s.overall_score}%, #e5e7eb ${s.overall_score}%)"><div class="w-10 h-10 rounded-full bg-white dark:bg-gray-900 flex items-center justify-center"><span class="${scoreColor(s.overall_score)}">${s.overall_score.toFixed(0)}</span></div></div>` : ''}
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2 flex-wrap">${reportKindBadge(r.kind)}<span class="text-sm font-semibold text-gray-900 dark:text-white">${r.datetime}</span></div>
        ${s.task_name ? `<div class="text-sm text-gray-600 dark:text-gray-400 truncate mt-0.5">${s.task_name}</div>` : ''}
        <div class="flex flex-wrap gap-x-4 text-xs text-gray-500 dark:text-gray-400 mt-1">
          ${s.total_sessions ? `<span>会话: <b>${s.total_sessions}</b></span>` : ''}
          ${s.overall_score_std ? `<span>±<b>${s.overall_score_std.toFixed(1)}</b></span>` : ''}
          ${s.overall_confidence ? `<span>置信: <b class="${confidenceColor(s.overall_confidence)}">${s.overall_confidence.toFixed(2)}</b></span>` : ''}
          ${s.branch_coverage != null ? `<span>分支: <b class="text-green-600">${(s.branch_coverage * 100).toFixed(0)}%</b></span>` : ''}
        </div>
      </div>
      <div class="flex gap-1 flex-shrink-0">${links}</div>
    </div>`;
  }).join('');
}

function renderJobList(jobs) {
  const div = document.getElementById('job-list');
  if (!div) return;
  const sorted = [...jobs].sort((a, b) => b.started_at.localeCompare(a.started_at));
  if (!sorted.length) {
    div.innerHTML = '<div class="text-center py-10 text-gray-400 dark:text-gray-500 text-sm">尚无任务</div>';
    return;
  }
  div.innerHTML = sorted.map(j => {
    const statusDot = { running: 'bg-primary-500 pulse-dot', finished: 'bg-green-500', failed: 'bg-red-500', pending: 'bg-gray-400' }[j.status] || 'bg-gray-400';
    const statusBg = { running: 'bg-primary-500', finished: 'bg-green-500', failed: 'bg-red-500', pending: 'bg-gray-400' }[j.status] || 'bg-gray-400';
    return `<div class="job-card border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden mb-3" data-job-id="${j.id}">
      <div class="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors" onclick="toggleJobLog('${j.id}')">
        <span class="w-2.5 h-2.5 rounded-full flex-shrink-0 ${statusDot}"></span>
        <span class="font-mono text-xs text-primary-600 dark:text-primary-400 font-medium">${j.id}</span>
        <span class="px-2 py-0.5 rounded-full text-xs font-medium text-white ${statusBg}">${j.status}</span>
        <span class="text-xs text-gray-500 dark:text-gray-400">${j.kind}</span>
        <span class="text-xs text-gray-400">${formatTime(j.started_at)}</span>
        <div class="ml-auto flex items-center gap-2">
          <span class="text-xs text-gray-400">${j.n_log_lines} lines</span>
          <svg class="w-4 h-4 text-gray-400 transition-transform log-chevron" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
        </div>
      </div>
      <div class="log-section hidden border-t border-gray-200 dark:border-gray-700">
        <div class="bg-gray-900 px-4 py-3">
          <div class="log-viewer max-h-80 overflow-y-auto text-gray-100" id="log-${j.id}">
            <span class="text-gray-600">点击「加载日志」查看输出</span>
          </div>
          <div class="flex gap-2 mt-2">
            <button onclick="event.stopPropagation(); loadLog('${j.id}')" class="text-xs px-3 py-1 rounded bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors">加载日志</button>
            <button onclick="event.stopPropagation(); autoRefreshLog('${j.id}')" class="text-xs px-3 py-1 rounded bg-primary-900 text-primary-300 hover:bg-primary-800 transition-colors">自动刷新</button>
          </div>
        </div>
      </div>
      ${j.report_files?.html ? `<div class="px-4 py-2 bg-gray-50 dark:bg-gray-850 border-t border-gray-100 dark:border-gray-700"><a href="${API}/files/${j.report_files.html}" target="_blank" class="text-xs text-primary-600 dark:text-primary-400 hover:underline">📄 查看报告</a></div>` : ''}
    </div>`;
  }).join('');
}

let openJobLogs = new Set();
function toggleJobLog(jobId) {
  const section = document.querySelector(`.job-card[data-job-id="${jobId}"] .log-section`);
  const chevron = document.querySelector(`.job-card[data-job-id="${jobId}"] .log-chevron`);
  if (!section) return;
  const isOpen = !section.classList.contains('hidden');
  section.classList.toggle('hidden');
  if (chevron) chevron.style.transform = isOpen ? '' : 'rotate(180deg)';
  if (!isOpen) loadLog(jobId);
}

function renderCalibrationResults(results, resultFiles) {
  const div = document.getElementById('cal-results');
  if (!div) return;
  const m = results.metrics || {};
  div.innerHTML = `
    <div class="grid grid-cols-3 gap-4 mb-4">
      <div class="bg-white dark:bg-gray-900 rounded-xl p-5 shadow-sm border border-gray-200 dark:border-gray-800 text-center">
        <div class="text-xs font-medium text-gray-500 uppercase tracking-wide">MAE</div>
        <div class="mt-1 text-3xl font-bold ${(m.mae || 0) < 5 ? 'text-green-600' : 'text-yellow-600'}">${(m.mae || 0).toFixed(2)}</div>
        <div class="text-xs text-gray-400 mt-1">平均绝对误差</div>
      </div>
      <div class="bg-white dark:bg-gray-900 rounded-xl p-5 shadow-sm border border-gray-200 dark:border-gray-800 text-center">
        <div class="text-xs font-medium text-gray-500 uppercase tracking-wide">Pearson r</div>
        <div class="mt-1 text-3xl font-bold ${(m.pearson_r || 0) > 0.9 ? 'text-green-600' : 'text-yellow-600'}">${(m.pearson_r || 0).toFixed(3)}</div>
        <div class="text-xs text-gray-400 mt-1">相关系数</div>
      </div>
      <div class="bg-white dark:bg-gray-900 rounded-xl p-5 shadow-sm border border-gray-200 dark:border-gray-800 text-center">
        <div class="text-xs font-medium text-gray-500 uppercase tracking-wide">通过率</div>
        <div class="mt-1 text-3xl font-bold text-green-600">${((m.within_tolerance_rate || 0) * 100).toFixed(0)}%</div>
        <div class="text-xs text-gray-400 mt-1">容差范围内</div>
      </div>
    </div>
    ${resultFiles ? `<div class="flex gap-2"><a href="${API}/files/${resultFiles.html}" target="_blank" class="px-4 py-2 rounded-lg text-sm font-medium bg-primary-500 text-white hover:bg-primary-600 transition-colors">查看完整 HTML 报告</a></div>` : ''}
  `;
}
