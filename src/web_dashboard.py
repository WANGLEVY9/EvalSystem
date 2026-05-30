"""
Web Dashboard (v3.0)

提供:
- 历史报告浏览 (列表 + 一键打开)
- 一键发起新评测 (in-process, 异步流式日志)
- 模型对比报告查看
- 校准报告查看

启动:
    pip install fastapi uvicorn
    export DEEPSEEK_API_KEY=...
    python -m src.web_dashboard
    # 浏览器访问 http://localhost:8765/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dashboard")


# ============ 内存状态 ============

JOBS: dict[str, dict[str, Any]] = {}  # job_id -> {status, log_lines, report_files, ...}
JOBS_LOCK = threading.Lock()


def make_job(kind: str, params: dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())[:8]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "params": params,
            "status": "pending",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "log_lines": deque(maxlen=400),
            "report_files": [],
            "error": None,
        }
    return job_id


def update_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def append_log(job_id: str, line: str):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["log_lines"].append(line)


def list_reports() -> list[dict[str, Any]]:
    """扫描 output 目录, 返回所有报告"""
    output_dir = ROOT / "output"
    if not output_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    seen_ts = {}  # ts -> {html, json, md, pdf}
    for f in sorted(output_dir.iterdir(), reverse=True):
        m = re.match(r"^(report|comparison|calibration)_(\d{8}_\d{6})\.(html|json|md|pdf)$", f.name)
        if not m:
            continue
        kind, ts, ext = m.group(1), m.group(2), m.group(3)
        key = f"{kind}_{ts}"
        if key not in seen_ts:
            seen_ts[key] = {"kind": kind, "ts": ts, "files": {}}
        seen_ts[key]["files"][ext] = f.name

    for key, info in seen_ts.items():
        try:
            ts_dt = datetime.strptime(info["ts"], "%Y%m%d_%H%M%S")
        except ValueError:
            ts_dt = datetime.now()
        # 提取摘要
        summary: dict[str, Any] = {}
        if "json" in info["files"]:
            try:
                with open(output_dir / info["files"]["json"], encoding="utf-8") as f:
                    rep = json.load(f)
                summary = {
                    "task_name": rep.get("task_name", "")[:50],
                    "overall_score": rep.get("overall_score"),
                    "overall_score_std": rep.get("overall_score_std"),
                    "overall_confidence": rep.get("overall_confidence"),
                    "total_sessions": rep.get("total_sessions"),
                    "branch_coverage": (rep.get("branch_coverage", {}) or {}).get("coverage_rate"),
                }
            except Exception:
                pass
        items.append({
            "kind": info["kind"],
            "ts": info["ts"],
            "datetime": ts_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "files": info["files"],
            "summary": summary,
        })
    items.sort(key=lambda x: x["ts"], reverse=True)
    return items


# ============ 后台跑评测 ============

def run_eval_job(job_id: str, instruction_path: str, personas: list[str],
                 sessions: int, branch_test: bool, self_consistency: int,
                 concurrency: int, seed: Optional[int], generate_pdf: bool):
    """在后台线程跑评测"""
    update_job(job_id, status="running")
    append_log(job_id, f"[{job_id}] 开始评测: {instruction_path}")

    try:
        # Hijack logger
        class _DashboardHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                append_log(job_id, msg)

        h = _DashboardHandler(level=logging.INFO)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(h)

        try:
            from main import run_full_evaluation
            cfg_path = ROOT / "config" / "default_config.yaml"
            with open(cfg_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            run_full_evaluation(
                config=config,
                instruction_path=instruction_path,
                personas=personas if personas else None,
                sessions_per_persona=sessions,
                enable_branch_test=branch_test,
                self_consistency=self_consistency,
                concurrency=concurrency,
                seed=seed,
                generate_pdf=generate_pdf,
            )
        finally:
            logging.getLogger().removeHandler(h)

        # 查找最新报告
        latest = list_reports()
        if latest:
            update_job(job_id, report_files=latest[0]["files"])
        update_job(job_id, status="finished", finished_at=datetime.now().isoformat())
        append_log(job_id, f"[{job_id}] ✅ 评测完成")
    except Exception as e:
        logger.exception("Job failed")
        update_job(job_id, status="failed", error=str(e), finished_at=datetime.now().isoformat())
        append_log(job_id, f"[{job_id}] ❌ 失败: {e}")


def run_calibration_job(job_id: str, self_consistency: int):
    update_job(job_id, status="running")
    append_log(job_id, f"[{job_id}] 开始校准回测 (SC={self_consistency})")
    try:
        class _DashboardHandler(logging.Handler):
            def emit(self, record):
                append_log(job_id, self.format(record))

        h = _DashboardHandler(level=logging.INFO)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(h)
        try:
            sys.path.insert(0, str(ROOT / "tests"))
            from run_calibration import run_calibration, gen_calibration_html
            api_key = os.environ["DEEPSEEK_API_KEY"]
            t = time.time()
            summary = run_calibration(
                cases_path=str(ROOT / "tests/calibration_set.json"),
                self_consistency=self_consistency,
                api_key=api_key,
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
            )
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = ROOT / "output" / f"calibration_{ts}.json"
            html_path = ROOT / "output" / f"calibration_{ts}.html"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            gen_calibration_html(summary, html_path)
            update_job(job_id, status="finished", finished_at=datetime.now().isoformat(),
                       report_files={"html": html_path.name, "json": json_path.name})
            append_log(job_id, f"[{job_id}] ✅ 完成 用时 {time.time()-t:.1f}s")
        finally:
            logging.getLogger().removeHandler(h)
    except Exception as e:
        logger.exception("Calibration failed")
        update_job(job_id, status="failed", error=str(e), finished_at=datetime.now().isoformat())
        append_log(job_id, f"[{job_id}] ❌ 失败: {e}")


# ============ FastAPI 应用 ============

def create_app():
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as e:
        raise ImportError("Web Dashboard 需要 fastapi+uvicorn: pip install fastapi uvicorn") from e

    app = FastAPI(title="多轮对话评测系统 Dashboard")

    # 挂载 output 目录为静态文件
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    app.mount("/files", StaticFiles(directory=str(output_dir)), name="files")

    @app.get("/", response_class=HTMLResponse)
    def index():
        # 列出指令、报告、jobs
        instr_dir = ROOT / "config" / "sample_instructions"
        instructions = []
        if instr_dir.exists():
            for f in sorted(instr_dir.iterdir()):
                if f.suffix in {".md", ".json"}:
                    instructions.append(f"config/sample_instructions/{f.name}")

        return DASHBOARD_HTML.replace("__INSTRUCTIONS__", json.dumps(instructions))

    @app.get("/api/reports")
    def api_reports():
        return list_reports()

    @app.get("/api/jobs")
    def api_jobs():
        with JOBS_LOCK:
            jobs = []
            for jid, j in JOBS.items():
                jobs.append({
                    "id": j["id"],
                    "kind": j["kind"],
                    "status": j["status"],
                    "started_at": j["started_at"],
                    "finished_at": j["finished_at"],
                    "params": j["params"],
                    "report_files": j["report_files"],
                    "error": j["error"],
                    "n_log_lines": len(j["log_lines"]),
                })
        return jobs

    @app.get("/api/jobs/{job_id}/log")
    def api_job_log(job_id: str, tail: int = Query(60)):
        with JOBS_LOCK:
            if job_id not in JOBS:
                raise HTTPException(404, "job not found")
            lines = list(JOBS[job_id]["log_lines"])[-tail:]
            return {
                "id": job_id,
                "status": JOBS[job_id]["status"],
                "report_files": JOBS[job_id]["report_files"],
                "lines": lines,
            }

    @app.post("/api/eval/start")
    def api_eval_start(payload: dict):
        instruction_path = payload.get("instruction_path", "")
        if not instruction_path:
            raise HTTPException(400, "missing instruction_path")
        if not (ROOT / instruction_path).exists():
            raise HTTPException(400, f"file not found: {instruction_path}")
        personas = payload.get("personas", [])
        sessions = int(payload.get("sessions", 1))
        branch_test = bool(payload.get("branch_test", False))
        self_consistency = int(payload.get("self_consistency", 1))
        concurrency = int(payload.get("concurrency", 4))
        seed = payload.get("seed")
        generate_pdf = bool(payload.get("generate_pdf", False))

        job_id = make_job("eval", payload)
        threading.Thread(
            target=run_eval_job,
            args=(job_id, instruction_path, personas, sessions, branch_test,
                  self_consistency, concurrency, seed, generate_pdf),
            daemon=True,
        ).start()
        return {"job_id": job_id}

    @app.post("/api/calibration/start")
    def api_calibration_start(payload: dict):
        sc = int(payload.get("self_consistency", 1))
        job_id = make_job("calibration", payload)
        threading.Thread(target=run_calibration_job, args=(job_id, sc), daemon=True).start()
        return {"job_id": job_id}

    @app.get("/files/{name}")
    def file_redirect(name: str):
        # 防 staticfiles 不命中, fallback
        f = output_dir / name
        if not f.exists():
            raise HTTPException(404)
        return FileResponse(f)

    return app


# ============ Dashboard HTML ============

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>评测系统 Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:#f3f5f9;color:#222;line-height:1.5}
.container{max-width:1320px;margin:0 auto;padding:18px}
.hero{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:24px 30px;border-radius:14px;margin-bottom:18px}
.hero h1{font-size:24px;margin-bottom:6px}
.hero .meta{opacity:.92;font-size:13px}
.grid{display:grid;gap:16px}
.grid-2{grid-template-columns:1.2fr 1fr}
.card{background:#fff;border-radius:12px;padding:18px 20px;box-shadow:0 2px 8px rgba(0,0,0,.05);margin-bottom:14px}
.card h2{font-size:16px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center;padding-bottom:8px;border-bottom:2px solid #f0f2f7}
.card h2 .badge{background:#667eea;color:#fff;font-size:10px;padding:2px 7px;border-radius:8px;font-weight:400}
label{display:block;font-size:12px;color:#666;margin-top:8px;margin-bottom:3px}
input,select,textarea{width:100%;padding:6px 9px;border:1px solid #d9dde6;border-radius:6px;font-size:13px;font-family:inherit}
button{padding:8px 16px;background:#1890ff;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;margin-top:8px}
button:hover{background:#40a9ff}
button.ghost{background:#fff;color:#1890ff;border:1px solid #1890ff}
button.danger{background:#f5222d}
.row{display:flex;gap:8px;align-items:center}
.row > *{flex:1}
.persona-chip{display:inline-block;padding:3px 10px;margin:2px;background:#fafbfd;border:1px solid #d9dde6;border-radius:14px;cursor:pointer;font-size:11px;user-select:none}
.persona-chip.active{background:#1890ff;color:#fff;border-color:#1890ff}
.report-row{display:grid;grid-template-columns:auto 1fr auto auto auto;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid #f0f2f7;font-size:12px}
.report-row .kind{font-size:10px;padding:2px 6px;border-radius:4px;color:#fff}
.kind-report{background:#1890ff}.kind-comparison{background:#722ed1}.kind-calibration{background:#fa8c16}
.score{font-weight:700;color:#1890ff;min-width:50px;text-align:right}
.btn-mini{padding:3px 9px;background:#fafbfd;color:#444;border:1px solid #d9dde6;border-radius:4px;font-size:11px;cursor:pointer;margin-left:4px;text-decoration:none;display:inline-block}
.btn-mini:hover{background:#1890ff;color:#fff;border-color:#1890ff}
.job-row{padding:10px 12px;background:#fafbfd;border-radius:6px;margin:6px 0;font-size:12px}
.job-row .id{font-family:Menlo,monospace;color:#1890ff;font-size:11px}
.status{display:inline-block;padding:1px 7px;border-radius:8px;font-size:10px;color:#fff;margin-left:6px}
.status-running{background:#1890ff}
.status-finished{background:#52c41a}
.status-failed{background:#f5222d}
.status-pending{background:#aaa}
.log{background:#1e1e2e;color:#d9d9e2;padding:10px 12px;border-radius:6px;font-family:Menlo,monospace;font-size:11px;max-height:280px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;margin-top:8px}
.tag{display:inline-block;padding:1px 7px;background:rgba(255,255,255,.18);border-radius:8px;font-size:10px;margin-right:5px}
</style>
</head>
<body>
<div class="container">

<div class="hero">
  <h1>📊 多轮对话评测系统 v3.0 — Dashboard</h1>
  <div class="meta">
    可解释 · 可量化 · 可复算 · 可对比
    <span class="tag">12 画像</span><span class="tag">8 维度</span>
    <span class="tag">证据三元组</span><span class="tag">Self-Consistency</span>
    <span class="tag">分支覆盖</span>
  </div>
</div>

<div class="grid grid-2">
  <!-- 左: 新评测 -->
  <div>
    <div class="card">
      <h2>🚀 发起新评测 <span class="badge">异步</span></h2>
      <label>任务指令文件</label>
      <select id="instruction">
        <!-- JS 动态填 -->
      </select>
      <label>用户画像 (点击切换)</label>
      <div id="personas">
        <span class="persona-chip" data-p="cooperative">配合型</span>
        <span class="persona-chip" data-p="hesitant">犹豫型</span>
        <span class="persona-chip" data-p="resistant">抗拒型</span>
        <span class="persona-chip" data-p="off_topic">跑题型</span>
        <span class="persona-chip" data-p="contradictory">矛盾型</span>
        <span class="persona-chip" data-p="busy">忙碌型</span>
        <span class="persona-chip" data-p="confused">困惑型</span>
        <span class="persona-chip" data-p="impatient">急躁型</span>
        <span class="persona-chip" data-p="boundary">边界型</span>
        <span class="persona-chip" data-p="red_team_l1">红队 L1</span>
        <span class="persona-chip" data-p="red_team_l2">红队 L2</span>
        <span class="persona-chip" data-p="red_team_l3">红队 L3</span>
      </div>
      <div class="row" style="margin-top:6px">
        <div><label>每画像会话数</label><input id="sessions" type="number" value="1" min="1"/></div>
        <div><label>Self-Consistency N</label><input id="sc" type="number" value="1" min="1"/></div>
        <div><label>并发数</label><input id="conc" type="number" value="4" min="1"/></div>
        <div><label>Seed</label><input id="seed" type="number" value="42"/></div>
      </div>
      <div style="margin-top:10px">
        <label><input type="checkbox" id="branch"/> 强制分支覆盖测试</label>
        <label><input type="checkbox" id="pdf"/> 同时导出 PDF (需 weasyprint)</label>
      </div>
      <div class="row" style="margin-top:10px">
        <button onclick="startEval()">▶ 开始评测</button>
        <button class="ghost" onclick="quickPersonas(['cooperative','busy'])">2 画像 mini</button>
        <button class="ghost" onclick="quickPersonas(['cooperative','busy','red_team_l1'])">3 画像</button>
      </div>
    </div>

    <div class="card">
      <h2>🔬 校准回测 <span class="badge">8 case 学术级证明</span></h2>
      <p style="font-size:12px;color:#666;margin-bottom:8px">用 8 个人工标注 case 评估 evaluator 的 MAE / Pearson-r, 证明可靠性</p>
      <div class="row">
        <div><label>Self-Consistency N</label><input id="cal_sc" type="number" value="1" min="1"/></div>
      </div>
      <button onclick="startCalibration()">▶ 开始校准回测</button>
    </div>

    <!-- 进行中 jobs -->
    <div class="card">
      <h2>⚙️ 任务状态 <button class="btn-mini" onclick="refreshJobs()">🔄 刷新</button></h2>
      <div id="jobs"></div>
    </div>
  </div>

  <!-- 右: 历史报告 -->
  <div>
    <div class="card">
      <h2>📂 历史报告 <button class="btn-mini" onclick="refreshReports()">🔄 刷新</button></h2>
      <div id="reports" style="max-height:780px;overflow-y:auto"></div>
    </div>
  </div>
</div>

</div>

<script>
const INSTRUCTIONS = __INSTRUCTIONS__;

document.addEventListener('DOMContentLoaded', () => {
  // 填充指令下拉
  const sel = document.getElementById('instruction');
  for (const i of INSTRUCTIONS) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = i.split('/').pop();
    sel.appendChild(opt);
  }
  // 默认勾选 cooperative
  document.querySelector('.persona-chip[data-p="cooperative"]').classList.add('active');

  // chip 点击
  document.querySelectorAll('.persona-chip').forEach(el => {
    el.addEventListener('click', () => el.classList.toggle('active'));
  });

  refreshReports();
  refreshJobs();
  setInterval(refreshJobs, 3500);
});

function getSelectedPersonas() {
  return Array.from(document.querySelectorAll('.persona-chip.active')).map(el => el.dataset.p);
}

function quickPersonas(arr) {
  document.querySelectorAll('.persona-chip').forEach(el => {
    el.classList.toggle('active', arr.includes(el.dataset.p));
  });
}

async function startEval() {
  const payload = {
    instruction_path: document.getElementById('instruction').value,
    personas: getSelectedPersonas(),
    sessions: parseInt(document.getElementById('sessions').value) || 1,
    branch_test: document.getElementById('branch').checked,
    self_consistency: parseInt(document.getElementById('sc').value) || 1,
    concurrency: parseInt(document.getElementById('conc').value) || 4,
    seed: parseInt(document.getElementById('seed').value) || null,
    generate_pdf: document.getElementById('pdf').checked,
  };
  if (payload.personas.length === 0) {
    alert('至少选择一个画像');
    return;
  }
  const r = await fetch('/api/eval/start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const data = await r.json();
  alert('已开始, job_id=' + data.job_id);
  refreshJobs();
}

async function startCalibration() {
  const payload = {self_consistency: parseInt(document.getElementById('cal_sc').value) || 1};
  const r = await fetch('/api/calibration/start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  const data = await r.json();
  alert('已开始, job_id=' + data.job_id);
  refreshJobs();
}

async function refreshReports() {
  const r = await fetch('/api/reports');
  const reports = await r.json();
  const div = document.getElementById('reports');
  if (!reports.length) {
    div.innerHTML = '<div style="color:#999;padding:14px 0;text-align:center;font-size:12px">尚无报告</div>';
    return;
  }
  div.innerHTML = reports.map(r => {
    const s = r.summary || {};
    const score = s.overall_score ? `<span class="score">${s.overall_score.toFixed(1)}</span>` : '';
    const subline = s.task_name ? `<div style="font-size:11px;color:#999">${s.task_name}${s.total_sessions?(' · '+s.total_sessions+' 会话'):''}${s.overall_score_std?(' · ±'+s.overall_score_std.toFixed(1)):''}</div>` : '';
    const links = Object.keys(r.files).map(ext => `<a class="btn-mini" href="/files/${r.files[ext]}" target="_blank">${ext.toUpperCase()}</a>`).join('');
    return `<div class="report-row">
      <span class="kind kind-${r.kind}">${r.kind}</span>
      <div>
        <div style="font-weight:500">${r.datetime}</div>
        ${subline}
      </div>
      ${score}
      <div></div>
      <div>${links}</div>
    </div>`;
  }).join('');
}

async function refreshJobs() {
  const r = await fetch('/api/jobs');
  const jobs = await r.json();
  const div = document.getElementById('jobs');
  if (!jobs.length) {
    div.innerHTML = '<div style="color:#999;padding:8px 0;text-align:center;font-size:12px">尚无任务</div>';
    return;
  }
  // 最新 5 个
  const recent = jobs.slice().sort((a,b)=>b.started_at.localeCompare(a.started_at)).slice(0, 5);
  div.innerHTML = recent.map(j => {
    return `<div class="job-row">
      <div>
        <span class="id">${j.id}</span>
        <span class="status status-${j.status}">${j.status}</span>
        <span style="font-size:11px;color:#666;margin-left:5px">${j.kind} · ${new Date(j.started_at).toLocaleTimeString()}</span>
        <button class="btn-mini" onclick="toggleLog('${j.id}')">日志</button>
        ${j.report_files && j.report_files.html ? `<a class="btn-mini" href="/files/${j.report_files.html}" target="_blank">查看报告</a>` : ''}
      </div>
      <div id="log-${j.id}" style="display:none"><div class="log" id="log-content-${j.id}">加载中...</div></div>
    </div>`;
  }).join('');
}

async function toggleLog(jobId) {
  const div = document.getElementById('log-' + jobId);
  if (div.style.display === 'none') {
    div.style.display = 'block';
    refreshJobLog(jobId);
  } else {
    div.style.display = 'none';
  }
}

async function refreshJobLog(jobId) {
  const r = await fetch(`/api/jobs/${jobId}/log?tail=120`);
  const data = await r.json();
  const c = document.getElementById('log-content-' + jobId);
  if (c) c.textContent = data.lines.join('\n') || '(无日志)';
  // 如果 running 持续刷
  if (data.status === 'running') {
    setTimeout(() => refreshJobLog(jobId), 2000);
  }
}
</script>
</body></html>
"""


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    app = create_app()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
