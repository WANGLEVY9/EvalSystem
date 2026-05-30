"""
PDF 报告导出 (v3.0)

设计:
- 用 weasyprint 把 HTML 渲染成 PDF
- 由于 weasyprint 不支持 Canvas, 我们用 SVG 重画雷达图与热力图
- 移除可点击 tab, 改为线性铺开 (评委友好)
- 会话原文限制只展示首会话, 但保留所有 quote 证据

用法:
    from src.pdf_exporter import generate_pdf_report
    generate_pdf_report(report, "output/report.pdf")
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    BranchCoverageMatrix,
    DialogueRole,
    EvaluationReport,
    PersonaType,
)

logger = logging.getLogger(__name__)


PERSONA_NAMES_CN = {
    "cooperative": "配合型", "hesitant": "犹豫型",
    "resistant": "抗拒型", "off_topic": "跑题型",
    "contradictory": "矛盾型", "boundary": "边界型",
    "busy": "忙碌型", "confused": "困惑型", "impatient": "急躁型",
    "red_team_l1": "红队-轻度", "red_team_l2": "红队-社工",
    "red_team_l3": "红队-注入",
}

EVAL_METHOD_LABELS = {
    "rule": "规则确定性",
    "llm_judge": "LLM 语义判定",
    "hybrid": "规则+LLM 混合",
    "heuristic": "启发式",
}


# ============ SVG 绘图工具 ============

def _radar_svg(dims: list[tuple[str, float]], width: int = 400, height: int = 380) -> str:
    """绘制雷达图 SVG"""
    if not dims:
        return ""
    cx, cy, R = width / 2, height / 2, width * 0.36
    n = len(dims)
    step = 2 * math.pi / n

    pts: list[str] = []
    label_svgs: list[str] = []
    for i, (name, score) in enumerate(dims):
        angle = i * step - math.pi / 2
        r = R * score / 100.0
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        pts.append(f"{x:.1f},{y:.1f}")
        # 标签
        lr = R + 28
        lx = cx + lr * math.cos(angle)
        ly = cy + lr * math.sin(angle)
        # 在标签下方加一个分数
        sx = cx + (R + 12) * math.cos(angle)
        sy = cy + (R + 12) * math.sin(angle)
        anchor = "middle"
        if math.cos(angle) > 0.3:
            anchor = "start"
        elif math.cos(angle) < -0.3:
            anchor = "end"
        label_svgs.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="11" fill="#333">{name}</text>'
        )

    # 多层网格
    grid_polys: list[str] = []
    for lvl in range(1, 6):
        gpts = []
        for i in range(n + 1):
            angle = (i % n) * step - math.pi / 2
            r = R * lvl / 5.0
            gpts.append(f"{cx + r*math.cos(angle):.1f},{cy + r*math.sin(angle):.1f}")
        grid_polys.append(f'<polyline points="{" ".join(gpts)}" fill="none" stroke="#e8eaf0" stroke-width="1"/>')

    axis_lines: list[str] = []
    for i in range(n):
        angle = i * step - math.pi / 2
        x2 = cx + R * math.cos(angle)
        y2 = cy + R * math.sin(angle)
        axis_lines.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#dcdfe6" stroke-width="0.8"/>')

    poly_pts = " ".join(pts + [pts[0]])
    dots: list[str] = []
    for i, (name, score) in enumerate(dims):
        angle = i * step - math.pi / 2
        r = R * score / 100.0
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        color = "#52c41a" if score >= 80 else ("#1890ff" if score >= 60 else ("#faad14" if score >= 40 else "#f5222d"))
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')

    return f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>
        {''.join(grid_polys)}
        {''.join(axis_lines)}
        <polygon points='{poly_pts}' fill='rgba(102,126,234,0.22)' stroke='#667eea' stroke-width='2'/>
        {''.join(dots)}
        {''.join(label_svgs)}
    </svg>"""


def _heatmap_svg(personas: list[str], dims: list[str], data: list[tuple[int, int, float]],
                 width: int = 800, height: int = 320) -> str:
    """绘制热力图 SVG"""
    if not personas or not dims or not data:
        return ""
    pad_l, pad_t, pad_r, pad_b = 110, 50, 16, 22
    gw = width - pad_l - pad_r
    gh = height - pad_t - pad_b
    cell_w = gw / len(dims)
    cell_h = gh / len(personas)

    def color(v: float) -> str:
        if v == 0: return "#f0f0f0"
        if v >= 80: return "#52c41a"
        if v >= 70: return "#73d13d"
        if v >= 60: return "#1890ff"
        if v >= 50: return "#fadb14"
        if v >= 40: return "#faad14"
        if v >= 30: return "#fa541c"
        return "#f5222d"

    cells: list[str] = []
    for pi, di, v in data:
        x = pad_l + di * cell_w + 1
        y = pad_t + pi * cell_h + 1
        w = cell_w - 2
        h = cell_h - 2
        text_color = "#fff" if v >= 60 else "#333"
        cells.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{color(v)}" rx="2"/>'
            f'<text x="{x + w/2:.1f}" y="{y + h/2 + 4:.1f}" text-anchor="middle" font-size="11" fill="{text_color}">{v:.0f}</text>'
        )

    # x 标签
    x_labels: list[str] = []
    for i, d in enumerate(dims):
        cx = pad_l + i * cell_w + cell_w / 2
        cy = pad_t - 8
        x_labels.append(
            f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="10" text-anchor="middle" fill="#444"'
            f' transform="rotate(-15 {cx:.1f} {cy:.1f})">{d}</text>'
        )

    # y 标签
    y_labels: list[str] = []
    for i, p in enumerate(personas):
        y_labels.append(
            f'<text x="{pad_l - 8}" y="{pad_t + i * cell_h + cell_h/2 + 4:.1f}" font-size="11" text-anchor="end" fill="#333">{p}</text>'
        )

    return f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>
        {''.join(cells)}
        {''.join(x_labels)}
        {''.join(y_labels)}
    </svg>"""


def _bar_svg(score: float, width: int = 160, height: int = 14) -> str:
    """绘制水平条形图 (用于维度行)"""
    color = "#52c41a" if score >= 80 else ("#1890ff" if score >= 60 else ("#faad14" if score >= 40 else "#f5222d"))
    fill_w = max(0, min(width, score / 100 * width))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<rect width="{width}" height="{height}" fill="#eef0f4" rx="{height/2}"/>'
            f'<rect width="{fill_w:.1f}" height="{height}" fill="{color}" rx="{height/2}"/>'
            f'</svg>')


# ============ HTML 模板 (PDF 优化) ============

PDF_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>{task_name}</title>
<style>
@page {{ size: A4; margin: 1.5cm 1.5cm; @bottom-center {{ content: counter(page) " / " counter(pages); font-size: 9pt; color: #999; }} }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; color:#222; font-size:11pt; line-height:1.55; }}
h1 {{ font-size:22pt; margin-bottom:6pt; }}
h2 {{ font-size:14pt; margin: 14pt 0 8pt 0; padding-bottom:6pt; border-bottom: 2px solid #f0f2f7; color:#333; }}
h3 {{ font-size:11pt; margin: 8pt 0 4pt 0; }}
.hero {{ background:#667eea; color:#fff; padding:18pt 22pt; border-radius:8pt; margin-bottom:14pt; page-break-inside:avoid; }}
.hero .meta {{ opacity:0.92; font-size:10pt; margin-top:4pt; }}
.tag {{ display:inline-block; background:rgba(255,255,255,0.18); padding:2pt 7pt; border-radius:8pt; font-size:8pt; margin:3pt 4pt 0 0; }}
.stats {{ display: table; width:100%; margin-bottom:12pt; }}
.stats .row {{ display:table-row; }}
.stats .cell {{ display:table-cell; text-align:center; padding:8pt; background:#fafbfd; border-radius:6pt; width:25%; }}
.stats .cell .v {{ font-size:18pt; font-weight:700; color:#1890ff; }}
.stats .cell .l {{ font-size:8pt; color:#666; margin-top:2pt; }}
.score-excellent {{ color:#52c41a; }} .score-good {{ color:#1890ff; }} .score-fair {{ color:#faad14; }} .score-poor {{ color:#f5222d; }}
table {{ width:100%; border-collapse:collapse; font-size:9pt; margin-top:5pt; }}
th, td {{ padding:5pt 8pt; border-bottom:1px solid #eee; text-align:left; vertical-align:top; }}
th {{ background:#fafbfd; font-weight:600; color:#666; font-size:9pt; }}
.dim-row td {{ padding:6pt 8pt; }}
.method {{ display:inline-block; padding:1pt 5pt; border-radius:6pt; font-size:8pt; }}
.method-rule {{ background:#e6f7ff; color:#0958d9; }}
.method-llm_judge {{ background:#f9f0ff; color:#722ed1; }}
.method-hybrid {{ background:#fff7e6; color:#d46b08; }}
.method-heuristic {{ background:#f6ffed; color:#389e0d; }}
.fm {{ background:#fff7e6; border-left:3px solid #ffd591; padding:7pt 10pt; margin:5pt 0; border-radius:4pt; font-size:9.5pt; page-break-inside:avoid; }}
.fm .name {{ font-weight:600; color:#d46b08; }}
.fm .quote {{ background:#fff; padding:3pt 6pt; margin:3pt 0; font-style:italic; color:#555; border-radius:3pt; font-size:9pt; }}
.fm .suggestion {{ font-size:8.5pt; color:#666; margin-top:2pt; }}
.cat {{ display:inline-block; background:#ff7a45; color:#fff; font-size:8pt; padding:1pt 5pt; border-radius:4pt; margin-right:4pt; }}
.branch-line {{ font-size:9pt; padding:2pt 0; }}
.branch-line .dot {{ display:inline-block; width:7pt; height:7pt; border-radius:50%; margin-right:6pt; vertical-align:middle; }}
.branch-covered .dot {{ background:#52c41a; }} .branch-uncovered .dot {{ background:#ccc; }}
.branch-uncovered {{ color:#999; }}
.summary {{ background:#fafbfd; padding:10pt 14pt; border-radius:5pt; white-space:pre-line; line-height:1.8; font-size:10pt; }}
.dialogue {{ background:#fafbfd; padding:8pt; border-radius:5pt; font-size:9pt; }}
.turn {{ margin:3pt 0; padding:4pt 8pt; border-radius:5pt; }}
.turn-system {{ background:#e6f7ff; border-left:2px solid #91d5ff; }}
.turn-user {{ background:#f6ffed; border-left:2px solid #b7eb8f; }}
.turn-violation {{ background:#fff2f0 !important; border-left-color:#ff7875 !important; }}
.turn-label {{ font-size:8pt; color:#999; margin-bottom:2pt; }}
.rec {{ background:#fafbfd; border-left:3px solid #1890ff; padding:5pt 10pt; margin:3pt 0; border-radius:3pt; font-size:9pt; }}
.rec.urgent {{ border-left-color:#f5222d; background:#fff2f0; }}
.rec.failure {{ border-left-color:#fa541c; background:#fff2e8; }}
.meta-grid {{ display:table; width:100%; }}
.meta-cell {{ display:table-cell; padding:3pt 8pt; font-size:8pt; vertical-align:top; }}
.meta-cell .l {{ color:#888; }}
.meta-cell .v {{ font-family:"Menlo","Courier New",monospace; word-break:break-all; }}
.col-2 {{ display:flex; gap:12pt; }}
.col-2 > * {{ flex:1; }}
.page-break {{ page-break-before:always; }}
.no-break {{ page-break-inside:avoid; }}
.persona-grid {{ display:table; width:100%; border-collapse:separate; border-spacing:5pt; }}
.persona-grid .item {{ display:table-cell; text-align:center; padding:8pt; background:#fafbfd; border-radius:5pt; vertical-align:middle; width:120pt; }}
.persona-grid .item .v {{ font-size:14pt; font-weight:700; }}
.persona-grid .item .l {{ font-size:9pt; color:#666; margin-bottom:2pt; }}
</style>
</head>
<body>

<!-- 1. HERO -->
<div class="hero">
  <h1>📊 多轮对话评测报告 v3.0</h1>
  <div class="meta">任务: <strong>{task_name}</strong></div>
  <div class="meta">{task_description_short}</div>
  <div class="meta">生成时间: {generated_at} | 测试会话数: {total_sessions}</div>
  <div style="margin-top:6pt">
    {complexity_tags}
    {run_tags}
  </div>
</div>

<!-- 2. 概览 -->
<div class="stats">
  <div class="row">
    <div class="cell"><div class="v {score_class}">{overall_score:.1f}</div><div class="l">总体得分</div></div>
    <div class="cell"><div class="v">{total_sessions}</div><div class="l">测试会话数</div></div>
    <div class="cell"><div class="v">{n_dims}</div><div class="l">评估维度</div></div>
    <div class="cell"><div class="v">{n_personas}</div><div class="l">用户画像</div></div>
  </div>
</div>

<!-- 3. 总分 + 摘要 -->
<div class="col-2">
  <div>
    <h2>🎯 总分</h2>
    <div style="text-align:center;font-size:48pt;font-weight:700;line-height:1" class="{score_class}">{overall_score:.1f}</div>
    <div style="text-align:center;font-size:9pt;color:#999;margin-top:3pt">{score_level} | ±{overall_score_std:.1f} | 置信 {overall_confidence:.2f}</div>
  </div>
  <div>
    <h2>📋 评测摘要</h2>
    <div class="summary">{summary}</div>
  </div>
</div>

<!-- 4. 雷达图 -->
<h2>📐 多维度评分</h2>
<div style="text-align:center;margin:6pt 0">{radar_svg}</div>

<table>
<tr><th>维度</th><th>得分</th><th>分布</th><th>权重</th><th>置信度</th><th>判定方式</th></tr>
{dim_rows}
</table>

<div class="page-break"></div>

<!-- 5. 画像表现 -->
<h2>👥 用户画像表现</h2>
<div class="persona-grid"><div class="row" style="display:table-row">{persona_cells}</div></div>

<h3 style="margin-top:14pt">🔥 画像 × 维度 表现热力图</h3>
<div style="text-align:center">{heatmap_svg}</div>

<!-- 6. 短板诊断 -->
<h2>🩺 模型短板诊断</h2>
<div class="col-2">
  <div>
    <h3 style="color:#f5222d">🔻 最弱维度</h3>
    <ul style="margin-left:14pt;font-size:9.5pt">{weakest_dims}</ul>
  </div>
  <div>
    <h3 style="color:#52c41a">🔺 最强维度</h3>
    <ul style="margin-left:14pt;font-size:9.5pt">{strongest_dims}</ul>
  </div>
</div>
<p style="font-size:9.5pt;margin:5pt 0"><strong>风险总览:</strong> {risk_summary}</p>

<h3 style="margin-top:10pt">⚡ Top 失败模式</h3>
{failure_modes}

<!-- 7. 分支覆盖 -->
{branch_section}

<div class="page-break"></div>

<!-- 8. 改进建议 -->
<h2>💡 改进建议</h2>
{recommendations}

<!-- 9. 复算信息 -->
{run_metadata_section}

<!-- 10. 示例对话 (头 1-2 个会话) -->
<h2>💬 示例对话节选</h2>
{sample_dialogues}

<p style="text-align:center;color:#999;font-size:8pt;margin-top:14pt">
  评测系统 v3.0 | 报告 ID {report_id_short} | 📐可解释 · 📊可量化 · 🎯可复算
</p>

</body></html>"""


# ============ 主函数 ============

def generate_pdf_report(report: EvaluationReport, output_path: Path):
    """生成 PDF 报告"""
    # macOS 下自动加 Homebrew 库路径
    import os, platform
    if platform.system() == "Darwin":
        brew_lib = "/opt/homebrew/lib"
        if Path(brew_lib).exists():
            cur = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            if brew_lib not in cur:
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (cur + ":" + brew_lib).strip(":")

    try:
        from weasyprint import HTML
    except (ImportError, OSError) as e:
        raise ImportError(
            "PDF 导出需要 weasyprint 和系统库。请安装:\n"
            "  pip install weasyprint\n"
            "  macOS: brew install pango glib cairo gdk-pixbuf libffi\n"
            "  Linux: apt-get install libpango-1.0-0 libpangoft2-1.0-0\n"
            f"原始错误: {e}"
        ) from e

    html_str = _build_pdf_html(report)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_str).write_pdf(str(output_path))
    logger.info(f"PDF 报告: {output_path}")


def _build_pdf_html(report: EvaluationReport) -> str:
    score_class = "score-excellent" if report.overall_score >= 90 else (
        "score-good" if report.overall_score >= 75 else (
            "score-fair" if report.overall_score >= 60 else "score-poor"))
    score_level = ("优秀" if report.overall_score >= 90 else
                   "良好" if report.overall_score >= 75 else
                   "合格" if report.overall_score >= 60 else "待改进")

    # 复杂度 tags
    complexity_tags = ""
    if report.instruction_complexity:
        c = report.instruction_complexity
        complexity_tags = (
            f'<span class="tag">复杂度 {int(c.complexity_score)} ({c.complexity_level})</span>'
            f'<span class="tag">流程 {c.n_flow_nodes}</span>'
            f'<span class="tag">分支 {c.n_branches}</span>'
            f'<span class="tag">约束 {c.n_constraints}</span>'
        )
    run_tags = ""
    if report.run_metadata:
        m = report.run_metadata
        run_tags = (
            f'<span class="tag">SC=N{m.self_consistency_n}</span>'
            f'<span class="tag">seed={m.seed}</span>'
            f'<span class="tag">{m.duration_seconds:.0f}s</span>'
        )

    # 雷达 SVG
    dim_pairs = [(d.dimension_name, d.score) for d in report.dimension_averages]
    radar_svg = _radar_svg(dim_pairs, 460, 380)

    # 维度表
    dim_rows_list = []
    for d in sorted(report.dimension_averages, key=lambda x: -x.weight):
        score_color = "#52c41a" if d.score >= 80 else ("#1890ff" if d.score >= 60 else ("#faad14" if d.score >= 40 else "#f5222d"))
        std_str = f' ±{d.score_std:.1f}' if d.score_std else ''
        method_cn = EVAL_METHOD_LABELS.get(d.evaluation_method.value, d.evaluation_method.value)
        dim_rows_list.append(
            f'<tr class="dim-row">'
            f'<td><strong>{d.dimension_name}</strong></td>'
            f'<td style="color:{score_color};font-weight:700">{d.score:.1f}{std_str}</td>'
            f'<td>{_bar_svg(d.score, 130, 12)}</td>'
            f'<td>{d.weight*100:.0f}%</td>'
            f'<td>{d.confidence:.2f}</td>'
            f'<td><span class="method method-{d.evaluation_method.value}">{method_cn}</span></td>'
            f'</tr>'
        )

    # 画像 cells
    persona_cells = []
    for p, s in sorted(report.persona_scores.items(), key=lambda x: -x[1]):
        color = "#52c41a" if s >= 80 else ("#1890ff" if s >= 60 else ("#faad14" if s >= 40 else "#f5222d"))
        persona_cells.append(
            f'<div class="item"><div class="l">{PERSONA_NAMES_CN.get(p, p)}</div>'
            f'<div class="v" style="color:{color}">{s:.1f}</div></div>'
        )

    # 热力图 (画像 × 维度)
    from collections import defaultdict
    persona_dim_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ev in report.session_evaluations:
        for d in ev.dimension_scores:
            persona_dim_scores[ev.persona_type.value][d.dimension_key].append(d.score)

    heat_personas = [PERSONA_NAMES_CN.get(p, p) for p in sorted(report.persona_scores.keys(), key=lambda x: -report.persona_scores[x])]
    heat_persona_keys = sorted(report.persona_scores.keys(), key=lambda x: -report.persona_scores[x])
    heat_dims = [d.dimension_name for d in report.dimension_averages]
    heat_dim_keys = [d.dimension_key for d in report.dimension_averages]
    heat_data: list[tuple[int, int, float]] = []
    for pi, pk in enumerate(heat_persona_keys):
        for di, dk in enumerate(heat_dim_keys):
            vals = persona_dim_scores.get(pk, {}).get(dk, [])
            avg = (sum(vals) / len(vals)) if vals else 0.0
            heat_data.append((pi, di, avg))
    heatmap_svg = _heatmap_svg(heat_personas, heat_dims, heat_data, 760, max(180, 40 * len(heat_personas) + 70))

    # 短板诊断
    weakest_html = ""
    strongest_html = ""
    risk_summary = ""
    fm_html = ""
    if report.weakness_profile:
        wp = report.weakness_profile
        weakest_html = "".join(f"<li>{w}</li>" for w in wp.weakest_dimensions)
        strongest_html = "".join(f"<li>{w}</li>" for w in wp.strongest_dimensions)
        risk_summary = wp.risk_summary
        for fm in wp.top_failure_modes[:6]:
            quote_html = f'<div class="quote">"{fm.typical_quote[:100]}"</div>' if fm.typical_quote else ""
            sugg_html = f'<div class="suggestion">💡 {fm.suggestion}</div>' if fm.suggestion else ""
            fm_html += (
                f'<div class="fm"><div><span class="cat">{fm.category}</span>'
                f'<span class="name">{fm.name}</span> '
                f'<span style="color:#999;font-size:8pt">x{fm.occurrences} · 影响 {fm.impact_score:.1f}</span></div>'
                f'{quote_html}{sugg_html}</div>'
            )

    # 分支覆盖
    branch_section = ""
    if report.branch_coverage and report.branch_coverage.total_branches > 0:
        bc = report.branch_coverage
        bar_w = int(bc.coverage_rate * 100)
        branch_lines = []
        for tb in bc.target_branches_to_cover:
            covered = tb not in bc.uncovered_branches
            branch_lines.append(
                f'<div class="branch-line branch-{"covered" if covered else "uncovered"}">'
                f'<span class="dot"></span><span>{tb}</span></div>'
            )
        branch_section = (
            f'<h2>🌳 分支覆盖率</h2>'
            f'<div style="display:flex;align-items:center;gap:10pt;margin-bottom:8pt">'
            f'<div style="flex:1;height:14pt;background:#eef0f4;border-radius:7pt;overflow:hidden">'
            f'<div style="height:100%;background:linear-gradient(90deg,#52c41a,#73d13d);width:{bar_w}%"></div></div>'
            f'<div style="font-weight:700">{bc.covered_branches}/{bc.total_branches} ({int(bc.coverage_rate*100)}%)</div>'
            f'</div>'
            + "".join(branch_lines)
        )

    # 改进建议
    recs_html = ""
    for r in report.recommendations:
        cls = "urgent" if "紧急" in r else ("failure" if "失败模式" in r else "")
        recs_html += f'<div class="rec {cls}">{r}</div>'

    # 复算信息
    run_metadata_section = ""
    if report.run_metadata:
        m = report.run_metadata
        run_metadata_section = f"""
<h2>🔬 复算信息 (Reproducibility)</h2>
<div class="meta-grid">
  <div class="meta-cell"><div class="l">run_id</div><div class="v">{m.run_id}</div></div>
  <div class="meta-cell"><div class="l">evaluator</div><div class="v">{m.evaluator_version}</div></div>
  <div class="meta-cell"><div class="l">target_model</div><div class="v">{m.target_model_id}</div></div>
  <div class="meta-cell"><div class="l">judge_model</div><div class="v">{m.judge_model_id}</div></div>
</div>
<div class="meta-grid" style="margin-top:3pt">
  <div class="meta-cell"><div class="l">seed</div><div class="v">{m.seed}</div></div>
  <div class="meta-cell"><div class="l">SC N</div><div class="v">{m.self_consistency_n}</div></div>
  <div class="meta-cell"><div class="l">concurrency</div><div class="v">{m.concurrency}</div></div>
  <div class="meta-cell"><div class="l">duration</div><div class="v">{m.duration_seconds:.1f} s</div></div>
</div>"""

    # 示例对话: 取前 1-2 个 session 简要展示
    dialogue_html = ""
    n_show = min(2, len(report.dialogue_sessions))
    for idx in range(n_show):
        sess = report.dialogue_sessions[idx]
        ev = report.session_evaluations[idx]
        violated_turns: set[int] = set()
        if ev.utterance_constraint_eval:
            violated_turns.update(int(v.get("turn_id", -1)) for v in ev.utterance_constraint_eval.violations)
        for c in ev.constraint_evaluations:
            if c.violated:
                violated_turns.update(q.turn_id for q in c.quotes)

        dialogue_html += (
            f'<h3 style="margin-top:10pt">会话 #{idx + 1}: {PERSONA_NAMES_CN.get(sess.persona_type.value, sess.persona_type.value)} '
            f'(总分 {ev.total_score:.1f})</h3>'
            f'<div class="dialogue">'
        )
        for t in sess.turns[:14]:  # 限长
            cls_v = " turn-violation" if (t.role == DialogueRole.SYSTEM and t.turn_id in violated_turns) else ""
            role_label = "🤖 系统" if t.role == DialogueRole.SYSTEM else "👤 用户"
            content = (t.content[:140] + "...") if len(t.content) > 140 else t.content
            dialogue_html += (
                f'<div class="turn turn-{t.role.value}{cls_v}">'
                f'<div class="turn-label">[轮{t.turn_id}] {role_label}</div>'
                f'{content}</div>'
            )
        dialogue_html += '</div>'

    return PDF_HTML_TEMPLATE.format(
        task_name=_html_escape(report.task_name),
        task_description_short=_html_escape((report.task_description or "")[:80]),
        generated_at=report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        total_sessions=report.total_sessions,
        complexity_tags=complexity_tags,
        run_tags=run_tags,
        score_class=score_class,
        overall_score=report.overall_score,
        score_level=score_level,
        overall_score_std=report.overall_score_std,
        overall_confidence=report.overall_confidence,
        n_dims=len(report.dimension_averages),
        n_personas=len(report.persona_scores),
        summary=_html_escape(report.summary),
        radar_svg=radar_svg,
        dim_rows="".join(dim_rows_list),
        persona_cells="".join(persona_cells),
        heatmap_svg=heatmap_svg,
        weakest_dims=weakest_html,
        strongest_dims=strongest_html,
        risk_summary=_html_escape(risk_summary),
        failure_modes=fm_html,
        branch_section=branch_section,
        recommendations=recs_html,
        run_metadata_section=run_metadata_section,
        sample_dialogues=dialogue_html,
        report_id_short=report.report_id[:8],
    )


def _html_escape(s: str) -> str:
    if not s:
        return ""
    return (s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))
