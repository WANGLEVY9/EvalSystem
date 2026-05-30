"""
评估器可靠性回测 (v3.0 学术级证明)

用法:
    export DEEPSEEK_API_KEY=...
    python tests/run_calibration.py
    python tests/run_calibration.py --self-consistency 3       # 多次评估
    python tests/run_calibration.py --output reports/calibration.json

产出:
    output/calibration_<ts>.json   完整回测数据
    output/calibration_<ts>.html   可视化对比报告
    控制台输出 MAE / Pearson-r / Top-line 指标

回测内容:
- 8 个人工标注 case (tests/calibration_set.json), 涵盖:
  · 满分配合 / 严重违反字数 / 禁词 / critical 约束 / 知识幻觉
  · 忙碌用户 / 抗拒用户 / 红队 prompt-injection
- 评估器对每个 case 的预制对话打分, 与人工标注对比
- 计算: MAE, RMSE, Pearson-r, 维度级误差

合格阈值:
- 总分 MAE < 10
- 总分 Pearson-r > 0.85
- 关键维度 (task/constraint/knowledge) MAE < 12
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from src.evaluator import Evaluator
from src.instruction_parser import InstructionParser
from src.models import (
    EVALUATOR_VERSION,
    DialogueRole,
    DialogueSession,
    DialogueTurn,
    PersonaType,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("calibration")


# ============ 工具 ============

def pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def make_session(case: dict, instruction) -> DialogueSession:
    persona = PersonaType(case["persona_type"])
    sess = DialogueSession(task_instruction=instruction, persona_type=persona, seed=42)
    for i, turn in enumerate(case["dialogue"]):
        sess.turns.append(DialogueTurn(
            turn_id=i,
            role=DialogueRole.SYSTEM if turn["role"] == "system" else DialogueRole.USER,
            content=turn["content"],
            char_count=len(turn["content"]),
        ))
    sess.terminated_reason = "回测"
    sess.end_time = datetime.now()
    return sess


# ============ 主流程 ============

def run_calibration(
    cases_path: str,
    self_consistency: int,
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    with open(cases_path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"]
    logger.info(f"加载 {len(cases)} 个校准 case")

    llm_client = OpenAI(api_key=api_key, base_url=base_url)
    parser = InstructionParser(llm_client=llm_client, model=model)
    evaluator = Evaluator(
        llm_client=llm_client, model=model,
        n_self_consistency=self_consistency,
    )

    # 缓存指令解析结果 (同一 instruction_path 复用)
    instruction_cache: dict[str, Any] = {}

    results: list[dict[str, Any]] = []

    for case in cases:
        case_id = case["case_id"]
        ipath = case["instruction_path"]
        logger.info(f"--- {case_id} ({case['description']}) ---")

        if ipath not in instruction_cache:
            instruction_cache[ipath] = parser.parse_from_file(ipath)
        instruction = instruction_cache[ipath]

        sess = make_session(case, instruction)
        # 替换变量 (与对话引擎一致)
        # 注: 我们的标注对话已用 default 值, 跳过替换以保留原文
        # 但 evaluator 会用 session.task_instruction, 所以需要也提供 resolved
        # 这里直接评估 — instruction.knowledge_base 中的 X/Y 占位符靠 prompt 修复处理
        ev = evaluator.evaluate_session(sess, instruction)

        gt = case["ground_truth"]
        gt_total = gt["total_score"]
        evaluator_total = ev.total_score
        diff = abs(gt_total - evaluator_total)
        within = diff <= gt.get("tolerance", 12)

        # 维度级误差
        dim_diffs = {}
        for d in ev.dimension_scores:
            if d.dimension_key in gt:
                dim_diffs[d.dimension_key] = {
                    "gt": gt[d.dimension_key],
                    "evaluator": d.score,
                    "diff": abs(gt[d.dimension_key] - d.score),
                }

        results.append({
            "case_id": case_id,
            "description": case["description"],
            "persona": case["persona_type"],
            "gt_total": gt_total,
            "tolerance": gt.get("tolerance", 12),
            "evaluator_total": evaluator_total,
            "evaluator_confidence": ev.overall_confidence,
            "diff": diff,
            "within_tolerance": within,
            "dim_diffs": dim_diffs,
            "expected_issues": gt.get("expected_issues", []),
            "actual_issues": ev.issues,
            "expected_violations": gt.get("expected_violations", []),
            "actual_constraint_violations": [
                f"{c.constraint_name} (x{c.violation_count})"
                for c in ev.constraint_evaluations if c.violated
            ],
            "actual_utterance_violations": [
                f"{v.get('violation_type')}"
                for v in (ev.utterance_constraint_eval.violations if ev.utterance_constraint_eval else [])
            ],
        })
        logger.info(f"   GT={gt_total} eval={evaluator_total} diff={diff:.1f} {'✓' if within else '✗'}")

    # 汇总
    gt_scores = [r["gt_total"] for r in results]
    ev_scores = [r["evaluator_total"] for r in results]
    diffs = [r["diff"] for r in results]
    mae = statistics.mean(diffs)
    rmse = math.sqrt(statistics.mean(d ** 2 for d in diffs))
    pcc = pearson_r(gt_scores, ev_scores)
    within_rate = sum(1 for r in results if r["within_tolerance"]) / len(results)

    # 维度级聚合
    per_dim: dict[str, list[float]] = {}
    for r in results:
        for k, v in r["dim_diffs"].items():
            per_dim.setdefault(k, []).append(v["diff"])
    dim_mae = {k: round(statistics.mean(v), 2) for k, v in per_dim.items()}

    summary = {
        "n_cases": len(results),
        "self_consistency": self_consistency,
        "evaluator_version": EVALUATOR_VERSION,
        "model": model,
        "metrics": {
            "total_score_mae": round(mae, 2),
            "total_score_rmse": round(rmse, 2),
            "total_score_pearson_r": round(pcc, 3),
            "within_tolerance_rate": round(within_rate, 3),
            "dim_mae": dim_mae,
        },
        "thresholds": {
            "mae_target": 10,
            "pearson_target": 0.85,
            "within_tolerance_target": 0.75,
        },
        "passed": {
            "mae": mae < 10,
            "pearson": pcc > 0.85,
            "within_tolerance": within_rate >= 0.75,
        },
        "results": results,
    }
    return summary


def gen_calibration_html(summary: dict, path: Path):
    s = summary["metrics"]
    p = summary["passed"]
    rows = []
    rows.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Calibration Report</title>")
    rows.append("""<style>
body{font-family:-apple-system,sans-serif;background:#f3f5f9;margin:0;padding:24px;color:#222}
.container{max-width:1100px;margin:0 auto}
.hero{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:24px 28px;border-radius:14px;margin-bottom:18px}
.hero h1{margin:0 0 6px 0}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:20px}
.metric{background:#fff;border-radius:10px;padding:18px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.05)}
.metric .v{font-size:28px;font-weight:700;color:#1890ff}
.metric .v.pass{color:#52c41a}.metric .v.fail{color:#f5222d}
.metric .l{font-size:11px;color:#666;margin-top:4px}
.card{background:#fff;border-radius:12px;padding:18px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,.05)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 11px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
th{background:#fafbfd;font-weight:600;color:#666}
.ok{color:#52c41a}.fail{color:#f5222d}.warn{color:#fa8c16}
.diff-bar{display:inline-block;width:60px;height:10px;background:#eef0f4;border-radius:5px;overflow:hidden;vertical-align:middle;margin-right:5px}
.diff-bar > .fill{height:100%}
</style></head><body><div class='container'>""")
    rows.append(f"<div class='hero'><h1>🔬 评估器校准报告</h1><div>{summary['n_cases']} 个人工标注 case · 模型 {summary['model']} · self-consistency N={summary['self_consistency']}</div></div>")

    # 关键指标
    rows.append("<div class='metrics'>")
    rows.append(f"""<div class='metric'>
        <div class='v {"pass" if p["mae"] else "fail"}'>{s["total_score_mae"]:.2f}</div>
        <div class='l'>总分 MAE (目标 &lt; {summary["thresholds"]["mae_target"]})</div>
    </div>""")
    rows.append(f"""<div class='metric'>
        <div class='v'>{s["total_score_rmse"]:.2f}</div>
        <div class='l'>总分 RMSE</div>
    </div>""")
    rows.append(f"""<div class='metric'>
        <div class='v {"pass" if p["pearson"] else "fail"}'>{s["total_score_pearson_r"]:.3f}</div>
        <div class='l'>Pearson-r (目标 &gt; {summary["thresholds"]["pearson_target"]})</div>
    </div>""")
    rows.append(f"""<div class='metric'>
        <div class='v {"pass" if p["within_tolerance"] else "fail"}'>{s["within_tolerance_rate"]*100:.0f}%</div>
        <div class='l'>容差内 (目标 ≥ {summary["thresholds"]["within_tolerance_target"]*100:.0f}%)</div>
    </div>""")
    rows.append("</div>")

    # 通过/失败汇总
    all_pass = p["mae"] and p["pearson"] and p["within_tolerance"]
    if all_pass:
        rows.append("<div class='card'><h2 style='color:#52c41a'>✅ 评估器校准通过</h2>"
                    "<p>所有关键指标达标, 评估器结果与人工标注高度一致, 可信度高。</p></div>")
    else:
        rows.append("<div class='card'><h2 style='color:#fa8c16'>⚠️ 部分指标待优化</h2><ul>")
        if not p["mae"]: rows.append(f"<li>MAE {s['total_score_mae']} 超过目标 {summary['thresholds']['mae_target']}</li>")
        if not p["pearson"]: rows.append(f"<li>Pearson-r {s['total_score_pearson_r']} 低于目标 {summary['thresholds']['pearson_target']}</li>")
        if not p["within_tolerance"]: rows.append(f"<li>容差内比例 {s['within_tolerance_rate']*100:.0f}% 低于目标 {summary['thresholds']['within_tolerance_target']*100:.0f}%</li>")
        rows.append("</ul></div>")

    # 维度 MAE
    rows.append("<div class='card'><h2>各维度 MAE (越小越好)</h2><table><tr><th>维度</th><th>MAE</th><th>状态</th></tr>")
    for k, v in summary["metrics"]["dim_mae"].items():
        cls = "ok" if v < 12 else ("warn" if v < 20 else "fail")
        rows.append(f"<tr><td>{k}</td><td><strong class='{cls}'>{v}</strong></td>"
                    f"<td>{'✓ 优秀' if v < 8 else ('✓ 良好' if v < 12 else ('⚠ 一般' if v < 20 else '✗ 待改进'))}</td></tr>")
    rows.append("</table></div>")

    # 逐 case 表
    rows.append("<div class='card'><h2>逐 Case 详情</h2><table><tr><th>Case</th><th>画像</th><th>人工分</th><th>评估分</th><th>误差</th><th>容差</th><th>状态</th></tr>")
    for r in summary["results"]:
        diff_w = min(int(r["diff"] * 3), 60)
        diff_color = "#52c41a" if r["within_tolerance"] else "#f5222d"
        rows.append(f"""<tr>
            <td><strong>{r['case_id']}</strong><br><small style='color:#999'>{r['description'][:40]}...</small></td>
            <td>{r['persona']}</td>
            <td>{r['gt_total']}</td>
            <td>{r['evaluator_total']:.1f}</td>
            <td><span class='diff-bar'><span class='fill' style='width:{diff_w}px;background:{diff_color}'></span></span><strong>{r['diff']:.1f}</strong></td>
            <td>{r['tolerance']}</td>
            <td>{'<span class=ok>✓ 通过</span>' if r['within_tolerance'] else '<span class=fail>✗ 超差</span>'}</td>
        </tr>""")
    rows.append("</table></div>")

    # 每个 case 的预期 vs 实际 issue
    rows.append("<div class='card'><h2>预期 vs 实际问题/违规对比</h2>")
    for r in summary["results"]:
        rows.append(f"<h3 style='font-size:14px;margin-top:14px'>{r['case_id']}</h3>")
        rows.append("<table><tr><th>类型</th><th>预期</th><th>实际</th></tr>")
        rows.append(f"<tr><td>Issues</td><td>{'<br>'.join(r['expected_issues']) or '-'}</td>"
                    f"<td>{'<br>'.join(r['actual_issues']) or '-'}</td></tr>")
        rows.append(f"<tr><td>约束违反</td><td>{'<br>'.join(r['expected_violations']) or '-'}</td>"
                    f"<td>{'<br>'.join(r['actual_constraint_violations']) or '-'}</td></tr>")
        rows.append(f"<tr><td>话术违规</td><td>(规则检测)</td><td>{'<br>'.join(set(r['actual_utterance_violations'])) or '-'}</td></tr>")
        rows.append("</table>")
    rows.append("</div>")

    rows.append(f"<p style='text-align:center;color:#999;font-size:11px;margin-top:14px'>"
                f"Generated at {datetime.now():%Y-%m-%d %H:%M:%S} | Evaluator v{summary['evaluator_version']}</p>")
    rows.append("</div></body></html>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="tests/calibration_set.json")
    p.add_argument("--self-consistency", type=int, default=1)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--model", default="deepseek-chat")
    args = p.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("缺少 DEEPSEEK_API_KEY")
        sys.exit(1)

    t = time.time()
    summary = run_calibration(
        cases_path=args.cases,
        self_consistency=args.self_consistency,
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model=args.model,
    )
    dur = time.time() - t

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"calibration_{ts}.json"
    html_path = output_dir / f"calibration_{ts}.html"

    summary["duration_seconds"] = round(dur, 1)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    gen_calibration_html(summary, html_path)

    logger.info("=" * 60)
    logger.info(f"✅ 校准回测完成 (用时 {dur:.1f}s)")
    logger.info(f"   总分 MAE: {summary['metrics']['total_score_mae']} (目标 <10) {'✓' if summary['passed']['mae'] else '✗'}")
    logger.info(f"   Pearson-r: {summary['metrics']['total_score_pearson_r']} (目标 >0.85) {'✓' if summary['passed']['pearson'] else '✗'}")
    logger.info(f"   容差内比例: {summary['metrics']['within_tolerance_rate']*100:.1f}% (目标 ≥75%) {'✓' if summary['passed']['within_tolerance'] else '✗'}")
    logger.info(f"   各维度 MAE: {summary['metrics']['dim_mae']}")
    logger.info(f"   报告: {html_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
