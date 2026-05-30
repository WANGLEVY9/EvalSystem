"""
报告生成器 (v3.0)

核心创新点 C2 + C6 的展示层:
1. 交互式 HTML - 雷达图、画像热力图、分支覆盖图、可点击对话浏览器、quote 高亮
2. 对话级深入 - 每个会话独立 tab, 系统话语显示「字数标签 + 违规标记」, quote 高亮
3. 短板诊断卡 - Top-3 失败模式 + 可点击跳转
4. 复算信息卡 - run_id, seed, 模型版本, evaluator 版本, 各 dim 评估方法
5. 对比模式 - 同一 instruction 多模型并排报告
6. JSON / Markdown 摘要 - 多格式导出

使用纯 vanilla JS + Canvas, 不依赖外部 CDN, 完全离线可用。
"""

from __future__ import annotations

import html
import json
import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from jinja2 import Template

from .models import (
    BranchCoverageMatrix,
    DialogueRole,
    DialogueSession,
    DimensionScore,
    EvaluationMethod,
    EvaluationReport,
    EvidenceQuote,
    FailureMode,
    InstructionComplexity,
    ModelWeaknessProfile,
    PersonaType,
    RunMetadata,
    SessionEvaluation,
    TaskInstruction,
)
from .weakness_analyzer import WeaknessAnalyzer

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


class ReportGenerator:
    """评测报告生成器 v3"""

    def __init__(
        self,
        output_dir: str = "output",
        template_path: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = template_path

    def generate_report(
        self,
        instruction: TaskInstruction,
        sessions: list[DialogueSession],
        evaluations: list[SessionEvaluation],
        generate_html: bool = True,
        generate_json: bool = True,
        generate_markdown: bool = True,
        generate_pdf: bool = False,
        run_metadata: Optional[RunMetadata] = None,
    ) -> EvaluationReport:
        logger.info("生成评测报告...")

        report = self._build_report(instruction, sessions, evaluations, run_metadata)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if generate_json:
            jp = self.output_dir / f"report_{timestamp}.json"
            self._gen_json(report, jp)
            logger.info(f"JSON 报告: {jp}")
        if generate_html:
            hp = self.output_dir / f"report_{timestamp}.html"
            self._gen_html(report, hp)
            logger.info(f"HTML 报告: {hp}")
        if generate_markdown:
            mp = self.output_dir / f"report_{timestamp}.md"
            self._gen_markdown(report, mp)
            logger.info(f"Markdown 摘要: {mp}")
        if generate_pdf:
            pp = self.output_dir / f"report_{timestamp}.pdf"
            try:
                from .pdf_exporter import generate_pdf_report
                generate_pdf_report(report, pp)
            except Exception as e:
                logger.warning(f"PDF 导出失败: {e}")

        return report

    # ============ 构建报告对象 ============

    def _build_report(
        self,
        instruction: TaskInstruction,
        sessions: list[DialogueSession],
        evaluations: list[SessionEvaluation],
        run_metadata: Optional[RunMetadata],
    ) -> EvaluationReport:
        # 总分
        if evaluations:
            total_scores = [e.total_score for e in evaluations]
            overall = statistics.mean(total_scores)
            overall_std = statistics.stdev(total_scores) if len(total_scores) > 1 else 0.0
            confidence = statistics.mean(e.overall_confidence for e in evaluations)
        else:
            overall = 0.0
            overall_std = 0.0
            confidence = 0.0

        # 维度均值
        dim_avg = self._calc_dim_averages(evaluations)
        # 画像均值
        persona_scores = self._calc_persona_scores(evaluations)
        # 分支覆盖
        branch_cov = self._calc_branch_coverage(instruction, sessions, evaluations)
        # 短板诊断
        weakness = WeaknessAnalyzer().analyze(instruction, sessions, evaluations)

        summary = self._gen_summary(overall, overall_std, dim_avg, persona_scores, weakness)
        recommendations = self._gen_recommendations(evaluations, dim_avg, instruction, weakness)

        return EvaluationReport(
            task_name=instruction.task_name,
            task_description=instruction.task_description,
            total_sessions=len(sessions),
            overall_score=round(overall, 2),
            overall_score_std=round(overall_std, 2),
            overall_confidence=round(confidence, 2),
            dimension_averages=dim_avg,
            persona_scores=persona_scores,
            session_evaluations=evaluations,
            dialogue_sessions=sessions,
            branch_coverage=branch_cov,
            summary=summary,
            recommendations=recommendations,
            instruction_complexity=instruction.complexity,
            weakness_profile=weakness,
            run_metadata=run_metadata,
        )

    @staticmethod
    def _calc_dim_averages(evaluations: list[SessionEvaluation]) -> list[DimensionScore]:
        if not evaluations:
            return []
        bucket: dict[str, list[DimensionScore]] = defaultdict(list)
        for ev in evaluations:
            for d in ev.dimension_scores:
                bucket[d.dimension_key].append(d)
        out = []
        for key, scores in bucket.items():
            avg = statistics.mean(s.score for s in scores)
            confidence = statistics.mean(s.confidence for s in scores)
            stds = [s.score_std for s in scores if s.score_std]
            std_avg = statistics.mean(stds) if stds else 0.0
            method = scores[0].evaluation_method  # 同一 key 的 method 应一致
            out.append(DimensionScore(
                dimension_key=key,
                dimension_name=scores[0].dimension_name,
                score=round(avg, 1),
                weight=scores[0].weight,
                weighted_score=round(avg * scores[0].weight, 2),
                explanation=f"基于 {len(scores)} 个会话的均值",
                evaluation_method=method,
                confidence=round(confidence, 2),
                score_std=round(std_avg, 2),
            ))
        return out

    @staticmethod
    def _calc_persona_scores(evaluations: list[SessionEvaluation]) -> dict[str, float]:
        groups: dict[str, list[float]] = defaultdict(list)
        for e in evaluations:
            groups[e.persona_type.value].append(e.total_score)
        return {k: round(statistics.mean(v), 2) for k, v in groups.items()}

    @staticmethod
    def _calc_branch_coverage(
        instruction: TaskInstruction,
        sessions: list[DialogueSession],
        evaluations: list[SessionEvaluation],
    ) -> BranchCoverageMatrix:
        # 所有目标分支
        target = []
        node_branch_map: dict[str, list[str]] = defaultdict(list)
        for n in instruction.flow_nodes:
            for b in n.branches:
                key = f"{n.name}::{b.condition}"
                target.append(key)
                node_branch_map[n.name].append(b.condition)
            for nx, c in n.conditions.items():
                if c:
                    target.append(f"{n.name}->{nx}::{c}")

        if not target:
            return BranchCoverageMatrix(coverage_rate=1.0)

        covered_count: Counter = Counter()
        # 1. session.branch_path (强制覆盖创建的会话)
        for s in sessions:
            for bp in s.branch_path:
                covered_count[bp] += 1
        # 2. evaluation.branch_coverage (LLM 推断的)
        for e in evaluations:
            for bp in e.branch_coverage:
                covered_count[bp] += 1

        covered = set()
        for bp in covered_count:
            # 模糊匹配
            for t in target:
                if t == bp or t.split("::")[1] in bp or bp in t:
                    covered.add(t)

        uncovered = [t for t in target if t not in covered]
        rate = len(covered) / len(target) if target else 1.0

        return BranchCoverageMatrix(
            total_branches=len(target),
            covered_branches=len(covered),
            coverage_rate=round(rate, 3),
            uncovered_branches=uncovered,
            branch_details=dict(covered_count),
            branch_node_map=dict(node_branch_map),
            target_branches_to_cover=target,
        )

    @staticmethod
    def _gen_summary(
        overall: float,
        std: float,
        dim_avg: list[DimensionScore],
        persona_scores: dict[str, float],
        weakness: ModelWeaknessProfile,
    ) -> str:
        if overall >= 90:
            level = "优秀"
        elif overall >= 75:
            level = "良好"
        elif overall >= 60:
            level = "合格"
        else:
            level = "待改进"

        parts = [f"总体评级: {level} ({overall:.1f}±{std:.1f}分)"]
        if dim_avg:
            best = max(dim_avg, key=lambda x: x.score)
            worst = min(dim_avg, key=lambda x: x.score)
            parts.append(f"最强维度: {best.dimension_name} ({best.score:.1f}分)")
            parts.append(f"最弱维度: {worst.dimension_name} ({worst.score:.1f}分)")
        if persona_scores:
            best_p = max(persona_scores.items(), key=lambda x: x[1])
            worst_p = min(persona_scores.items(), key=lambda x: x[1])
            parts.append(f"画像最佳: {PERSONA_NAMES_CN.get(best_p[0], best_p[0])} ({best_p[1]:.1f}分)")
            parts.append(f"画像最差: {PERSONA_NAMES_CN.get(worst_p[0], worst_p[0])} ({worst_p[1]:.1f}分)")
        if weakness.risk_summary:
            parts.append(f"风险: {weakness.risk_summary}")
        return "\n".join(parts)

    @staticmethod
    def _gen_recommendations(
        evaluations: list[SessionEvaluation],
        dim_avg: list[DimensionScore],
        instruction: TaskInstruction,
        weakness: ModelWeaknessProfile,
    ) -> list[str]:
        recs: list[str] = []

        for d in sorted(dim_avg, key=lambda x: x.score)[:3]:
            if d.score < 60:
                recs.append(f"[紧急] {d.dimension_name} 得分 {d.score:.1f}, 建议重点改进")
            elif d.score < 75:
                recs.append(f"[建议] {d.dimension_name} 得分 {d.score:.1f}, 有改进空间")

        # 失败模式 Top-3
        for fm in weakness.top_failure_modes[:3]:
            recs.append(f"[失败模式] {fm.name} (出现{fm.occurrences}次) - {fm.suggestion}")

        # 知识幻觉
        for ev in evaluations:
            kb = ev.knowledge_accuracy_eval
            if kb and kb.fabricated_info:
                recs.append(f"[知识幻觉] 发现编造信息: {'; '.join(kb.fabricated_info[:2])}")
                break

        # 话术违规
        utt_issues: list[str] = []
        for ev in evaluations:
            if ev.utterance_constraint_eval:
                utt_issues.extend(v.get("violation_type", "") for v in ev.utterance_constraint_eval.violations)
        if utt_issues:
            counter = Counter(utt_issues)
            for vt, n in counter.most_common(2):
                recs.append(f"[话术约束] {vt} 共 {n} 次违反")

        if not recs:
            recs.append("整体表现良好, 建议保持")
        return recs

    # ============ 输出 ============

    def _gen_json(self, report: EvaluationReport, path: Path):
        data = report.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def _gen_markdown(self, report: EvaluationReport, path: Path):
        """1 页评委友好的 markdown 摘要"""
        lines = []
        lines.append(f"# 评测报告: {report.task_name}")
        lines.append("")
        lines.append(f"- **生成时间**: {report.generated_at:%Y-%m-%d %H:%M:%S}")
        lines.append(f"- **总分**: {report.overall_score:.1f} ± {report.overall_score_std:.1f} (置信度 {report.overall_confidence:.2f})")
        lines.append(f"- **测试会话数**: {report.total_sessions}")
        if report.instruction_complexity:
            c = report.instruction_complexity
            lines.append(f"- **指令复杂度**: {c.complexity_score:.0f}/100 ({c.complexity_level}) — 流程节点 {c.n_flow_nodes} / 分支 {c.n_branches} / 约束 {c.n_constraints} / 知识 {c.n_knowledge_entries}")
        if report.run_metadata:
            md = report.run_metadata
            lines.append(f"- **复算元数据**: run_id={md.run_id[:8]} | evaluator={md.evaluator_version} | seed={md.seed} | self-consistency N={md.self_consistency_n}")
        lines.append("")
        lines.append("## 1. 摘要")
        lines.append("")
        lines.append("```")
        lines.append(report.summary)
        lines.append("```")
        lines.append("")
        lines.append("## 2. 维度评分")
        lines.append("")
        lines.append("| 维度 | 得分 | 权重 | 标准差 | 置信度 | 判定方式 |")
        lines.append("|---|---|---|---|---|---|")
        for d in sorted(report.dimension_averages, key=lambda x: -x.weight):
            method_cn = EVAL_METHOD_LABELS.get(d.evaluation_method.value, d.evaluation_method.value)
            lines.append(
                f"| {d.dimension_name} | {d.score:.1f} | {d.weight*100:.0f}% | "
                f"{d.score_std:.1f} | {d.confidence:.2f} | {method_cn} |"
            )
        lines.append("")
        lines.append("## 3. 画像表现")
        lines.append("")
        for p, s in sorted(report.persona_scores.items(), key=lambda x: -x[1]):
            lines.append(f"- **{PERSONA_NAMES_CN.get(p, p)}**: {s:.1f}")
        lines.append("")
        if report.branch_coverage and report.branch_coverage.total_branches:
            bc = report.branch_coverage
            lines.append("## 4. 分支覆盖")
            lines.append("")
            lines.append(f"覆盖率: {bc.coverage_rate*100:.1f}%  ({bc.covered_branches}/{bc.total_branches})")
            if bc.uncovered_branches:
                lines.append("")
                lines.append("**未覆盖**:")
                for u in bc.uncovered_branches[:8]:
                    lines.append(f"- `{u}`")
            lines.append("")
        if report.weakness_profile:
            wp = report.weakness_profile
            lines.append("## 5. 模型短板诊断")
            lines.append("")
            if wp.weakest_dimensions:
                lines.append("**最弱维度**: " + " / ".join(wp.weakest_dimensions))
            if wp.weakest_personas:
                lines.append("**最弱画像**: " + " / ".join(wp.weakest_personas))
            if wp.top_failure_modes:
                lines.append("")
                lines.append("**Top-N 失败模式**:")
                for fm in wp.top_failure_modes:
                    lines.append(f"- [{fm.category}|x{fm.occurrences}] {fm.name}")
                    if fm.typical_quote:
                        lines.append(f"  - 典型: `{fm.typical_quote[:80]}`")
                    if fm.suggestion:
                        lines.append(f"  - 建议: {fm.suggestion}")
            lines.append("")
        lines.append("## 6. 改进建议")
        lines.append("")
        for r in report.recommendations:
            lines.append(f"- {r}")
        lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _gen_html(self, report: EvaluationReport, path: Path):
        # 准备 quote 高亮的 turn 索引: 每个 session 中有违规/有 quote 的 turn_id
        session_violations: dict[str, set[int]] = {}
        session_quotes: dict[str, list[dict]] = {}

        for sess, ev in zip(report.dialogue_sessions, report.session_evaluations):
            v_turns: set[int] = set()
            quotes_info: list[dict] = []
            # utterance violations
            if ev.utterance_constraint_eval:
                for v in ev.utterance_constraint_eval.violations:
                    v_turns.add(int(v.get("turn_id", -1)))
            # constraint violations
            for c in ev.constraint_evaluations:
                if c.violated:
                    for q in c.quotes:
                        v_turns.add(q.turn_id)
                        quotes_info.append({
                            "turn_id": q.turn_id, "type": "约束违反",
                            "label": c.constraint_name, "text": q.text, "note": q.note,
                            "color": "#ff4d4f",
                        })
            # checkpoint quotes
            for cp in ev.checkpoint_evaluations:
                for q in cp.quotes:
                    quotes_info.append({
                        "turn_id": q.turn_id, "type": "检查点证据",
                        "label": cp.checkpoint_name, "text": q.text, "note": q.note,
                        "color": "#52c41a" if cp.status.value == "completed" else "#faad14",
                    })
            # knowledge quotes
            if ev.knowledge_accuracy_eval:
                for q in ev.knowledge_accuracy_eval.quotes:
                    quotes_info.append({
                        "turn_id": q.turn_id, "type": "知识引用",
                        "label": "知识库", "text": q.text, "note": q.note,
                        "color": "#722ed1",
                    })
            session_violations[sess.session_id] = v_turns
            session_quotes[sess.session_id] = quotes_info

        # Heatmap data: persona × dimension
        heat_personas = sorted(report.persona_scores.keys(),
                               key=lambda p: -report.persona_scores[p])
        heat_dim_keys = [d.dimension_key for d in report.dimension_averages]
        heat_data = []  # list of [persona_idx, dim_idx, score]
        # 计算 (persona, dim) -> 平均分
        persona_dim_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for ev in report.session_evaluations:
            for d in ev.dimension_scores:
                persona_dim_acc[ev.persona_type.value][d.dimension_key].append(d.score)
        for pi, p in enumerate(heat_personas):
            for di, dk in enumerate(heat_dim_keys):
                vals = persona_dim_acc.get(p, {}).get(dk, [])
                avg = round(statistics.mean(vals), 1) if vals else 0.0
                heat_data.append([pi, di, avg])

        ctx = {
            "report": report,
            "dim_avg": report.dimension_averages,
            "persona_scores": report.persona_scores,
            "sessions": report.dialogue_sessions,
            "evaluations": report.session_evaluations,
            "branch_cov": report.branch_coverage,
            "weakness": report.weakness_profile,
            "complexity": report.instruction_complexity,
            "run_metadata": report.run_metadata,
            "session_violations_json": json.dumps(
                {sid: list(v) for sid, v in session_violations.items()}, ensure_ascii=False
            ),
            "session_quotes_json": json.dumps(session_quotes, ensure_ascii=False),
            "heat_personas": heat_personas,
            "heat_persona_names": [PERSONA_NAMES_CN.get(p, p) for p in heat_personas],
            "heat_dim_names": [d.dimension_name for d in report.dimension_averages],
            "heat_data_json": json.dumps(heat_data),
            "PERSONA_NAMES_CN": PERSONA_NAMES_CN,
            "EVAL_METHOD_LABELS": EVAL_METHOD_LABELS,
            "DialogueRole": DialogueRole,
            "generated_at": report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
            "html_escape": html.escape,
        }
        template = Template(HTML_TEMPLATE_V3, autoescape=False)
        out = template.render(**ctx)
        with open(path, "w", encoding="utf-8") as f:
            f.write(out)


# ============ HTML 模板 ============

HTML_TEMPLATE_V3 = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>{{ report.task_name }} - 评测报告</title>
  <style>
    *{margin:0;padding:0;box-sizing:border-box}
    body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;background:#f3f5f9;color:#222;line-height:1.6}
    .container{max-width:1280px;margin:0 auto;padding:18px}
    .hero{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:32px 36px;border-radius:14px;margin-bottom:18px;box-shadow:0 4px 16px rgba(102,126,234,.25)}
    .hero h1{font-size:26px;margin-bottom:6px}
    .hero .meta{opacity:.92;font-size:13px}
    .hero .meta-tags{margin-top:12px}
    .hero .tag{display:inline-block;background:rgba(255,255,255,.18);padding:3px 10px;border-radius:12px;font-size:11px;margin-right:6px}
    .grid{display:grid;gap:16px}
    .grid-4{grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
    .grid-2{grid-template-columns:1fr 1fr}
    .card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 10px rgba(0,0,0,.05);margin-bottom:16px}
    .card h2{font-size:17px;margin-bottom:14px;padding-bottom:10px;border-bottom:2px solid #f0f2f7;display:flex;align-items:center;gap:8px}
    .card h2 .badge{background:#667eea;color:#fff;font-size:10px;padding:2px 7px;border-radius:9px;font-weight:400}
    .stat{text-align:center;padding:14px;background:#fafbfd;border-radius:10px}
    .stat .v{font-size:28px;font-weight:700;color:#1890ff}
    .stat .l{font-size:11px;color:#666;margin-top:3px}
    .stat .sub{font-size:11px;color:#999;margin-top:2px}
    .score-excellent{color:#52c41a}.score-good{color:#1890ff}.score-fair{color:#faad14}.score-poor{color:#f5222d}
    .big-score{font-size:60px;font-weight:700;text-align:center;line-height:1.1}
    .conf{font-size:11px;color:#999;text-align:center;margin-top:4px}
    .dim-row{display:grid;grid-template-columns:160px 1fr 80px 90px 110px;align-items:center;gap:10px;padding:10px 0;border-bottom:1px solid #f5f5f8;font-size:13px}
    .dim-name{font-weight:500}
    .dim-bar{height:18px;background:#eef0f4;border-radius:9px;overflow:hidden;position:relative}
    .dim-fill{height:100%;border-radius:9px;transition:width .3s}
    .dim-score{font-weight:700;text-align:right}
    .dim-confidence{font-size:11px;color:#888}
    .dim-method{font-size:10px;padding:2px 8px;border-radius:8px;text-align:center}
    .method-rule{background:#e6f7ff;color:#1890ff}
    .method-llm_judge{background:#f9f0ff;color:#722ed1}
    .method-hybrid{background:#fff7e6;color:#fa8c16}
    .method-heuristic{background:#f6ffed;color:#52c41a}
    .persona-card{text-align:center;padding:14px;border-radius:10px;background:#fafbfd;cursor:pointer;transition:all .2s}
    .persona-card:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,.08)}
    .persona-card .v{font-size:22px;font-weight:700;margin:6px 0}
    .persona-card .l{font-size:11px;color:#666}

    /* tab */
    .tab-buttons{display:flex;gap:5px;margin-bottom:14px;flex-wrap:wrap}
    .tab-btn{padding:7px 14px;border:1px solid #d9dde6;background:#fff;border-radius:7px;cursor:pointer;font-size:12px;color:#444}
    .tab-btn:hover{border-color:#1890ff;color:#1890ff}
    .tab-btn.active{background:#1890ff;color:#fff;border-color:#1890ff}
    .tab-btn .score-pill{background:rgba(255,255,255,.25);padding:1px 6px;border-radius:6px;margin-left:5px;font-size:10px}
    .tab-btn:not(.active) .score-pill{background:#eef0f4;color:#444}
    .tab-content{display:none}
    .tab-content.active{display:block}

    /* 对话 */
    .dialogue-box{background:#fafbfd;border:1px solid #eef0f4;border-radius:10px;padding:14px;max-height:560px;overflow-y:auto}
    .turn{margin:8px 0;padding:9px 13px;border-radius:9px;max-width:78%;position:relative}
    .turn-system{background:#e6f7ff;border:1px solid #91d5ff;margin-right:auto}
    .turn-user{background:#f6ffed;border:1px solid #b7eb8f;margin-left:auto}
    .turn-violation{background:#fff2f0!important;border-color:#ff7875!important;box-shadow:0 0 0 2px rgba(245,34,45,.1)}
    .turn-violation::after{content:"⚠";position:absolute;top:-8px;right:-8px;width:20px;height:20px;background:#f5222d;color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700}
    .turn-label{font-size:10px;color:#999;margin-bottom:3px;display:flex;justify-content:space-between}
    .turn-flow{font-size:9px;color:#bbb;font-style:italic}
    .turn-content{white-space:pre-wrap;word-break:break-word}
    .turn-meta{font-size:10px;color:#aaa;margin-top:4px;display:flex;justify-content:space-between;align-items:center}
    .quote-mark{background:#fffacb;padding:0 2px;border-radius:2px}

    /* 评估明细 */
    .cp-item{padding:9px 13px;margin:5px 0;border-radius:7px;display:grid;grid-template-columns:1fr auto auto;gap:10px;align-items:center;font-size:12px}
    .cp-completed{background:#f6ffed;border-left:4px solid #52c41a}
    .cp-partially{background:#fffbe6;border-left:4px solid #faad14}
    .cp-failed{background:#fff2f0;border-left:4px solid #f5222d}
    .cp-na{background:#f5f5f5;border-left:4px solid #aaa;color:#888}
    .pill{padding:1px 7px;border-radius:8px;font-size:10px;font-weight:600;color:#fff}
    .pill-completed{background:#52c41a}.pill-partially{background:#faad14}
    .pill-failed{background:#f5222d}.pill-na{background:#aaa}
    .pill-conf{background:#1890ff}

    /* 失败模式 */
    .fm-card{background:#fff7e6;border:1px solid #ffd591;border-radius:9px;padding:11px 14px;margin:8px 0;font-size:12px}
    .fm-card .name{font-weight:600;color:#d46b08}
    .fm-card .quote{background:#fff;padding:5px 8px;border-radius:5px;margin:6px 0;font-style:italic;color:#555}
    .fm-card .suggestion{color:#666;font-size:11px}
    .fm-cat{display:inline-block;background:#ff7a45;color:#fff;font-size:10px;padding:1px 6px;border-radius:6px;margin-right:5px}

    /* 分支覆盖 */
    .branch-line{display:flex;align-items:center;gap:10px;font-size:12px;padding:5px 0}
    .branch-line .dot{width:10px;height:10px;border-radius:50%}
    .branch-covered .dot{background:#52c41a}
    .branch-uncovered .dot{background:#ccc}
    .branch-uncovered{color:#999}

    /* 推荐 */
    .rec-list li{padding:9px 13px;margin:6px 0;background:#fafbfd;border-radius:7px;border-left:3px solid #1890ff;list-style:none;font-size:13px}
    .rec-list .urgent{border-left-color:#f5222d;background:#fff2f0}
    .rec-list .knowledge{border-left-color:#722ed1;background:#f9f0ff}
    .rec-list .utterance{border-left-color:#fa8c16;background:#fff7e6}
    .rec-list .failure{border-left-color:#fa541c;background:#fff2e8}

    /* 复算信息 */
    .meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;font-size:12px}
    .meta-item{padding:8px 12px;background:#fafbfd;border-radius:6px}
    .meta-item .l{color:#888;font-size:10px}
    .meta-item .v{font-weight:500;font-family:Menlo,monospace;font-size:12px;word-break:break-all}

    /* heatmap canvas */
    canvas{max-width:100%;display:block;margin:0 auto}

    /* footer */
    .footer{text-align:center;color:#999;font-size:11px;padding:18px 0}
    .footer a{color:#666;text-decoration:none}
  </style>
</head>
<body>
<div class="container">

<!-- HERO -->
<div class="hero">
  <h1>📊 多轮对话评测报告 v3.0</h1>
  <div class="meta">
    任务: <strong>{{ report.task_name }}</strong>
    {% if report.task_description %}— {{ report.task_description[:80] }}{% if report.task_description|length > 80 %}...{% endif %}{% endif %}
  </div>
  <div class="meta">生成时间: {{ generated_at }} | 测试会话数: {{ report.total_sessions }}</div>
  <div class="meta-tags">
    {% if complexity %}
    <span class="tag">指令复杂度 {{ complexity.complexity_score|int }} ({{ complexity.complexity_level }})</span>
    <span class="tag">流程节点 {{ complexity.n_flow_nodes }}</span>
    <span class="tag">分支 {{ complexity.n_branches }}</span>
    <span class="tag">约束 {{ complexity.n_constraints }}</span>
    <span class="tag">知识 {{ complexity.n_knowledge_entries }}</span>
    {% endif %}
    {% if run_metadata %}
    <span class="tag">run_id={{ run_metadata.run_id[:8] }}</span>
    <span class="tag">N={{ run_metadata.self_consistency_n }}</span>
    {% endif %}
  </div>
</div>

<!-- 概览统计 -->
<div class="grid grid-4">
  <div class="card stat">
    <div class="v {% if report.overall_score>=90 %}score-excellent{% elif report.overall_score>=75 %}score-good{% elif report.overall_score>=60 %}score-fair{% else %}score-poor{% endif %}">
      {{ "%.1f"|format(report.overall_score) }}
    </div>
    <div class="l">总体得分 / 100</div>
    <div class="sub">±{{ "%.1f"|format(report.overall_score_std) }}, 置信 {{ "%.2f"|format(report.overall_confidence) }}</div>
  </div>
  <div class="card stat">
    <div class="v">{{ report.total_sessions }}</div>
    <div class="l">测试会话数</div>
  </div>
  <div class="card stat">
    <div class="v">{{ dim_avg|length }}</div>
    <div class="l">评估维度</div>
  </div>
  <div class="card stat">
    <div class="v">{{ persona_scores|length }}</div>
    <div class="l">用户画像数</div>
  </div>
</div>

<!-- 总分 + 摘要 -->
<div class="grid grid-2">
  <div class="card">
    <h2>🎯 总分</h2>
    <div class="big-score {% if report.overall_score>=90 %}score-excellent{% elif report.overall_score>=75 %}score-good{% elif report.overall_score>=60 %}score-fair{% else %}score-poor{% endif %}">
      {{ "%.1f"|format(report.overall_score) }}
    </div>
    <div class="conf">
      {% if report.overall_score>=90 %}优秀{% elif report.overall_score>=75 %}良好{% elif report.overall_score>=60 %}合格{% else %}待改进{% endif %}
      | 标准差 ±{{ "%.1f"|format(report.overall_score_std) }}
      | 置信度 {{ "%.2f"|format(report.overall_confidence) }}
    </div>
  </div>
  <div class="card">
    <h2>📋 评测摘要</h2>
    <div style="white-space:pre-line;font-size:13px;line-height:1.85;color:#444">{{ report.summary }}</div>
  </div>
</div>

<!-- 雷达 + 维度 -->
<div class="card">
  <h2>📐 多维度评分 <span class="badge">{{ dim_avg|length }} 维</span></h2>
  <canvas id="radar" width="420" height="420" style="max-width:420px"></canvas>
  <div style="margin-top:16px">
    {% for d in dim_avg %}
    <div class="dim-row">
      <span class="dim-name">{{ d.dimension_name }}</span>
      <div class="dim-bar">
        <div class="dim-fill" style="width:{{ d.score }}%;background:{% if d.score>=80 %}#52c41a{% elif d.score>=60 %}#1890ff{% elif d.score>=40 %}#faad14{% else %}#f5222d{% endif %}"></div>
      </div>
      <span class="dim-score" style="color:{% if d.score>=80 %}#52c41a{% elif d.score>=60 %}#1890ff{% elif d.score>=40 %}#faad14{% else %}#f5222d{% endif %}">
        {{ "%.1f"|format(d.score) }}
        {% if d.score_std > 0 %}<span class="dim-confidence"> ±{{ "%.1f"|format(d.score_std) }}</span>{% endif %}
      </span>
      <span class="dim-confidence">权重 {{ (d.weight*100)|int }}%, 置信 {{ "%.2f"|format(d.confidence) }}</span>
      <span class="dim-method method-{{ d.evaluation_method.value }}">
        {{ EVAL_METHOD_LABELS[d.evaluation_method.value] }}
      </span>
    </div>
    {% endfor %}
  </div>
</div>

<!-- 画像表现 + 热力图 -->
<div class="card">
  <h2>👥 用户画像表现 <span class="badge">{{ persona_scores|length }} 类</span></h2>
  <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(110px,1fr));margin-bottom:18px">
    {% for p, s in persona_scores.items()|sort(attribute=1, reverse=true) %}
    <div class="persona-card" onclick="filterByPersona('{{ p }}')">
      <div class="l">{{ PERSONA_NAMES_CN.get(p, p) }}</div>
      <div class="v" style="color:{% if s>=80 %}#52c41a{% elif s>=60 %}#1890ff{% elif s>=40 %}#faad14{% else %}#f5222d{% endif %}">{{ "%.1f"|format(s) }}</div>
    </div>
    {% endfor %}
  </div>
  <h2 style="margin-top:14px">🔥 画像 × 维度 表现热力图</h2>
  <canvas id="heatmap" width="900" height="320" style="max-width:100%"></canvas>
</div>

<!-- 短板诊断 -->
{% if weakness %}
<div class="card">
  <h2>🩺 模型短板诊断 <span class="badge">v3 创新</span></h2>
  <div class="grid grid-2" style="margin-bottom:14px">
    <div>
      <h3 style="font-size:13px;color:#f5222d">🔻 最弱维度 Top-3</h3>
      <ul style="margin-top:8px;font-size:13px">
        {% for w in weakness.weakest_dimensions %}
        <li style="padding:5px 0">{{ w }}</li>
        {% endfor %}
      </ul>
    </div>
    <div>
      <h3 style="font-size:13px;color:#52c41a">🔺 最强维度 Top-3</h3>
      <ul style="margin-top:8px;font-size:13px">
        {% for s in weakness.strongest_dimensions %}
        <li style="padding:5px 0">{{ s }}</li>
        {% endfor %}
      </ul>
    </div>
  </div>
  {% if weakness.weakest_personas %}
  <div style="margin-bottom:14px;font-size:13px">
    <strong>最弱画像:</strong>
    {% for p in weakness.weakest_personas %}<span style="background:#fff2f0;color:#cf1322;padding:2px 9px;border-radius:9px;margin:0 4px;font-size:12px">{{ p }}</span>{% endfor %}
  </div>
  {% endif %}
  {% if weakness.top_failure_modes %}
  <h3 style="font-size:14px;margin-top:14px">⚡ Top 失败模式 ({{ weakness.top_failure_modes|length }})</h3>
  {% for fm in weakness.top_failure_modes %}
  <div class="fm-card">
    <div><span class="fm-cat">{{ fm.category }}</span><span class="name">{{ fm.name }}</span> <span style="color:#999;font-size:11px">| 出现 {{ fm.occurrences }} 次 | 影响分 {{ "%.1f"|format(fm.impact_score) }}</span></div>
    {% if fm.typical_quote %}<div class="quote">"{{ fm.typical_quote }}"</div>{% endif %}
    {% if fm.affected_personas %}<div style="font-size:11px;color:#999">影响画像: {{ fm.affected_personas|join(', ') }}</div>{% endif %}
    {% if fm.suggestion %}<div class="suggestion">💡 建议: {{ fm.suggestion }}</div>{% endif %}
  </div>
  {% endfor %}
  {% endif %}
</div>
{% endif %}

<!-- 分支覆盖 -->
{% if branch_cov and branch_cov.total_branches > 0 %}
<div class="card">
  <h2>🌳 分支覆盖率</h2>
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">
    <div style="flex:1;height:24px;background:#eef0f4;border-radius:12px;overflow:hidden">
      <div style="height:100%;background:linear-gradient(90deg,#52c41a,#73d13d);width:{{ (branch_cov.coverage_rate*100)|int }}%"></div>
    </div>
    <div style="font-weight:700;font-size:18px">{{ branch_cov.covered_branches }}/{{ branch_cov.total_branches }} ({{ (branch_cov.coverage_rate*100)|round }}%)</div>
  </div>
  <div>
    <h3 style="font-size:13px;margin-bottom:8px">✓ 已覆盖分支</h3>
    {% for tb in branch_cov.target_branches_to_cover %}
    {% if tb not in branch_cov.uncovered_branches %}
    <div class="branch-line branch-covered"><span class="dot"></span><span>{{ tb }}</span></div>
    {% endif %}
    {% endfor %}
    {% if branch_cov.uncovered_branches %}
    <h3 style="font-size:13px;margin-top:14px;margin-bottom:8px;color:#f5222d">✗ 未覆盖分支</h3>
    {% for ub in branch_cov.uncovered_branches %}
    <div class="branch-line branch-uncovered"><span class="dot"></span><span>{{ ub }}</span></div>
    {% endfor %}
    {% endif %}
  </div>
</div>
{% endif %}

<!-- 对话浏览器 -->
<div class="card">
  <h2>💬 对话浏览器 <span class="badge">点击 tab 切换会话</span></h2>
  <div class="tab-buttons" id="sessionTabs">
    {% for sess in sessions %}
    {% set ev = evaluations[loop.index0] %}
    <button class="tab-btn{% if loop.index == 1 %} active{% endif %}"
            data-persona="{{ sess.persona_type.value }}"
            onclick="showSession('s{{ loop.index }}', this)">
      会话{{ loop.index }} {{ PERSONA_NAMES_CN.get(sess.persona_type.value, sess.persona_type.value) }}
      <span class="score-pill">{{ "%.0f"|format(ev.total_score) }}</span>
    </button>
    {% endfor %}
  </div>
  {% for sess in sessions %}
  {% set ev = evaluations[loop.index0] %}
  <div id="s{{ loop.index }}" class="tab-content{% if loop.index == 1 %} active{% endif %}">
    <div style="font-size:12px;color:#777;margin-bottom:10px">
      画像: <strong>{{ PERSONA_NAMES_CN.get(sess.persona_type.value, sess.persona_type.value) }}</strong>
      | 轮次: {{ sess.turns|length }}
      | 终止: {{ sess.terminated_reason or "正常" }}
      | 总分: <strong>{{ "%.1f"|format(ev.total_score) }}</strong> (置信 {{ "%.2f"|format(ev.overall_confidence) }})
      {% if sess.branch_path %} | 分支: {{ sess.branch_path|join('; ') }}{% endif %}
      {% if sess.seed is not none %} | seed={{ sess.seed }}{% endif %}
    </div>
    <div class="dialogue-box">
      {% for turn in sess.turns %}
      <div class="turn turn-{{ turn.role.value }}"
           data-turn-id="{{ turn.turn_id }}" data-session="{{ sess.session_id }}">
        <div class="turn-label">
          <span>[轮次{{ turn.turn_id }}] {% if turn.role.value == 'system' %}🤖 系统{% else %}👤 用户{% endif %}</span>
          {% if turn.inferred_flow_node %}<span class="turn-flow">📍 {{ turn.inferred_flow_node }}</span>{% endif %}
        </div>
        <div class="turn-content">{{ html_escape(turn.content) }}</div>
        {% if turn.role.value == 'system' %}
        <div class="turn-meta"><span>{{ turn.content|length }} 字</span></div>
        {% endif %}
      </div>
      {% endfor %}
    </div>

    <!-- 检查点 -->
    {% if ev.checkpoint_evaluations %}
    <h3 style="margin-top:16px;font-size:14px;display:flex;justify-content:space-between">
      <span>📌 检查点评估 ({{ ev.checkpoint_evaluations|length }})</span>
    </h3>
    <ul style="margin-top:8px">
    {% for cp in ev.checkpoint_evaluations %}
      <li class="cp-item cp-{% if cp.status.value == 'n/a' %}na{% else %}{{ cp.status.value }}{% endif %}">
        <div>
          <strong>{{ cp.checkpoint_name }}</strong>
          {% if cp.evidence %}<div style="color:#666;font-size:11px;margin-top:3px">{{ cp.evidence[:200] }}</div>{% endif %}
          {% if cp.quotes %}<div style="margin-top:4px">
            {% for q in cp.quotes %}<span class="quote-mark" style="font-size:10px">[轮{{ q.turn_id }}] "{{ q.text[:50] }}"</span> {% endfor %}
          </div>{% endif %}
        </div>
        <span class="pill pill-{% if cp.status.value == 'n/a' %}na{% else %}{{ cp.status.value }}{% endif %}">{{ "%.0f"|format(cp.score) }}</span>
        {% if cp.confidence < 0.8 %}<span class="pill pill-conf">置信 {{ "%.2f"|format(cp.confidence) }}</span>{% else %}<span></span>{% endif %}
      </li>
    {% endfor %}
    </ul>
    {% endif %}

    <!-- 约束违反 -->
    {% set violated = ev.constraint_evaluations|selectattr("violated")|list %}
    {% if violated %}
    <h3 style="margin-top:14px;font-size:14px;color:#f5222d">⚠ 约束违反 ({{ violated|length }})</h3>
    <ul style="margin-top:6px">
    {% for c in violated %}
      <li class="cp-item cp-failed">
        <div>
          <strong>[{{ c.severity.value }}|{{ EVAL_METHOD_LABELS[c.evaluation_method.value] }}] {{ c.constraint_name }}</strong>
          <span style="color:#999;font-size:11px">x {{ c.violation_count }}</span>
          {% if c.violation_details %}<div style="color:#666;font-size:11px;margin-top:3px">{{ c.violation_details[0][:150] }}</div>{% endif %}
        </div>
        <span class="pill pill-failed">违反</span>
        <span></span>
      </li>
    {% endfor %}
    </ul>
    {% endif %}

    <!-- 话术违规 -->
    {% if ev.utterance_constraint_eval and ev.utterance_constraint_eval.violations %}
    <h3 style="margin-top:14px;font-size:14px;color:#fa8c16">📏 话术约束违规 ({{ ev.utterance_constraint_eval.violations|length }})</h3>
    <ul style="margin-top:6px">
    {% for v in ev.utterance_constraint_eval.violations %}
      <li style="padding:7px 11px;margin:4px 0;background:#fff7e6;border-left:3px solid #fa8c16;border-radius:5px;font-size:12px">
        轮次{{ v.turn_id }}: <strong>{{ v.violation_type }}</strong> - {{ v.detail }}
      </li>
    {% endfor %}
    </ul>
    {% endif %}

    <!-- 知识幻觉 -->
    {% if ev.knowledge_accuracy_eval and (ev.knowledge_accuracy_eval.fabricated_info or ev.knowledge_accuracy_eval.incorrect_answers > 0) %}
    <h3 style="margin-top:14px;font-size:14px;color:#722ed1">🧠 知识引用问题</h3>
    <div style="margin-top:6px;font-size:12px">
      {% if ev.knowledge_accuracy_eval.fabricated_info %}
      <div style="background:#f9f0ff;padding:8px 12px;border-radius:6px;margin-bottom:5px">
        <strong>编造信息:</strong>
        <ul style="margin-left:18px">
        {% for fi in ev.knowledge_accuracy_eval.fabricated_info %}<li>{{ fi }}</li>{% endfor %}
        </ul>
      </div>
      {% endif %}
      {% if ev.knowledge_accuracy_eval.incorrect_answers > 0 %}
      <div>错误回答: <strong>{{ ev.knowledge_accuracy_eval.incorrect_answers }}</strong> 个 (共 {{ ev.knowledge_accuracy_eval.total_knowledge_queries }} 次查询)</div>
      {% endif %}
    </div>
    {% endif %}

    <!-- 优点&问题 -->
    {% if ev.strengths or ev.issues %}
    <div class="grid grid-2" style="margin-top:14px">
      <div>
        {% if ev.strengths %}<h3 style="font-size:13px;color:#52c41a">✓ 优点</h3>
        <div style="margin-top:6px">
        {% for st in ev.strengths %}<span style="display:inline-block;background:#f6ffed;border:1px solid #b7eb8f;padding:2px 9px;margin:2px;border-radius:9px;font-size:11px">{{ st }}</span>{% endfor %}
        </div>{% endif %}
      </div>
      <div>
        {% if ev.issues %}<h3 style="font-size:13px;color:#f5222d">✗ 问题</h3>
        <div style="margin-top:6px">
        {% for it in ev.issues %}<span style="display:inline-block;background:#fff2f0;border:1px solid #ffccc7;padding:2px 9px;margin:2px;border-radius:9px;font-size:11px">{{ it }}</span>{% endfor %}
        </div>{% endif %}
      </div>
    </div>
    {% endif %}

  </div>
  {% endfor %}
</div>

<!-- 改进建议 -->
<div class="card">
  <h2>💡 改进建议</h2>
  <ul class="rec-list">
  {% for r in report.recommendations %}
    <li class="{% if '紧急' in r %}urgent{% elif '知识' in r %}knowledge{% elif '话术' in r %}utterance{% elif '失败模式' in r %}failure{% endif %}">{{ r }}</li>
  {% endfor %}
  </ul>
</div>

<!-- 复算信息 -->
{% if run_metadata %}
<div class="card">
  <h2>🔬 复算信息 (Reproducibility) <span class="badge">v3 新增</span></h2>
  <div class="meta-grid">
    <div class="meta-item"><div class="l">run_id</div><div class="v">{{ run_metadata.run_id }}</div></div>
    <div class="meta-item"><div class="l">evaluator_version</div><div class="v">{{ run_metadata.evaluator_version }}</div></div>
    <div class="meta-item"><div class="l">target_model</div><div class="v">{{ run_metadata.target_model_id }}</div></div>
    <div class="meta-item"><div class="l">simulator_model</div><div class="v">{{ run_metadata.simulator_model_id }}</div></div>
    <div class="meta-item"><div class="l">judge_model</div><div class="v">{{ run_metadata.judge_model_id }}</div></div>
    <div class="meta-item"><div class="l">seed</div><div class="v">{{ run_metadata.seed }}</div></div>
    <div class="meta-item"><div class="l">self-consistency N</div><div class="v">{{ run_metadata.self_consistency_n }}</div></div>
    <div class="meta-item"><div class="l">concurrency</div><div class="v">{{ run_metadata.concurrency }}</div></div>
    <div class="meta-item"><div class="l">duration</div><div class="v">{{ "%.1f"|format(run_metadata.duration_seconds) }} s</div></div>
  </div>
</div>
{% endif %}

<div class="footer">
  多轮对话评测系统 v3.0 | 报告 ID: {{ report.report_id[:8] }}<br>
  📐 可解释 (证据三元组) · 📊 可量化 (Self-Consistency) · 🎯 可复算 (seed + version)
</div>

</div>

<script>
// session_violations 静态注入 (用于 turn-violation 高亮)
const SESSION_VIOLATIONS = {{ session_violations_json|safe }};
const SESSION_QUOTES = {{ session_quotes_json|safe }};

// 给已经渲染的每个 turn 元素加 violation class
document.querySelectorAll('.turn[data-session][data-turn-id]').forEach(el => {
  const sid = el.dataset.session;
  const tid = parseInt(el.dataset.turnId, 10);
  const vList = SESSION_VIOLATIONS[sid] || [];
  if (vList.includes(tid) && el.classList.contains('turn-system')) {
    el.classList.add('turn-violation');
  }
});

function showSession(id, btn) {
  document.querySelectorAll('.tab-content').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(e => e.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}

function filterByPersona(p) {
  // 切换到第一个匹配的 session
  const tabs = document.querySelectorAll('#sessionTabs .tab-btn');
  for (const t of tabs) {
    if (t.dataset.persona === p) {
      t.click();
      t.scrollIntoView({behavior:'smooth', block:'center'});
      return;
    }
  }
}

// 雷达图
function drawRadar() {
  const c = document.getElementById('radar');
  if (!c) return;
  const ctx = c.getContext('2d');
  const dims = [
    {% for d in dim_avg %}{name:{{ d.dimension_name|tojson }}, score:{{ d.score }}, conf:{{ d.confidence }}}{% if not loop.last %},{% endif %}{% endfor %}
  ];
  if (dims.length === 0) return;
  const cx=210, cy=210, R=160, n=dims.length, step=2*Math.PI/n;
  ctx.clearRect(0,0,420,420);
  // grid
  for (let lv=1; lv<=5; lv++) {
    const r = R*lv/5;
    ctx.beginPath();
    for (let i=0;i<=n;i++){
      const a=i*step-Math.PI/2;
      const x=cx+r*Math.cos(a), y=cy+r*Math.sin(a);
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.strokeStyle='#e8eaf0'; ctx.stroke();
    if (lv===5){ ctx.fillStyle='#bbb'; ctx.font='10px sans-serif'; ctx.textAlign='right';
      ctx.fillText('100', cx-3, cy-r+3);}
  }
  // axes
  for (let i=0;i<n;i++){
    const a=i*step-Math.PI/2;
    ctx.beginPath(); ctx.moveTo(cx,cy);
    ctx.lineTo(cx+R*Math.cos(a),cy+R*Math.sin(a));
    ctx.strokeStyle='#dcdfe6'; ctx.stroke();
  }
  // data
  ctx.beginPath();
  for (let i=0;i<=n;i++){
    const idx=i%n; const a=idx*step-Math.PI/2; const r=R*dims[idx].score/100;
    const x=cx+r*Math.cos(a), y=cy+r*Math.sin(a);
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.fillStyle='rgba(102,126,234,.25)'; ctx.fill();
  ctx.strokeStyle='#667eea'; ctx.lineWidth=2; ctx.stroke();
  // points
  for (let i=0;i<n;i++){
    const a=i*step-Math.PI/2; const r=R*dims[i].score/100;
    const x=cx+r*Math.cos(a), y=cy+r*Math.sin(a);
    ctx.beginPath(); ctx.arc(x,y,4,0,2*Math.PI);
    ctx.fillStyle = dims[i].score>=80?'#52c41a':dims[i].score>=60?'#1890ff':dims[i].score>=40?'#faad14':'#f5222d';
    ctx.fill();
  }
  // labels
  ctx.fillStyle='#333'; ctx.font='11px sans-serif'; ctx.textAlign='center';
  for (let i=0;i<n;i++){
    const a=i*step-Math.PI/2; const lr=R+22;
    const x=cx+lr*Math.cos(a), y=cy+lr*Math.sin(a);
    ctx.fillText(dims[i].name, x, y+4);
  }
}

// Heatmap
function drawHeatmap() {
  const data = {{ heat_data_json|safe }};
  const personas = {{ heat_persona_names|tojson }};
  const dims = {{ heat_dim_names|tojson }};
  const c = document.getElementById('heatmap');
  if (!c || !data.length) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  const padL=110, padT=60, padR=20, padB=30;
  const gw = W-padL-padR, gh = H-padT-padB;
  const cellW = gw/dims.length, cellH = gh/personas.length;
  ctx.clearRect(0,0,W,H);
  function color(v){
    if (v===0) return '#f0f0f0';
    if (v>=80) return '#52c41a';
    if (v>=70) return '#73d13d';
    if (v>=60) return '#1890ff';
    if (v>=50) return '#fadb14';
    if (v>=40) return '#faad14';
    if (v>=30) return '#fa541c';
    return '#f5222d';
  }
  // cells
  for (const [pi, di, v] of data) {
    ctx.fillStyle = color(v);
    ctx.fillRect(padL+di*cellW+1, padT+pi*cellH+1, cellW-2, cellH-2);
    ctx.fillStyle = v>=60?'#fff':'#333';
    ctx.font='11px sans-serif'; ctx.textAlign='center';
    ctx.fillText(v.toFixed(0), padL+di*cellW+cellW/2, padT+pi*cellH+cellH/2+4);
  }
  // x labels (dims)
  ctx.save(); ctx.fillStyle='#333'; ctx.font='11px sans-serif'; ctx.textAlign='center';
  for (let i=0;i<dims.length;i++){
    ctx.save();
    ctx.translate(padL+i*cellW+cellW/2, padT-8);
    ctx.rotate(-Math.PI/8);
    ctx.fillText(dims[i], 0, 0);
    ctx.restore();
  }
  ctx.restore();
  // y labels (personas)
  ctx.fillStyle='#333'; ctx.font='11px sans-serif'; ctx.textAlign='right';
  for (let i=0;i<personas.length;i++){
    ctx.fillText(personas[i], padL-8, padT+i*cellH+cellH/2+4);
  }
}

drawRadar();
drawHeatmap();
</script>

</body>
</html>
"""
