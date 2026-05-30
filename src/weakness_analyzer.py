"""
模型短板诊断器 (v3.0)

核心创新点 C6:
  把"评测系统"升级为"分析诊断工具"

输出:
- 维度雷达 - 最弱 Top-3 / 最强 Top-3
- 画像短板 - 哪类用户场景下表现最差
- 失败模式 Top-N - 自动从 issues / violations / failed checkpoints 中聚类
- 风险总览 - 综合 narratives
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict

from .models import (
    CheckpointStatus,
    DialogueSession,
    EvaluationReport,
    FailureMode,
    ModelWeaknessProfile,
    SessionEvaluation,
    TaskInstruction,
)

logger = logging.getLogger(__name__)


class WeaknessAnalyzer:
    """模型短板诊断器"""

    def analyze(
        self,
        instruction: TaskInstruction,
        sessions: list[DialogueSession],
        evaluations: list[SessionEvaluation],
    ) -> ModelWeaknessProfile:
        if not evaluations:
            return ModelWeaknessProfile()

        # 1. 维度短板
        dim_avg: dict[str, list[float]] = defaultdict(list)
        for ev in evaluations:
            for d in ev.dimension_scores:
                dim_avg[d.dimension_name].append(d.score)
        dim_means = {name: sum(s) / len(s) for name, s in dim_avg.items()}
        weakest = sorted(dim_means.items(), key=lambda x: x[1])[:3]
        strongest = sorted(dim_means.items(), key=lambda x: -x[1])[:3]

        # 2. 画像短板
        persona_scores: dict[str, list[float]] = defaultdict(list)
        for ev in evaluations:
            persona_scores[ev.persona_type.value].append(ev.total_score)
        persona_means = {p: sum(s) / len(s) for p, s in persona_scores.items()}
        weakest_personas = [p for p, _ in sorted(persona_means.items(), key=lambda x: x[1])[:3]]

        # 3. 失败模式聚类
        failure_modes = self._cluster_failure_modes(sessions, evaluations)

        # 4. 风险总览
        risk_summary = self._compose_risk_summary(weakest, weakest_personas, failure_modes)

        return ModelWeaknessProfile(
            weakest_dimensions=[f"{name} ({score:.1f}分)" for name, score in weakest],
            strongest_dimensions=[f"{name} ({score:.1f}分)" for name, score in strongest],
            weakest_personas=[f"{p} ({persona_means[p]:.1f}分)" for p in weakest_personas],
            top_failure_modes=failure_modes,
            risk_summary=risk_summary,
        )

    def _cluster_failure_modes(
        self,
        sessions: list[DialogueSession],
        evaluations: list[SessionEvaluation],
    ) -> list[FailureMode]:
        """从评估结果中聚类失败模式"""
        fm_counter: dict[tuple[str, str], dict] = {}

        # 1. 失败/部分完成的检查点
        for ev in evaluations:
            for cp in ev.checkpoint_evaluations:
                if cp.status in (CheckpointStatus.FAILED, CheckpointStatus.PARTIALLY):
                    key = ("task", f"检查点未完成: {cp.checkpoint_name}")
                    self._add_fm(fm_counter, key, ev, "task",
                                affected=ev.persona_type.value,
                                quote=cp.quotes[0].text if cp.quotes else cp.evidence)

        # 2. 违反的约束
        for ev in evaluations:
            for c in ev.constraint_evaluations:
                if c.violated:
                    key = ("constraint", f"约束违反 [{c.severity.value}]: {c.constraint_name}")
                    quote = c.quotes[0].text if c.quotes else (c.violation_details[0] if c.violation_details else "")
                    self._add_fm(fm_counter, key, ev, "constraint",
                                affected=ev.persona_type.value,
                                quote=quote)

        # 3. 话术违规
        for ev in evaluations:
            if ev.utterance_constraint_eval and ev.utterance_constraint_eval.violations:
                vt_counter = Counter(v["violation_type"] for v in ev.utterance_constraint_eval.violations)
                for vt, n in vt_counter.items():
                    key = ("utterance", f"话术违规: {vt}")
                    sample = next((v for v in ev.utterance_constraint_eval.violations if v["violation_type"] == vt), {})
                    quote = sample.get("content", sample.get("detail", ""))[:80]
                    self._add_fm(fm_counter, key, ev, "utterance",
                                affected=ev.persona_type.value,
                                quote=quote, occurrences_extra=n - 1)

        # 4. 知识幻觉
        for ev in evaluations:
            if ev.knowledge_accuracy_eval:
                kb = ev.knowledge_accuracy_eval
                if kb.fabricated_info:
                    for fi in kb.fabricated_info[:3]:
                        key = ("knowledge", f"知识幻觉: 编造信息")
                        self._add_fm(fm_counter, key, ev, "knowledge",
                                    affected=ev.persona_type.value, quote=fi)
                if kb.incorrect_answers > 0:
                    key = ("knowledge", "知识引用错误")
                    self._add_fm(fm_counter, key, ev, "knowledge",
                                affected=ev.persona_type.value,
                                quote=str(kb.details[:1]) if kb.details else "")

        # 5. 通用 issues (LLM 总评里的 issue 列表)
        issue_counter = Counter()
        for ev in evaluations:
            for it in ev.issues:
                issue_counter[it] += 1
        # 只取出现 >=2 次的 issue
        for it, cnt in issue_counter.most_common(10):
            if cnt >= 2:
                key = ("flow", f"通用问题: {it[:50]}")
                # 找一个出现该 issue 的 session
                sample_ev = next((ev for ev in evaluations if it in ev.issues), None)
                self._add_fm(fm_counter, key, sample_ev, "flow",
                            affected=sample_ev.persona_type.value if sample_ev else "",
                            quote=it, occurrences_extra=cnt - 1)

        # 转化为 FailureMode 列表 + 排序
        modes: list[FailureMode] = []
        for (cat, name), info in fm_counter.items():
            modes.append(FailureMode(
                name=name,
                category=cat,
                occurrences=info["count"],
                affected_personas=sorted(set(info["personas"])),
                affected_sessions=info["sessions"][:6],
                typical_quote=info.get("quote", "")[:120],
                impact_score=self._estimate_impact(cat, info["count"]),
                suggestion=self._gen_suggestion(cat, name),
            ))
        # 按 impact 降序, 取 Top 8
        modes.sort(key=lambda x: -x.impact_score)
        return modes[:8]

    @staticmethod
    def _add_fm(
        counter: dict,
        key: tuple[str, str],
        ev,
        category: str,
        affected: str = "",
        quote: str = "",
        occurrences_extra: int = 0,
    ):
        info = counter.setdefault(key, {"count": 0, "personas": [], "sessions": [], "quote": ""})
        info["count"] += 1 + max(0, occurrences_extra)
        if affected:
            info["personas"].append(affected)
        if ev is not None:
            sid = ev.session_id[:8]
            if sid not in info["sessions"]:
                info["sessions"].append(sid)
        if quote and not info["quote"]:
            info["quote"] = str(quote)

    @staticmethod
    def _estimate_impact(category: str, count: int) -> float:
        weight = {
            "constraint": 8.0,    # 约束违反, 影响最大
            "task": 6.0,          # 任务未完成
            "knowledge": 5.0,     # 知识幻觉
            "utterance": 2.0,     # 话术
            "flow": 3.0,          # 流程
        }.get(category, 1.0)
        return round(weight * count, 1)

    @staticmethod
    def _gen_suggestion(category: str, name: str) -> str:
        if category == "constraint":
            return "检查 system_prompt 中相关约束的表达, 加强约束置顶, 或加 few-shot 示例"
        if category == "task":
            return "在 system_prompt 中明确该检查点是必达项, 或在流程描述中加 must-do 标记"
        if category == "knowledge":
            return "提供知识库 FAQ 标准答案, 强调'未知信息不要编造'"
        if category == "utterance":
            return "在 prompt 中给出长度/禁词约束的反例和正例对比"
        if category == "flow":
            return "梳理 flow_nodes 之间的过渡话术, 在 prompt 中加示例对话"
        return "见详细分析"

    @staticmethod
    def _compose_risk_summary(
        weakest_dims: list[tuple[str, float]],
        weakest_personas: list[str],
        failure_modes: list[FailureMode],
    ) -> str:
        parts: list[str] = []
        if weakest_dims:
            top = weakest_dims[0]
            parts.append(f"最薄弱维度是【{top[0]}】({top[1]:.1f}分)")
        if weakest_personas:
            parts.append(f"在【{weakest_personas[0]}】用户画像下表现最差")
        if failure_modes:
            top_fm = failure_modes[0]
            parts.append(f"高频失败模式: {top_fm.name} (出现{top_fm.occurrences}次)")
        if not parts:
            return "整体表现稳定, 未发现高频风险点"
        return "; ".join(parts) + "。"
