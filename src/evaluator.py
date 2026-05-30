"""
评估器 v3.0 (总编排)

设计:
- RuleEvaluator (确定性) + LLMJudge (语义) 协作
- 8 维评分, 每维标注 evaluation_method (rule/llm/hybrid/heuristic)
- Self-consistency 仅对核心语义维度启用
- 输出复算友好的元数据 (run_id, seed, n_sc, std)

8 维度:
1. task_completion        - LLM_JUDGE  - 检查点完成度 (派生检查点交给 flow)
2. flow_adherence         - LLM_JUDGE  - 流程 + 分支正确性
3. constraint_compliance  - HYBRID     - 规则约束 + 语义约束分别评, 加权
4. exception_handling     - LLM_JUDGE  - 异常画像下的应对
5. dialogue_efficiency    - HEURISTIC  - 轮次 + LLM 辅助判定冗余
6. utterance_quality      - LLM_JUDGE  - 语言专业度
7. knowledge_accuracy     - LLM_JUDGE  - 知识引用与幻觉
8. response_brevity       - RULE       - 字数/禁词/未替换变量
"""
from __future__ import annotations

import logging
import statistics
from typing import Any, Optional

from openai import OpenAI

from .llm_judge import LLMJudge
from .models import (
    CheckpointEvaluation,
    CheckpointStatus,
    ConstraintEvaluation,
    ConstraintSeverity,
    DialogueRole,
    DialogueSession,
    DimensionScore,
    EvaluationMethod,
    EvidenceQuote,
    KnowledgeAccuracyEvaluation,
    PersonaType,
    SessionEvaluation,
    TaskInstruction,
    UtteranceConstraintEvaluation,
)
from .rule_evaluator import RuleEvaluator

logger = logging.getLogger(__name__)


DEFAULT_DIMENSION_WEIGHTS = {
    "task_completion": 0.20,
    "flow_adherence": 0.20,
    "constraint_compliance": 0.15,
    "exception_handling": 0.10,
    "dialogue_efficiency": 0.05,
    "utterance_quality": 0.05,
    "knowledge_accuracy": 0.15,
    "response_brevity": 0.10,
}

DIMENSION_NAMES_CN = {
    "task_completion": "任务完成度",
    "flow_adherence": "流程遵循度",
    "constraint_compliance": "约束遵守度",
    "exception_handling": "异常处理能力",
    "dialogue_efficiency": "对话效率",
    "utterance_quality": "话术质量",
    "knowledge_accuracy": "知识准确度",
    "response_brevity": "话术简洁度",
}


class Evaluator:
    """评估器总编排 v3"""

    SEVERITY_PENALTY = {
        ConstraintSeverity.CRITICAL: 40,
        ConstraintSeverity.MAJOR: 20,
        ConstraintSeverity.MINOR: 10,
    }

    def __init__(
        self,
        llm_client: OpenAI,
        model: str = "deepseek-chat",
        dimension_weights: Optional[dict[str, float]] = None,
        n_self_consistency: int = 1,
    ):
        self.llm_client = llm_client
        self.judge = LLMJudge(
            llm_client=llm_client,
            model=model,
            n_self_consistency=n_self_consistency,
        )
        self.rule_eval = RuleEvaluator()
        self.dimension_weights = dimension_weights or DEFAULT_DIMENSION_WEIGHTS
        # 归一化权重
        total = sum(self.dimension_weights.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            self.dimension_weights = {k: v / total for k, v in self.dimension_weights.items()}
        self.n_sc = max(1, n_self_consistency)

    # ============ 主入口 ============

    def evaluate_session(
        self,
        session: DialogueSession,
        instruction: TaskInstruction,
    ) -> SessionEvaluation:
        logger.info(f"[评估] 会话={session.session_id[:8]} 画像={session.persona_type.value}")
        # 优先使用 session.task_instruction (已 _resolve_variables 替换过), 保证 KB 与对话里的具体数值匹配
        if session.task_instruction is not None:
            instruction = session.task_instruction
        dialogue_text = self._format_dialogue(session)

        # 1. 检查点 (LLM Judge)
        cp_evals = self.judge.evaluate_checkpoints(instruction, dialogue_text, session)

        # 2. 约束: 规则 + LLM
        rule_cons: list[ConstraintEvaluation] = []
        for c in instruction.constraints:
            r = self.rule_eval.evaluate_deterministic_constraint(c, session)
            if r is not None:
                rule_cons.append(r)
        llm_cons = self.judge.evaluate_semantic_constraints(instruction, dialogue_text, session)
        # 合并: 规则结果优先
        rule_ids = {r.constraint_id for r in rule_cons}
        all_cons = rule_cons + [c for c in llm_cons if c.constraint_id not in rule_ids]
        # 补齐未评估的约束 (默认未违反)
        evaluated_ids = {c.constraint_id for c in all_cons}
        for c in instruction.constraints:
            if c.id not in evaluated_ids:
                all_cons.append(ConstraintEvaluation(
                    constraint_id=c.id,
                    constraint_name=c.name,
                    violated=False,
                    severity=c.severity,
                    evaluation_method=EvaluationMethod.HYBRID,
                    confidence=0.5,
                ))

        # 3. 流程
        flow_result = self.judge.evaluate_flow(instruction, dialogue_text, session)

        # 4. 质量 (异常 / 效率 / 话术)
        quality_result = self.judge.evaluate_quality(instruction, dialogue_text, session.persona_type)

        # 5. 话术约束 (规则)
        utt_eval = self.rule_eval.evaluate_utterance_constraints(session, instruction)

        # 6. 知识库
        kb_eval = self.judge.evaluate_knowledge(instruction, dialogue_text, session)

        # 7. 变量未替换
        unresolved = self.rule_eval.detect_unresolved_variables(session)
        if unresolved:
            # 把未替换变量作为 critical 扣分: 加进 utt_eval 违反列表
            for q in unresolved:
                utt_eval.violations.append({
                    "turn_id": q.turn_id,
                    "content": q.text[:60],
                    "violation_type": "未替换变量",
                    "detail": q.note,
                    "constraint_name": "变量必须替换为实值",
                })
                utt_eval.violated_turns += 1
            # 重新计算分数
            utt_eval.score = max(0.0, utt_eval.score - len(unresolved) * 10)

        # 8. 计算维度分
        dimension_scores = self._calculate_dimensions(
            cp_evals, all_cons, flow_result, quality_result, utt_eval, kb_eval, session,
        )
        total = sum(d.weighted_score for d in dimension_scores)
        # 总分置信度 = 各维度置信度按权重加权
        weight_sum = sum(d.weight for d in dimension_scores)
        if weight_sum > 0:
            overall_conf = sum(d.confidence * d.weight for d in dimension_scores) / weight_sum
        else:
            overall_conf = 1.0
        # 总分标准差: weighted std of dimension samples
        # 简化: 取最大 std 作为代表
        max_std = max((d.score_std for d in dimension_scores if d.score_std), default=0.0)

        # 评估方法已经记录, 整体注释
        flow_quotes_data = flow_result.get("evidence_quotes", [])

        return SessionEvaluation(
            session_id=session.session_id,
            persona_type=session.persona_type,
            total_score=round(total, 2),
            dimension_scores=dimension_scores,
            checkpoint_evaluations=cp_evals,
            constraint_evaluations=all_cons,
            utterance_constraint_eval=utt_eval,
            knowledge_accuracy_eval=kb_eval,
            flow_analysis=flow_result.get("flow_analysis", ""),
            branch_coverage=flow_result.get("branches_triggered", []),
            overall_comment=quality_result.get("overall_comment", ""),
            issues=quality_result.get("issues", []),
            strengths=quality_result.get("strengths", []),
            overall_confidence=round(overall_conf, 2),
            total_score_std=round(max_std, 2),
        )

    def evaluate_batch(
        self,
        sessions: list[DialogueSession],
        instruction: TaskInstruction,
    ) -> list[SessionEvaluation]:
        out = []
        for i, s in enumerate(sessions):
            logger.info(f"评估进度: [{i + 1}/{len(sessions)}]")
            out.append(self.evaluate_session(s, instruction))
        return out

    # ============ 维度计算 ============

    def _calculate_dimensions(
        self,
        cp_evals: list[CheckpointEvaluation],
        cons_evals: list[ConstraintEvaluation],
        flow_result: dict[str, Any],
        quality_result: dict[str, Any],
        utt_eval: UtteranceConstraintEvaluation,
        kb_eval: KnowledgeAccuracyEvaluation,
        session: DialogueSession,
    ) -> list[DimensionScore]:
        scores: list[DimensionScore] = []

        # 1. task_completion
        if cp_evals:
            valid = [e for e in cp_evals if e.status != CheckpointStatus.NOT_APPLICABLE]
            if valid:
                # must_do 权重 1.0, should_do 0.5, optional 0.2
                weight_map = {
                    "must_do": 1.0, "should_do": 0.5, "optional": 0.2,
                }
                # 由于 CheckpointEvaluation 没存 type, 默认全按 1.0
                task_score = sum(e.score for e in valid) / len(valid)
                conf = statistics.mean(e.confidence for e in valid) if valid else 1.0
                stds = [e.score_std for e in valid if e.score_std]
                std_avg = statistics.mean(stds) if stds else 0.0
            else:
                task_score = 75.0
                conf = 0.5
                std_avg = 0.0
        else:
            task_score = 75.0
            conf = 0.5
            std_avg = 0.0

        scores.append(self._make_dim(
            "task_completion", task_score, conf,
            method=EvaluationMethod.LLM_JUDGE, std=std_avg,
            explanation=f"评估{len(cp_evals)}个检查点, 有效{sum(1 for e in cp_evals if e.status != CheckpointStatus.NOT_APPLICABLE)}个",
            details=[
                f"{e.checkpoint_name}: {e.status.value}({e.score:.0f}分){' [conf=' + format(e.confidence, '.2f') + ']' if e.confidence < 0.8 else ''} - {e.evidence[:60]}"
                for e in cp_evals[:8]
            ],
        ))

        # 2. flow_adherence
        flow_score = self.judge._safe_score(flow_result.get("flow_score", 75))
        flow_conf = self.judge._safe_conf(flow_result.get("confidence", 0.7))
        flow_std = float(flow_result.get("score_std", 0.0))
        scores.append(self._make_dim(
            "flow_adherence", flow_score, flow_conf,
            method=EvaluationMethod.LLM_JUDGE, std=flow_std,
            explanation=flow_result.get("flow_analysis", "")[:200],
            details=flow_result.get("deviations", []),
            score_samples=flow_result.get("score_samples", []),
        ))

        # 3. constraint_compliance (HYBRID)
        violated = [c for c in cons_evals if c.violated]
        penalty = 0
        for v in violated:
            penalty += self.SEVERITY_PENALTY.get(v.severity, 20) * max(1, v.violation_count)
        cons_score = max(0.0, 100.0 - penalty)
        # 置信度: 规则的 1.0, LLM 的取自身 - 加权平均
        if cons_evals:
            cons_conf = statistics.mean(c.confidence for c in cons_evals)
        else:
            cons_conf = 1.0
        # 评估方法: 看混合占比
        n_rule = sum(1 for c in cons_evals if c.evaluation_method == EvaluationMethod.RULE)
        method = EvaluationMethod.HYBRID if 0 < n_rule < len(cons_evals) else (
            EvaluationMethod.RULE if n_rule == len(cons_evals) and cons_evals else EvaluationMethod.LLM_JUDGE
        )
        scores.append(self._make_dim(
            "constraint_compliance", cons_score, cons_conf, method=method,
            explanation=f"共{len(cons_evals)}个约束, 违反{len(violated)}个 (扣{penalty}分)",
            details=[
                f"[{c.severity.value}|{c.evaluation_method.value}] {c.constraint_name}: "
                f"{'违反 x' + str(c.violation_count) if c.violated else '遵守'}"
                + (f' - {c.violation_details[0][:60]}' if c.violation_details else '')
                for c in cons_evals
            ],
        ))

        # 4. exception_handling
        exc = quality_result.get("exception_handling", {}) or {}
        exc_score = self.judge._safe_score(exc.get("score", 70))
        exc_conf = self.judge._safe_conf(exc.get("confidence", quality_result.get("confidence", 0.7)))
        scores.append(self._make_dim(
            "exception_handling", exc_score, exc_conf,
            method=EvaluationMethod.LLM_JUDGE,
            explanation=exc.get("explanation", ""),
            details=exc.get("details", []),
        ))

        # 5. dialogue_efficiency (HEURISTIC + LLM 辅助)
        eff = quality_result.get("dialogue_efficiency", {}) or {}
        eff_score_llm = self.judge._safe_score(eff.get("score", 75))
        # 启发式: 轮次效率
        n_user_turns = sum(1 for t in session.turns if t.role == DialogueRole.USER)
        # 经验: 5-10 轮最佳, 越偏离扣分
        if n_user_turns == 0:
            eff_h = 0
        elif n_user_turns <= 3:
            eff_h = 65
        elif 4 <= n_user_turns <= 10:
            eff_h = 90
        elif 11 <= n_user_turns <= 15:
            eff_h = 75
        else:
            eff_h = 60
        # 加权融合: LLM 60%, 启发式 40%
        eff_score = round(0.6 * eff_score_llm + 0.4 * eff_h, 1)
        scores.append(self._make_dim(
            "dialogue_efficiency", eff_score, 0.85,
            method=EvaluationMethod.HYBRID,
            explanation=f"轮次={n_user_turns}, LLM评{eff_score_llm:.0f}, 启发{eff_h}, 融合={eff_score}",
            details=[f"必要轮次={eff.get('necessary_turns', 'n/a')}", f"冗余轮次={eff.get('redundant_turns', 'n/a')}"],
        ))

        # 6. utterance_quality
        utq = quality_result.get("utterance_quality", {}) or {}
        utq_score = self.judge._safe_score(utq.get("score", 75))
        scores.append(self._make_dim(
            "utterance_quality", utq_score, 0.8,
            method=EvaluationMethod.LLM_JUDGE,
            explanation=utq.get("explanation", ""),
            details=utq.get("details", []),
        ))

        # 7. knowledge_accuracy
        kb_score = kb_eval.score
        # 如果用户没问知识库相关, 这是 100 但置信度降低
        kb_conf = 1.0 if kb_eval.total_knowledge_queries > 0 else 0.6
        kb_details = []
        if kb_eval.fabricated_info:
            kb_details.append("编造信息: " + "; ".join(kb_eval.fabricated_info[:3]))
        if kb_eval.incorrect_answers > 0:
            kb_details.append(f"错误回答 {kb_eval.incorrect_answers} 个")
        scores.append(self._make_dim(
            "knowledge_accuracy", kb_score, kb_conf,
            method=EvaluationMethod.LLM_JUDGE,
            explanation=f"查询{kb_eval.total_knowledge_queries}次, 正确{kb_eval.correct_answers}次",
            details=kb_details,
        ))

        # 8. response_brevity (RULE)
        br_details = []
        if utt_eval.violations:
            from collections import Counter
            counter = Counter(v["violation_type"] for v in utt_eval.violations)
            for vt, n in counter.most_common():
                br_details.append(f"{vt}: {n}次")
        scores.append(self._make_dim(
            "response_brevity", utt_eval.score, 1.0,
            method=EvaluationMethod.RULE,
            explanation=f"系统发言{utt_eval.total_system_turns}轮, 违反{utt_eval.violated_turns}轮 (违反率{utt_eval.violation_rate:.1%})",
            details=br_details,
        ))

        return scores

    def _make_dim(
        self,
        key: str,
        score: float,
        confidence: float,
        method: EvaluationMethod,
        explanation: str = "",
        details: list[str] = None,
        std: float = 0.0,
        score_samples: list[float] = None,
    ) -> DimensionScore:
        weight = self.dimension_weights.get(key, 0.10)
        return DimensionScore(
            dimension_key=key,
            dimension_name=DIMENSION_NAMES_CN.get(key, key),
            score=round(self.judge._safe_score(score), 1),
            weight=round(weight, 4),
            weighted_score=round(self.judge._safe_score(score) * weight, 2),
            explanation=explanation,
            details=details or [],
            evaluation_method=method,
            confidence=round(self.judge._safe_conf(confidence), 2),
            score_std=round(std, 2),
            score_samples=score_samples or [],
        )

    # ============ 工具 ============

    @staticmethod
    def _format_dialogue(session: DialogueSession) -> str:
        lines = []
        for t in session.turns:
            label = "系统" if t.role == DialogueRole.SYSTEM else "用户"
            lines.append(f"[轮次{t.turn_id}] {label}: {t.content}")
        return "\n".join(lines)
