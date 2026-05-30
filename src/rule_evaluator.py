"""
规则评估器 (v3.0)

确定性评估: 不需要 LLM, 完全可复算

涵盖:
- 单句字数限制 (max_chars / min_chars)
- 禁用词列表
- 必出关键词
- 系统级禁词约束 (从 Constraint.description 反向解析)
- 变量未替换检测 (输出含 ${var}/**X 单** 即为 bug)
"""
from __future__ import annotations

import re
from typing import Any

from .models import (
    Constraint,
    ConstraintEvaluation,
    ConstraintSeverity,
    DialogueRole,
    DialogueSession,
    EvaluationMethod,
    EvidenceQuote,
    TaskInstruction,
    UtteranceConstraint,
    UtteranceConstraintEvaluation,
)


class RuleEvaluator:
    """确定性规则评估器 - 无需 LLM, 完全可复算"""

    # ============ 话术约束 (utterance_constraints) ============

    def evaluate_utterance_constraints(
        self,
        session: DialogueSession,
        instruction: TaskInstruction,
    ) -> UtteranceConstraintEvaluation:
        if not instruction.utterance_constraints:
            return UtteranceConstraintEvaluation(score=100)

        violations: list[dict[str, Any]] = []
        system_turns = [t for t in session.turns if t.role == DialogueRole.SYSTEM]
        total = len(system_turns)
        # 句级评估: 把每条 system turn 按 ; 。 ! ? ? . 拆成句子, 字数限制按句子算
        for turn in system_turns:
            content = turn.content
            sentences = self._split_sentences(content)

            for uc in instruction.utterance_constraints:
                if uc.applies_to == "opening" and turn.turn_id != 0:
                    continue
                if uc.applies_to == "closing" and turn != system_turns[-1]:
                    continue

                # 字数 (按句子)
                if uc.max_chars is not None:
                    for s in sentences:
                        if len(s) > uc.max_chars:
                            violations.append({
                                "turn_id": turn.turn_id,
                                "content": (content[:60] + "...") if len(content) > 60 else content,
                                "violation_type": "超出字数限制",
                                "detail": f"句子'{s[:30]}...'实际{len(s)}字, 限制{uc.max_chars}字",
                                "constraint_name": uc.name,
                                "sentence": s,
                            })
                if uc.min_chars is not None:
                    if len(content) < uc.min_chars:
                        violations.append({
                            "turn_id": turn.turn_id,
                            "content": content,
                            "violation_type": "低于最少字数",
                            "detail": f"实际{len(content)}字, 要求至少{uc.min_chars}字",
                            "constraint_name": uc.name,
                        })

                # 禁词
                for word in uc.forbidden_words:
                    if not word:
                        continue
                    # 匹配 word 完整出现, 简单包含
                    if word in content:
                        violations.append({
                            "turn_id": turn.turn_id,
                            "content": (content[:60] + "...") if len(content) > 60 else content,
                            "violation_type": "使用禁止词",
                            "detail": f"使用了禁止词 '{word}'",
                            "constraint_name": uc.name,
                            "matched_word": word,
                        })

        violated_turn_ids = set(v["turn_id"] for v in violations)
        rate = len(violated_turn_ids) / total if total else 0
        # 扣分: 每违反一次扣 5 分, 但不再线性叠加超过 100
        score = max(0.0, 100.0 - len(violations) * 5)
        return UtteranceConstraintEvaluation(
            total_system_turns=total,
            violated_turns=len(violated_turn_ids),
            violation_rate=round(rate, 3),
            violations=violations,
            score=round(score, 1),
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """将一条 utterance 切分成句子"""
        # 中英文标点
        parts = re.split(r"[。！？；!?;]", text)
        return [p.strip() for p in parts if p.strip()]

    # ============ 系统级约束 (Constraint) - 仅处理 deterministic 部分 ============

    def evaluate_deterministic_constraint(
        self,
        constraint: Constraint,
        session: DialogueSession,
    ) -> ConstraintEvaluation | None:
        """对单个 deterministic 约束做规则化评估; 无法判定时返回 None"""
        if not constraint.is_deterministic:
            return None

        text = constraint.description
        system_turns = [t for t in session.turns if t.role == DialogueRole.SYSTEM]

        # 字数类约束
        m = re.search(r"(\d+)\s*[-~到至]?\s*(\d+)?\s*[个]?[字字符]", text)
        if m and ("不超过" in text or "最多" in text or "控制" in text or "约" in text or "以内" in text):
            high = int(m.group(2)) if m.group(2) else int(m.group(1))
            violations = []
            quotes: list[EvidenceQuote] = []
            for t in system_turns:
                # 句子级别字数
                for s in re.split(r"[。！？；!?;]", t.content):
                    s = s.strip()
                    if s and len(s) > high:
                        violations.append(f"轮次{t.turn_id}句子超{high}字: '{s[:30]}...' ({len(s)}字)")
                        quotes.append(EvidenceQuote(
                            turn_id=t.turn_id, role=DialogueRole.SYSTEM, text=s,
                            note=f"超出 {high} 字限制 (实际 {len(s)} 字)",
                        ))
            return ConstraintEvaluation(
                constraint_id=constraint.id,
                constraint_name=constraint.name,
                violated=bool(violations),
                violation_count=len(violations),
                violation_details=violations,
                relevant_turns=[q.turn_id for q in quotes],
                severity=constraint.severity,
                quotes=quotes,
                evaluation_method=EvaluationMethod.RULE,
                confidence=1.0,
            )

        # 禁词类
        forbidden = self._extract_forbidden_from_constraint(text)
        if forbidden:
            violations = []
            quotes = []
            for t in system_turns:
                for w in forbidden:
                    if w in t.content:
                        violations.append(f"轮次{t.turn_id}使用禁词'{w}'")
                        quotes.append(EvidenceQuote(
                            turn_id=t.turn_id, role=DialogueRole.SYSTEM, text=t.content,
                            note=f"使用禁词 '{w}'",
                        ))
            return ConstraintEvaluation(
                constraint_id=constraint.id,
                constraint_name=constraint.name,
                violated=bool(violations),
                violation_count=len(violations),
                violation_details=violations,
                relevant_turns=sorted(set(q.turn_id for q in quotes)),
                severity=constraint.severity,
                quotes=quotes,
                evaluation_method=EvaluationMethod.RULE,
                confidence=1.0,
            )

        return None

    @staticmethod
    def _extract_forbidden_from_constraint(text: str) -> list[str]:
        words: list[str] = []
        for pat in [r'不[说用]"([^"]{1,15})"', r'不[说用]"([^"]{1,15})"', r"不[说用]'([^']{1,15})'"]:
            words.extend(re.findall(pat, text))
        # "不说"好的"、"哈哈"" 形式
        for raw in re.findall(r"不[说用](.+?)等[语气词词汇]", text):
            for w in re.split(r"[、，,/]+", raw):
                w = w.strip().strip('"').strip('"')
                if w:
                    words.append(w)
        return [w for w in words if w]

    # ============ 变量未替换检测 ============

    def detect_unresolved_variables(self, session: DialogueSession) -> list[EvidenceQuote]:
        """检测系统话语里是否有未替换的占位符 (硬 bug)"""
        bad: list[EvidenceQuote] = []
        pattern = re.compile(r"(\$\{[^}]+\})|\*\*([A-Z]\s*[单天元个次分时])\*\*")
        for t in session.turns:
            if t.role != DialogueRole.SYSTEM:
                continue
            for m in pattern.finditer(t.content):
                bad.append(EvidenceQuote(
                    turn_id=t.turn_id, role=DialogueRole.SYSTEM, text=t.content,
                    note=f"未替换占位符: {m.group(0)}",
                ))
        return bad
