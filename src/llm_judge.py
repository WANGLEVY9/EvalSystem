"""
LLM-as-Judge 评估器 (v3.0)

核心改进:
1. 结构化证据强约束 - 每个评估必须返回 quotes (turn_id + verbatim text + note)
2. 后处理校验 - quote.text 必须能在原对话中找到 (字面包含或近似匹配)
3. Self-Consistency - 关键维度可多次评估取中位数 + 标准差
4. 分维度独立调用 - 避免单次大 prompt 输出截断
5. 置信度输出 - 0-1 数值, 多次评估发散时降低置信度
"""

from __future__ import annotations

import json
import logging
import statistics
from typing import Any, Optional

from openai import OpenAI

from .models import (
    Checkpoint,
    CheckpointEvaluation,
    CheckpointStatus,
    Constraint,
    ConstraintEvaluation,
    ConstraintSeverity,
    DialogueRole,
    DialogueSession,
    EvaluationMethod,
    EvidenceQuote,
    KnowledgeAccuracyEvaluation,
    PersonaType,
    TaskInstruction,
)

logger = logging.getLogger(__name__)


# ============ Prompt 模板 ============

CHECKPOINT_PROMPT = """你是对话质量评估专家 (LLM-as-Judge)。基于完整对话, 严格判定每个检查点是否完成。

【任务背景】
{task_description}

【检查点 (待评)】
{checkpoints_json}

【对话原文】(每行格式: [轮次N] 系统/用户: ...)
{dialogue_text}

【输出要求】严格 JSON, 包含字段:
```json
{{"evaluations":[
  {{"checkpoint_id":"...","checkpoint_name":"...",
    "status":"completed|partially|failed|n_a",
    "score":0,
    "reasoning":"判定理由",
    "quotes":[{{"turn_id":N,"text":"对话中的原文片段(逐字引用)","note":"为何引用"}}],
    "confidence":0.0-1.0
  }}
]}}
```

【评分规则】
- completed=100: 完整准确完成
- partially=40-70: 部分完成或质量瑕疵
- failed=0: 未完成
- n_a: 条件未触发(如"若用户说不在家则改约"但用户没说不在家)
- 必须为每个非 n_a 的判定提供至少 1 条 quotes (从对话原文逐字截取)
- confidence 反映你的判定把握度: <0.6 表示证据不足

只输出 JSON, 不要解释。"""


CONSTRAINT_PROMPT = """你是对话合规评估专家。判定被测系统是否违反了下述语义类约束。

【任务背景】
{task_description}

【约束 (待评 - 仅含语义类)】
{constraints_json}

【对话原文】
{dialogue_text}

【输出要求】严格 JSON:
```json
{{"evaluations":[
  {{"constraint_id":"...","constraint_name":"...",
    "violated":true|false,"violation_count":N,
    "severity":"critical|major|minor",
    "violation_details":["违反描述1"],
    "quotes":[{{"turn_id":N,"text":"违规原文(逐字)","note":"为何违规"}}],
    "confidence":0.0-1.0
  }}
]}}
```

只输出 JSON, 不要解释。"""


FLOW_PROMPT = """你是对话流程分析专家。请分析对话是否按预期流程推进、分支处理是否正确。

【任务背景】
{task_description}

【期望流程】
{flow_description}

【条件分支】
{branch_details}

【对话原文】
{dialogue_text}

【输出要求】严格 JSON:
```json
{{
  "flow_score":0-100,
  "flow_analysis":"流程分析(包含偏离说明)",
  "expected_path":"...",
  "actual_path":"步骤1→步骤2→...",
  "deviations":["偏离1"],
  "branches_triggered":["在步骤4触发了'未显示'分支"],
  "is_reasonable":true|false,
  "evidence_quotes":[{{"turn_id":N,"text":"...","note":"..."}}],
  "confidence":0.0-1.0
}}
```

【评分参考】
- 90-100: 完全按流程推进, 分支处理正确
- 70-89: 基本按流程, 小偏离合理
- 50-69: 有明显偏离但完成核心
- <50: 流程混乱

只输出 JSON。"""


QUALITY_PROMPT = """你是对话质量评估专家。评估异常处理 / 对话效率 / 话术质量。

【任务背景】
{task_description}

【用户画像】{persona_type}

【对话原文】
{dialogue_text}

【输出要求】严格 JSON:
```json
{{
  "exception_handling":{{
    "score":0-100,"explanation":"...","details":["..."],
    "quotes":[{{"turn_id":N,"text":"...","note":"..."}}]
  }},
  "dialogue_efficiency":{{
    "score":0-100,"explanation":"...",
    "total_turns":N,"necessary_turns":N,"redundant_turns":N
  }},
  "utterance_quality":{{
    "score":0-100,"explanation":"...","details":["..."]
  }},
  "overall_comment":"...",
  "issues":["问题1"],
  "strengths":["优点1"],
  "confidence":0.0-1.0
}}
```

只输出 JSON。"""


KNOWLEDGE_PROMPT = """你是知识准确度评估专家。评估系统是否准确引用了知识库, 有无幻觉。

【任务背景】
{task_description}

【知识库】(以下是事实, 系统提及这些事实就算正确)
{knowledge_base_json}

【对话原文】
{dialogue_text}

【输出要求】严格 JSON:
```json
{{
  "total_knowledge_queries":N,
  "correct_answers":N,
  "incorrect_answers":N,
  "fabricated_info":["真正编造、与知识库矛盾的内容"],
  "score":0-100,
  "details":[
    {{"query":"用户问题","expected":"知识库标准答案","actual":"系统实际回答","correct":true|false,"turn_id":N}}
  ],
  "quotes":[{{"turn_id":N,"text":"...","note":"..."}}],
  "confidence":0.0-1.0
}}
```

【关键判定原则】
- 系统说的数字/事实 **如果与知识库一致就算正确**, 即使知识库里这个数字以"X 单"形式出现, 系统说"5 单"也算正确(系统已替换占位符)
- 只有当系统说的内容 **与知识库矛盾** 或 **知识库根本没提的事实** 才算 fabricated_info
- 用户没问知识库相关问题 -> total_knowledge_queries=0, score=100
- 系统主动告知的事实(如开场白里的核心数字) 也算 knowledge_query, 需要核对

只输出 JSON。"""


# ============ LLM Judge 类 ============

class LLMJudge:
    """LLM-as-Judge - 带证据三元组、self-consistency、置信度"""

    def __init__(
        self,
        llm_client: OpenAI,
        model: str = "deepseek-chat",
        n_self_consistency: int = 1,
        temperature: float = 0.0,
    ):
        self.llm_client = llm_client
        self.model = model
        self.n_sc = max(1, n_self_consistency)
        self.temperature = temperature

    # ---- 检查点 ----

    def evaluate_checkpoints(
        self,
        instruction: TaskInstruction,
        dialogue_text: str,
        session: DialogueSession,
    ) -> list[CheckpointEvaluation]:
        """评估检查点; 仅评估 *非派生* 的 checkpoint, 派生的留给流程评估"""
        # 优先评估非派生检查点; 若全是派生的 (markdown 规则模式), 则全部评估
        target_cps = [cp for cp in instruction.checkpoints if not cp.derived_from_flow]
        if not target_cps:
            target_cps = instruction.checkpoints
        if not target_cps:
            return []

        cps_data = [
            {
                "id": cp.id,
                "name": cp.name,
                "description": cp.description,
                "type": cp.type.value,
                "evaluation_criteria": cp.evaluation_criteria,
                "condition": cp.condition,
                "at_flow_step": cp.at_flow_step,
            }
            for cp in target_cps
        ]
        prompt = CHECKPOINT_PROMPT.format(
            task_description=instruction.task_description,
            checkpoints_json=json.dumps(cps_data, ensure_ascii=False, indent=2),
            dialogue_text=dialogue_text,
        )

        # Self-consistency
        all_runs: list[dict[str, Any]] = []
        for i in range(self.n_sc):
            data = self._call_judge(prompt, label=f"checkpoints#{i + 1}")
            if data and "evaluations" in data:
                all_runs.append(data)

        if not all_runs:
            return []

        # 合并: 按 checkpoint_id 聚合
        merged: dict[str, list[dict[str, Any]]] = {}
        for run in all_runs:
            for ev in run.get("evaluations", []):
                cid = str(ev.get("checkpoint_id") or ev.get("checkpoint_name", ""))
                merged.setdefault(cid, []).append(ev)

        # 构建 CheckpointEvaluation
        result: list[CheckpointEvaluation] = []
        cp_by_id = {cp.id: cp for cp in target_cps}
        cp_by_name = {cp.name: cp for cp in target_cps}

        for cid, runs in merged.items():
            cp = cp_by_id.get(cid) or cp_by_name.get(runs[0].get("checkpoint_name", ""))
            if not cp:
                continue

            scores = [self._safe_score(r.get("score", 0)) for r in runs]
            statuses = [r.get("status", "failed") for r in runs]
            confs = [self._safe_conf(r.get("confidence", 0.8)) for r in runs]

            # 中位数分 + 多数投票状态
            score = statistics.median(scores) if scores else 0
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            status_str = max(set(statuses), key=statuses.count)
            confidence = float(statistics.mean(confs))
            # 多次评估发散惩罚
            if std > 15:
                confidence *= 0.7

            # quotes: 取第一次的, 校验
            quotes_raw = runs[0].get("quotes", [])
            quotes = self._validate_quotes(quotes_raw, session)

            evidence_text = "; ".join(r.get("reasoning", "") for r in runs[:1]).strip()
            if not evidence_text and quotes:
                evidence_text = quotes[0].text[:80]

            status_map = {
                "completed": CheckpointStatus.COMPLETED,
                "partially": CheckpointStatus.PARTIALLY,
                "partial": CheckpointStatus.PARTIALLY,
                "failed": CheckpointStatus.FAILED,
                "fail": CheckpointStatus.FAILED,
                "n_a": CheckpointStatus.NOT_APPLICABLE,
                "n/a": CheckpointStatus.NOT_APPLICABLE,
                "na": CheckpointStatus.NOT_APPLICABLE,
            }
            result.append(CheckpointEvaluation(
                checkpoint_id=cp.id,
                checkpoint_name=cp.name,
                status=status_map.get(status_str, CheckpointStatus.FAILED),
                score=round(score, 1),
                evidence=evidence_text,
                relevant_turns=sorted({q.turn_id for q in quotes}),
                quotes=quotes,
                confidence=round(confidence, 2),
                evaluation_method=EvaluationMethod.LLM_JUDGE,
                score_samples=[round(s, 1) for s in scores],
                score_std=round(std, 2),
            ))
        return result

    # ---- 约束 ----

    def evaluate_semantic_constraints(
        self,
        instruction: TaskInstruction,
        dialogue_text: str,
        session: DialogueSession,
    ) -> list[ConstraintEvaluation]:
        """仅评估语义类约束 (is_deterministic=False)"""
        targets = [c for c in instruction.constraints if not c.is_deterministic]
        if not targets:
            return []

        data = [
            {
                "id": c.id, "name": c.name, "description": c.description,
                "severity": c.severity.value, "violation_examples": c.violation_examples,
            }
            for c in targets
        ]
        prompt = CONSTRAINT_PROMPT.format(
            task_description=instruction.task_description,
            constraints_json=json.dumps(data, ensure_ascii=False, indent=2),
            dialogue_text=dialogue_text,
        )
        result_json = self._call_judge(prompt, label="semantic_constraints")
        if not result_json:
            return []

        c_by_id = {c.id: c for c in targets}
        c_by_name = {c.name: c for c in targets}

        out: list[ConstraintEvaluation] = []
        for ev in result_json.get("evaluations", []):
            cid = str(ev.get("constraint_id") or "")
            cobj = c_by_id.get(cid) or c_by_name.get(ev.get("constraint_name", ""))
            if not cobj:
                continue
            try:
                sev = ConstraintSeverity(ev.get("severity", cobj.severity.value))
            except ValueError:
                sev = cobj.severity
            quotes = self._validate_quotes(ev.get("quotes", []), session)
            out.append(ConstraintEvaluation(
                constraint_id=cobj.id,
                constraint_name=cobj.name,
                violated=bool(ev.get("violated", False)),
                violation_count=int(ev.get("violation_count", 0) or 0),
                violation_details=ev.get("violation_details", []),
                relevant_turns=sorted({q.turn_id for q in quotes}),
                severity=sev,
                quotes=quotes,
                evaluation_method=EvaluationMethod.LLM_JUDGE,
                confidence=self._safe_conf(ev.get("confidence", 0.8)),
            ))
        return out

    # ---- 流程 ----

    def evaluate_flow(
        self,
        instruction: TaskInstruction,
        dialogue_text: str,
        session: DialogueSession,
    ) -> dict[str, Any]:
        if not instruction.flow_nodes:
            return {"flow_score": 75, "flow_analysis": "未定义流程节点", "branches_triggered": [], "confidence": 0.5}

        flow_desc = "\n".join(
            f"- [{n.step_number}] {n.name}: {n.description} (后续: {', '.join(n.next_nodes) if n.next_nodes else '结束'})"
            for n in instruction.flow_nodes
        )
        branch_lines: list[str] = []
        for n in instruction.flow_nodes:
            for b in n.branches:
                branch_lines.append(f"  - 在{n.name}中: 若{b.condition} → {b.action}")
            for nx, c in n.conditions.items():
                branch_lines.append(f"  - {n.name} → {nx}: 当 {c}")

        prompt = FLOW_PROMPT.format(
            task_description=instruction.task_description,
            flow_description=flow_desc,
            branch_details="\n".join(branch_lines) if branch_lines else "无显式分支",
            dialogue_text=dialogue_text,
        )

        runs = []
        for i in range(self.n_sc):
            r = self._call_judge(prompt, label=f"flow#{i + 1}")
            if r:
                runs.append(r)

        if not runs:
            return {"flow_score": 70, "flow_analysis": "评估失败, 默认 70 分", "branches_triggered": [], "confidence": 0.4}

        scores = [self._safe_score(r.get("flow_score", 70)) for r in runs]
        score = statistics.median(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        confs = [self._safe_conf(r.get("confidence", 0.8)) for r in runs]
        confidence = statistics.mean(confs)
        if std > 15:
            confidence *= 0.7

        first = runs[0]
        first["flow_score"] = round(score, 1)
        first["confidence"] = round(confidence, 2)
        first["score_std"] = round(std, 2)
        first["score_samples"] = [round(s, 1) for s in scores]
        # 校验 quotes
        first["evidence_quotes"] = [
            q.model_dump() for q in self._validate_quotes(first.get("evidence_quotes", []), session)
        ]
        return first

    # ---- 质量 (异常处理 / 效率 / 话术) ----

    def evaluate_quality(
        self,
        instruction: TaskInstruction,
        dialogue_text: str,
        persona_type: PersonaType,
    ) -> dict[str, Any]:
        persona_names = {
            PersonaType.COOPERATIVE: "配合型", PersonaType.HESITANT: "犹豫型",
            PersonaType.RESISTANT: "抗拒型", PersonaType.OFF_TOPIC: "跑题型",
            PersonaType.CONTRADICTORY: "信息矛盾型", PersonaType.BOUNDARY: "边界测试型",
            PersonaType.BUSY: "忙碌型", PersonaType.CONFUSED: "困惑型",
            PersonaType.IMPATIENT: "急躁型",
            PersonaType.RED_TEAM_L1: "红队-轻度刺探",
            PersonaType.RED_TEAM_L2: "红队-社工诱导",
            PersonaType.RED_TEAM_L3: "红队-prompt注入",
        }
        prompt = QUALITY_PROMPT.format(
            task_description=instruction.task_description,
            dialogue_text=dialogue_text,
            persona_type=persona_names.get(persona_type, persona_type.value),
        )
        return self._call_judge(prompt, label="quality") or {}

    # ---- 知识 ----

    def evaluate_knowledge(
        self,
        instruction: TaskInstruction,
        dialogue_text: str,
        session: DialogueSession,
    ) -> KnowledgeAccuracyEvaluation:
        if not instruction.knowledge_base:
            return KnowledgeAccuracyEvaluation(score=100)

        kb_data = [
            {"id": kb.id, "question": kb.question, "answer": kb.answer,
             "keywords": kb.keywords, "category": kb.category}
            for kb in instruction.knowledge_base
        ]
        prompt = KNOWLEDGE_PROMPT.format(
            task_description=instruction.task_description,
            knowledge_base_json=json.dumps(kb_data, ensure_ascii=False, indent=2),
            dialogue_text=dialogue_text,
        )
        r = self._call_judge(prompt, label="knowledge") or {}
        quotes = self._validate_quotes(r.get("quotes", []), session)
        return KnowledgeAccuracyEvaluation(
            total_knowledge_queries=int(r.get("total_knowledge_queries", 0) or 0),
            correct_answers=int(r.get("correct_answers", 0) or 0),
            incorrect_answers=int(r.get("incorrect_answers", 0) or 0),
            fabricated_info=r.get("fabricated_info", []),
            score=self._safe_score(r.get("score", 100)),
            details=r.get("details", []),
            quotes=quotes,
        )

    # ============ 工具 ============

    def _call_judge(self, prompt: str, label: str = "") -> Optional[dict[str, Any]]:
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "你是严谨的对话评估专家。严格按要求输出 JSON 对象, 不输出任何额外文字。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            return self._parse_json(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"LLM Judge 调用失败 [{label}]: {e}")
            return None

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        if not text:
            return {}
        if "```json" in text:
            s = text.find("```json") + 7
            e = text.find("```", s)
            if e != -1:
                text = text[s:e].strip()
        elif "```" in text:
            s = text.find("```") + 3
            e = text.find("```", s)
            if e != -1:
                text = text[s:e].strip()
        if not text.lstrip().startswith("{"):
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1:
                text = text[s:e + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _safe_score(v: Any) -> float:
        try:
            return max(0.0, min(100.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_conf(v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _validate_quotes(
        raw_quotes: list[Any],
        session: DialogueSession,
    ) -> list[EvidenceQuote]:
        """校验 quote.text 是否真的出现在对话中

        - 完全包含: 接受, 置信度 1.0
        - 高度相似 (子串/前缀): 接受, 置信度 0.7
        - 不匹配: 丢弃 (反幻觉)
        """
        if not raw_quotes:
            return []

        turn_lookup: dict[int, str] = {t.turn_id: t.content for t in session.turns}
        out: list[EvidenceQuote] = []
        for q in raw_quotes:
            if not isinstance(q, dict):
                continue
            try:
                tid = int(q.get("turn_id"))
            except (TypeError, ValueError):
                continue
            text = str(q.get("text", "")).strip().strip('"').strip('"').strip("'")
            note = str(q.get("note", "")).strip()
            if not text or tid not in turn_lookup:
                continue
            content = turn_lookup[tid]
            # 验证: 子串
            if text in content:
                out.append(EvidenceQuote(
                    turn_id=tid,
                    role=DialogueRole.SYSTEM if any(t.turn_id == tid and t.role == DialogueRole.SYSTEM for t in session.turns) else DialogueRole.USER,
                    text=text,
                    note=note,
                ))
            elif len(text) >= 4 and text[:6] in content:
                # 前缀匹配 (LLM 可能轻微改写)
                out.append(EvidenceQuote(
                    turn_id=tid,
                    role=DialogueRole.SYSTEM if any(t.turn_id == tid and t.role == DialogueRole.SYSTEM for t in session.turns) else DialogueRole.USER,
                    text=text,
                    note=(note + " [近似匹配]").strip(),
                ))
            # 否则丢弃 (反幻觉)
        return out
