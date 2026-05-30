"""
指令编译器 (v3.0)

在 InstructionParser 解析的基础上，进一步将 TaskInstruction 编译为评测系统可直接执行的表征：

1. 复杂度量化  — InstructionComplexity
2. 约束分类   — 区分 deterministic vs semantic
3. 流程 DAG   — 节点 + 分支的有向图，附带 BFS 拓扑序
4. 变量绑定表 — 自动从 ${var}/**X 单** 抽取所有占位符
5. 目标分支   — 列出本次评测应覆盖的所有 (节点, 条件) 元组
6. 检查点去重 — 标记从 flow_node 派生的 checkpoint，避免双重计分

设计原则:
- 编译过程不调用 LLM，纯本地计算
- 编译结果是不可变的、可序列化的、可复算的
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

from .models import (
    Constraint,
    ConstraintCheckMethod,
    ConstraintSeverity,
    InstructionComplexity,
    TaskInstruction,
)

logger = logging.getLogger(__name__)


# ============ 复杂度计算 ============

def compute_complexity(instruction: TaskInstruction) -> InstructionComplexity:
    """计算指令的多维复杂度并合成单一指标"""
    n_cp = len(instruction.checkpoints)
    n_cons = len(instruction.constraints)
    n_flow = len(instruction.flow_nodes)
    n_branches = sum(len(n.branches) for n in instruction.flow_nodes)
    n_kb = len(instruction.knowledge_base)
    n_uc = len(instruction.utterance_constraints)
    n_st = len(instruction.scenario_triggers)
    n_var = len(instruction.variables)
    raw_chars = len(instruction.raw_instruction or "")

    # 最大分支深度: 流程节点中 sub_steps + branches 嵌套深度的近似
    max_depth = 0
    for node in instruction.flow_nodes:
        d = 1
        if node.branches:
            d += 1
        if node.sub_steps:
            d += 1
        max_depth = max(max_depth, d)

    # 综合复杂度分 0-100, 经验权重 (基于 xlsx 两条样本调参)
    raw = (
        n_cp * 1.5
        + n_cons * 2.0
        + n_flow * 2.5
        + n_branches * 4.0
        + n_kb * 1.2
        + n_uc * 2.0
        + n_st * 2.5
        + n_var * 1.5
        + max_depth * 3
        + math.log10(max(raw_chars, 10)) * 5
    )
    # 平滑映射: 0-100
    score = min(100.0, raw)
    if score < 25:
        level = "simple"
    elif score < 50:
        level = "medium"
    elif score < 75:
        level = "complex"
    else:
        level = "very_complex"

    return InstructionComplexity(
        n_checkpoints=n_cp,
        n_constraints=n_cons,
        n_flow_nodes=n_flow,
        n_branches=n_branches,
        n_knowledge_entries=n_kb,
        n_utterance_constraints=n_uc,
        n_scenario_triggers=n_st,
        n_variables=n_var,
        max_branch_depth=max_depth,
        raw_chars=raw_chars,
        complexity_score=round(score, 1),
        complexity_level=level,
    )


# ============ 约束分类: 确定性 vs 语义 ============

DET_PATTERNS = [
    # 字数约束
    (re.compile(r"(?:每次回复|单句|每[句条]|回复).{0,12}?(?:最多|不超过|控制在|限制|≤|≦|约|不要超过).{0,8}?(\d+)\s*[-~到至个]?\s*(\d+)?\s*[个字字符]"),
     "char_count"),
    (re.compile(r"(\d+)\s*[-~到至]\s*(\d+)\s*[个字字符]"), "char_count"),
    (re.compile(r"(?:不超过|最多)\s*(\d+)\s*[个字字符]"), "char_count"),
    # 禁词 (中文/英文/普通引号)
    (re.compile(r'不[说用]"([^"]+)"'), "forbidden_words"),
    (re.compile(r'不[说用]"([^"]+)"'), "forbidden_words"),  # 普通双引号
    (re.compile(r'不能?(?:说|用|使用)([^\n。；]{0,40}?)等(?:语气词|词汇|表达)'), "forbidden_words"),
    (re.compile(r'避免[说用]([^\n。；]{0,40}?)等'), "forbidden_words"),
    # 承诺/折扣类显式禁止
    (re.compile(r"不能?(?:承诺|保证|许诺)"), "forbidden_words"),
    (re.compile(r"(?:禁止|不[得能要])(?:承诺|保证|提供)(?:折扣|优惠|赠品)"), "forbidden_words"),
]


def classify_constraint(constraint: Constraint) -> tuple[bool, str]:
    """判断约束是否可由规则确定性判定"""
    text = (constraint.description or "") + " " + (constraint.name or "")
    for pattern, method in DET_PATTERNS:
        if pattern.search(text):
            return True, method

    # 已有的 check_method 显式标注
    if constraint.check_method in {"char_count", "keyword", "regex", "forbidden_words", "keyword_presence"}:
        return True, constraint.check_method

    return False, "llm_judge"


def annotate_constraints(instruction: TaskInstruction) -> None:
    """就地为约束打上 is_deterministic 标签"""
    for c in instruction.constraints:
        is_det, method = classify_constraint(c)
        c.is_deterministic = is_det
        # 不覆盖已显式设置的 check_method, 仅做补充建议
        if c.check_method == "llm_judge" and is_det:
            c.check_method = method


# ============ 流程 DAG 与分支编译 ============

def build_flow_dag(instruction: TaskInstruction) -> dict[str, dict[str, Any]]:
    """构建流程节点的有向图

    Returns:
        {node_name: {"node": FlowNode, "successors": [name], "branches": [...]}}
    """
    dag: dict[str, dict[str, Any]] = {}
    for n in instruction.flow_nodes:
        dag[n.name] = {
            "node": n,
            "successors": list(n.next_nodes),
            "branches": list(n.branches),
            "is_start": n.is_start,
            "is_end": n.is_end,
        }
    return dag


def list_target_branches(instruction: TaskInstruction) -> list[str]:
    """生成本次评测应当覆盖的所有分支字符串列表

    格式: "{node_name}::{condition}"
    """
    targets: list[str] = []
    for node in instruction.flow_nodes:
        for branch in node.branches:
            targets.append(f"{node.name}::{branch.condition}")
        # conditions 字典 (老格式兼容)
        for next_node, cond in node.conditions.items():
            if cond:
                targets.append(f"{node.name}->{next_node}::{cond}")
    return targets


# ============ 变量绑定表 ============

VAR_PATTERNS = [
    re.compile(r"\$\{([^}]+)\}"),  # ${name}
    re.compile(r"\*\*([A-Z])\s+([单天元个次分时])\*\*"),  # **X 单**
    re.compile(r"\*\*([A-Z]\s*[单天元个次分时])\*\*"),  # **X单**
]


def extract_variable_placeholders(text: str) -> set[str]:
    """从文本里抽取所有变量占位符"""
    placeholders = set()
    for pattern in VAR_PATTERNS:
        for m in pattern.finditer(text):
            placeholders.add(m.group(0))
    return placeholders


# ============ 检查点去重标记 ============

def mark_derived_checkpoints(instruction: TaskInstruction) -> None:
    """标记由 flow_node 派生的检查点

    规则: 如果一个 checkpoint 的 at_flow_step 等于某个 flow_node 的 name,
    或 checkpoint name 是 "步骤N" 形式且与 flow_node 一一对应，则视为派生。
    """
    flow_names = {n.name for n in instruction.flow_nodes}

    for cp in instruction.checkpoints:
        # 显式关联
        if cp.at_flow_step and cp.at_flow_step in flow_names:
            cp.derived_from_flow = True
            continue
        # 命名一致
        if cp.name in flow_names:
            cp.derived_from_flow = True
            continue
        # "步骤N" 形式
        if re.fullmatch(r"步骤\d+", cp.name) and cp.name in flow_names:
            cp.derived_from_flow = True


# ============ 主编译器 ============

class InstructionCompiler:
    """指令编译器 - 把 TaskInstruction 编译为可执行表征"""

    def __init__(self, instruction: TaskInstruction):
        self.instruction = instruction
        self.dag: dict[str, dict[str, Any]] = {}
        self.target_branches: list[str] = []
        self.variable_placeholders: set[str] = set()

    def compile(self) -> TaskInstruction:
        """执行完整编译流程，原地修改 instruction 并返回"""
        instr = self.instruction
        logger.info(f"开始编译指令: {instr.task_name}")

        # 1. 复杂度
        instr.complexity = compute_complexity(instr)

        # 2. 约束分类
        annotate_constraints(instr)

        # 3. 检查点去重标记
        mark_derived_checkpoints(instr)

        # 4. DAG
        self.dag = build_flow_dag(instr)

        # 5. 目标分支
        self.target_branches = list_target_branches(instr)

        # 6. 变量占位符
        self.variable_placeholders = (
            extract_variable_placeholders(instr.raw_instruction or "")
            | extract_variable_placeholders(instr.opening_line or "")
            | extract_variable_placeholders(instr.system_prompt or "")
        )

        det_count = sum(1 for c in instr.constraints if c.is_deterministic)
        derived_count = sum(1 for cp in instr.checkpoints if cp.derived_from_flow)

        logger.info(
            f"编译完成: 复杂度={instr.complexity.complexity_score} ({instr.complexity.complexity_level}) | "
            f"DAG节点={len(self.dag)} | 目标分支={len(self.target_branches)} | "
            f"确定性约束={det_count}/{len(instr.constraints)} | "
            f"派生检查点={derived_count}/{len(instr.checkpoints)}"
        )

        return instr

    def get_compile_summary(self) -> dict[str, Any]:
        """返回编译产物的摘要 (用于报告)"""
        return {
            "complexity": self.instruction.complexity.model_dump() if self.instruction.complexity else None,
            "dag_nodes": list(self.dag.keys()),
            "target_branches": self.target_branches,
            "variable_placeholders": sorted(self.variable_placeholders),
            "deterministic_constraints": [
                c.name for c in self.instruction.constraints if c.is_deterministic
            ],
            "semantic_constraints": [
                c.name for c in self.instruction.constraints if not c.is_deterministic
            ],
            "derived_checkpoints": [
                cp.name for cp in self.instruction.checkpoints if cp.derived_from_flow
            ],
        }
