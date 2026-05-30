"""
数据模型定义 (v3.0)

核心设计目标:
1. 可解释 (Explainability)
   - 每个评分都能引用具体对话轮次和原文片段 (EvidenceQuote)
   - 区分判定方式: 规则确定性 vs LLM 语义判断 (evaluation_method)
2. 可量化 (Quantifiability)
   - 多次评估的置信区间 / 标准差 (confidence, std_dev)
   - 复算元数据 (run_id, seed, evaluator_version, model_version)
3. 可靠性 (Reliability)
   - 指令复杂度量化 (InstructionComplexity)
   - 分支覆盖矩阵 (BranchCoverageMatrix)
   - 模型短板诊断 (ModelWeaknessProfile)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

EVALUATOR_VERSION = "3.0.0"


# ============ 枚举定义 ============

class CheckpointType(str, Enum):
    MUST_DO = "must_do"
    SHOULD_DO = "should_do"
    OPTIONAL = "optional"


class CheckpointStatus(str, Enum):
    COMPLETED = "completed"
    PARTIALLY = "partially"
    FAILED = "failed"
    NOT_APPLICABLE = "n/a"


class ConstraintSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class PersonaType(str, Enum):
    COOPERATIVE = "cooperative"
    HESITANT = "hesitant"
    RESISTANT = "resistant"
    OFF_TOPIC = "off_topic"
    CONTRADICTORY = "contradictory"
    BOUNDARY = "boundary"
    BUSY = "busy"
    CONFUSED = "confused"
    IMPATIENT = "impatient"
    # v3 新增 - 红队分层
    RED_TEAM_L1 = "red_team_l1"  # 轻度刺探越权请求
    RED_TEAM_L2 = "red_team_l2"  # 社工诱导
    RED_TEAM_L3 = "red_team_l3"  # prompt-injection 攻击


class DialogueRole(str, Enum):
    SYSTEM = "system"
    USER = "user"


class EvaluationMethod(str, Enum):
    """评估判定方式 - 用于可解释性"""
    RULE = "rule"             # 确定性规则 (字数, 禁词, 正则)
    LLM_JUDGE = "llm_judge"   # LLM 语义判定
    HYBRID = "hybrid"         # 规则 + LLM 混合
    HEURISTIC = "heuristic"   # 启发式 (轮次效率等)


class InstructionFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    AUTO = "auto"


class ConstraintCheckMethod(str, Enum):
    """约束的检查方式 - 区分确定性 vs 语义"""
    CHAR_COUNT = "char_count"       # 字数限制
    FORBIDDEN_WORDS = "forbidden_words"  # 禁用词列表
    REGEX = "regex"                 # 正则匹配
    KEYWORD_PRESENCE = "keyword_presence"  # 关键词必须出现
    LLM_JUDGE = "llm_judge"         # LLM 语义判定


# ============ 知识库与变量模型 ============

class KnowledgeEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str = Field(..., description="用户可能问的问题")
    answer: str = Field(..., description="期望的正确回答")
    keywords: list[str] = Field(default_factory=list)
    category: str = Field(default="general")
    priority: int = Field(default=0)


class VariableDefinition(BaseModel):
    name: str
    description: str = ""
    sample_values: list[str] = Field(default_factory=list)
    default_value: str = ""
    placeholder: str = ""


class UtteranceConstraint(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    max_chars: Optional[int] = None
    min_chars: Optional[int] = None
    forbidden_words: list[str] = Field(default_factory=list)
    required_tone: str = ""
    applies_to: str = "all"
    description: str = ""


class ScenarioTrigger(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    trigger_condition: str
    trigger_keywords: list[str] = Field(default_factory=list)
    expected_system_action: str
    forbidden_system_action: str = ""
    should_end_conversation: bool = False
    at_flow_step: str = ""


class ConditionalBranch(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    condition: str
    condition_keywords: list[str] = Field(default_factory=list)
    action: str
    next_node: str = ""
    is_default: bool = False


# ============ 指令相关模型 ============

class Checkpoint(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str
    type: CheckpointType = Field(default=CheckpointType.MUST_DO)
    keywords: list[str] = Field(default_factory=list)
    evaluation_criteria: str = ""
    order: int = 0
    condition: str = ""
    at_flow_step: str = ""
    # v3 新增: 标记是否从 flow_node 派生 (用于避免双重计分)
    derived_from_flow: bool = Field(default=False, description="是否由流程节点派生 - 用于评分去重")


class Constraint(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str
    severity: ConstraintSeverity = Field(default=ConstraintSeverity.MAJOR)
    violation_examples: list[str] = Field(default_factory=list)
    check_method: str = Field(default="llm_judge")
    # v3 新增: 显式区分判定方式
    is_deterministic: bool = Field(default=False, description="是否可由规则确定性判定")


class FlowNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str
    next_nodes: list[str] = Field(default_factory=list)
    is_start: bool = False
    is_end: bool = False
    conditions: dict[str, str] = Field(default_factory=dict)
    branches: list[ConditionalBranch] = Field(default_factory=list)
    step_number: int = 0
    reference_script: str = ""
    sub_steps: list[str] = Field(default_factory=list)


class ExceptionRule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    trigger: str
    expected_behavior: str
    forbidden_behavior: str = ""


class InstructionComplexity(BaseModel):
    """指令复杂度量化 - v3 新增

    用于:
    - 同模型不同任务横向对比
    - 评测系统自身的难度分级
    """
    n_checkpoints: int = 0
    n_constraints: int = 0
    n_flow_nodes: int = 0
    n_branches: int = 0
    n_knowledge_entries: int = 0
    n_utterance_constraints: int = 0
    n_scenario_triggers: int = 0
    n_variables: int = 0
    max_branch_depth: int = 0
    raw_chars: int = 0
    complexity_score: float = Field(default=0, description="0-100 综合复杂度分")
    complexity_level: str = Field(default="simple", description="simple/medium/complex/very_complex")


class TaskInstruction(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_name: str
    task_description: str
    system_prompt: str
    context_info: dict[str, Any] = Field(default_factory=dict)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    flow_nodes: list[FlowNode] = Field(default_factory=list)
    exception_rules: list[ExceptionRule] = Field(default_factory=list)
    opening_line: str = ""
    knowledge_base: list[KnowledgeEntry] = Field(default_factory=list)
    variables: list[VariableDefinition] = Field(default_factory=list)
    utterance_constraints: list[UtteranceConstraint] = Field(default_factory=list)
    scenario_triggers: list[ScenarioTrigger] = Field(default_factory=list)
    instruction_format: str = "json"
    role_description: str = ""
    raw_instruction: str = ""
    # v3 新增: 编译产物
    complexity: Optional[InstructionComplexity] = None


# ============ 对话相关模型 ============

class DialogueTurn(BaseModel):
    turn_id: int
    role: DialogueRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    char_count: int = 0
    constraint_violations: list[str] = Field(default_factory=list)
    # v3 新增: 状态机推断 - 当前对话所处的流程节点
    inferred_flow_node: str = Field(default="", description="状态跟踪器推断的当前所在节点")


class DialogueSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_instruction: TaskInstruction
    persona_type: PersonaType
    turns: list[DialogueTurn] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    terminated_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    branch_path: list[str] = Field(default_factory=list)
    triggered_scenarios: list[str] = Field(default_factory=list)
    # v3 新增: 复算元数据
    seed: Optional[int] = None
    target_model_version: str = ""
    simulator_model_version: str = ""


# ============ 评估相关模型 ============

class EvidenceQuote(BaseModel):
    """证据片段 - v3 新增的核心可解释组件"""
    turn_id: int = Field(..., description="所在对话轮次")
    role: DialogueRole = Field(default=DialogueRole.SYSTEM)
    text: str = Field(..., description="原文引用 (verbatim quote)")
    note: str = Field(default="", description="可选说明，比如为何引用")


class CheckpointEvaluation(BaseModel):
    checkpoint_id: str
    checkpoint_name: str
    status: CheckpointStatus
    score: float = Field(..., ge=0, le=100)
    evidence: str = Field(default="", description="评分依据 (可读)")
    relevant_turns: list[int] = Field(default_factory=list)
    # v3 新增: 结构化证据三元组
    quotes: list[EvidenceQuote] = Field(default_factory=list, description="原文证据片段")
    confidence: float = Field(default=1.0, ge=0, le=1, description="评估置信度 0-1")
    evaluation_method: EvaluationMethod = Field(default=EvaluationMethod.LLM_JUDGE)
    # 多次评估时的统计信息
    score_samples: list[float] = Field(default_factory=list, description="多次评估的原始分数")
    score_std: float = Field(default=0.0, description="多次评估的标准差")


class ConstraintEvaluation(BaseModel):
    constraint_id: str
    constraint_name: str
    violated: bool = False
    violation_count: int = 0
    violation_details: list[str] = Field(default_factory=list)
    relevant_turns: list[int] = Field(default_factory=list)
    severity: ConstraintSeverity = Field(default=ConstraintSeverity.MAJOR)
    # v3 新增
    quotes: list[EvidenceQuote] = Field(default_factory=list)
    evaluation_method: EvaluationMethod = Field(default=EvaluationMethod.LLM_JUDGE)
    confidence: float = Field(default=1.0, ge=0, le=1)


class UtteranceConstraintEvaluation(BaseModel):
    total_system_turns: int = 0
    violated_turns: int = 0
    violation_rate: float = 0
    violations: list[dict[str, Any]] = Field(default_factory=list)
    score: float = Field(default=100, ge=0, le=100)


class KnowledgeAccuracyEvaluation(BaseModel):
    total_knowledge_queries: int = 0
    correct_answers: int = 0
    incorrect_answers: int = 0
    fabricated_info: list[str] = Field(default_factory=list)
    score: float = Field(default=100, ge=0, le=100)
    details: list[dict[str, Any]] = Field(default_factory=list)
    quotes: list[EvidenceQuote] = Field(default_factory=list)


class DimensionScore(BaseModel):
    dimension_key: str
    dimension_name: str
    score: float = Field(..., ge=0, le=100)
    weight: float = Field(..., ge=0, le=1)
    weighted_score: float = 0
    explanation: str = ""
    details: list[str] = Field(default_factory=list)
    # v3 新增
    evaluation_method: EvaluationMethod = Field(default=EvaluationMethod.LLM_JUDGE)
    confidence: float = Field(default=1.0, ge=0, le=1)
    score_std: float = Field(default=0.0, description="多次评估的标准差")
    score_samples: list[float] = Field(default_factory=list)


class SessionEvaluation(BaseModel):
    session_id: str
    persona_type: PersonaType
    total_score: float = Field(default=0, ge=0, le=100)
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    checkpoint_evaluations: list[CheckpointEvaluation] = Field(default_factory=list)
    constraint_evaluations: list[ConstraintEvaluation] = Field(default_factory=list)
    utterance_constraint_eval: Optional[UtteranceConstraintEvaluation] = None
    knowledge_accuracy_eval: Optional[KnowledgeAccuracyEvaluation] = None
    flow_analysis: str = ""
    branch_coverage: list[str] = Field(default_factory=list)
    overall_comment: str = ""
    issues: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    # v3 新增: 复算与稳定性
    overall_confidence: float = Field(default=1.0, ge=0, le=1, description="整体置信度")
    total_score_std: float = Field(default=0.0, description="多次评估总分标准差")


# ============ 报告相关模型 ============

class BranchCoverageMatrix(BaseModel):
    """分支覆盖矩阵 - v3 升级"""
    total_branches: int = 0
    covered_branches: int = 0
    coverage_rate: float = 0
    uncovered_branches: list[str] = Field(default_factory=list)
    branch_details: dict[str, int] = Field(default_factory=dict)
    # v3 新增
    branch_node_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description="每个流程节点 -> 该节点下的分支条件"
    )
    target_branches_to_cover: list[str] = Field(
        default_factory=list,
        description="本次评测目标要覆盖的分支列表"
    )


# 兼容老代码: 保留 BranchCoverage 作为别名
BranchCoverage = BranchCoverageMatrix


class FailureMode(BaseModel):
    """失败模式 - v3 新增"""
    name: str = Field(..., description="失败模式名称")
    category: str = Field(..., description="类别: task/constraint/flow/utterance/knowledge")
    occurrences: int = Field(default=1)
    affected_personas: list[str] = Field(default_factory=list)
    affected_sessions: list[str] = Field(default_factory=list)
    typical_quote: str = Field(default="", description="典型违规话语")
    impact_score: float = Field(default=0, description="对总分的影响估计")
    suggestion: str = Field(default="")


class ModelWeaknessProfile(BaseModel):
    """模型短板诊断 - v3 新增 (核心创新点 C6)"""
    weakest_dimensions: list[str] = Field(default_factory=list, description="最弱维度 Top-3")
    strongest_dimensions: list[str] = Field(default_factory=list, description="最强维度 Top-3")
    weakest_personas: list[str] = Field(default_factory=list, description="表现最差画像")
    top_failure_modes: list[FailureMode] = Field(default_factory=list, description="高频失败模式 Top-N")
    risk_summary: str = Field(default="", description="风险总览")


class CalibrationInfo(BaseModel):
    """评估器校准信息 - 用于证明可靠性"""
    calibrated: bool = False
    n_calibration_cases: int = 0
    mae_vs_human: float = Field(default=-1, description="与人工标注的 MAE，-1 表示未校准")
    correlation_vs_human: float = Field(default=-1)


class RunMetadata(BaseModel):
    """复算所需元数据 - v3 新增"""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: Optional[datetime] = None
    seed: Optional[int] = None
    evaluator_version: str = EVALUATOR_VERSION
    target_model_id: str = ""
    simulator_model_id: str = ""
    judge_model_id: str = ""
    self_consistency_n: int = Field(default=1, description="Self-consistency 评估次数")
    concurrency: int = 1
    duration_seconds: float = 0


class EvaluationReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str
    task_description: str
    generated_at: datetime = Field(default_factory=datetime.now)
    total_sessions: int = 0
    overall_score: float = Field(default=0, ge=0, le=100)
    dimension_averages: list[DimensionScore] = Field(default_factory=list)
    persona_scores: dict[str, float] = Field(default_factory=dict)
    session_evaluations: list[SessionEvaluation] = Field(default_factory=list)
    dialogue_sessions: list[DialogueSession] = Field(default_factory=list)
    branch_coverage: Optional[BranchCoverageMatrix] = None
    summary: str = ""
    recommendations: list[str] = Field(default_factory=list)
    # v3 新增
    instruction_complexity: Optional[InstructionComplexity] = None
    weakness_profile: Optional[ModelWeaknessProfile] = None
    calibration: Optional[CalibrationInfo] = None
    run_metadata: Optional[RunMetadata] = None
    overall_score_std: float = Field(default=0.0, description="跨会话总分标准差")
    overall_confidence: float = Field(default=1.0, ge=0, le=1)
