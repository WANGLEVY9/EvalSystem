"""
多轮对话评测系统 - 主入口 v3.0

针对履约数字人外呼场景, 复杂指令遵循能力的可解释、可量化、可复算自动评估系统。

核心特性:
  📐 可解释  - 证据三元组 (turn_id + 原文 quote + 评分理由)
  📊 可量化  - Self-Consistency 多次评估 + 置信区间 + 标准差
  🎯 可复算  - run_id + seed + version + 模型版本元数据
  🚀 高效    - 异步并发, 9画像 × 3次会话约 5-8 分钟跑完
  🩺 诊断    - Top-3 维度短板 + 失败模式聚类 + 改进建议

使用:
    # Demo 模式 (无需 API)
    python main.py --demo

    # 完整评测 (需 DEEPSEEK_API_KEY)
    python main.py -i config/sample_instructions/delivery_notification.json
    python main.py -i config/sample_instructions/rider_feimaotui.md
    python main.py --excel "命题二：外呼任务对话模型指令示例.xlsx"

    # 调参
    python main.py -i task.json --personas cooperative,busy,red_team_l2 --sessions 3
    python main.py -i task.json --branch-test            # 强制覆盖所有分支
    python main.py -i task.json --self-consistency 3      # 关键维度评 3 次
    python main.py -i task.json --concurrency 8 --seed 42 # 并发 + 复算
    python main.py -i task.json --mini                    # 快速模式 (3 画像 × 1 次)
    python main.py -i task.json --full                    # 完整模式 (全画像 × 3 次 + 分支)

    # 校准 (内置标注集回测)
    python main.py --calibrate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.dialogue_engine import DialogueEngine
from src.evaluator import Evaluator
from src.instruction_parser import InstructionParser
from src.models import (
    EVALUATOR_VERSION,
    PersonaType,
    RunMetadata,
)
from src.report_generator import ReportGenerator
from src.target_model import DeepSeekModel, MockTargetModel, OpenAICompatibleModel
from src.user_simulator import PersonaSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_system")


# ============ 工具 ============

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_llm_client(api_key: str, base_url: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


def get_api_key(llm_config: dict) -> str:
    api_key = llm_config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("缺少 DEEPSEEK_API_KEY (环境变量或 config.llm.api_key)")
        sys.exit(1)
    return api_key


# ============ Demo 模式 ============

def run_demo():
    """无需 API 的演示模式 - 用 MockTargetModel + 预制对话生成报告"""
    logger.info("=" * 60)
    logger.info("DEMO 模式 (Mock 模型, 无需 API)")
    logger.info("=" * 60)

    instr_path = Path(__file__).parent / "config" / "sample_instructions" / "delivery_notification.json"
    if not instr_path.exists():
        logger.error(f"示例指令不存在: {instr_path}")
        sys.exit(1)

    parser = InstructionParser()
    instruction = parser.parse_from_file(instr_path)

    for w in parser.validate_instruction(instruction):
        logger.warning(w)

    # 手动构造一个示例 session
    from src.models import (
        CheckpointEvaluation,
        CheckpointStatus,
        ConstraintEvaluation,
        DialogueRole,
        DialogueSession,
        DialogueTurn,
        DimensionScore,
        EvaluationMethod,
        EvidenceQuote,
        KnowledgeAccuracyEvaluation,
        SessionEvaluation,
        UtteranceConstraintEvaluation,
    )

    sess = DialogueSession(task_instruction=instruction, persona_type=PersonaType.COOPERATIVE)
    demo = [
        (DialogueRole.SYSTEM, "您好, 请问是李先生吗? 我这边是XX物流, 来电是关于您的快递配送通知。"),
        (DialogueRole.USER, "嗯, 我是, 你说。"),
        (DialogueRole.SYSTEM, "好的李先生, 您的快递包裹预计今天下午14:00-16:00送达, 订单尾号8856。"),
        (DialogueRole.USER, "哦好的, 是那双运动鞋吧?"),
        (DialogueRole.SYSTEM, "是的。收货地址是北京市朝阳区建国路88号2单元501, 请问地址正确吗?"),
        (DialogueRole.USER, "对对, 地址没问题。"),
        (DialogueRole.SYSTEM, "好的, 请问今天下午您方便签收吗?"),
        (DialogueRole.USER, "可以的, 我下午在家。"),
        (DialogueRole.SYSTEM, "太好了, 快递员王师傅会在14:00-16:00为您配送。如有问题请联系电话尾号6677。"),
        (DialogueRole.USER, "好的, 谢谢。"),
        (DialogueRole.SYSTEM, "不客气, 祝您生活愉快, 再见!"),
        (DialogueRole.USER, "再见。"),
    ]
    for i, (r, c) in enumerate(demo):
        sess.turns.append(DialogueTurn(turn_id=i, role=r, content=c, char_count=len(c)))
    sess.terminated_reason = "对话自然结束"
    sess.end_time = datetime.now()

    ev = SessionEvaluation(
        session_id=sess.session_id,
        persona_type=PersonaType.COOPERATIVE,
        total_score=88.5,
        overall_confidence=0.85,
        total_score_std=2.0,
        dimension_scores=[
            DimensionScore(dimension_key="task_completion", dimension_name="任务完成度",
                           score=92.0, weight=0.20, weighted_score=18.4,
                           evaluation_method=EvaluationMethod.LLM_JUDGE, confidence=0.9,
                           explanation="6个检查点完成5个", details=["确认身份: completed", "通知包裹: completed"]),
            DimensionScore(dimension_key="flow_adherence", dimension_name="流程遵循度",
                           score=88.0, weight=0.20, weighted_score=17.6,
                           evaluation_method=EvaluationMethod.LLM_JUDGE, confidence=0.85),
            DimensionScore(dimension_key="constraint_compliance", dimension_name="约束遵守度",
                           score=100.0, weight=0.15, weighted_score=15.0,
                           evaluation_method=EvaluationMethod.HYBRID, confidence=1.0),
            DimensionScore(dimension_key="exception_handling", dimension_name="异常处理能力",
                           score=75.0, weight=0.10, weighted_score=7.5,
                           evaluation_method=EvaluationMethod.LLM_JUDGE, confidence=0.7),
            DimensionScore(dimension_key="dialogue_efficiency", dimension_name="对话效率",
                           score=85.0, weight=0.05, weighted_score=4.25,
                           evaluation_method=EvaluationMethod.HYBRID, confidence=0.85),
            DimensionScore(dimension_key="utterance_quality", dimension_name="话术质量",
                           score=82.0, weight=0.05, weighted_score=4.1,
                           evaluation_method=EvaluationMethod.LLM_JUDGE, confidence=0.8),
            DimensionScore(dimension_key="knowledge_accuracy", dimension_name="知识准确度",
                           score=100.0, weight=0.15, weighted_score=15.0,
                           evaluation_method=EvaluationMethod.LLM_JUDGE, confidence=0.6,
                           explanation="未触发知识库查询"),
            DimensionScore(dimension_key="response_brevity", dimension_name="话术简洁度",
                           score=85.0, weight=0.10, weighted_score=8.5,
                           evaluation_method=EvaluationMethod.RULE, confidence=1.0),
        ],
        checkpoint_evaluations=[
            CheckpointEvaluation(
                checkpoint_id="1", checkpoint_name="确认用户身份",
                status=CheckpointStatus.COMPLETED, score=100,
                evidence="开场白中明确询问身份",
                relevant_turns=[0],
                quotes=[EvidenceQuote(turn_id=0, text="请问是李先生吗?", note="身份确认")],
                evaluation_method=EvaluationMethod.LLM_JUDGE, confidence=1.0,
            ),
            CheckpointEvaluation(
                checkpoint_id="3", checkpoint_name="确认收货地址",
                status=CheckpointStatus.COMPLETED, score=100,
                evidence="完整报出地址并请求确认",
                relevant_turns=[4],
                quotes=[EvidenceQuote(turn_id=4, text="收货地址是北京市朝阳区建国路88号2单元501", note="完整地址")],
                evaluation_method=EvaluationMethod.LLM_JUDGE, confidence=0.95,
            ),
        ],
        constraint_evaluations=[],
        utterance_constraint_eval=UtteranceConstraintEvaluation(
            total_system_turns=6, violated_turns=0, violation_rate=0, violations=[], score=100,
        ),
        knowledge_accuracy_eval=KnowledgeAccuracyEvaluation(score=100),
        overall_comment="演示模式: 配合型用户对话流畅, 信息传达准确",
        issues=[],
        strengths=["流程完整", "信息准确", "语言得体"],
    )

    rg = ReportGenerator(output_dir=str(Path(__file__).parent / "output"))
    rep = rg.generate_report(
        instruction=instruction,
        sessions=[sess],
        evaluations=[ev],
        run_metadata=RunMetadata(
            run_id=str(uuid.uuid4()),
            evaluator_version=EVALUATOR_VERSION,
            target_model_id="MockTargetModel",
            simulator_model_id="N/A (demo)",
            judge_model_id="N/A (demo)",
            self_consistency_n=1,
            concurrency=1,
            duration_seconds=0,
        ),
    )

    logger.info("=" * 60)
    logger.info(f"DEMO 完成! 总分 {rep.overall_score} | 报告已生成到 output/")
    logger.info("=" * 60)


# ============ 完整评测 ============

def run_full_evaluation(
    config: dict,
    instruction_path: Optional[str] = None,
    instruction_obj=None,
    personas: Optional[list[str]] = None,
    sessions_per_persona: int = 1,
    enable_branch_test: bool = False,
    self_consistency: int = 1,
    concurrency: int = 4,
    seed: Optional[int] = None,
    mode_label: str = "default",
    generate_pdf: bool = False,
):
    llm_config = config["llm"]
    api_key = get_api_key(llm_config)
    base_url = llm_config.get("base_url", "https://api.deepseek.com")
    llm_client = make_llm_client(api_key, base_url)

    # 解析或使用现成 instruction
    if instruction_obj is not None:
        instruction = instruction_obj
    else:
        parser = InstructionParser(
            llm_client=llm_client,
            model=llm_config.get("evaluator_model", "deepseek-chat"),
        )
        instruction = parser.parse_from_file(instruction_path)
        for w in parser.validate_instruction(instruction):
            logger.warning(w)

    cx = instruction.complexity
    if cx:
        logger.info(
            f"📋 任务: {instruction.task_name} | "
            f"复杂度 {cx.complexity_score:.0f} ({cx.complexity_level}) | "
            f"流程 {cx.n_flow_nodes} 节点 / {cx.n_branches} 分支 | "
            f"约束 {cx.n_constraints} ({sum(1 for c in instruction.constraints if c.is_deterministic)} 确定性) | "
            f"知识 {cx.n_knowledge_entries} | 话术 {cx.n_utterance_constraints} | 触发 {cx.n_scenario_triggers}"
        )

    # 被测模型
    target_base_url = llm_config.get("target_base_url") or base_url
    target_api_key = llm_config.get("target_api_key") or api_key
    target_model_name = llm_config.get("target_model", "deepseek-chat")
    target_model = DeepSeekModel(
        api_key=target_api_key,
        base_url=target_base_url,
        model=target_model_name,
        temperature=float(llm_config.get("temperature", 0.7)),
        max_tokens=int(llm_config.get("max_tokens", 1024)),
    )

    # 画像
    if personas:
        try:
            persona_types = [PersonaType(p.strip()) for p in personas]
        except ValueError as e:
            logger.error(f"无效画像: {e}, 可选: {[p.value for p in PersonaType]}")
            sys.exit(1)
    else:
        selector = PersonaSelector(config.get("simulator", {}).get("personas"))
        persona_types = selector.get_personas_for_task(instruction)
        logger.info(f"自动推荐画像: {[p.value for p in persona_types]}")

    logger.info(f"🎭 测试画像: {[p.value for p in persona_types]} × {sessions_per_persona} 次")
    logger.info(f"⚙ self-consistency N={self_consistency} | concurrency={concurrency} | seed={seed}")

    # 引擎
    dialogue_config = config.get("dialogue", {})
    engine = DialogueEngine(
        target_model=target_model,
        llm_client=llm_client,
        simulator_model=llm_config.get("simulator_model", "deepseek-chat"),
        max_turns=int(dialogue_config.get("max_turns", 18)),
        timeout=int(dialogue_config.get("timeout", 300)),
        concurrency=concurrency,
        seed=seed,
    )

    # 跑会话
    t_start = time.time()
    sessions = engine.run_batch_sessions(
        instruction=instruction,
        persona_types=persona_types,
        sessions_per_persona=sessions_per_persona,
        seed=seed,
    )

    if enable_branch_test or dialogue_config.get("enable_branch_testing", False):
        logger.info("🌳 强制分支覆盖测试...")
        branch_sessions = engine.run_branched_sessions(instruction=instruction, seed=seed)
        if branch_sessions:
            sessions.extend(branch_sessions)
            logger.info(f"分支测试 + {len(branch_sessions)} 会话")

    # 评估
    eval_config = config.get("evaluation", {})
    dim_weights = {}
    for k, v in eval_config.get("dimensions", {}).items():
        dim_weights[k] = float(v.get("weight", 0.10))

    evaluator = Evaluator(
        llm_client=llm_client,
        model=llm_config.get("evaluator_model", "deepseek-chat"),
        dimension_weights=dim_weights or None,
        n_self_consistency=self_consistency,
    )
    logger.info(f"🔬 开始评估 {len(sessions)} 个会话...")
    evaluations = evaluator.evaluate_batch(sessions, instruction)

    duration = time.time() - t_start

    # 报告
    report_config = config.get("report", {})
    rg = ReportGenerator(output_dir=report_config.get("output_dir", "output"))
    run_metadata = RunMetadata(
        run_id=str(uuid.uuid4()),
        evaluator_version=EVALUATOR_VERSION,
        target_model_id=target_model_name,
        simulator_model_id=llm_config.get("simulator_model", "deepseek-chat"),
        judge_model_id=llm_config.get("evaluator_model", "deepseek-chat"),
        seed=seed,
        self_consistency_n=self_consistency,
        concurrency=concurrency,
        duration_seconds=round(duration, 2),
        finished_at=datetime.now(),
    )
    report = rg.generate_report(
        instruction=instruction,
        sessions=sessions,
        evaluations=evaluations,
        generate_html=bool(report_config.get("generate_html", True)),
        generate_json=bool(report_config.get("generate_json", True)),
        generate_markdown=bool(report_config.get("generate_markdown", True)),
        generate_pdf=generate_pdf or bool(report_config.get("generate_pdf", False)),
        run_metadata=run_metadata,
    )

    # 摘要
    logger.info("=" * 60)
    logger.info(f"✅ 评测完成 (耗时 {duration:.1f}s)")
    logger.info(f"📊 总分: {report.overall_score:.1f} ± {report.overall_score_std:.1f} (置信 {report.overall_confidence:.2f})")
    logger.info(f"会话: {report.total_sessions} | 模式: {mode_label}")
    logger.info("各维度:")
    for d in report.dimension_averages:
        bar = "█" * int(d.score / 5) + "░" * (20 - int(d.score / 5))
        method = {"rule": "📏", "llm_judge": "🧠", "hybrid": "🔀", "heuristic": "🔧"}.get(d.evaluation_method.value, "?")
        logger.info(f"  {method} {d.dimension_name:8s} {bar} {d.score:5.1f} ±{d.score_std:.1f}")

    if report.weakness_profile and report.weakness_profile.top_failure_modes:
        logger.info("\n🩺 Top 失败模式:")
        for fm in report.weakness_profile.top_failure_modes[:5]:
            logger.info(f"  - [{fm.category}|x{fm.occurrences}] {fm.name}")

    if report.branch_coverage and report.branch_coverage.total_branches:
        bc = report.branch_coverage
        logger.info(f"\n🌳 分支覆盖: {bc.covered_branches}/{bc.total_branches} ({bc.coverage_rate*100:.0f}%)")

    logger.info("\n报告已生成到 output/")
    logger.info("=" * 60)
    return report


# ============ Excel 批量 ============

def run_excel_evaluation(config: dict, excel_path: str, **kwargs):
    llm_config = config["llm"]
    api_key = get_api_key(llm_config)
    base_url = llm_config.get("base_url", "https://api.deepseek.com")
    llm_client = make_llm_client(api_key, base_url)

    parser = InstructionParser(llm_client=llm_client, model=llm_config.get("evaluator_model", "deepseek-chat"))
    instructions = parser.parse_from_excel(excel_path)
    logger.info(f"从 Excel 解析出 {len(instructions)} 个任务指令")

    for i, instr in enumerate(instructions):
        logger.info(f"\n{'=' * 60}\n📋 任务 [{i + 1}/{len(instructions)}]: {instr.task_name}\n{'=' * 60}")
        run_full_evaluation(config=config, instruction_obj=instr, **kwargs)


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(
        description="多轮对话评测系统 v3.0 - 复杂指令遵循能力可解释、可量化、可复算评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --demo                                            # Demo (无需 API)
  python main.py -i task.json                                      # 默认评测
  python main.py -i task.md --branch-test --self-consistency 3     # 强分支覆盖 + SC=3
  python main.py --excel "命题二:外呼任务对话模型指令示例.xlsx"     # Excel 批量
  python main.py -i task.json --mini                               # 快跑 (3画像×1次)
  python main.py -i task.json --full                               # 全跑 (全画像×3次+分支)
""",
    )
    parser.add_argument("--demo", action="store_true", help="演示模式 (无需 API)")
    parser.add_argument("-c", "--config", default="config/default_config.yaml", help="配置文件路径")
    parser.add_argument("-i", "--instruction", help="任务指令文件 (JSON/Markdown)")
    parser.add_argument("--excel", help="从 Excel 批量导入")
    parser.add_argument("--personas", help="逗号分隔: cooperative,hesitant,resistant,off_topic,contradictory,boundary,busy,confused,impatient,red_team_l1,red_team_l2,red_team_l3")
    parser.add_argument("--sessions", type=int, default=1, help="每画像会话数 (默认 1)")
    parser.add_argument("--branch-test", action="store_true", help="启用强制分支覆盖测试")
    parser.add_argument("--self-consistency", type=int, default=1, help="LLM Judge 多次评估次数 (1=单次, 3=推荐)")
    parser.add_argument("--concurrency", type=int, default=4, help="并发会话数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (复算)")
    parser.add_argument("--pdf", action="store_true", help="额外导出 PDF 报告 (需要 weasyprint)")
    parser.add_argument("--mini", action="store_true", help="快速模式: 3画像×1次, 不分支")
    parser.add_argument("--full", action="store_true", help="完整模式: 全画像×3次 + 分支覆盖 + SC=3")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    os.chdir(Path(__file__).parent)

    if args.demo:
        run_demo()
        return

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"配置文件不存在: {cfg_path}")
        sys.exit(1)
    config = load_config(str(cfg_path))

    # 模式预设
    if args.mini:
        args.personas = args.personas or "cooperative,busy,resistant"
        args.sessions = max(1, args.sessions)
        args.branch_test = False
        args.self_consistency = max(1, args.self_consistency)
        mode_label = "mini"
    elif args.full:
        args.personas = args.personas or None  # 用 selector 推荐
        args.sessions = max(args.sessions, 3)
        args.branch_test = True
        args.self_consistency = max(args.self_consistency, 3)
        mode_label = "full"
    else:
        mode_label = "default"

    if args.excel:
        if not Path(args.excel).exists():
            logger.error(f"Excel 文件不存在: {args.excel}")
            sys.exit(1)
        run_excel_evaluation(
            config=config,
            excel_path=args.excel,
            personas=args.personas.split(",") if args.personas else None,
            sessions_per_persona=args.sessions,
            enable_branch_test=args.branch_test,
            self_consistency=args.self_consistency,
            concurrency=args.concurrency,
            seed=args.seed,
            mode_label=mode_label,
            generate_pdf=args.pdf,
        )
        return

    if not args.instruction:
        args.instruction = "config/sample_instructions/delivery_notification.json"

    run_full_evaluation(
        config=config,
        instruction_path=args.instruction,
        personas=args.personas.split(",") if args.personas else None,
        sessions_per_persona=args.sessions,
        enable_branch_test=args.branch_test,
        self_consistency=args.self_consistency,
        concurrency=args.concurrency,
        seed=args.seed,
        mode_label=mode_label,
        generate_pdf=args.pdf,
    )


if __name__ == "__main__":
    main()
