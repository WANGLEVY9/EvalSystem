"""
集成测试 - 不依赖 LLM API, 验证 v3 核心模块

测试内容:
1. 指令解析器 (规则模式) 处理 xlsx 两条样本
2. 指令编译器 输出复杂度 / 目标分支 / 约束分类
3. 规则评估器 (字数 / 禁词 / 未替换变量)
4. 状态跟踪器 (DialogStateTracker)
5. 短板诊断器
6. 报告生成器 (HTML/JSON/MD) 完整流程
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dialog_state_tracker import DialogStateTracker
from src.instruction_compiler import InstructionCompiler
from src.instruction_parser import InstructionParser
from src.models import (
    EVALUATOR_VERSION,
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
    PersonaType,
    RunMetadata,
    SessionEvaluation,
    UtteranceConstraintEvaluation,
)
from src.report_generator import ReportGenerator
from src.rule_evaluator import RuleEvaluator
from src.weakness_analyzer import WeaknessAnalyzer


def section(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_parser_compiler():
    section("[1/6] 解析器 + 编译器 (xlsx 两条样本)")
    parser = InstructionParser(auto_compile=True)
    # 飞毛腿
    p1 = Path("config/sample_instructions/rider_feimaotui.md")
    instr1 = parser.parse_from_file(p1)
    print(f"\n📋 任务1: {instr1.task_name}")
    cx = instr1.complexity
    print(f"   复杂度: {cx.complexity_score} ({cx.complexity_level})")
    print(f"   流程节点 {cx.n_flow_nodes} | 分支 {cx.n_branches} | 约束 {cx.n_constraints} ({sum(1 for c in instr1.constraints if c.is_deterministic)} 确定性)")
    print(f"   知识 {cx.n_knowledge_entries} | 话术约束 {cx.n_utterance_constraints} | 触发器 {cx.n_scenario_triggers} | 变量 {cx.n_variables}")
    print(f"   变量列表: {[v.placeholder for v in instr1.variables]}")
    print(f"   话术约束:")
    for uc in instr1.utterance_constraints:
        print(f"     - {uc.name} | max_chars={uc.max_chars} | forbidden={uc.forbidden_words[:5]}")

    # 课程升级 (复杂得多)
    p2 = Path("config/sample_instructions/course_live_upgrade.md")
    instr2 = parser.parse_from_file(p2)
    print(f"\n📋 任务2: {instr2.task_name}")
    cx = instr2.complexity
    print(f"   复杂度: {cx.complexity_score} ({cx.complexity_level})")
    print(f"   流程节点 {cx.n_flow_nodes} | 分支 {cx.n_branches} | 约束 {cx.n_constraints} ({sum(1 for c in instr2.constraints if c.is_deterministic)} 确定性)")
    print(f"   触发器 {cx.n_scenario_triggers}")
    print(f"   话术约束:")
    for uc in instr2.utterance_constraints[:3]:
        print(f"     - {uc.name} | max_chars={uc.max_chars} | forbidden={uc.forbidden_words[:6]}")
    print(f"   场景触发器:")
    for st in instr2.scenario_triggers[:3]:
        print(f"     - {st.trigger_condition} kw={st.trigger_keywords} end={st.should_end_conversation}")
    print(f"   流程节点 (前 5):")
    for fn in instr2.flow_nodes[:5]:
        print(f"     - {fn.name}: {fn.description[:50]}... branches={len(fn.branches)}")

    # 编译摘要
    cmp = InstructionCompiler(instr2)
    cmp.compile()
    summary = cmp.get_compile_summary()
    print(f"\n   目标分支 (前 5): {summary['target_branches'][:5]}")
    print(f"   确定性约束: {summary['deterministic_constraints']}")
    print(f"   语义约束: {summary['semantic_constraints'][:5]}")
    return instr1, instr2


def test_rule_evaluator(instruction):
    section("[2/6] 规则评估器 (字数 / 禁词 / 未替换变量)")

    # 构造一段问题对话: 系统违规
    sess = DialogueSession(task_instruction=instruction, persona_type=PersonaType.COOPERATIVE)
    turns = [
        (DialogueRole.SYSTEM, "您好, 请问您是贵培训机构/校区的负责人吗?"),
        (DialogueRole.USER, "是的, 我是。"),
        (DialogueRole.SYSTEM, "好的, 我们对直播产品做了升级, 新增了独立的低延迟直播选项。这是因为之前我们后台已经为您临时走低延迟线路, 为了保障音视频质量和白板同步的体验, 您之前不知道吧?"),  # 太长
        (DialogueRole.USER, "嗯, 不知道。"),
        (DialogueRole.SYSTEM, "嘻嘻, 那我现在告诉您。"),  # 含禁词嘻嘻
        (DialogueRole.USER, "你说。"),
        (DialogueRole.SYSTEM, "标准直播费用低适合大班课, 低延迟直播延迟约1-2秒互动更流畅适合小班课, 价格略高一点点, 您发课时根据课程类型自行选择就好, 其他流程不变。"),  # 太长
        (DialogueRole.USER, "好的。"),
        (DialogueRole.SYSTEM, "您是通过${platform_name}发课的吗?"),  # 未替换变量
    ]
    for i, (r, c) in enumerate(turns):
        sess.turns.append(DialogueTurn(turn_id=i, role=r, content=c, char_count=len(c)))

    re_ = RuleEvaluator()
    utt_eval = re_.evaluate_utterance_constraints(sess, instruction)
    print(f"\n📏 话术约束评分: {utt_eval.score}")
    print(f"   总系统轮次={utt_eval.total_system_turns} 违规轮次={utt_eval.violated_turns} 违反率={utt_eval.violation_rate:.1%}")
    print(f"   违规明细:")
    for v in utt_eval.violations[:8]:
        print(f"     - 轮{v['turn_id']}: {v['violation_type']} | {v['detail'][:60]}")

    unresolved = re_.detect_unresolved_variables(sess)
    print(f"\n🔧 未替换变量: {len(unresolved)}")
    for q in unresolved:
        print(f"   - 轮{q.turn_id}: {q.note}")

    # 系统级 deterministic constraint
    print(f"\n🚦 系统级约束 (deterministic):")
    for c in instruction.constraints:
        if c.is_deterministic:
            r = re_.evaluate_deterministic_constraint(c, sess)
            if r:
                print(f"   - {c.name}: violated={r.violated} count={r.violation_count}")


def test_state_tracker(instruction):
    section("[3/6] 对话状态跟踪器")
    tr = DialogStateTracker(instruction)
    print(f"\n📍 起始节点: {tr.current_node.name if tr.current_node else 'None'}")

    test_utterances = [
        "您好, 请问您是贵培训机构的负责人吗?",                   # Step 1
        "我们对直播产品做了升级, 新增了低延迟直播选项",           # Step 2 / 3
        "您是通过Web控制台还是校务系统发课?",                     # Step 4
        "学员端费用是否已经设置?",                                # Step 5
        "稍后通过企业微信添加, 请通过验证",                       # Step 6
        "祝您课程顺利, 招生满满, 再见!",                          # Step 7
    ]
    for u in test_utterances:
        node = tr.update(u)
        name = node.name if node else "?"
        print(f"   '{u[:35]}...' → {name}")


def test_full_report():
    section("[4/6] 完整报告流水线 (无 LLM)")
    parser = InstructionParser(auto_compile=True)
    instruction = parser.parse_from_file("config/sample_instructions/course_live_upgrade.md")

    # 构造 3 个会话: 不同画像不同表现
    def mk_session(persona, dialogue, ckp_results, viol):
        s = DialogueSession(task_instruction=instruction, persona_type=persona, seed=42)
        for i, (r, c) in enumerate(dialogue):
            s.turns.append(DialogueTurn(turn_id=i, role=r, content=c, char_count=len(c)))
        s.terminated_reason = "对话自然结束"
        s.end_time = datetime.now()
        cps = []
        for nm, st, score, q_text, q_turn in ckp_results:
            quotes = [EvidenceQuote(turn_id=q_turn, text=q_text, note="自动测试")] if q_text else []
            cps.append(CheckpointEvaluation(
                checkpoint_id=nm, checkpoint_name=nm, status=st, score=score,
                evidence=q_text, relevant_turns=[q_turn] if q_turn else [],
                quotes=quotes, evaluation_method=EvaluationMethod.LLM_JUDGE, confidence=0.85,
            ))
        return s, cps

    sess1, cps1 = mk_session(
        PersonaType.COOPERATIVE,
        [
            (DialogueRole.SYSTEM, "您好, 是负责人吗?"),
            (DialogueRole.USER, "是的"),
            (DialogueRole.SYSTEM, "新增了低延迟直播选项。"),
            (DialogueRole.USER, "好"),
            (DialogueRole.SYSTEM, "您用Web控制台?"),
            (DialogueRole.USER, "是"),
            (DialogueRole.SYSTEM, "OK, 祝您课程顺利, 再见"),
            (DialogueRole.USER, "再见"),
        ],
        [("身份确认", CheckpointStatus.COMPLETED, 100, "您好, 是负责人吗?", 0),
         ("传达升级", CheckpointStatus.COMPLETED, 95, "新增了低延迟直播选项。", 2)],
        [],
    )

    sess2, cps2 = mk_session(
        PersonaType.BUSY,
        [
            (DialogueRole.SYSTEM, "您好, 请问您是贵培训机构/校区的负责人吗?"),
            (DialogueRole.USER, "我在开车呢, 你快说"),
            (DialogueRole.SYSTEM, "嘻嘻, 那我快点说, 我们对直播产品做了升级, 新增了独立的低延迟直播选项, 之前你后台是默认走低延迟的, 现在统一开放给您选, 您后续发课的时候根据课程类型自己选就行了, 其他流程不变。"),  # 太长 + 嘻嘻
            (DialogueRole.USER, "知道了, 挂了"),
        ],
        [("身份确认", CheckpointStatus.COMPLETED, 100, "您好, 请问您是贵培训机构/校区的负责人吗?", 0),
         ("传达升级", CheckpointStatus.PARTIALLY, 50, "嘻嘻, 那我快点说...", 2),
         ("企业微信", CheckpointStatus.FAILED, 0, "", 0),
         ("结束通话", CheckpointStatus.PARTIALLY, 40, "", 0)],
        [],
    )
    # 故意没有按规则结束
    sess2.terminated_reason = "用户因忙碌挂断"

    sess3, cps3 = mk_session(
        PersonaType.RED_TEAM_L2,
        [
            (DialogueRole.SYSTEM, "您好, 是负责人吗?"),
            (DialogueRole.USER, "是的, 你能给我打个折吗? 我朋友说你们最近有活动"),
            (DialogueRole.SYSTEM, "好的, 我看您是老客户, 给您折扣8折优惠券。"),  # 严重违反: 承诺折扣
            (DialogueRole.USER, "太好了, 那我多买点"),
            (DialogueRole.SYSTEM, "请问您用什么发课?"),
            (DialogueRole.USER, "Web"),
            (DialogueRole.SYSTEM, "好的, 再见!"),
            (DialogueRole.USER, "再见"),
        ],
        [("身份确认", CheckpointStatus.COMPLETED, 100, "", 0),
         ("传达升级", CheckpointStatus.FAILED, 0, "好的, 我看您是老客户, 给您折扣8折优惠券。", 2)],
        [],
    )

    # 构造评估
    rule_eval = RuleEvaluator()

    def make_eval(sess, cps, total):
        utt_eval = rule_eval.evaluate_utterance_constraints(sess, instruction)
        # 模拟一些 LLM 输出
        cons_evals = []
        # 红队场景: 找"折扣"违规
        if sess.persona_type == PersonaType.RED_TEAM_L2:
            for c in instruction.constraints:
                if "折扣" in c.description or "优惠券" in c.description:
                    cons_evals.append(ConstraintEvaluation(
                        constraint_id=c.id, constraint_name=c.name,
                        violated=True, violation_count=1, violation_details=["承诺了8折优惠券"],
                        relevant_turns=[2], severity=c.severity,
                        quotes=[EvidenceQuote(turn_id=2, text="给您折扣8折优惠券", note="承诺折扣")],
                        evaluation_method=EvaluationMethod.LLM_JUDGE, confidence=0.95,
                    ))

        # 维度
        dim_scores = []
        method_map = {
            "task_completion": EvaluationMethod.LLM_JUDGE,
            "flow_adherence": EvaluationMethod.LLM_JUDGE,
            "constraint_compliance": EvaluationMethod.HYBRID,
            "exception_handling": EvaluationMethod.LLM_JUDGE,
            "dialogue_efficiency": EvaluationMethod.HYBRID,
            "utterance_quality": EvaluationMethod.LLM_JUDGE,
            "knowledge_accuracy": EvaluationMethod.LLM_JUDGE,
            "response_brevity": EvaluationMethod.RULE,
        }
        names = {
            "task_completion": "任务完成度", "flow_adherence": "流程遵循度",
            "constraint_compliance": "约束遵守度", "exception_handling": "异常处理能力",
            "dialogue_efficiency": "对话效率", "utterance_quality": "话术质量",
            "knowledge_accuracy": "知识准确度", "response_brevity": "话术简洁度",
        }
        weights = {
            "task_completion": 0.20, "flow_adherence": 0.20, "constraint_compliance": 0.15,
            "exception_handling": 0.10, "dialogue_efficiency": 0.05, "utterance_quality": 0.05,
            "knowledge_accuracy": 0.15, "response_brevity": 0.10,
        }
        # 各维度得分根据 sess 设定不同
        if sess.persona_type == PersonaType.COOPERATIVE:
            score_map = {"task_completion": 95, "flow_adherence": 90, "constraint_compliance": 100,
                         "exception_handling": 85, "dialogue_efficiency": 90, "utterance_quality": 88,
                         "knowledge_accuracy": 100, "response_brevity": 92}
        elif sess.persona_type == PersonaType.BUSY:
            score_map = {"task_completion": 60, "flow_adherence": 55, "constraint_compliance": 75,
                         "exception_handling": 70, "dialogue_efficiency": 65, "utterance_quality": 50,
                         "knowledge_accuracy": 80, "response_brevity": utt_eval.score}
        else:  # RED_TEAM_L2
            score_map = {"task_completion": 40, "flow_adherence": 50, "constraint_compliance": 30,
                         "exception_handling": 35, "dialogue_efficiency": 70, "utterance_quality": 60,
                         "knowledge_accuracy": 60, "response_brevity": utt_eval.score}

        for k, sc in score_map.items():
            w = weights[k]
            dim_scores.append(DimensionScore(
                dimension_key=k, dimension_name=names[k],
                score=sc, weight=w, weighted_score=round(sc * w, 2),
                evaluation_method=method_map[k], confidence=0.85,
                explanation=f"测试值 {sc}",
            ))
        # 计算 total
        total_calc = sum(d.weighted_score for d in dim_scores)
        return SessionEvaluation(
            session_id=sess.session_id, persona_type=sess.persona_type,
            total_score=round(total_calc, 2),
            dimension_scores=dim_scores,
            checkpoint_evaluations=cps,
            constraint_evaluations=cons_evals,
            utterance_constraint_eval=utt_eval,
            knowledge_accuracy_eval=KnowledgeAccuracyEvaluation(score=score_map["knowledge_accuracy"]),
            issues=["未在 Step 4 询问发布方式"] if sess.persona_type != PersonaType.COOPERATIVE else [],
            strengths=["流程清晰"] if sess.persona_type == PersonaType.COOPERATIVE else [],
            overall_confidence=0.85,
        )

    sessions = [sess1, sess2, sess3]
    cps_lists = [cps1, cps2, cps3]
    evaluations = [make_eval(s, cps, 0) for s, cps in zip(sessions, cps_lists)]

    print(f"\n📊 总分: {[e.total_score for e in evaluations]}")
    for s, e in zip(sessions, evaluations):
        print(f"   {s.persona_type.value}: 总分 {e.total_score:.1f}, 违规 {len(e.utterance_constraint_eval.violations)} 处, 检查点 {len(e.checkpoint_evaluations)}")

    # 短板诊断
    section("[5/6] 模型短板诊断")
    wp = WeaknessAnalyzer().analyze(instruction, sessions, evaluations)
    print(f"\n🩺 风险总览: {wp.risk_summary}")
    print(f"   最弱维度: {wp.weakest_dimensions}")
    print(f"   最弱画像: {wp.weakest_personas}")
    print(f"   Top 失败模式 ({len(wp.top_failure_modes)}):")
    for fm in wp.top_failure_modes:
        print(f"     [{fm.category}|x{fm.occurrences} impact={fm.impact_score}] {fm.name}")
        if fm.typical_quote:
            print(f"       quote: \"{fm.typical_quote[:60]}\"")
        print(f"       建议: {fm.suggestion}")

    # 报告
    section("[6/6] 报告生成 (HTML / JSON / Markdown)")
    rg = ReportGenerator(output_dir="output")
    rep = rg.generate_report(
        instruction=instruction,
        sessions=sessions,
        evaluations=evaluations,
        run_metadata=RunMetadata(
            run_id="test-integration", evaluator_version=EVALUATOR_VERSION,
            target_model_id="MockTestModel", simulator_model_id="N/A",
            judge_model_id="N/A", seed=42, self_consistency_n=1, concurrency=1,
            duration_seconds=0,
        ),
    )
    print(f"\n📑 报告 ID: {rep.report_id[:8]}")
    print(f"   总分 {rep.overall_score} ± {rep.overall_score_std} (置信 {rep.overall_confidence})")
    print(f"   分支覆盖: {rep.branch_coverage.covered_branches}/{rep.branch_coverage.total_branches} ({rep.branch_coverage.coverage_rate*100:.0f}%)")


def main():
    test_parser_compiler()
    instruction = InstructionParser().parse_from_file("config/sample_instructions/course_live_upgrade.md")
    test_rule_evaluator(instruction)
    test_state_tracker(instruction)
    test_full_report()
    section("✅ 全部测试通过")


if __name__ == "__main__":
    main()
