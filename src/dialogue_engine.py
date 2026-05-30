"""
对话引擎 (v3.0)

核心改进:
1. 异步并发 - 用 asyncio + ThreadPoolExecutor 并发跑多会话, 大幅缩短端到端耗时
2. 强制分支覆盖 - 基于 ScenarioSimulator.list_target_branches() 为每条分支创建专门会话
3. 统一终止条件链 (按优先级):
     超时 > 强结束信号 > 自然双方告别 > 模拟器自判 > 场景触发 > 上限
4. seed 复算 - target/simulator 使用同一 seed
5. 状态跟踪 - 每个 system_turn 都被 DialogStateTracker 推断节点; 写回 turn.inferred_flow_node
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from openai import OpenAI

from .dialog_state_tracker import DialogStateTracker
from .models import (
    DialogueRole,
    DialogueSession,
    DialogueTurn,
    PersonaType,
    TaskInstruction,
)
from .target_model import TargetModel
from .user_simulator import PersonaSelector, ScenarioSimulator, UserSimulator

logger = logging.getLogger(__name__)


class DialogueEngine:
    """对话引擎 v3"""

    def __init__(
        self,
        target_model: TargetModel,
        llm_client: OpenAI,
        simulator_model: str = "deepseek-chat",
        max_turns: int = 18,
        timeout: int = 300,
        end_signals_user: Optional[list[str]] = None,
        end_signals_system: Optional[list[str]] = None,
        concurrency: int = 4,
        seed: Optional[int] = None,
    ):
        self.target_model = target_model
        self.llm_client = llm_client
        self.simulator_model = simulator_model
        self.max_turns = max_turns
        self.timeout = timeout
        self.concurrency = max(1, concurrency)
        self.seed = seed
        self.end_signals_user = end_signals_user or [
            "再见", "拜拜", "好的再见", "行吧再见", "挂了",
            "没什么了", "就这样吧", "知道了再见", "行再见",
        ]
        self.end_signals_system = end_signals_system or [
            "再见", "祝您", "感谢您的", "拜拜",
            "稍后再打", "改天再联系",
        ]

    # ============ 单会话 ============

    def run_single_session(
        self,
        instruction: TaskInstruction,
        persona_type: PersonaType,
        user_context: str = "",
        branch_config: Optional[dict[str, str]] = None,
        target_branch: Optional[tuple[str, str]] = None,
        arc_to: Optional[PersonaType] = None,
        seed: Optional[int] = None,
    ) -> DialogueSession:
        sess_seed = seed if seed is not None else self.seed
        logger.info(
            f"[会话] 任务={instruction.task_name} | 画像={persona_type.value}"
            + (f" | 目标分支='{target_branch[1]}'" if target_branch else "")
        )

        resolved_instruction = self._resolve_variables(instruction)

        session = DialogueSession(
            task_instruction=resolved_instruction,
            persona_type=persona_type,
            seed=sess_seed,
            target_model_version=getattr(self.target_model, "model", "unknown"),
            simulator_model_version=self.simulator_model,
        )

        # 模拟器
        simulator = UserSimulator(
            llm_client=self.llm_client,
            model=self.simulator_model,
            seed=sess_seed,
        )
        simulator.initialize(
            persona_type=persona_type,
            instruction=resolved_instruction,
            user_context=user_context,
            branch_config=branch_config,
            target_branch=target_branch,
            arc_to=arc_to,
        )

        # 状态跟踪器 (用于评估时记录推断节点)
        eval_tracker = DialogStateTracker(resolved_instruction)

        # 开场
        opening = self.target_model.get_opening_line(resolved_instruction)
        eval_tracker.update(opening)
        session.turns.append(DialogueTurn(
            turn_id=0,
            role=DialogueRole.SYSTEM,
            content=opening,
            char_count=len(opening),
            metadata={"type": "opening"},
            inferred_flow_node=eval_tracker.current_node.name if eval_tracker.current_node else "",
        ))

        model_history: list[dict[str, str]] = []
        start = time.time()
        turn_id = 1
        triggered_scenarios: set[str] = set()

        while turn_id <= self.max_turns * 2:
            # 优先级 1: 超时
            if time.time() - start > self.timeout:
                session.terminated_reason = "对话超时"
                break

            # 模拟器生成
            last_sys = session.turns[-1].content
            user_response = simulator.generate_response(last_sys)

            session.turns.append(DialogueTurn(
                turn_id=turn_id,
                role=DialogueRole.USER,
                content=user_response,
                char_count=len(user_response),
                inferred_flow_node=eval_tracker.current_node.name if eval_tracker.current_node else "",
            ))
            turn_id += 1

            # 优先级 2: 模拟器自判
            should_end, reason = simulator.should_end_conversation()
            if should_end:
                session.terminated_reason = reason
                break

            # 优先级 3: 自然结束 (双向告别)
            if self._detect_natural_end(user_response, last_sys):
                session.terminated_reason = "对话自然结束"
                break

            # 优先级 4: 场景触发结束
            scenario_end = self._check_scenario_end(user_response, resolved_instruction)
            if scenario_end:
                session.terminated_reason = f"场景触发结束: {scenario_end}"
                triggered_scenarios.add(scenario_end)
                # 让系统再回一句以便评估
                model_history.append({"role": "user", "content": user_response})
                sys_resp = self.target_model.generate_response(
                    system_prompt=resolved_instruction.system_prompt,
                    dialogue_history=model_history,
                    context_info=resolved_instruction.context_info,
                )
                model_history.append({"role": "assistant", "content": sys_resp})
                eval_tracker.update(sys_resp)
                session.turns.append(DialogueTurn(
                    turn_id=turn_id,
                    role=DialogueRole.SYSTEM,
                    content=sys_resp,
                    char_count=len(sys_resp),
                    inferred_flow_node=eval_tracker.current_node.name if eval_tracker.current_node else "",
                ))
                break

            # 系统生成
            model_history.append({"role": "user", "content": user_response})
            sys_resp = self.target_model.generate_response(
                system_prompt=resolved_instruction.system_prompt,
                dialogue_history=model_history,
                context_info=resolved_instruction.context_info,
            )
            model_history.append({"role": "assistant", "content": sys_resp})
            eval_tracker.update(sys_resp)
            session.turns.append(DialogueTurn(
                turn_id=turn_id,
                role=DialogueRole.SYSTEM,
                content=sys_resp,
                char_count=len(sys_resp),
                inferred_flow_node=eval_tracker.current_node.name if eval_tracker.current_node else "",
            ))
            turn_id += 1

            # 优先级 5: 系统强结束信号
            if self._detect_system_end(sys_resp):
                session.terminated_reason = "系统主动结束对话"
                break

            # 优先级 6: 轮次上限
            user_turns = sum(1 for t in session.turns if t.role == DialogueRole.USER)
            if user_turns >= self.max_turns:
                session.terminated_reason = "达到最大轮次限制"
                break

        session.end_time = datetime.now()

        # 记录 branch_path & triggered_scenarios
        if target_branch:
            session.branch_path = [f"{target_branch[0]}::{target_branch[1]}"]
        elif branch_config:
            session.branch_path = [f"{k}::{v}" for k, v in branch_config.items()]
        session.triggered_scenarios = sorted(triggered_scenarios)

        user_turns = sum(1 for t in session.turns if t.role == DialogueRole.USER)
        logger.info(f"[会话] 结束 轮次={user_turns} 原因={session.terminated_reason or '正常'}")

        return session

    # ============ 批量 (并发) ============

    def run_batch_sessions(
        self,
        instruction: TaskInstruction,
        persona_types: Optional[list[PersonaType]] = None,
        sessions_per_persona: int = 1,
        user_context: str = "",
        seed: Optional[int] = None,
    ) -> list[DialogueSession]:
        if persona_types is None:
            persona_types = PersonaSelector().get_all_personas()

        # 任务列表
        tasks: list[tuple[PersonaType, int]] = []
        for p in persona_types:
            for j in range(sessions_per_persona):
                tasks.append((p, j))

        logger.info(f"[批量] {len(persona_types)} 画像 × {sessions_per_persona} 次 = {len(tasks)} 会话 (并发={self.concurrency})")
        return self._run_concurrent(instruction, tasks, user_context, seed)

    def _run_concurrent(
        self,
        instruction: TaskInstruction,
        tasks: list[tuple[PersonaType, int]],
        user_context: str,
        seed: Optional[int],
    ) -> list[DialogueSession]:
        """用 ThreadPoolExecutor 并发跑会话"""
        results: list[Optional[DialogueSession]] = [None] * len(tasks)

        def _run_one(idx: int, persona: PersonaType, j: int) -> tuple[int, DialogueSession]:
            # 不同 session 用不同 seed (保证多样性 + 仍可复算)
            sub_seed = (seed if seed is not None else 0) * 1000 + idx
            # 重置 mock target model
            if hasattr(self.target_model, "reset"):
                try:
                    self.target_model.reset()
                except Exception:
                    pass
            sess = self.run_single_session(
                instruction=instruction,
                persona_type=persona,
                user_context=user_context,
                seed=sub_seed,
            )
            return idx, sess

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(_run_one, i, p, j) for i, (p, j) in enumerate(tasks)]
            for fut in futures:
                try:
                    idx, sess = fut.result()
                    results[idx] = sess
                except Exception as e:
                    logger.error(f"[批量] 会话失败: {e}")

        return [r for r in results if r is not None]

    def run_branched_sessions(
        self,
        instruction: TaskInstruction,
        base_persona: PersonaType = PersonaType.COOPERATIVE,
        user_context: str = "",
        seed: Optional[int] = None,
    ) -> list[DialogueSession]:
        """强制分支覆盖测试 - 为每条 (节点, 分支) 创建一个会话"""
        scenario_sim = ScenarioSimulator(instruction)
        targets = scenario_sim.list_target_branches()
        if not targets:
            logger.info("[分支] 无显式分支需要覆盖, 跳过")
            return []

        logger.info(f"[分支] 开始覆盖 {len(targets)} 条分支 (并发={self.concurrency})")
        persona_map = scenario_sim.get_branch_persona_mapping()

        def _select_persona(condition: str) -> PersonaType:
            personas = persona_map.get(condition, [base_persona])
            return personas[0] if personas else base_persona

        results: list[Optional[DialogueSession]] = [None] * len(targets)

        def _run_one(idx: int, node_name: str, condition: str):
            sub_seed = (seed if seed is not None else 0) * 5000 + idx
            if hasattr(self.target_model, "reset"):
                try:
                    self.target_model.reset()
                except Exception:
                    pass
            persona = _select_persona(condition)
            sess = self.run_single_session(
                instruction=instruction,
                persona_type=persona,
                user_context=user_context,
                target_branch=(node_name, condition),
                seed=sub_seed,
            )
            return idx, sess

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(_run_one, i, n, c) for i, (n, c) in enumerate(targets)]
            for fut in futures:
                try:
                    idx, sess = fut.result()
                    results[idx] = sess
                except Exception as e:
                    logger.error(f"[分支] 会话失败: {e}")

        out = [r for r in results if r is not None]
        logger.info(f"[分支] 完成 {len(out)}/{len(targets)} 个会话")
        return out

    # ============ 工具 ============

    def _resolve_variables(self, instruction: TaskInstruction) -> TaskInstruction:
        if not instruction.variables:
            return instruction
        var_map: dict[str, str] = {}
        for var in instruction.variables:
            value = var.default_value or (var.sample_values[0] if var.sample_values else var.name)
            # 多种 placeholder 形式都要替换 (兼容 LLM 抽取的不同格式)
            if var.placeholder:
                var_map[var.placeholder] = value
            var_map[f"${{{var.name}}}"] = value
            # 兼容 **X 单** 形式 (XYZ 单字母变量)
            if len(var.name) == 1 and var.name.isupper():
                # 把 **X 单** **X单** **X 天** 等形式都替换为带 ** 的样本值
                for unit in ["单", "天", "元", "次", "时", "分", "个"]:
                    var_map[f"**{var.name} {unit}**"] = f"**{value} {unit}**"
                    var_map[f"**{var.name}{unit}**"] = f"**{value}{unit}**"
                    # 句中可能没 ** 包裹
                    var_map[f"{var.name} {unit}"] = f"{value} {unit}"
        if not var_map:
            return instruction

        def _sub(text: str) -> str:
            if not text:
                return text
            for ph, v in sorted(var_map.items(), key=lambda x: -len(x[0])):  # 长的先替换
                text = text.replace(ph, v)
            return text

        resolved = instruction.model_copy(deep=True)
        resolved.system_prompt = _sub(resolved.system_prompt)
        resolved.opening_line = _sub(resolved.opening_line)
        resolved.task_description = _sub(resolved.task_description)
        # 也替换知识库 (因为知识库里的 X 单/Y 单 必须显示为具体数字, 否则模型会理解成占位符)
        for kb in resolved.knowledge_base:
            kb.question = _sub(kb.question)
            kb.answer = _sub(kb.answer)
        # 替换检查点描述
        for cp in resolved.checkpoints:
            cp.description = _sub(cp.description)
            cp.evaluation_criteria = _sub(cp.evaluation_criteria)
        for k, v in resolved.context_info.items():
            if isinstance(v, str):
                resolved.context_info[k] = _sub(v)
        return resolved

    def _detect_natural_end(self, user_msg: str, system_msg: str) -> bool:
        u_end = any(s in user_msg for s in self.end_signals_user)
        s_end = any(s in system_msg for s in self.end_signals_system)
        return u_end and s_end

    def _detect_system_end(self, system_msg: str) -> bool:
        strong = [
            "再见！", "再见。", "祝您生活愉快，再见", "祝您生活愉快，再见！",
            "感谢您的配合，再见", "为您转接", "稍后再联系您",
            "改天再给您打", "那我就不打扰您了",
        ]
        return any(s in system_msg for s in strong)

    def _check_scenario_end(self, user_msg: str, instruction: TaskInstruction) -> str:
        for tr in instruction.scenario_triggers:
            if not tr.should_end_conversation:
                continue
            if tr.trigger_keywords and any(kw in user_msg for kw in tr.trigger_keywords):
                return tr.trigger_condition
        return ""
