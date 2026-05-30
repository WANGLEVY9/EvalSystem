"""
对话状态跟踪器 (DialogStateTracker) - v3.0

任务: 在每一轮根据当前 system 话语 / 已发生的对话, 推断对话当前所处的流程节点

实现方式 (轻量, 无需 LLM):
- 关键词重叠匹配 (节点描述 / sub_steps / reference_script)
- 步骤号显式提及 (如 "进入第3步")
- 单调推进先验: 默认情况下节点编号单调不减
- 输出: 当前节点名 + 置信度

应用:
- 用户模拟器在合适节点强制注入场景 (说"在开车")
- 评估器记录 branch_path / triggered_scenarios
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from .models import DialogueRole, DialogueSession, FlowNode, TaskInstruction

logger = logging.getLogger(__name__)


class DialogStateTracker:
    """对话状态跟踪器"""

    def __init__(self, instruction: TaskInstruction):
        self.instruction = instruction
        self.flow_nodes: list[FlowNode] = list(instruction.flow_nodes)
        # 索引: name -> node, step_number -> node
        self.by_name = {n.name: n for n in self.flow_nodes}
        self.by_step = {}
        for n in self.flow_nodes:
            self.by_step.setdefault(n.step_number, n)
        self.current_node: Optional[FlowNode] = None
        # 起始节点
        for n in self.flow_nodes:
            if n.is_start:
                self.current_node = n
                break
        if self.current_node is None and self.flow_nodes:
            self.current_node = self.flow_nodes[0]

    def reset(self):
        for n in self.flow_nodes:
            if n.is_start:
                self.current_node = n
                return
        self.current_node = self.flow_nodes[0] if self.flow_nodes else None

    def update(self, system_utterance: str, user_utterance: str = "") -> Optional[FlowNode]:
        """根据最新 system 话语更新当前节点; 返回新当前节点"""
        if not self.flow_nodes:
            return None

        u = system_utterance or ""
        # 1. 显式步骤提及 ("进入第N步" / "Step N" / "现在到第N步")
        m = re.search(r"(?:第|进入|来到|到达)?\s*(\d+(?:\.\d+)?)\s*步", u)
        if m:
            try:
                step_num = int(float(m.group(1)))
                if step_num in self.by_step:
                    self.current_node = self.by_step[step_num]
                    return self.current_node
            except ValueError:
                pass

        # 2. 关键词匹配
        cur_idx = self._index_of(self.current_node)
        cur_score = self._match_score(u, self.current_node) if self.current_node else 0
        best_node = self.current_node
        best_score = cur_score

        for i, node in enumerate(self.flow_nodes):
            # 单调推进: 不允许跳回上一节点 (除非是 end)
            if cur_idx is not None and i < cur_idx and not node.is_end:
                continue
            score = self._match_score(u, node)
            if score > best_score:
                best_score = score
                best_node = node

        # 当前节点的得分至少要 1, 或者新节点比当前节点得分更高
        if best_node is not None and best_score >= 1 and best_score > cur_score:
            self.current_node = best_node

        return self.current_node

    def _index_of(self, node: Optional[FlowNode]) -> Optional[int]:
        if node is None:
            return None
        for i, n in enumerate(self.flow_nodes):
            if n.name == node.name:
                return i
        return None

    @staticmethod
    def _match_score(text: str, node: Optional[FlowNode]) -> int:
        """计算文本与节点的关键词匹配数"""
        if node is None:
            return 0
        score = 0
        # 节点名 / 描述中的关键 token
        keywords: list[str] = []
        for src in [node.name, node.description, node.reference_script, *node.sub_steps]:
            if not src:
                continue
            # 抽 2-6 字的中文 token
            for tok in re.findall(r"[\u4e00-\u9fa5]{2,6}", src):
                # 排除常见动词 / 通用词
                if tok in {"进入", "若是", "其他", "通过", "如果", "可以", "请问", "好的", "我们", "您好"}:
                    continue
                keywords.append(tok)
        # 加入节点名本身 (不论几个字)
        if node.name:
            keywords.append(node.name)
        # 去重
        keywords = list(dict.fromkeys(keywords))[:40]
        for kw in keywords:
            if kw in text:
                score += 1
        return score
