"""
被测模型适配层

提供统一的模型接口定义，支持：
1. MockTargetModel - 内置规则模拟模型（用于演示和测试）
2. DeepSeekModel - 对接 DeepSeek API
3. OpenAICompatibleModel - 对接任意OpenAI兼容API
4. HTTPAPIModel - 对接自定义HTTP API
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from .models import TaskInstruction

logger = logging.getLogger(__name__)


class TargetModel(ABC):
    """被测模型抽象接口"""

    @abstractmethod
    def generate_response(
        self,
        system_prompt: str,
        dialogue_history: list[dict[str, str]],
        context_info: dict[str, Any] = None,
    ) -> str:
        """
        生成模型回复

        Args:
            system_prompt: 系统提示词
            dialogue_history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            context_info: 上下文信息

        Returns:
            模型回复文本
        """
        pass

    @abstractmethod
    def get_opening_line(self, instruction: TaskInstruction) -> str:
        """获取模型开场白"""
        pass


class DeepSeekModel(TargetModel):
    """DeepSeek API 模型"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response(
        self,
        system_prompt: str,
        dialogue_history: list[dict[str, str]],
        context_info: dict[str, Any] = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        # 如果有上下文信息，添加到系统提示中
        if context_info:
            context_str = "\n\n## 当前上下文信息：\n"
            for key, value in context_info.items():
                context_str += f"- {key}: {value}\n"
            messages[0]["content"] += context_str

        messages.extend(dialogue_history)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}")
            raise

    def get_opening_line(self, instruction: TaskInstruction) -> str:
        if instruction.opening_line:
            return instruction.opening_line

        # 让模型生成开场白
        messages = [
            {"role": "system", "content": instruction.system_prompt},
            {"role": "user", "content": "请生成你的开场白，直接开始与用户的通话。"}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=256,
        )
        return response.choices[0].message.content


class OpenAICompatibleModel(TargetModel):
    """兼容OpenAI格式的API模型（通用适配器）"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response(
        self,
        system_prompt: str,
        dialogue_history: list[dict[str, str]],
        context_info: dict[str, Any] = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        if context_info:
            context_str = "\n\n## 当前上下文信息：\n"
            for key, value in context_info.items():
                context_str += f"- {key}: {value}\n"
            messages[0]["content"] += context_str

        messages.extend(dialogue_history)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise

    def get_opening_line(self, instruction: TaskInstruction) -> str:
        if instruction.opening_line:
            return instruction.opening_line

        messages = [
            {"role": "system", "content": instruction.system_prompt},
            {"role": "user", "content": "请生成你的开场白，直接开始与用户的通话。"}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=256,
        )
        return response.choices[0].message.content


class MockTargetModel(TargetModel):
    """
    内置模拟被测模型（基于规则，用于演示和系统测试）

    模拟一个物流外呼场景的对话模型，具有一定的规则逻辑：
    - 开场白 -> 确认身份 -> 通知信息 -> 确认/处理变更 -> 结束
    """

    def __init__(self):
        self.state = "opening"
        self.turn_count = 0

    def generate_response(
        self,
        system_prompt: str,
        dialogue_history: list[dict[str, str]],
        context_info: dict[str, Any] = None,
    ) -> str:
        self.turn_count = len([m for m in dialogue_history if m["role"] == "user"])

        if not dialogue_history:
            return self._get_opening()

        last_user_msg = ""
        for msg in reversed(dialogue_history):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break

        last_user_msg_lower = last_user_msg.lower()

        # 基于关键词的简单规则响应
        if any(kw in last_user_msg for kw in ["不在", "没空", "改时间", "改约"]):
            self.state = "reschedule"
            return "好的，请问您方便的收货时间是什么时候呢？我帮您备注一下。"

        if any(kw in last_user_msg for kw in ["退货", "不要了", "取消"]):
            self.state = "cancel"
            return "好的，我理解您的需求。请问是对商品不满意还是有其他原因呢？我帮您记录一下，稍后会有专人跟进退货事宜。"

        if any(kw in last_user_msg for kw in ["转人工", "找人工", "客服"]):
            self.state = "transfer"
            return "好的，我现在为您转接人工客服，请您稍等片刻。祝您生活愉快，再见！"

        if any(kw in last_user_msg for kw in ["好的", "可以", "没问题", "对", "嗯"]):
            if self.state == "opening":
                self.state = "confirmed_identity"
                return "好的，您有一个快递包裹预计今天下午14:00-16:00送达，收货地址是朝阳区建国路88号，请问这个地址方便收货吗？"
            elif self.state == "confirmed_identity":
                self.state = "confirmed_delivery"
                return "太好了，那我们的快递员会在预计时间内为您配送。如果届时有任何变动，我们会提前联系您。请问还有其他需要帮助的吗？"
            elif self.state in ["confirmed_delivery", "reschedule"]:
                self.state = "closing"
                return "好的，感谢您的配合。祝您生活愉快，再见！"

        if any(kw in last_user_msg for kw in ["什么快递", "什么包裹", "哪个订单"]):
            return "是您在5月15日下单的商品，订单尾号是8856，快递单号尾号3342。请问您是否方便在今天下午14:00-16:00收货呢？"

        if any(kw in last_user_msg for kw in ["你是谁", "哪里", "什么公司"]):
            return "我是XX物流的智能客服，今天给您致电是关于您的快递配送事宜。请问您现在方便听一下吗？"

        # 默认回复
        if self.turn_count > 5:
            self.state = "closing"
            return "好的，感谢您的时间。如果您后续有任何疑问，可以随时联系我们。祝您生活愉快，再见！"

        return "好的，我了解了。请问还有其他可以帮您的吗？"

    def get_opening_line(self, instruction: TaskInstruction) -> str:
        if instruction.opening_line:
            return instruction.opening_line
        return self._get_opening()

    def _get_opening(self) -> str:
        self.state = "opening"
        return "您好，这里是XX物流，我是智能客服小助手。请问是李先生/女士吗？今天给您来电是关于您的快递配送事宜。"

    def reset(self):
        """重置模型状态（用于新一轮对话）"""
        self.state = "opening"
        self.turn_count = 0
