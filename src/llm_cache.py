"""
LLM 调用缓存 (v3.0)

设计:
- 用 prompt 的 SHA-256 作为 key
- 缓存到本地 JSONL, 跨运行有效
- 装饰器 + 显式调用两种用法

价值:
- 在多次跑同一个 instruction 调试时, 评估器对同一对话的判定可复用 (节省 70%+ token)
- 用于 self-consistency 也很合适: temperature=0 + cache 后 N 次调用只算 1 次

使用:
    from src.llm_cache import LLMCache
    cache = LLMCache("output/.llm_cache.jsonl")
    response = cache.get_or_call(prompt, lambda: llm_client.chat.completions.create(...))
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class LLMCache:
    """文件级 LLM 调用缓存"""

    def __init__(self, cache_path: str = "output/.llm_cache.jsonl", enabled: bool = True):
        self.cache_path = Path(cache_path)
        self.enabled = enabled
        self.lock = threading.Lock()
        self._cache: dict[str, str] = {}
        self._load()

    def _load(self):
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["value"]
                    except json.JSONDecodeError:
                        continue
            logger.info(f"[LLMCache] 加载 {len(self._cache)} 条缓存")
        except Exception as e:
            logger.warning(f"[LLMCache] 加载失败: {e}")

    def _save(self, key: str, value: str):
        if not self.enabled:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[LLMCache] 写入失败: {e}")

    @staticmethod
    def make_key(prompt: str, model: str = "", extra: str = "") -> str:
        s = f"{model}|{extra}|{prompt}"
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        if not self.enabled:
            return None
        return self._cache.get(key)

    def put(self, key: str, value: str):
        with self.lock:
            self._cache[key] = value
            self._save(key, value)

    def get_or_call(
        self,
        prompt: str,
        callable_fn: Callable[[], Any],
        model: str = "",
        extra: str = "",
    ) -> str:
        """命中即返回, 未命中调用并缓存"""
        if not self.enabled:
            return self._extract_text(callable_fn())
        key = self.make_key(prompt, model, extra)
        if key in self._cache:
            return self._cache[key]
        result = callable_fn()
        text = self._extract_text(result)
        self.put(key, text)
        return text

    @staticmethod
    def _extract_text(result: Any) -> str:
        # OpenAI ChatCompletion 兼容
        try:
            return result.choices[0].message.content or ""
        except AttributeError:
            return str(result)
