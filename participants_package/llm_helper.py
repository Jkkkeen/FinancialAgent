# ============================================================
# 🔧 MODIFIABLE — LLM 调用辅助类
#
# 本文件提供即插即用的 LLM 客户端，选手可以:
#   1. 直接使用: from llm_helper import LLMHelper
#   2. 复制并修改: 替换为自己的 LLM 客户端
#   3. 完全替换: 使用其他 SDK (Anthropic、LangChain 等)
#
# 使用方式:
#   from llm_helper import LLMHelper, create_llm_client
#   llm = create_llm_client("config.yaml")
#   response = llm.chat(system_prompt, user_prompt)
#
# 💡 TIP: 也可以直接使用 OpenAI SDK:
#   from openai import OpenAI
#   client = OpenAI(api_key="...", base_url="...")
# ============================================================

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional

import yaml

logger = logging.getLogger(__name__)

# ============================================================
# 类型别名 — 与 investment_agent.py 中 LLMClient 兼容
# LLMClient = Callable[[str, str], str]
#   第一个 str: system prompt（系统提示词，设定 Agent 角色）
#   第二个 str: user prompt（用户提示词，包含市场数据、信念状态等）
#   返回值 str: LLM 原始响应文本（通常为 JSON 格式的决策）
# ============================================================


class LLMHelper:
    """OpenAI 兼容的 LLM 调用助手。

    🔧 MODIFIABLE: 如果使用 Anthropic 或其他 SDK，替换此类即可。
    只需保持 ``chat(system, user) -> str`` 签名不变。

    支持的提供商:
        - deepseek（DeepSeek API）
        - openai（OpenAI API）
        - 任何 OpenAI 兼容端点（custom）

    Attributes:
        config: 完整配置字典（从 config.yaml 加载）。
        provider: 提供商名称（"deepseek" / "openai" / "custom"）。
        model: 模型名称。
        client: OpenAI 客户端实例。
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """从 YAML 配置文件初始化 LLM 客户端。

        Args:
            config_path: 配置文件路径。默认为当前目录下的 config.yaml。
        """
        self.config = self._load_config(config_path)
        llm_cfg = self.config.get("llm", {})
        self.provider = llm_cfg.get("provider", "deepseek")
        self.model = llm_cfg.get("model", "deepseek-chat")
        self.temperature = float(llm_cfg.get("temperature", 0.7))
        self.max_tokens = int(llm_cfg.get("max_tokens", 256))
        self.base_url = llm_cfg.get("base_url", "https://api.deepseek.com/v1")

        # 从环境变量读取 API key
        api_key_env = llm_cfg.get("api_key_env", "API_KEY")
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            # 尝试从配置文件直接读取（不推荐）
            api_key = llm_cfg.get("api_key", "")
        if not api_key:
            logger.warning(
                "⚠️ 未找到 API key！请设置环境变量 %s 或在 config.yaml 中配置。",
                api_key_env,
            )

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError(
                "需要安装 openai 包: pip install openai>=1.0.0"
            )

    def chat(self, system: str, user: str) -> str:
        """发送一次对话请求并返回 LLM 响应文本。

        🔧 MODIFIABLE: 这是核心调用点。替换为其他 SDK 时修改此方法。

        Args:
            system: 系统提示词（设定 Agent 角色和行为约束）。
            user: 用户提示词（包含市场观察数据和决策要求）。

        Returns:
            LLM 原始响应文本。解析工作由调用方完成。

        Raises:
            RuntimeError: API 调用失败时抛出。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as exc:
            logger.error("LLM API 调用失败: %s", exc)
            raise RuntimeError(f"LLM API 调用失败: {exc}") from exc

    def chat_with_retry(
        self, system: str, user: str, max_retries: int = 3
    ) -> str:
        """带重试的对话请求。

        🔧 MODIFIABLE: 可根据需要调整重试策略。

        Args:
            system: 系统提示词。
            user: 用户提示词。
            max_retries: 最大重试次数。

        Returns:
            LLM 响应文本。
        """
        import time

        last_error = None
        for attempt in range(max_retries):
            try:
                return self.chat(system, user)
            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次），%d 秒后重试...",
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"LLM 调用在 {max_retries} 次重试后仍然失败: {last_error}"
        )

    def parse_json_response(
        self, raw: str, default: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """从 LLM 原始响应中解析 JSON。

        💡 TIP: LLM 有时会在 JSON 前后加上 markdown 代码块标记。
        此方法会自动去除这些标记。

        Args:
            raw: LLM 原始响应文本。
            default: 解析失败时返回的默认值。

        Returns:
            解析后的字典，或 default（如果解析失败）。
        """
        # 去除 markdown 代码块标记
        text = raw.strip()
        if text.startswith("```"):
            # 找到第一个换行后的内容
            lines = text.split("\n")
            # 去除首行 ```json 或 ``` 以及末行 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试查找 JSON 片段
            import re

            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        logger.warning("无法解析 LLM 响应为 JSON: %s", raw[:200])
        return default if default is not None else {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载 YAML 配置文件。

        Args:
            config_path: 配置文件路径。

        Returns:
            配置字典。文件不存在时返回空字典。
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("配置文件 %s 未找到，使用默认值。", config_path)
            return {}


def create_llm_client(
    config_path: str = "config.yaml",
) -> Callable[[str, str], str]:
    """创建与 InvestmentAgent 兼容的 LLM 客户端函数。

    此函数的返回值可以直接作为 ``InvestmentAgent.llm_client`` 参数。

    💡 TIP 使用示例:
        from llm_helper import create_llm_client
        from competition_solution.investment_agent import InvestmentAgent

        llm_client = create_llm_client("config.yaml")
        agent = InvestmentAgent(
            agent_id="A0001",
            personality="trend",
            llm_client=llm_client,  # 注入 LLM 客户端
        )

    Args:
        config_path: YAML 配置文件路径。

    Returns:
        Callable[[str, str], str]: 签名兼容 LLMClient 的函数。
    """
    helper = LLMHelper(config_path)

    def _client(system: str, user: str) -> str:
        return helper.chat_with_retry(system, user)

    return _client
