"""
Provider 无关的结构化输出 LLM 客户端。

设计目标：Planner / Narrator 都通过同一个 `complete_structured(system, user, schema)` 接口调用，
底层换成哪个 provider（OpenRouter / OpenAI / DeepSeek / Anthropic）对上层代码透明。
"""
from __future__ import annotations

import json
import os
from typing import Optional, Protocol, TypeVar

from pydantic import BaseModel

from .env_loader import load_dotenv

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def complete_structured(self, *, system: str, user: str, schema: type[T]) -> T: ...


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 4096):
        import anthropic  # 延迟 import，避免没装这个 SDK 的环境也能 import 本模块的其它部分

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("未找到 ANTHROPIC_API_KEY，请在 .env 或环境变量里设置。")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete_structured(self, *, system: str, user: str, schema: type[T]) -> T:
        tool_name = "emit_result"
        json_schema = schema.model_json_schema()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": tool_name,
                    "description": "输出符合 schema 的结构化结果。",
                    "input_schema": json_schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return schema.model_validate(block.input)

        raise RuntimeError(f"Anthropic 没有返回 tool_use 结果，原始响应：{response.content}")


class OpenRouterClient:
    """OpenRouter 是 OpenAI 兼容 API（换个 base_url + key 就能转发给任意支持的模型，
    包括 anthropic/claude-*、openai/gpt-* 等）。结构化输出走 json_object 模式 + pydantic
    校验，校验失败则把错误信息喂回去重试（不是所有经 OpenRouter 转发的模型都支持严格的
    json_schema 模式，这样实现对模型的兼容性更好）。
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4.5",
        max_tokens: int = 4096,
        max_retries: int = 2,
    ):
        from openai import OpenAI  # 复用 requirements.txt 里已有的 openai SDK

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("未找到 OPENROUTER_API_KEY，请在 .env 或环境变量里设置。")
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def complete_structured(self, *, system: str, user: str, schema: type[T]) -> T:
        json_schema = schema.model_json_schema()
        schema_instructions = (
            "你必须只输出一个 JSON 对象本身，不要有任何额外文字、不要用 markdown 代码块包裹。"
            f"JSON 必须严格符合下面这个 JSON Schema：\n{json.dumps(json_schema, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": system + "\n\n" + schema_instructions},
            {"role": "user", "content": user},
        ]

        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content
            try:
                data = json.loads(content)
                return schema.model_validate(data)
            except Exception as e:  # noqa: BLE001
                last_error = e
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"你上面的输出没有通过校验，错误信息：{e}\n"
                        "请重新输出一个严格符合 schema 的、单独的 JSON 对象，不要包含其他内容。",
                    }
                )

        raise RuntimeError(f"OpenRouter 结构化输出多次重试后仍失败：{last_error}")


DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter")


def get_client(provider: Optional[str] = None, **kwargs) -> LLMClient:
    provider = provider or DEFAULT_PROVIDER
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    if provider == "openrouter":
        return OpenRouterClient(**kwargs)
    raise ValueError(f"Provider '{provider}' 尚未接入，目前支持 'anthropic' / 'openrouter'。")
