"""DeepSeek OpenAI-compatible Chat Completions client."""
from __future__ import annotations

import json
import re
from typing import Any

import requests

from my_deepagent.config import get_config
from my_deepagent.cost import record_usage


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    cfg = get_config()
    if not cfg.deepseek_api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在 my_deepagent/.env 中配置")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 必须是非空列表")
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0")

    model_name = model or cfg.deepseek_model
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        f"{cfg.deepseek_base_url}/chat/completions",
        json=payload,
        headers={
            "Authorization": f"Bearer {cfg.deepseek_api_key}",
            "Content-Type": "application/json",
        },
        timeout=cfg.deepseek_timeout,
    )
    response.raise_for_status()

    try:
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("DeepSeek 返回结构不符合 Chat Completions 格式") from exc

    content = message.get("content")
    if isinstance(content, list):
        # 兼容部分 OpenAI-compatible 服务返回的分段文本。
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        )
    if not isinstance(content, str):
        raise RuntimeError("DeepSeek 返回的 message.content 为空或不是文本")

    record_usage(
        data.get("model") or model_name,
        data.get("usage") or {},
        pricing_tier=cfg.deepseek_pricing_tier,
    )
    return content


def _extract_json(text: str) -> Any:
    """解析 JSON 对象；允许模型在 JSON 前后附带说明或 Markdown 围栏。"""
    text = _strip_fences(str(text))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 可选修复器；未安装时不影响正常 JSON 解析。
    try:
        from json_repair import repair_json  # type: ignore

        repaired = json.loads(repair_json(text))
        return repaired
    except Exception:
        pass

    # 从文本中提取第一个完整 JSON 对象，正确处理字符串中的括号。
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def chat_json(messages, *, strict: bool = True, **kwargs) -> dict:
    """请求 JSON 对象；strict=False 时解析失败返回空字典。"""
    raw = chat(messages, json_mode=True, **kwargs)
    parsed = _extract_json(raw)
    if isinstance(parsed, dict):
        return parsed
    if strict:
        raise ValueError("模型未返回有效 JSON 对象")
    return {}