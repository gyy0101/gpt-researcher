"""DeepSeek 调用封装（直接走 OpenAI 兼容的 /chat/completions 接口）。"""
import json
import re

import requests

from .config import get_config
from .cost import record_usage


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def chat(messages, *, model=None, temperature=0.3, max_tokens=4096, json_mode=False) -> str:
    cfg = get_config()
    if not cfg.deepseek_api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")

    url = cfg.deepseek_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model or cfg.deepseek_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {cfg.deepseek_api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=cfg.deepseek_timeout)
    resp.raise_for_status()
    data = resp.json()
    model_used = data.get("model") or (model or cfg.deepseek_model)
    record_usage(model_used, data.get("usage") or {})
    return data["choices"][0]["message"]["content"]


def _extract_json(text: str):
    """从模型输出中提取 JSON，容忍前后文字、代码块、残缺 JSON。

    依次尝试：直接解析 → json_repair（若已安装）→ 括号匹配提取第一个完整片段。
    失败返回 None。
    """
    if not isinstance(text, str):
        text = str(text)
    text = _strip_fences(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        from json_repair import repair_json  # type: ignore

        return json.loads(repair_json(text))
    except Exception:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    return None


def _normalize_json(obj):
    """把解析结果归一化为 dict，避免模型返回数组时下游 .get 崩溃。"""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                return item
        return {"result": obj}
    return {}


def chat_json(messages, **kwargs):
    """调用 DeepSeek 并容错解析 JSON 返回。

    模型可能返回带前后文字、代码块、数组甚至残缺 JSON，这里统一做容错；
    最坏情况返回空 dict，由调用方回退到原始 query。
    """
    text = chat(messages, json_mode=True, **kwargs)
    obj = _extract_json(text)
    return _normalize_json(obj)