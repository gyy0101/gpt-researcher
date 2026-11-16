"""RAG 服务 HTTP 客户端。"""
from __future__ import annotations

from typing import Any

import requests

from my_deepagent.config import get_config


def search(
    question: str,
    *,
    top_k: int = 5,
    kb_ids=None,
    doc_ids=None,
    chunk_type: str = "text",
    source_type=None,
    as_of_date=None,
) -> list[dict[str, Any]]:
    cfg = get_config()
    if not cfg.rag_base_url:
        raise RuntimeError("缺少 RAG_BASE_URL，请在 my_deepagent/.env 中配置")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("RAG search question 不能为空")
    if int(top_k) <= 0:
        raise ValueError("top_k 必须大于 0")

    body = {
        "question": question.strip(),
        "top_k": int(top_k),
        "kb_ids": kb_ids if kb_ids is not None else (cfg.rag_kb_ids or None),
        "doc_ids": doc_ids,
        "chunk_type": chunk_type,
        "source_type": source_type,
        "as_of_date": as_of_date if as_of_date is not None else (cfg.rag_as_of_date or None),
    }
    body = {key: value for key, value in body.items() if value is not None}

    response = requests.post(
        f"{cfg.rag_base_url}/rag/search",
        json=body,
        timeout=cfg.rag_timeout,
    )
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("RAG 服务返回的不是合法 JSON") from exc

    if not isinstance(data, dict):
        raise RuntimeError("RAG 服务返回结构错误：顶层应为 JSON 对象")
    chunks = data.get("chunks", [])
    if not isinstance(chunks, list):
        raise RuntimeError("RAG 服务返回结构错误：chunks 应为数组")
    return [chunk for chunk in chunks if isinstance(chunk, dict)]