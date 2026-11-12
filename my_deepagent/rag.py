"""你的 RAG 服务 HTTP 客户端。"""
import requests

from .config import get_config


def search(
    question: str,
    *,
    top_k: int = 5,
    kb_ids=None,
    doc_ids=None,
    chunk_type: str = "text",
    source_type=None,
    as_of_date=None,
):
    cfg = get_config()
    if not cfg.rag_base_url:
        raise RuntimeError("缺少 RAG_BASE_URL，请在 .env 或环境变量中配置")

    body = {
        "question": question,
        "top_k": top_k,
        "kb_ids": kb_ids if kb_ids is not None else (cfg.rag_kb_ids or None),
        "doc_ids": doc_ids,
        "chunk_type": chunk_type,
        "source_type": source_type,
        "as_of_date": as_of_date if as_of_date is not None else (cfg.rag_as_of_date or None),
    }
    body = {k: v for k, v in body.items() if v is not None}

    url = cfg.rag_base_url.rstrip("/") + "/rag/search"
    resp = requests.post(url, json=body, timeout=cfg.rag_timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("chunks", [])