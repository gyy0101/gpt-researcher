"""研究工具：本地 RAG 检索 + DeepSeek 研究合成。"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from my_deepagent import llm
from my_deepagent import rag
from my_deepagent.citations import SourceRegistry


CITATION_RULE = """引用规范：
- 只使用上下文中明确提供的 [SRC:xxxxxxxxxxxx] 稳定来源标识。
- 每个重要事实或判断在对应句子附近标注来源标识。
- 不得自行创建、改写、猜测或重新编号来源标识。
- 不限制来源类型；来源名称必须忠实于 RAG 原始文档标题。
- 不输出内部文件路径、doc_id 或其他内部元数据。
"""


def _chunk_content(chunk: dict) -> str:
    value = (
        chunk.get("content_with_weight")
        or chunk.get("content")
        or chunk.get("content_ltks")
        or ""
    )
    return str(value).strip()


def _chunk_key(chunk: dict) -> str:
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"

    doc_id = chunk.get("doc_id") or chunk.get("title") or "unknown"
    position = next(
        (
            chunk.get(key)
            for key in (
                "page_num_int", "position_int", "page",
                "position", "chunk_index"
            )
            if chunk.get(key) is not None
        ),
        "",
    )
    digest = hashlib.sha256(_chunk_content(chunk).encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}|{position}|{digest}"


def _dedup_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for chunk in chunks:
        key = _chunk_key(chunk)
        if key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


def _format_quick_results(chunks: list[dict], max_chunk_chars: int) -> str:
    lines = [
        "以下是快速检索的定位材料，仅用于规划和发现线索；"
        "临时序号不是引用编号，最终报告必须使用 deep_research 提供的稳定来源标识。"
    ]
    for index, chunk in enumerate(chunks, 1):
        title = str(chunk.get("title") or "未命名文档")
        content = _chunk_content(chunk)
        if len(content) > max_chunk_chars:
            content = content[:max_chunk_chars] + "……[片段截断]"
        lines.append(f"\n### 检索片段 {index}\n来源标题：{title}\n{content}")
    return "\n".join(lines)


def _plan_sub_queries(
    query: str,
    parent_query: str,
    max_sub_queries: int,
) -> list[str]:
    prompt = (
        "你是公司研究助理。把研究问题拆成互补、可直接检索的子问题。"
        "优先覆盖不同维度，避免同义重复；不要把一个问题拆成多个只替换关键词的 query。\n\n"
        f"研究问题：{query}\n"
        f"公司与报告背景：{parent_query or '未提供'}\n"
        f"最多返回 {max_sub_queries} 个子问题。只返回 JSON："
        '{"sub_queries":["问题1","问题2"]}'
    )
    try:
        data = llm.chat_json(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
            strict=False,
        )
    except Exception:
        return [query]

    subs = data.get("sub_queries")
    if not isinstance(subs, list):
        return [query]

    cleaned = []
    for item in subs:
        if isinstance(item, str) and item.strip():
            value = item.strip()
            if value not in cleaned:
                cleaned.append(value)
    return cleaned[:max_sub_queries] or [query]


def _build_cited_context(
    chunks: list[dict],
    registry: SourceRegistry,
    *,
    max_context_chars: int,
    max_chunk_chars: int,
) -> tuple[str, list[tuple[str, str]], int]:
    """构建带 Run 级稳定来源标识的上下文；超预算时整块跳过并显式计数。"""
    lines: list[str] = []
    sources: dict[str, str] = {}
    used_chars = 0
    skipped = 0

    for chunk in chunks:
        content = _chunk_content(chunk)
        if not content:
            continue
        if len(content) > max_chunk_chars:
            content = content[:max_chunk_chars] + "\n[该检索片段因长度限制被截断]"

        marker = registry.register(chunk)
        title = str(chunk.get("title") or "未命名文档（来源元数据缺失）")
        block = f"[{marker}] 来源：{title}\n{content}"

        if used_chars + len(block) > max_context_chars:
            skipped += 1
            continue

        lines.append(block)
        used_chars += len(block)
        sources[marker] = title

    return "\n\n".join(lines), list(sources.items()), skipped


def _synthesize(query: str, context: str, skipped_chunks: int) -> str:
    truncation_note = (
        f"\n注意：受上下文预算限制，另有 {skipped_chunks} 个检索片段未纳入本次合成；"
        "不要假装已经审阅这些片段。"
        if skipped_chunks else ""
    )
    prompt = (
        "请围绕研究问题，对给定知识库材料进行分析，形成供公司研究报告使用的研究发现。"
        "重点解释事实之间的关系、业务机制和真正影响判断的变量，不要把资料逐条堆砌。\n\n"
        f"研究问题：{query}\n\n"
        "要求：\n"
        "1. 只依据提供的材料，不得补造数据、事实或来源。\n"
        "2. 区分已证实事实、来源观点、推算和待核实信息。\n"
        "3. 关键事实和判断使用上下文中的稳定来源标识，例如 [SRC:012345abcdef]。\n"
        "4. 不得输出 [1]、[2] 这种局部数字引用；不得自行创建 SRC 标识。\n"
        "5. 不要把多个重复数字堆在同一句话里；只保留对判断有新增价值的证据。\n"
        "6. 如材料存在口径冲突、时间差或证据不足，明确说明。\n"
        f"{CITATION_RULE}"
        f"{truncation_note}\n\n"
        f"资料：\n{context}"
    )
    return llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=5000,
    )


def build_research_tools(task: dict, run_dir: str | None = None):
    """构建本次 Run 共享的检索工具和来源注册表。"""
    quick_top_k = int(task.get("quick_search_top_k", 5))
    deep_top_k = int(task.get("rag_top_k", 8))
    max_sub_queries = int(task.get("max_sub_queries", 3))
    max_deep_calls = int(task.get("max_deep_research_calls", 12))
    max_quick_calls = int(task.get("max_quick_search_calls", 30))
    max_context_chars = int(task.get("max_context_chars", 35000))
    max_chunk_chars = int(task.get("max_chunk_chars", 7000))
    max_parallel = int(task.get("max_parallel_rag_searches", 3))
    verbose = bool(task.get("verbose", True))

    for name, value in {
        "quick_search_top_k": quick_top_k,
        "rag_top_k": deep_top_k,
        "max_sub_queries": max_sub_queries,
        "max_deep_research_calls": max_deep_calls,
        "max_quick_search_calls": max_quick_calls,
        "max_context_chars": max_context_chars,
        "max_chunk_chars": max_chunk_chars,
        "max_parallel_rag_searches": max_parallel,
    }.items():
        if value <= 0:
            raise ValueError(f"task.json 中 {name} 必须大于 0")

    registry = SourceRegistry(run_dir or ".")
    search_semaphore = asyncio.Semaphore(max_parallel)

    def _normalize_filter_list(value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    filters = {
        "kb_ids": _normalize_filter_list(task.get("kb_ids")),
        "doc_ids": _normalize_filter_list(task.get("doc_ids")),
        "chunk_type": task.get("chunk_type", "text"),
        "source_type": _normalize_filter_list(task.get("source_type")),
        "as_of_date": task.get("as_of_date"),
    }
    filters = {
        key: value for key, value in filters.items()
        if value is not None and value != []
    }

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    async def search_one(query: str, top_k: int) -> list[dict]:
        async with search_semaphore:
            return await asyncio.to_thread(
                rag.search, query, top_k=top_k, **filters
            )

    async def quick_search(query: str) -> str:
        """快速定位知识库材料；结果仅用于 scoping，不作为最终引用编号。"""
        if not isinstance(query, str) or not query.strip():
            return "quick_search 未执行：query 不能为空。"
        log(f"[quick_search] {query}")
        chunks = await search_one(query.strip(), quick_top_k)
        if not chunks:
            return "快速检索未命中相关材料。"
        log(f"[quick_search] 命中 {len(chunks)} 个 chunk")
        return _format_quick_results(chunks, max_chunk_chars)

    async def deep_research(query: str, parent_query: str = "") -> str:
        """对一个具体研究问题进行多 query 检索、去重、引用合成。"""
        if not isinstance(query, str) or not query.strip():
            return "deep_research 未执行：query 不能为空。"

        query = query.strip()
        log(f"[deep_research] 主题：{query}")
        sub_queries = await asyncio.to_thread(
            _plan_sub_queries, query, parent_query, max_sub_queries
        )
        log(f"[deep_research] 拆分为 {len(sub_queries)} 个互补子问题")
        for sub_query in sub_queries:
            log(f"  · {sub_query}")

        results = await asyncio.gather(
            *(search_one(sub_query, deep_top_k) for sub_query in sub_queries),
            return_exceptions=True,
        )

        chunks: list[dict] = []
        errors: list[str] = []
        for sub_query, result in zip(sub_queries, results):
            if isinstance(result, Exception):
                errors.append(f"{sub_query}: {type(result).__name__}: {result}")
                log(f"[deep_research] 子查询失败：{errors[-1]}")
            else:
                log(f"[deep_research] 检索「{sub_query}」→ {len(result)} 个 chunk")
                chunks.extend(result)

        chunks = _dedup_chunks(chunks)
        if not chunks:
            if errors:
                return (
                    "本次深度研究的所有子查询均未成功。错误摘要："
                    + "；".join(errors)
                    + "。请将其记录为检索失败，不要写成行业不存在相关数据。"
                )
            return f"知识库未检索到与「{query}」相关的材料。"

        context, sources, skipped = _build_cited_context(
            chunks,
            registry,
            max_context_chars=max_context_chars,
            max_chunk_chars=max_chunk_chars,
        )
        if not context:
            return "检索命中内容为空或超出上下文预算，未形成可引用证据。"

        log(
            f"[deep_research] 去重后 {len(chunks)} 个 chunk，"
            f"纳入上下文 {len(sources)} 个来源，跳过 {skipped} 个 chunk"
        )
        draft = await asyncio.to_thread(_synthesize, query, context, skipped)

        source_lines = ["\n\n## 来源映射（请在 Findings 中原样保留 SRC 标识）"]
        source_lines.extend(f"[{marker}] {title}" for marker, title in sources)
        if errors:
            source_lines.append(
                "\n检索备注：部分子查询失败，以下发现基于成功返回的子查询；"
                + "；".join(errors)
            )
        log(f"[deep_research] 草稿完成（{len(sources)} 个来源）")
        return draft.rstrip() + "\n" + "\n".join(source_lines)

    return quick_search, deep_research