"""研究工具：quick_search / deep_research，用你的 RAG + DeepSeek 实现。

接口签名与原 deep_agents/tools.py 保持一致（async 函数 + 返回字符串），
因此可以直接被真正的 Deep Agents harness（create_deep_agent）当作工具使用。
"""
import asyncio
import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_deepagent import llm, rag

CITATION_RULE = (
    "引用规范：正文关键结论后用 [n] 标注来源编号（如 [1]，或多来源 [2][3]）；"
    "末尾附「来源」列表，每行 [n] 来源名称。"
    "来源名称只写券商研报或年报（如「宁德时代 2025 年年报」「东吴证券研报」），"
    "不写页码、不写内部文件路径；不得伪造来源编号。"
)


def _chunk_source(chunk) -> str:
    """来源标注：只返回文档标题（券商研报/年报名称），不附页码；引用统一用 [n] 编号。"""
    return chunk.get("title") or "未命名文档"


def _chunk_content(chunk) -> str:
    return (
        chunk.get("content_with_weight")
        or chunk.get("content")
        or chunk.get("content_ltks")
        or ""
    ).strip()


def _chunk_key(chunk) -> str:
    """chunk 级去重键。

    优先使用稳定的 chunk_id；若没有，则用「doc_id + 位置信息 + 内容摘要哈希」。
    不能仅凭 doc_id 判定两个 chunk 相同，否则同一文档下内容不同的 chunk
    会被误判为重复而丢失。
    """
    cid = chunk.get("chunk_id")
    if cid:
        return f"chunk:{cid}"

    doc_id = chunk.get("doc_id") or "unknown_doc"
    position = (
        chunk.get("page_num_int")
        or chunk.get("position_int")
        or chunk.get("page")
        or chunk.get("position")
        or chunk.get("chunk_index")
        or ""
    )
    content = _chunk_content(chunk)
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:16] if content else "empty"
    return f"{doc_id}|{position}|{digest}"


def _dedup_chunks(chunks):
    """按 chunk 级键去重，保留首次出现的 chunk，不改变顺序。"""
    seen = set()
    out = []
    for c in chunks:
        key = _chunk_key(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _doc_key(chunk) -> str:
    """文档级去重键，用于把同一文档的多个 chunk 归到同一个引用编号。"""
    return chunk.get("doc_id") or chunk.get("title") or ""


def _format_chunks(chunks) -> str:
    """快速检索用：按 chunk 逐个编号，仅供 scoping。"""
    lines = []
    for i, c in enumerate(chunks, 1):
        src = _chunk_source(c)
        content = _chunk_content(c)
        lines.append(f"[{i}] 来源：{src}\n{content}")
    return "\n\n".join(lines)


def _build_cited_context(chunks):
    """按文档去重后编号，返回 (上下文文本, 去重来源列表)。

    同一文档的多个 chunk 共享同一引用编号，保证正文 [n] 与来源列表一一对应、
    无重复来源。
    """
    lines = []
    doc_to_num = {}
    sources = []  # sources[num - 1] 即第 num 号来源的标注文本
    for c in chunks:
        dkey = _doc_key(c)
        if dkey and dkey in doc_to_num:
            num = doc_to_num[dkey]
        else:
            num = len(sources) + 1
            if dkey:
                doc_to_num[dkey] = num
            sources.append(_chunk_source(c))
        content = _chunk_content(c)
        lines.append(f"[{num}] 来源：{sources[num - 1]}\n{content}")
    return "\n\n".join(lines), sources


def _plan_sub_queries(query: str, parent_query: str, max_sub_queries: int):
    prompt = (
        f"你是一名金融研究助理。请把下面的研究主题拆分成不超过 {max_sub_queries} 个、"
        f"彼此互补的检索子问题，用于从知识库中检索材料。每个子问题要具体、可检索。\n\n"
        f"研究主题：{query}\n"
        + (f"所属报告主题：{parent_query}\n" if parent_query else "")
        + "\n请只输出 json 格式，形如：{\"sub_queries\": [\"子问题1\", \"子问题2\"]}"
    )
    data = llm.chat_json([{"role": "user", "content": prompt}], temperature=0.2)
    if not isinstance(data, dict):
        data = {}
    subs = data.get("sub_queries") or []
    subs = [s for s in subs if isinstance(s, str) and s.strip()]
    if not subs:
        subs = [query]
    return subs[:max_sub_queries]


def _synthesize(query: str, context: str) -> str:
    prompt = (
        f"请根据以下检索到的资料，围绕研究主题撰写一份分析草稿。\n\n"
        f"研究主题：{query}\n\n"
        f"要求：\n"
        f"1. 使用中文，结构清晰，可用小标题组织；\n"
        f"2. 每个关键结论后用 [n] 标注对应的资料来源编号（与末尾 Sources 列表的序号一致）；\n"
        f"3. 只依据给定资料，不编造资料中没有的信息；\n"
        f"4. 数据要注明出处。\n"
        f"5. {CITATION_RULE}\n\n"
        f"资料如下：\n\n{context}"
    )
    return llm.chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=4096)


def build_research_tools(task: dict):
    """构建研究工具，固定走你的本地 RAG。"""
    top_k = int(task.get("rag_top_k", 8))
    max_sub_queries = int(task.get("max_sub_queries", 3))
    max_deep_research_calls = int(task.get("max_deep_research_calls", 12))
    verbose = bool(task.get("verbose", True))

    # 全流程共享的深度检索预算，工具层硬性执行，而非仅靠 Prompt 提醒。
    budget = {"deep_research": 0}

    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    async def quick_search(query: str) -> str:
        """Run a fast search over the knowledge base and return the top results with snippets and sources.

        Use this to scope a topic, verify a fact, or gather just enough context
        to plan a report outline. Results are raw snippets - iterate with refined
        queries if the first search does not surface what you need. For in-depth,
        citation-grade research on a section, use deep_research instead.
        """
        log(f"[quick_search] {query}")
        chunks = await asyncio.to_thread(rag.search, query, top_k=5)
        if not chunks:
            return "No results found."
        log(f"[quick_search] 命中 {len(chunks)} 个 chunk")
        return _format_chunks(chunks)

    async def deep_research(query: str, parent_query: str = "") -> str:
        """Run the full research pipeline on a query over the knowledge base.

        Plans sub-queries, retrieves and deduplicates chunks from the knowledge
        base, and returns a detailed markdown report with in-text citations and
        a source list. This is thorough - use it for in-depth, citation-grade
        research on a specific question. You may call it multiple times for
        different questions, but the total number of calls is capped by the
        research budget; use quick_search for cheap lookups instead.

        Args:
            query: The research question or section topic to investigate.
            parent_query: The overall report topic, when researching a section
                of a larger report. Keeps the research focused in context.
        """
        if budget["deep_research"] >= max_deep_research_calls:
            return (
                f"研究预算已耗尽：deep_research 已调用 {budget['deep_research']} 次，"
                f"达到上限 {max_deep_research_calls} 次。"
                "请基于已检索到的材料完成研究发现，不要继续发起新的深度检索。"
            )
        budget["deep_research"] += 1
        log(f"[deep_research] 主题：{query}")
        log("[deep_research] 规划子问题（调用 DeepSeek）…")
        subs = await asyncio.to_thread(
            _plan_sub_queries, query, parent_query, max_sub_queries
        )
        log(f"[deep_research] 拆分为 {len(subs)} 个子问题：")
        for s in subs:
            log(f"[deep_research]   · {s}")

        semaphore = asyncio.Semaphore(3)

        async def _search_sub(sq):
            async with semaphore:
                found = await asyncio.to_thread(rag.search, sq, top_k=top_k)
                log(f"[deep_research] 检索「{sq}」→ {len(found)} 个 chunk")
                return sq, found

        results = await asyncio.gather(*(_search_sub(sq) for sq in subs))

        chunks = []
        for _sq, found in results:
            chunks.extend(found)
        chunks = _dedup_chunks(chunks)

        if not chunks:
            log("[deep_research] 未检索到相关材料")
            return f"No relevant material found for '{query}'."

        log(f"[deep_research] 去重后 {len(chunks)} 个 chunk，调用 DeepSeek 合成草稿…")
        context, sources = _build_cited_context(chunks)
        draft = await asyncio.to_thread(_synthesize, query, context)

        log(f"[deep_research] 草稿完成（{len(sources)} 个去重来源）")
        if sources:
            refs = "\n".join(f"[{i}] {s}" for i, s in enumerate(sources, 1))
            draft += "\n\n### Sources\n\n" + refs
        return draft

    return quick_search, deep_research