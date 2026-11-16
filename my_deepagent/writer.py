"""独立报告 Writer：由 Python 主流程显式调用，不依赖 Chief 自觉 write_file。

采用「编辑决策 → 成稿」两阶段：先让 LLM 对 Findings 做信息取舍（哪些保留、哪些
展开、哪些舍弃、哪些缺口），再依据这份取舍综合成稿，避免把 Findings 逐篇堆砌进
正文。

Findings 总量超出单次上下文预算时，先做逐条证据提炼（保留稳定来源标识与关键
数据），再进入编辑决策与成稿，避免丢失关键证据。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

# 允许直接 `python writer.py` 运行：把项目根目录加入 sys.path，使包内相对导入可用。
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _PACKAGE_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from my_deepagent.llm import chat, chat_json


# 单批 Findings 交给模型的最大字符数，超出则逐条提炼。
_DEFAULT_MAX_CONTEXT_CHARS = 32000
# 单条 Findings 需要提炼时的触发阈值与提炼输出上限。
_DIGEST_TRIGGER_CHARS = 8000
_DIGEST_MAX_TOKENS = 4096
# 编辑决策与成稿阶段的输出上限。
_EDITORIAL_MAX_TOKENS = 2500
_EDITORIAL_MAX_ATTEMPTS = 2
_WRITER_MAX_TOKENS = 8192


# 内化的写作约束：只保留当前研究报告真正需要的取舍与克制要求。
WRITER_RULES = """## 写作要求

### 信息取舍
- 只为「核心判断」服务：一项信息只有在能解释核心判断、支撑重要结论或体现公司与行业差异时才写；普通背景简写，无关内容直接舍弃。不追求覆盖率，检索到的数据不等于必须写进去。

### 论据克制
- 一个判断只保留最有代表性、最有解释力的一项证据；多个数据重复证明同一结论时只留一个。论据服务于判断，不是展示研究成果，不要连续堆叠数字、案例与来源。

### 语言表达
- 直接陈述事实与判断，不写券商研报式宏大叙事，不用空洞的总结性语言；一句话能说完不写一段，不追求段落对称、句式整齐，不硬造专业概念。

### 分析重点
- 不把历史数据罗列当成行业研究。优先回答：行业发生了什么变化、当前最主要的供需矛盾、未来哪些因素可能改变格局、公司在变化中处于什么位置。只有资料确实支持时才展开过去/现在/未来。

### 篇幅
- 不设硬性字数，由真正需要记录的信息量决定；不填满每个章节，不为凑字增加内容。

### 忠实与引用
- 忠实原始 Findings，不编造事实、数据与来源；区分事实、来源观点、测算与推断，不把推测写成确定事实。
- 原样保留 [SRC:xxxxxxxxxxxx] 稳定来源标识，不转换为数字编号，不自行创建标识。
- 缺失信息如实标注（检索不足 / 无法核实 / 已确认不适用），不伪装成完整结论。
- 保留关键年份、单位与统计口径；口径不一致或数据冲突时显式说明。

### 输出前检查
- 内部检查并删减：不必要的小标题、数据堆砌、重复表达、无关背景、遗漏的关键信息或自造概念。只输出最终报告，不输出检查过程与前置说明。
"""


EDITORIAL_GUIDE = """你是写作前的编辑助手，只负责判断哪些研究资料值得写、哪些删减，不负责写正文。

基于 Section Definition、研究计划与 Findings，做简洁的编辑决策：

1. core_issues：本节最值得回答的核心问题（不要把行业规模、销量、市占率的罗列当成核心问题）。
2. keep：直接支撑核心问题、或能体现公司与行业关键差异的内容（用 Findings 文件名或原文短语定位）。
3. expand：需要解释原因、机制或未来变化的问题。
4. brief：一句话即可交代的信息。
5. discard：背景、重复数据、次要案例等可舍弃的内容。
6. important_gaps：会实质影响核心判断的缺失信息（一般性资料缺失不算）。

要求：
- 不重新总结全部 Findings，不写成缩小版研究报告。
- 尽量用 Findings 中的文件名或原文短语定位，不得编造定位。
- 只做信息取舍与篇幅分配，不提前撰写正文。
- 允许字段为空；没有重要缺口时 important_gaps 返回空数组。
- 输出必须是单个 JSON 对象，不要 Markdown 围栏或解释。

JSON 结构：
{"core_issues":[],"keep":[],"expand":[],"brief":[],"discard":[],"important_gaps":[]}
"""

_EDITORIAL_KEYS = (
    "core_issues",
    "keep",
    "expand",
    "brief",
    "discard",
    "important_gaps",
)


def _digest_finding(name: str, content: str) -> str:
    """把一条 Findings 压缩为保留 SRC 标识与关键证据的紧凑摘要。"""
    prompt = (
        "你是一名研究助理，需要把一份 Findings 压缩成用于撰写最终报告的紧凑摘要。\n"
        "要求：\n"
        "1. 保留所有 [SRC:xxxxxxxxxxxx] 稳定来源标识，不得改写、丢弃或创建新标识。\n"
        "2. 保留关键事实、数字、年份、单位、口径、机制链条与判断，删除冗长重复的叙述。\n"
        "3. 保留「检索不足 / 无法核实 / 已确认不适用 / 待补充」等缺口标记。\n"
        "4. 只做压缩，不新增任何事实、数据或来源。\n\n"
        f"Findings 文件名：{name}\n\n{content}"
    )
    return chat(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=_DIGEST_MAX_TOKENS,
    )


def _lossy_finding_fallback(name: str, content: str) -> str:
    """提炼失败时的有损压缩兜底：保留标题结构与 SRC 标识，正文截断。

    相比无脑截断，多保留一份结构索引与来源标识，让成稿阶段仍能判断
    该 Findings 覆盖了哪些子问题、引用了哪些来源。
    """
    srcs = sorted(set(re.findall(r"\[SRC:[0-9a-f]{12}\]", content)))
    headings = re.findall(r"^#{1,3}\s+.+$", content, flags=re.MULTILINE)
    body = content[:_DIGEST_TRIGGER_CHARS]
    lines = [f"[压缩兜底] 未能调用 LLM 提炼「{name}」，以下为截断内容："]
    if headings:
        lines.append("小节结构：" + " / ".join(headings[:12]))
    if srcs:
        lines.append(f"来源标识（{len(srcs)} 个）：" + "、".join(srcs[:20]))
    lines.append(body)
    return "\n".join(lines) + "\n……[已截断]"


def _assemble_findings_context(
    findings: list[tuple[str, str, bool]],
    max_context_chars: int,
) -> str:
    """按上下文预算组织 Findings；超预算时按体积从大到小逐条提炼，避免静默丢弃。

    原实现只对超过 8000 字符的 Findings 做一次提炼，仍超预算时在遇到第一份
    装不下的 Findings 处直接 ``break``，导致其后所有 Findings 被整体省略。这里
    改为：优先提炼体积最大的未提炼条目，直到装进预算或全部已提炼；最后仍超预算
    时，为被省略的 Findings 保留明确清单，而不是无痕迹丢弃。
    """
    # 每条记录为 [文件名, 正文, 完整性标记, 是否已提炼]。
    blocks: list[list] = []
    for name, content, complete in findings:
        flag = "" if complete else "（程序标记：研究不完整）"
        blocks.append([name, content, flag, False])

    def build() -> str:
        parts = [
            f"## Findings：{name}{flag}\n\n{content}"
            for name, content, flag, _digested in blocks
        ]
        return "\n\n---\n\n".join(parts)

    combined = build()
    if len(combined) <= max_context_chars:
        return combined

    # 超预算：优先提炼体积最大的尚未提炼条目，直至装入预算或全部已提炼。
    while len(build()) > max_context_chars:
        candidates = [i for i, block in enumerate(blocks) if not block[3]]
        if not candidates:
            break
        index = max(candidates, key=lambda i: len(blocks[i][1]))
        name, content, flag, _ = blocks[index]
        try:
            digested = _digest_finding(name, content)
        except Exception:
            digested = _lossy_finding_fallback(name, content)
        blocks[index] = [name, digested, flag, True]

    combined = build()
    if len(combined) <= max_context_chars:
        return combined

    # 全部提炼后仍超预算：按条目累计到预算内，其余在末尾列出明确清单。
    kept: list[str] = []
    omitted: list[str] = []
    used = 0
    for name, content, flag, _digested in blocks:
        block = f"## Findings：{name}{flag}\n\n{content}"
        if kept and used + len(block) > max_context_chars:
            omitted.append(f"{name}{flag}")
            continue
        kept.append(block)
        used += len(block)

    if omitted:
        kept.append(
            "\n> 以下 Findings 因上下文预算已省略，请勿在正文中引用其内容：\n> "
            + "\n> ".join(omitted)
        )
    return "\n\n---\n\n".join(kept)


def _build_task_context(task: dict, section_brief: str) -> str:
    lines = [
        f"公司：{task['company']}",
        f"报告期间：{task['report_period']}",
        f"Section：{task['section_id']}",
        "",
        section_brief.strip(),
    ]
    return "\n".join(lines).strip()


def _normalize_editorial(result: Any) -> dict:
    """校验编辑决策 JSON 结构；返回各字段均为字符串列表的字典。"""
    if not isinstance(result, dict):
        raise ValueError("编辑决策不是 JSON 对象")

    normalized: dict[str, list[str]] = {}
    for key in _EDITORIAL_KEYS:
        value = result.get(key, [])
        if value is None:
            value = []
        if isinstance(value, str):
            value = [value] if value.strip() else []
        if not isinstance(value, list):
            raise ValueError(f"编辑决策字段 {key} 必须是数组")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        normalized[key] = cleaned

    if not any(normalized.values()):
        raise ValueError("编辑决策返回了空决策")
    return normalized


def _run_editorial(
    task_context: str,
    plan_md: str,
    findings_context: str,
) -> dict:
    """编辑决策阶段；连续失败时抛出异常，阻止进入成稿。"""
    prompt_parts = [
        task_context,
        "研究计划（仅用于理解研究范围，不是事实证据）：\n" + (plan_md.strip() or "（无）"),
        "原始 Findings（事实与判断的唯一依据）：\n" + findings_context,
        EDITORIAL_GUIDE,
    ]
    prompt = "\n\n".join(prompt_parts)
    messages = [{"role": "user", "content": prompt}]

    last_error: Exception | None = None
    for attempt in range(1, _EDITORIAL_MAX_ATTEMPTS + 1):
        try:
            raw = chat_json(
                messages,
                temperature=0.3,
                max_tokens=_EDITORIAL_MAX_TOKENS,
            )
            return _normalize_editorial(raw)
        except Exception as exc:
            last_error = exc
            if attempt < _EDITORIAL_MAX_ATTEMPTS:
                messages.append({
                    "role": "user",
                    "content": (
                        "上次输出未通过 JSON 结构校验。请严格返回单个 JSON 对象，"
                        "所有字段均为数组（允许空数组），不要添加解释或 Markdown 围栏。"
                    ),
                })

    # 编辑决策连续失败即终止：不允许在缺少信息取舍决策的情况下直接成稿。
    raise RuntimeError(
        f"编辑决策连续 {_EDITORIAL_MAX_ATTEMPTS} 次失败，终止成稿：{last_error}"
    )


def _build_writer_prompt(
    task: dict,
    section_brief: str,
    plan_md: str,
    findings_context: str,
    editorial: dict,
) -> str:
    parts = [
        "你是一名资深上市公司研究分析师，负责根据研究团队产出的 Findings 撰写最终研究报告。",
        _build_task_context(task, section_brief),
        WRITER_RULES,
    ]

    if plan_md.strip():
        parts.append("## 研究计划（Chief 制定，仅用于理解研究范围）\n\n" + plan_md.strip())

    parts.append(
        "## 写作前编辑决策（用于控制信息取舍，不是事实来源）\n"
        "- core_issues：本节优先回答的问题。\n"
        "- keep：优先保留的信息。\n"
        "- expand：值得解释清楚的问题。\n"
        "- brief：简要交代即可的信息。\n"
        "- discard：默认不写；只有发现其对核心判断不可或缺、或决策遗漏了关键事实时才重新纳入。\n"
        "- important_gaps：证据缺口，不得用推测补齐。\n"
        "编辑决策不能覆盖原始 Findings；发现决策遗漏重要事实时可回查补充，但不要因此重新罗列全部资料。\n\n"
        + _dump_editorial(editorial)
    )

    parts.append("## 研究发现（Findings）\n\n" + findings_context)
    parts.append(
        "请根据 Section Definition 完成最终报告。直接输出完整 Markdown 正文，"
        "不要输出分析过程、编辑决策或自我评价。"
    )
    return "\n\n".join(parts)


def _dump_editorial(editorial: dict) -> str:
    import json

    return json.dumps(editorial, ensure_ascii=False, indent=2)


def write_report(
    task: dict,
    section_brief: str,
    plan_md: str,
    findings: list[tuple[str, str, bool]],
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
    progress: Callable[[str], None] | None = None,
) -> str:
    """根据已有研究成果生成最终报告正文，返回含稳定来源标识的 Markdown。

    参数 ``findings`` 为 ``(文件名, 正文, 是否完整)`` 三元组列表。流程：
    组装 Findings → 编辑决策（信息取舍）→ 依据决策与写作规则成稿。
    ``progress`` 用于向调用方报告阶段进度，缺省时静默。
    """
    if not findings:
        raise ValueError("没有可用 Findings，无法生成报告")

    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    _log(f"组装 {len(findings)} 份 Findings 上下文……")
    findings_context = _assemble_findings_context(findings, max_context_chars)
    task_context = _build_task_context(task, section_brief)

    _log("阶段 1/2：编辑决策（信息取舍）……")
    editorial = _run_editorial(task_context, plan_md, findings_context)
    _log(
        f"编辑决策完成：core_issues={len(editorial.get('core_issues', []))}，"
        f"keep={len(editorial.get('keep', []))}，"
        f"discard={len(editorial.get('discard', []))}"
    )

    _log("阶段 2/2：成稿……")
    prompt = _build_writer_prompt(
        task, section_brief, plan_md, findings_context, editorial
    )
    body = chat(
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=_WRITER_MAX_TOKENS,
    )
    if not isinstance(body, str) or len(body.strip()) < 40:
        raise RuntimeError("Writer 未生成有效报告正文")
    _log(f"成稿完成：{len(body.strip())} 字符")
    return body.rstrip() + "\n"


def read_text(path: str | Path) -> str:
    """读取计划等辅助文件；不存在时返回空串。"""
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def run_writer_only(run_dir: str | Path, verbose: bool = True) -> str:
    """复用已有 Findings 直接成稿，跳过 Chief 与研究阶段。

    从 run 目录读取 task_snapshot.json、section_snapshot.json、plan.md 与
    findings/*.md，调用 write_report 生成报告、做引用归一化，写回 report.md。
    返回 report.md 的绝对路径。
    """
    import json
    from datetime import datetime

    from my_deepagent.citations import finalize_report_citations
    from my_deepagent.main import (
        STATUS_COMPLETED_BY_WRITER,
        _apply_degraded,
        _atomic_write_json,
        _collect_findings,
        _collect_plan,
        _write_status,
    )
    from my_deepagent.section_def import check_delivery, render_section_brief

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    run_dir = Path(run_dir).resolve()
    _log(f"Output directory: {run_dir}")

    # 1. 任务快照
    task_path = run_dir / "task_snapshot.json"
    if not task_path.is_file():
        raise FileNotFoundError(f"缺少任务快照：{task_path}")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    _log(f"任务：{task['company']} / {task['report_period']} / {task['section_id']}")

    # 2. Section
    section_path = run_dir / "section_snapshot.json"
    section = (
        json.loads(section_path.read_text(encoding="utf-8"))
        if section_path.is_file()
        else {}
    )
    if section:
        section_brief = render_section_brief(
            section, task["company"], task["report_period"]
        )
        _log(f"Section：{section.get('name', task['section_id'])}（{task['section_id']}）")
    else:
        section_brief = f"Section：{task['section_id']}"
        _log("（未找到 section_snapshot.json，回退为 section_id 简述）")

    # 3. 研究计划
    plan_md = _collect_plan(run_dir)
    _log(
        "研究计划 plan.md："
        + (f"已读取（{len(plan_md)} 字符）" if plan_md.strip() else "未找到")
    )

    # 4. Findings
    findings = _collect_findings(run_dir)
    if not findings:
        raise ValueError(f"未在 {run_dir / 'findings'} 找到任何 Findings")
    complete_count = sum(1 for _, _, complete in findings if complete)
    _log(
        f"已收集 {len(findings)} 份 Findings"
        f"（完整 {complete_count} / 不完整 {len(findings) - complete_count}）："
    )
    for name, _content, complete in findings:
        _log(f"  - {name}  {'完整' if complete else '不完整'}")

    # 5. Writer 生成
    _write_status(run_dir, "running")
    _log("\n[Writer] 开始生成报告……")
    report = write_report(task, section_brief, plan_md, findings, progress=_log)

    # 6. 引用归一化
    _log("[Writer] 引用归一化……")
    registry_path = run_dir / "_internal" / "source_registry.json"
    report, citation_validation = finalize_report_citations(report, registry_path)
    _log(
        f"引用校验：{citation_validation.get('source_count', 0)} 个来源，"
        f"未知 {len(citation_validation.get('unknown_source_ids', []))} 个。"
    )

    # 7. 写盘 + 状态
    report_path = run_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    delivery_validation = (
        check_delivery(report, section)
        if section
        else {"complete": True, "deliverable_count": 0, "missing": []}
    )
    _atomic_write_json(run_dir / "delivery_validation.json", delivery_validation)

    has_incomplete = any(not complete for _, _, complete in findings)
    status = _apply_degraded(
        STATUS_COMPLETED_BY_WRITER,
        citation_validation.get("has_warnings", False),
        has_incomplete,
        None,
        has_delivery_gaps=not delivery_validation["complete"],
    )
    _write_status(run_dir, status)

    metadata_path = run_dir / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    metadata.update({
        "finished_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "used_writer": True,
        "findings_count": len(findings),
        "has_incomplete_findings": has_incomplete,
        "citation_validation": citation_validation,
        "delivery_validation": delivery_validation,
    })
    _atomic_write_json(metadata_path, metadata)

    _log(f"\n[结束] 状态：{status}")
    _log(f"报告：{report_path}")
    return str(report_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="复用已有 Findings 直接运行 Writer 生成报告（跳过 Chief 与研究阶段）"
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        help="run 目录路径；缺省时使用 outputs 下最新的 run_* 目录",
    )
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        outputs = _PACKAGE_DIR / "outputs"
        runs = sorted(
            outputs.glob("run_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not runs:
            parser.exit(1, f"outputs 下没有找到 run_* 目录：{outputs}\n")
        run_dir = runs[0]

    report_path = run_writer_only(run_dir)
    print(f"报告已写入：{report_path}")