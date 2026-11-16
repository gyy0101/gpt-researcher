"""研究任务入口：运行 Agent、校验报告、引用归一化与兜底。"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
load_dotenv(PACKAGE_DIR / ".env")

from my_deepagent.agent import build_agent
from my_deepagent.citations import finalize_report_citations
from my_deepagent.config import get_config
from my_deepagent.cost import UsageCallback, UsageTracker, format_report, usage_scope
from my_deepagent.section_def import check_delivery, load_section, render_section_brief
from my_deepagent.writer import read_text, write_report


RUN_STATUS_PATH = "_run_status.txt"

# 运行状态枚举：报告交付与研究成果持久化由程序保证后，状态可被精确区分。
STATUS_COMPLETED = "completed"                      # Chief 已写出有效终稿
STATUS_COMPLETED_BY_WRITER = "completed_by_writer"  # Chief 未写终稿，独立 Writer 成稿
STATUS_DEGRADED = "degraded"                        # 已成稿，但存在证据或结构缺口
STATUS_FALLBACK = "fallback"                        # Writer 失败，用 Findings 最低限度兜底
STATUS_FAILED = "failed"                            # 材料不足，无法形成有效报告


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def _atomic_write_json(path: Path, data: dict) -> None:
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def _write_status(run_dir: Path, status: str) -> None:
    _atomic_write_text(run_dir / RUN_STATUS_PATH, status + "\n")


def validate_task(task: Any) -> dict:
    if not isinstance(task, dict):
        raise ValueError("task.json 顶层必须是 JSON 对象")

    required = ("company", "report_period", "section_id")
    missing = [key for key in required if not str(task.get(key, "")).strip()]
    if missing:
        raise ValueError("task.json 缺少必填字段：" + ", ".join(missing))

    integer_defaults = {
        "rag_top_k": 8,
        "quick_search_top_k": 5,
        "max_sub_queries": 3,
        "max_research_rounds": 4,
        "max_deep_research_calls": 12,
        "max_quick_search_calls": 30,
        "max_subagent_tasks": 6,
        "per_task_quick_quota": 4,
        "per_task_deep_quota": 3,
        "chief_quick_reserve": 8,
        "max_context_chars": 35000,
        "max_chunk_chars": 7000,
        "max_parallel_rag_searches": 3,
        "recursion_limit": 200,
        "agent_max_tokens": 8192,
    }
    normalized = dict(task)
    for key, default in integer_defaults.items():
        value = normalized.get(key, default)
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"task.json 中 {key} 必须为整数") from exc
        if value <= 0:
            raise ValueError(f"task.json 中 {key} 必须大于 0")
        normalized[key] = value

    # 可选的运行级硬预算：max_tokens（Token 上限）与 max_cost（费用上限）。
    # 未配置或非正数视为关闭，不做强制约束。
    for key, converter in (("max_tokens", int), ("max_cost", float)):
        value = normalized.get(key)
        if value is None:
            normalized[key] = None
            continue
        try:
            value = converter(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"task.json 中 {key} 必须为数字") from exc
        normalized[key] = value if value > 0 else None

    normalized["company"] = str(normalized["company"]).strip()
    normalized["report_period"] = str(normalized["report_period"]).strip()
    normalized["section_id"] = str(normalized["section_id"]).strip()
    return normalized


def open_task(path: Path | None = None) -> dict:
    task_path = path or PACKAGE_DIR / "task.json"
    if not task_path.is_file():
        raise FileNotFoundError(f"未找到任务配置：{task_path}")
    with task_path.open("r", encoding="utf-8") as file:
        return validate_task(json.load(file))


def build_request(task: dict) -> str:
    return (
        "请完成以下公司研究任务：\n"
        f"- 公司：{task['company']}\n"
        f"- 报告期间：{task['report_period']}\n"
        f"- Section：{task['section_id']}\n\n"
        "请围绕 Section Definition 自主规划研究路径，最终使用 write_file 交付 report.md。"
    )


def print_progress(update: dict) -> None:
    if not isinstance(update, dict):
        return
    for node, payload in update.items():
        if not isinstance(payload, dict):
            continue
        for message in payload.get("messages", []):
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                name = tool_call.get("name", "")
                args = tool_call.get("args", {}) or {}
                if name == "task":
                    detail = (
                        f"{args.get('subagent_type', '')}: "
                        f"{args.get('description', '')}"
                    )
                elif name in {"write_file", "edit_file", "read_file"}:
                    detail = args.get("file_path", "")
                else:
                    detail = str(args.get("query", ""))[:120]
                print(f"  [{node}] -> {name}({detail})")

            text = getattr(message, "text", None)
            if text and getattr(message, "type", "") == "ai":
                print(f"  [{node}] {text}")


_INCOMPLETE_HEAD_MARK = "研究不完整"


def _is_incomplete_finding(content: str) -> bool:
    """依据 Researcher 在回复开头的明确标注判断研究是否不完整。"""
    return _INCOMPLETE_HEAD_MARK in content[:300]


_NON_FINDINGS_NAMES = {"plan.md"}


def _collect_findings(run_dir: Path) -> list[tuple[str, str, bool]]:
    """收集 findings 目录下的全部研究产出，返回 ``(文件名, 正文, 是否完整)``。

    排除误写入 findings 目录的 plan.md 等非 Findings 文件，避免研究计划被
    当作研究产出收集。
    """
    findings_dir = run_dir / "findings"
    if not findings_dir.is_dir():
        return []
    result = []
    for path in sorted(findings_dir.glob("*.md")):
        if path.name in _NON_FINDINGS_NAMES:
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            result.append((path.name, content, not _is_incomplete_finding(content)))
    return result


def _collect_plan(run_dir: Path) -> str:
    """读取 Chief 制定的研究计划 plan.md。

    容忍 Chief 将 plan.md 误写入 findings 目录的情况：优先读取 run 根目录，
    缺失时回退到 findings 目录，确保独立 Writer 能拿到研究计划。
    """
    plan = read_text(run_dir / "plan.md")
    if plan.strip():
        return plan
    return read_text(run_dir / "findings" / "plan.md")


def _is_valid_report(report: str) -> bool:
    if not isinstance(report, str):
        return False
    content = report.strip()
    if len(content) < 40:
        return False
    nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    return len(nonempty_lines) >= 2


def finalize_fallback(task: dict, run_dir: Path, error: str | None = None) -> str:
    findings = _collect_findings(run_dir)
    lines = [
        f"# {task['company']} {task['report_period']} · {task['section_id']}",
        "",
        "> **兜底报告**：独立 Writer 未能完成统一综合。以下内容为各 Researcher Findings 的原始汇总，"
        "未经统一编辑和交叉验证，不应视为正常完成的最终研究报告。",
    ]
    if error:
        lines.extend(["", f"> 运行异常：{error.splitlines()[0][:500]}"])
    if not findings:
        lines.extend(["", "本次运行未找到有效 Findings，未形成可用研究内容。"])
    else:
        lines.extend(["", f"本次收集到 {len(findings)} 份 Findings："])
        for filename, body, _complete in findings:
            lines.extend(["", "---", "", f"## Findings：{filename}", "", body])
    return "\n".join(lines).rstrip() + "\n"


def _decide_delivery(
    task: dict,
    run_dir: Path,
    section_brief: str,
    existing_report: str,
    agent_error: str | None,
    tracker: UsageTracker,
    writer_fn: Any = write_report,
) -> tuple[str, str, str | None, bool, list[tuple[str, str, bool]]]:
    """决定最终报告来源与运行状态，返回 ``(report, status, writer_error, used_writer, findings)``。

    决策顺序：Chief 已写有效终稿 → 独立 Writer 成稿 → Findings 兜底 → 失败。
    ``writer_fn`` 可注入以便测试；默认使用真实 Writer。
    """
    findings = _collect_findings(run_dir)
    plan_md = _collect_plan(run_dir)

    if _is_valid_report(existing_report):
        return existing_report, STATUS_COMPLETED, None, False, findings

    if not findings:
        return (
            finalize_fallback(task, run_dir, error=agent_error),
            STATUS_FAILED,
            None,
            False,
            findings,
        )

    try:
        with usage_scope(tracker):
            report = writer_fn(task, section_brief, plan_md, findings)
    except Exception as exc:
        writer_error = f"{type(exc).__name__}: {exc}"
        return (
            finalize_fallback(task, run_dir, error=agent_error or writer_error),
            STATUS_FALLBACK,
            writer_error,
            False,
            findings,
        )

    return report, STATUS_COMPLETED_BY_WRITER, None, True, findings


def _apply_degraded(
    status: str,
    has_warnings: bool,
    has_incomplete: bool,
    agent_error: str | None,
    has_delivery_gaps: bool = False,
) -> str:
    """已成稿但存在证据/结构缺口时，将状态降级为 degraded。"""
    if status in (STATUS_COMPLETED, STATUS_COMPLETED_BY_WRITER):
        if has_warnings or has_incomplete or agent_error or has_delivery_gaps:
            return STATUS_DEGRADED
    return status


def _save_run_snapshot(run_dir: Path, task: dict, section: dict) -> None:
    _atomic_write_json(run_dir / "task_snapshot.json", task)
    _atomic_write_json(run_dir / "section_snapshot.json", section)
    _atomic_write_json(
        run_dir / "run_metadata.json",
        {
            "started_at": datetime.now().astimezone().isoformat(),
            "company": task["company"],
            "report_period": task["report_period"],
            "section_id": task["section_id"],
            "model": task.get("strategic_model")
            or os.environ.get("STRATEGIC_LLM")
            or task.get("model"),
            "status": "running",
        },
    )


def _write_usage(run_dir: Path, tracker: UsageTracker) -> None:
    report = format_report(tracker)
    print("\n" + report)
    _atomic_write_text(run_dir / "usage.txt", report + "\n")
    from my_deepagent.cost import get_usage_summary

    _atomic_write_json(run_dir / "usage.json", get_usage_summary(tracker))


async def run_research_task(task: dict | None = None) -> str:
    task = validate_task(task or open_task())
    section = load_section(task["section_id"])
    section_brief = render_section_brief(
        section, task["company"], task["report_period"]
    )

    run_dir = PACKAGE_DIR / "outputs" / f"run_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_status(run_dir, "running")
    _save_run_snapshot(run_dir, task, section)

    print(f"Output directory: {run_dir}")
    print(f"Section: {section['name']}（{task['section_id']}）\n")

    tracker = UsageTracker()
    agent_error: str | None = None
    agent = None
    budget = None

    try:
        cfg = get_config()
        if not cfg.deepseek_api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY，请先配置 my_deepagent/.env")
        if not cfg.rag_base_url:
            raise RuntimeError("缺少 RAG_BASE_URL，请先配置 my_deepagent/.env")
        agent, budget = build_agent(task, str(run_dir), section_brief=section_brief)
        with usage_scope(tracker):
            async for update in agent.astream(
                {"messages": [{"role": "user", "content": build_request(task)}]},
                config={
                    "recursion_limit": task["recursion_limit"],
                    "callbacks": [
                        UsageCallback(
                            tracker=tracker,
                            pricing_tier=cfg.deepseek_pricing_tier,
                        )
                    ],
                },
                stream_mode="updates",
            ):
                if task.get("verbose", True):
                    print_progress(update)
    except Exception as exc:
        agent_error = f"{type(exc).__name__}: {exc}"
        _atomic_write_text(
            run_dir / "run_error.txt",
            traceback.format_exc(),
        )
        print(f"\n[Agent 异常] {agent_error}")

    report_path = run_dir / "report.md"
    existing_report = ""
    if report_path.is_file():
        try:
            existing_report = report_path.read_text(encoding="utf-8")
        except OSError:
            existing_report = ""

    report, status, writer_error, used_writer, findings = _decide_delivery(
        task, run_dir, section_brief, existing_report, agent_error, tracker
    )
    if used_writer:
        print("\n[Writer] 独立 Writer 已根据 Findings 生成终稿。")
    elif status == STATUS_FALLBACK:
        print(f"\n[Writer 失败] {writer_error}，已降级为 Findings 兜底。")
    elif status == STATUS_FAILED:
        print("\n[失败] 无有效 Findings，无法形成有效报告。")

    registry_path = run_dir / "_internal" / "source_registry.json"
    report, citation_validation = finalize_report_citations(report, registry_path)
    _atomic_write_text(report_path, report)
    _atomic_write_json(
        run_dir / "citation_validation.json",
        citation_validation,
    )

    delivery_validation = check_delivery(report, section)
    _atomic_write_json(run_dir / "delivery_validation.json", delivery_validation)

    has_incomplete = any(not complete for _, _, complete in findings)
    status = _apply_degraded(
        status,
        citation_validation["has_warnings"],
        has_incomplete,
        agent_error,
        has_delivery_gaps=not delivery_validation["complete"],
    )

    _write_status(run_dir, status)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "finished_at": datetime.now().astimezone().isoformat(),
        "status": status,
        "agent_error": agent_error,
        "writer_error": writer_error,
        "used_writer": used_writer,
        "findings_count": len(findings),
        "has_incomplete_findings": has_incomplete,
        "citation_validation": citation_validation,
        "delivery_validation": delivery_validation,
    })
    _atomic_write_json(metadata_path, metadata)

    try:
        _write_usage(run_dir, tracker)
    except Exception:
        _atomic_write_text(
            run_dir / "usage_error.txt",
            traceback.format_exc(),
        )

    if budget is not None:
        try:
            _atomic_write_json(run_dir / "budget.json", budget.summary())
        except Exception:
            _atomic_write_text(
                run_dir / "budget_error.txt",
                traceback.format_exc(),
            )

    print(f"\n[结束] 状态：{status}")
    print(f"报告：{report_path}")
    return report


async def main() -> str:
    return await run_research_task(open_task())


if __name__ == "__main__":
    asyncio.run(main())