"""入口：读取 task.json + Section 定义，用 Deep Agents harness 跑研究任务，输出 report.md。

任务定义（task.json）说明公司、报告期间与要研究的 Section；
Section 定义（sections/<id>.yaml）说明该 Section 的交付目标与质量要求。
最终报告由 Chief Researcher 写入 report.md；若未完成则走确定性兜底并明确标记。
"""
import asyncio
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

from my_deepagent.agent import build_agent
from my_deepagent.cost import UsageCallback, format_report, reset_usage
from my_deepagent.section_def import load_section, render_section_brief

RUN_STATUS_PATH = "_run_status.txt"


def open_task():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    task_json_path = os.path.join(current_dir, "task.json")

    with open(task_json_path, "r", encoding="utf-8") as f:
        task = json.load(f)

    if not task:
        raise Exception(
            "No task found. Please ensure a valid task.json file is present in "
            "the my_deepagent directory and contains the necessary task information."
        )
    return task


def build_request(task: dict) -> str:
    company = task.get("company", "")
    period = task.get("report_period", "")
    section_id = task.get("section_id", "")
    return (
        f"请完成以下研究任务：\n"
        f"- 公司：{company}\n"
        f"- 报告期间：{period}\n"
        f"- 研究 Section：{section_id}\n\n"
        f"交付要求见系统提示中「本次任务的交付要求」部分。"
        f"请按首席研究员的职责自主规划研究路径并产出 report.md。"
    )


def print_progress(update: dict):
    for node, payload in update.items():
        if not isinstance(payload, dict):
            continue
        for message in payload.get("messages", []):
            for tool_call in getattr(message, "tool_calls", None) or []:
                name = tool_call.get("name", "")
                args = tool_call.get("args", {})
                if name == "task":
                    detail = f"{args.get('subagent_type', '')}: {args.get('description', '')}"
                elif name in ("write_file", "edit_file", "read_file"):
                    detail = args.get("file_path", "")
                else:
                    detail = str(args.get("query", ""))[:120]
                print(f"  [{node}] -> {name}({detail})")
            text = getattr(message, "text", None)
            if text is not None and not isinstance(text, str) and callable(text):
                text = text()
            if text and getattr(message, "type", "") == "ai":
                print(f"  [{node}] {text}")


def _collect_findings(run_dir: str):
    """收集 researcher 子 agent 产出的 findings，按文件名排序返回 (文件名, 正文)。"""
    findings_dir = os.path.join(run_dir, "findings")
    if not os.path.isdir(findings_dir):
        return []
    out = []
    for f in sorted(os.listdir(findings_dir)):
        if not f.endswith(".md"):
            continue
        with open(os.path.join(findings_dir, f), "r", encoding="utf-8") as fh:
            out.append((f, fh.read()))
    return out


def finalize_fallback(task: dict, run_dir: str):
    """确定性兜底：Chief Researcher 未产出 report.md 时，用 findings 拼装一份并明确标记。"""
    findings = _collect_findings(run_dir)
    company = task.get("company", "")
    period = task.get("report_period", "")
    section_id = task.get("section_id", "")

    lines = [f"# {company} {period} · {section_id}\n"]
    lines.append(
        "\n> ⚠️ 本报告由**确定性兜底流程**生成：首席研究员未完成最终成稿，"
        "以下内容为各研究任务 findings 的原始汇总，未经统一综合。"
        "引用缺失、口径不一致、未完成的研究任务等问题可能存在于下文，请勿视为正常完成的研究报告。\n"
    )

    if not findings:
        lines.append("\n未找到任何 research findings。本次运行未产出有效研究内容。\n")
    else:
        lines.append(f"\n共找到 {len(findings)} 份 findings，按文件名罗列如下：\n")
        for fname, body in findings:
            lines.append(f"\n---\n\n## findings/{fname}\n\n{body}\n")

    report = "".join(lines)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    _write_status(run_dir, "fallback")
    return report


def _write_status(run_dir: str, status: str):
    with open(os.path.join(run_dir, RUN_STATUS_PATH), "w", encoding="utf-8") as fh:
        fh.write(f"{status}\n")


def _report_usage(run_dir: str):
    """打印本次运行的 token 用量与费用，并写入 run_dir/usage.txt。"""
    report = format_report()
    print(f"\n{report}")
    with open(os.path.join(run_dir, "usage.txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")


async def run_research_task(task=None):
    task = task or open_task()

    company = task.get("company", "")
    period = task.get("report_period", "")
    section_id = task.get("section_id", "")
    if not company or not period or not section_id:
        raise ValueError(
            "task.json 需包含 company、report_period、section_id 字段，"
            "分别指明公司、报告期间与研究 Section。"
        )

    # 读取 Section 定义，渲染交付要求给 Chief Researcher
    section = load_section(section_id)
    section_brief = render_section_brief(section, company, period)

    run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs", f"run_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}")
    print(f"Section: {section.get('name')}（{section_id}）\n")

    agent = build_agent(task, run_dir, section_brief=section_brief)
    reset_usage()

    try:
        async for update in agent.astream(
            {"messages": [{"role": "user", "content": build_request(task)}]},
            config={"recursion_limit": 200, "callbacks": [UsageCallback()]},
            stream_mode="updates",
        ):
            if task.get("verbose", True):
                print_progress(update)

        report_path = os.path.join(run_dir, "report.md")
        if os.path.exists(report_path):
            _write_status(run_dir, "completed")
            print(f"\n[完成] Chief Researcher 已生成报告：{report_path}")
            with open(report_path, "r", encoding="utf-8") as f:
                return f.read()

        # 兜底：Chief Researcher 未写出 report.md
        print("\n[兜底] Chief Researcher 未生成 report.md，改用 findings 确定性拼装。")
        report = finalize_fallback(task, run_dir)
        print(f"\n[兜底] 报告已写入（标记为 fallback）：{report_path}")
        return report
    finally:
        _report_usage(run_dir)


async def main():
    task = open_task()
    return await run_research_task(task)


if __name__ == "__main__":
    asyncio.run(main())