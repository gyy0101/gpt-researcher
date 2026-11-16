

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# 保留原项目的导入方式
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_deepagent.config import get_config
from my_deepagent.citations import finalize_report_citations
from my_deepagent.cost import get_usage_summary, reset_usage
from my_deepagent.llm import chat, chat_json
from my_deepagent.section_def import load_section, render_section_brief


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

SCRIPT_VERSION = "2.0"
TEMPERATURE = 0.3
WRITER_MAX_TOKENS = 8000
EDITORIAL_MAX_TOKENS = 2500
EDITORIAL_MAX_ATTEMPTS = 2

OUTPUT_DIR_NAME = "writer_experiment"

# V3 baseline：保留原 V3 的主要写作约束，用于同输入对照。
# 结构不在这里硬编码，统一由 Section Definition 提供。
WRITER_V3_RULES = """
## 写作要求

### 信息取舍
- 不追求信息覆盖率，不因为某个数据已经被检索到就认为必须写进去。
- 一项信息只有在能解释核心判断、支撑重要结论或体现公司差异时，才进入正文。
- 普通背景信息简略处理，无关信息直接舍弃。
- 不同信息重要程度不同，篇幅不应平均分配。

### 数据使用
- 数据用于支撑判断，不是填充文章。
- 不罗列所有已找到的行业数据；相同结论不用多个相似指标反复证明。
- 优先选择最有代表性、最有解释力的数据。
- 保留关键年份、单位和统计口径。

### 语言表达
- 直接表达事实和判断，不刻意创造专业概念。
- 不强制每段设置观点标题，不强制每段做因果推演。
- 不重复解释已经说清楚的内容；不追求段落对称、句式整齐。
- 不写券商研报式的宏大叙事，不用空洞的总结性语言。
- 一句话能表达的内容不写成一段。

### 分析重点
- 不把历史数据罗列当成行业研究。
- 优先识别行业过去发生了什么变化、当前最重要的供需矛盾、未来哪些因素可能改变格局、公司在行业变化中处于什么位置。
- 不强制所有部分都包含完整过去/现在/未来分析，只有资料确实支持时才展开。

### 篇幅
- 不设硬性字数，以研究员真正需要记录的信息量决定篇幅。
- 不需要填满每个章节，不为达到某个字数而增加内容。

### 输出前检查
在内部检查是否存在不必要的观点标题、数据堆砌、重复表达、无关背景、遗漏的重要信息或人为创造的概念。发现问题时优先删减和修正。
不要输出检查过程，只输出最终报告。

### 数据与引用
- 忠实于原始 Findings，不得编造事实、数据和来源。
- 保留关键年份、单位和统计口径；口径不一致或数据冲突时显式说明。
- 原样保留 Findings 中的 [SRC:...] 稳定来源标识，不得转换为数字编号或自行创建标识。
- 不得将研究资料中的推测改写为确定事实。
- 不要自行生成来源清单；程序会在报告落盘前统一转换引用并追加来源列表。
"""

# V4：在 V3 的基础上，唯一重点新增“论据使用克制”规则。
WRITER_V4_RULES = WRITER_V3_RULES + """
### 论据使用与表达（本轮实验重点）
- 论据是用来支撑判断的，不是用来展示研究成果的。不要把所有支持某个判断的资料都写进正文。
- 一个判断优先选择最有代表性、最有解释力的一项事实或数据。
- 其他论据只有在提供了不同维度的信息、揭示重要例外，或会改变读者对判断的理解时，才额外保留。
- 如果多个数据只是重复证明同一个结论，选择其中最有解释力的一项，其余删除。
- 不要在一句话后连续堆叠多个数字、案例、机制和来源。
- 不要为了证明一个已经清楚的判断，再追加一整段同义论据。
- 一个段落围绕一个主要表达任务展开；确实需要多项论据时，分清它们分别说明什么，不要挤进一个长句。
- 不要机械地采用“观点 + 多个数据 + 原因机制 + 未来展望”的固定段落套路。
- 重要问题可以展开，但展开应增加新的解释，而不是增加同一结论的重复证据。
- 不要求每句话只出现一个数字，也不机械限制每个判断的论据数量；判断标准是新增论据是否带来新的信息价值。
- 引用只对应实际使用的事实和判断，不为了显得资料丰富而堆叠引用。
"""

EDITORIAL_GUIDE = """你是公司研究员写作前的编辑助手。你不负责撰写报告，只负责帮助 Writer 判断哪些研究资料值得写、哪些可以删减。

请基于 Section Definition、研究任务和 Findings，做简洁的编辑决策。

重点回答：
1. 核心问题：当前 Section 最值得读者关注的核心问题是什么？不要把行业规模、销量、市占率的罗列当成核心问题。
2. 保留内容：哪些事实或判断直接支撑核心问题，或能体现公司与行业的关键差异？
3. 重点展开：哪些问题需要解释原因、机制或未来变化？哪些内容一句话即可？
4. 舍弃内容：哪些背景、重复数据、次要案例或低价值信息可以不写？
5. 重要缺口：哪些缺失信息会实质影响核心判断？不要把一般性资料缺失都列为缺口。

要求：
- 不需要重新总结全部 Findings，不要写成一份缩小版研究报告。
- 不设固定观点数量，不为了填满字段而创造内容。
- 尽量用 Findings 中已有的文件名、引用编号或原文短语定位，不得编造定位。
- 只做信息取舍与篇幅分配，不提前撰写正文。
- 允许某个字段为空；如果没有重要缺口，important_gaps 返回空数组。
- 输出必须是单个 JSON 对象，不要附加 Markdown 代码围栏或解释。

JSON 结构：
{
  "core_issues": ["核心问题，简洁表达"],
  "keep": ["值得保留的信息及其定位"],
  "expand": ["值得展开的问题及原因"],
  "brief": ["一句话即可说明的信息"],
  "discard": ["建议舍弃的信息及原因"],
  "important_gaps": ["确实影响核心判断的信息缺口"]
}
"""


# ---------------------------------------------------------------------------
# 文件与数据工具
# ---------------------------------------------------------------------------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    """原子写入，避免进程中断留下半个报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    temp_path.replace(path)


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_task(script_dir: Path) -> dict:
    task_path = script_dir / "task.json"
    if not task_path.is_file():
        raise FileNotFoundError(
            f"未找到 task.json：{task_path}。请将实验脚本放在原实验目录，"
            "或按项目实际目录调整 task.json 的定位方式。"
        )
    with task_path.open("r", encoding="utf-8") as file:
        task = json.load(file)
    if not isinstance(task, dict):
        raise ValueError("task.json 顶层必须是 JSON 对象")
    return task


def build_task_context(task: dict, section: dict) -> str:
    company = str(task.get("company") or "").strip()
    period = str(task.get("report_period") or "").strip()
    section_name = str(section.get("name") or task.get("section_id") or "").strip()

    if not company:
        raise ValueError("task.json 缺少 company")
    if not section_name:
        raise ValueError("无法确定当前 Section 名称")

    brief = render_section_brief(section, company, period)
    return (
        f"研究对象：{company}\n"
        f"报告期间：{period or '未指定'}\n"
        f"当前 Section：{section_name}\n\n"
        "以下 Section Definition 是本次交付范围和结构的依据。"
        "请按其中实际定义的内容组织报告，不要套用其他行业或其他 Section 的固定模板。\n\n"
        f"{brief}"
    )


def render_findings(findings_dir: Path) -> tuple[list[Path], str]:
    files = sorted(
        path for path in findings_dir.glob("*.md")
        if path.is_file() and not path.name.startswith(".")
    )
    if not files:
        raise ValueError(f"Findings 目录中没有可读取的 Markdown 文件：{findings_dir}")

    parts: list[str] = []
    for path in files:
        content = read_text(path).strip()
        if content:
            parts.append(f"===== findings/{path.name} =====\n{content}")

    if not parts:
        raise ValueError("Findings 文件存在，但内容全部为空")

    return files, "\n\n".join(parts)


def build_input_manifest(
    run_dir: Path,
    plan_path: Path,
    finding_files: list[Path],
    task: dict,
    model_name: str,
    run_id: str,
) -> dict:
    inputs = []
    task_path = Path(__file__).resolve().parent / "task.json"
    source_paths = [plan_path, *finding_files]
    if task_path.is_file():
        source_paths.append(task_path)
    for source_path in source_paths:
        try:
            display_path = str(source_path.relative_to(run_dir))
            is_external = False
        except ValueError:
            # task.json 通常位于实验脚本目录，不一定位于 outputs/run_xxx 内。
            display_path = str(source_path)
            is_external = True
        inputs.append({
            "path": display_path,
            "external_to_run_dir": is_external,
            "sha256": sha256_file(source_path),
            "size_bytes": source_path.stat().st_size,
            "chars": len(read_text(source_path)),
        })

    return {
        "script_version": SCRIPT_VERSION,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "company": task.get("company"),
        "report_period": task.get("report_period"),
        "section_id": task.get("section_id"),
        "model_configured": model_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": inputs,
        "total_input_chars": sum(item["chars"] for item in inputs),
        # 这里只能确认脚本没有主动截断；不对模型服务端的上下文处理作无依据保证。
        "script_truncated_input": False,
    }


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------

def build_editorial_prompt(task_context: str, plan_text: str, findings_text: str) -> str:
    return (
        f"{task_context}\n\n"
        "研究计划（仅用于理解研究范围，不是事实证据）：\n"
        f"{plan_text}\n\n"
        "原始 Findings（事实与判断的唯一依据）：\n"
        f"{findings_text}\n\n"
        f"{EDITORIAL_GUIDE}"
    )


def build_writer_prompt(
    task_context: str,
    rules: str,
    plan_text: str,
    findings_text: str,
    editorial: dict | None = None,
) -> str:
    sections = [
        task_context,
        rules,
        "研究计划（仅用于理解研究范围，不是事实证据）：\n" + plan_text,
    ]

    if editorial is not None:
        sections.append(
            "写作前编辑决策（用于控制信息取舍，不是事实来源）：\n"
            "- core_issues：本节优先回答的问题。\n"
            "- keep：优先保留的信息。\n"
            "- expand：值得解释清楚的问题。\n"
            "- brief：简要交代即可的信息。\n"
            "- discard：默认不写；只有在发现其对核心判断不可或缺、或原决策遗漏了关键事实时才重新纳入。\n"
            "- important_gaps：证据缺口，不得用推测补齐。\n"
            "编辑决策不能覆盖原始 Findings；如发现决策遗漏了重要事实，可以回查补充，但不要因此重新罗列全部资料。\n\n"
            + json.dumps(editorial, ensure_ascii=False, indent=2)
        )

    sections.append(
        "原始 Findings（事实与判断的唯一依据）：\n" + findings_text
    )
    sections.append(
        "请根据 Section Definition 完成最终报告。直接输出 Markdown 正文，"
        "不要输出你的分析过程、编辑决策或自我评价。"
    )
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Editorial Decision
# ---------------------------------------------------------------------------

EDITORIAL_KEYS = (
    "core_issues",
    "keep",
    "expand",
    "brief",
    "discard",
    "important_gaps",
)


def normalize_editorial(result: Any) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Editorial Decision 返回值不是 JSON 对象")

    normalized: dict[str, list] = {}
    for key in EDITORIAL_KEYS:
        value = result.get(key, [])
        if value is None:
            value = []
        if isinstance(value, str):
            value = [value] if value.strip() else []
        if not isinstance(value, list):
            raise ValueError(f"Editorial Decision 字段 {key} 必须是数组")

        # 保留字符串和对象，丢弃空字符串；不静默接受其他类型。
        cleaned = []
        for item in value:
            if isinstance(item, str) and item.strip():
                cleaned.append(item.strip())
            elif isinstance(item, dict):
                cleaned.append(item)
            else:
                raise ValueError(f"Editorial Decision 字段 {key} 包含不支持的元素")
        normalized[key] = cleaned

    if not any(normalized.values()):
        raise ValueError("Editorial Decision 返回了空决策，不能继续假装实验成功")

    return normalized


def run_editorial(task_context: str, plan_text: str, findings_text: str) -> tuple[dict, dict]:
    prompt = build_editorial_prompt(task_context, plan_text, findings_text)
    messages = [{"role": "user", "content": prompt}]
    last_error: Exception | None = None

    reset_usage()
    started = time.time()

    for attempt in range(1, EDITORIAL_MAX_ATTEMPTS + 1):
        try:
            raw = chat_json(
                messages,
                temperature=TEMPERATURE,
                max_tokens=EDITORIAL_MAX_TOKENS,
            )
            editorial = normalize_editorial(raw)
            metrics = {
                "elapsed_seconds": round(time.time() - started, 2),
                "usage": get_usage_summary(),
                "attempts": attempt,
            }
            return editorial, metrics
        except Exception as exc:
            last_error = exc
            if attempt < EDITORIAL_MAX_ATTEMPTS:
                messages.append({
                    "role": "user",
                    "content": (
                        "上次输出未通过 JSON 结构校验。请严格按要求返回单个 JSON 对象；"
                        "所有字段均为数组，允许空数组，不要添加解释或 Markdown 围栏。"
                    ),
                })

    raise RuntimeError(
        f"Editorial Decision 连续 {EDITORIAL_MAX_ATTEMPTS} 次失败，已停止本次实验。"
        f"最后错误：{last_error}"
    )


# ---------------------------------------------------------------------------
# Writer 与指标
# ---------------------------------------------------------------------------

def run_writer(
    task_context: str,
    rules: str,
    plan_text: str,
    findings_text: str,
    editorial: dict | None,
    output_path: Path,
    source_registry_path: Path | None = None,
) -> dict:
    prompt = build_writer_prompt(
        task_context=task_context,
        rules=rules,
        plan_text=plan_text,
        findings_text=findings_text,
        editorial=editorial,
    )

    reset_usage()
    started = time.time()
    report = chat(
        [{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=WRITER_MAX_TOKENS,
    )
    elapsed = time.time() - started

    if not isinstance(report, str) or not report.strip():
        raise RuntimeError(f"Writer 没有生成有效正文：{output_path.name}")

    citation_validation = None
    if source_registry_path and source_registry_path.is_file():
        report, citation_validation = finalize_report_citations(
            report, source_registry_path
        )

    write_text(output_path, report)
    return {
        "elapsed_seconds": round(elapsed, 2),
        "usage": get_usage_summary(),
        "output": output_path.name,
        "output_chars": len(report.strip()),
        "output_sha256": hashlib.sha256(report.encode("utf-8")).hexdigest(),
        "citation_validation": citation_validation,
    }


def analyze_report(text: str) -> dict:
    return {
        "chars": len(text.strip()),
        "citations": len(re.findall(r"\[\d+\]", text)),
        "h1": len(re.findall(r"(?m)^#\s", text)),
        "h2": len(re.findall(r"(?m)^##\s", text)),
        "h3": len(re.findall(r"(?m)^###\s", text)),
        "table_lines": len(re.findall(r"(?m)^\s*\|.*\|\s*$", text)),
    }


def write_comparison(
    output_dir: Path,
    baseline_path: Path,
    v4_path: Path,
    editorial: dict,
) -> None:
    baseline_text = read_text(baseline_path)
    v4_text = read_text(v4_path)
    baseline = analyze_report(baseline_text)
    v4 = analyze_report(v4_text)

    lines = [
        "# Writer V3 / V4 对照（供人工评审）",
        "",
        "> 两个 Writer 使用同一份 Findings、Section Definition、研究计划和 Editorial Decision。"
        "本轮重点观察论据使用约束对表达的影响；自动统计不代表质量评分。",
        "",
        f"- Editorial 核心问题：{len(editorial['core_issues'])} 条",
        f"- 保留：{len(editorial['keep'])} 条",
        f"- 重点展开：{len(editorial['expand'])} 条",
        f"- 简要说明：{len(editorial['brief'])} 条",
        f"- 舍弃：{len(editorial['discard'])} 条",
        f"- 重要缺口：{len(editorial['important_gaps'])} 条",
        "",
        "| 指标 | V3 baseline | V4 论据克制 |",
        "|---|---:|---:|",
        f"| 正文字符数 | {baseline['chars']} | {v4['chars']} |",
        f"| 引用标记数 | {baseline['citations']} | {v4['citations']} |",
        f"| 一级标题数 | {baseline['h1']} | {v4['h1']} |",
        f"| 二级标题数 | {baseline['h2']} | {v4['h2']} |",
        f"| 三级标题数 | {baseline['h3']} | {v4['h3']} |",
        f"| 表格行数 | {baseline['table_lines']} | {v4['table_lines']} |",
        "",
        f"- V3 报告：[{baseline_path.name}](./{baseline_path.name})",
        f"- V4 报告：[{v4_path.name}](./{v4_path.name})",
        "- 编辑决策：`editorial_decision_v4.json`",
        "",
        "请人工重点检查：同一判断是否减少了重复论据；是否仍保留关键事实；"
        "是否把真正不同维度的证据错误删掉；语言是否更自然。",
    ]
    write_text(output_dir / "comparison_v4.md", "\n".join(lines))


def usage_summary_line(label: str, metrics: dict) -> None:
    usage = metrics.get("usage", {}).get("total", {})
    cost = usage.get("cost_current")
    cost_text = f"¥{cost:.4f}" if cost is not None else "未配置费率"
    print(
        f"[{label}] {metrics.get('elapsed_seconds', 0)}s | "
        f"calls={usage.get('calls', 0)} | "
        f"input={usage.get('prompt_tokens', 0):,} | "
        f"output={usage.get('completion_tokens', 0):,} | "
        f"cost={cost_text}"
    )



# ---------------------------------------------------------------------------
# 实验运行
# ---------------------------------------------------------------------------

def run(run_dir_arg: str, mode: str) -> None:
    run_dir = Path(run_dir_arg).expanduser().resolve()
    plan_path = run_dir / "plan.md"
    findings_dir = run_dir / "findings"

    if not run_dir.is_dir():
        raise SystemExit(f"运行目录不存在：{run_dir}")
    if not plan_path.is_file():
        raise SystemExit(f"未找到 plan.md：{plan_path}")
    if not findings_dir.is_dir():
        raise SystemExit(f"未找到 findings 目录：{findings_dir}")

    cfg = get_config()
    if not getattr(cfg, "deepseek_api_key", None):
        raise SystemExit("缺少 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")

    script_dir = Path(__file__).resolve().parent
    task = load_task(script_dir)
    section_id = task.get("section_id") or "industry_competition"
    section = load_section(section_id)

    task_context = build_task_context(task, section)
    plan_text = read_text(plan_path).strip()
    if not plan_text:
        raise SystemExit(f"plan.md 为空：{plan_path}")

    finding_files, findings_text = render_findings(findings_dir)
    output_dir = run_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:12]
    manifest = build_input_manifest(
        run_dir, plan_path, finding_files, task, cfg.deepseek_model, run_id
    )
    write_json(output_dir / "input_manifest.json", manifest)

    print(f"输出目录：{output_dir}")
    print(f"本次 run_id：{run_id}")
    print(
        f"输入：plan.md + {len(finding_files)} 份 Findings，"
        f"共 {manifest['total_input_chars']:,} 字符；脚本未主动截断输入。"
    )
    print(f"模式：{mode}")

    metrics: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "run_id": run_id,
        "model_configured": cfg.deepseek_model,
        "temperature": TEMPERATURE,
        "writer_max_tokens": WRITER_MAX_TOKENS,
        "editorial_max_tokens": EDITORIAL_MAX_TOKENS,
        "versions": {},
        "generated_artifacts": [],
    }

    def run_direct_baseline() -> None:
        print("\n=== Direct Writer（无 Editorial 对照）===")
        result = run_writer(
            task_context=task_context,
            rules=WRITER_V3_RULES,
            plan_text=plan_text,
            findings_text=findings_text,
            editorial=None,
            output_path=output_dir / "direct_report_no_editorial.md",
            source_registry_path=run_dir / "_internal" / "source_registry.json",
        )
        metrics["versions"]["direct_no_editorial"] = result
        metrics["generated_artifacts"].append(result["output"])
        usage_summary_line("Direct（无 Editorial）", result)

    if mode == "direct":
        run_direct_baseline()

    elif mode in ("v4", "both", "compare"):
        # Editorial 只运行一次，compare 时 V3/V4 共用同一份决策。
        print("\n=== Editorial Decision ===")
        editorial, editorial_metrics = run_editorial(
            task_context, plan_text, findings_text
        )
        write_json(output_dir / "editorial_decision_v4.json", editorial)
        metrics["editorial"] = editorial_metrics
        metrics["generated_artifacts"].append("editorial_decision_v4.json")
        usage_summary_line("Editorial", editorial_metrics)

        if mode == "compare":
            print("\n=== V3 baseline Writer（与 V4 共用 Editorial）===")
            v3_result = run_writer(
                task_context,
                WRITER_V3_RULES,
                plan_text,
                findings_text,
                editorial,
                output_dir / "direct_report_v3_baseline.md",
                source_registry_path=run_dir / "_internal" / "source_registry.json",
            )
            metrics["versions"]["v3_baseline"] = v3_result
            metrics["generated_artifacts"].append(v3_result["output"])
            usage_summary_line("V3 baseline", v3_result)

        print("\n=== V4 Writer（增加论据克制规则）===")
        v4_result = run_writer(
            task_context,
            WRITER_V4_RULES,
            plan_text,
            findings_text,
            editorial,
            output_dir / "direct_report_v4.md",
            source_registry_path=run_dir / "_internal" / "source_registry.json",
        )
        metrics["versions"]["v4"] = v4_result
        metrics["generated_artifacts"].append(v4_result["output"])
        usage_summary_line("V4", v4_result)

        if mode == "compare":
            write_comparison(
                output_dir,
                output_dir / "direct_report_v3_baseline.md",
                output_dir / "direct_report_v4.md",
                editorial,
            )
            metrics["generated_artifacts"].append("comparison_v4.md")

        if mode == "both":
            run_direct_baseline()

    else:
        raise ValueError(f"不支持的运行模式：{mode}")

    metrics["generated_artifacts"].append("run_metrics.json")
    write_json(output_dir / "run_metrics.json", metrics)
    print("\n[完成] 本次实际生成的文件：")
    for name in metrics["generated_artifacts"]:
        print(f"  - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Writer 独立实验：V3 baseline / V4 论据克制对照"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="已有研究运行目录，例如 outputs/run_xxx",
    )
    parser.add_argument(
        "--mode",
        choices=("direct", "v4", "both", "compare"),
        default="compare",
        help=(
            "direct：只跑 Direct baseline；v4：Editorial + V4；"
            "both：Direct（无 Editorial）+ Editorial/V4；"
            "compare：同一 Editorial 下对比 V3 baseline 与 V4（推荐）"
        ),
    )
    args = parser.parse_args()
    run(args.run_dir, args.mode)


if __name__ == "__main__":
    main()
