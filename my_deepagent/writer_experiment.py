"""Writer 独立实验（Experiment A）。

不运行 Chief Researcher / Researcher Subagent，不调用 quick_search / deep_research / RAG，
只用已有运行目录（plan.md + findings/*.md）作为事实来源，验证 DeepSeek 的「观点提炼 + 写作」能力。

两种版本并行对比：
- Version A（Direct）：一次性写作调用，直接生成最终报告 -> direct_report.md
- Version B（Synthesis + Writer）：先提炼核心研究判断（research_synthesis.json），再据此 + 原始
  Findings 写报告 -> synthesis_report.md

用法：
    python writer_experiment.py --run-dir outputs/run_52a6c22b --mode both
    python writer_experiment.py --run-dir outputs/run_52a6c22b --mode direct
    python writer_experiment.py --run-dir outputs/run_52a6c22b --mode synthesis
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from my_deepagent.config import get_config
from my_deepagent.cost import get_usage_summary, reset_usage
from my_deepagent.llm import chat, chat_json
from my_deepagent.section_def import load_section, render_section_brief

# ---- 模型生成参数（两组保持一致）----
TEMPERATURE = 0.3
WRITER_MAX_TOKENS = 8000
SYNTHESIS_MAX_TOKENS = 8000
MAX_SYNTHESIS_RETRIES = 2

# ---- 交付结构（任务文档「四、业务交付要求」）----
TARGET_STRUCTURE = """最终报告结构要求（不必机械写成列表，但必须覆盖以下内容）：

## 动力电池行业
- 供给：……
- 需求：……
- 总结：……

## 储能电池行业
- 供给：……
- 需求：……
- 总结：……

## 公司优势
## 公司劣势
## 行业对比

行业对比需要有关键数据，必要时使用一张精简的 Markdown 表格。"""

# ---- 目标风格（任务文档「五、人类研究员风格」）----
STYLE_GUIDE = """目标风格（接近专业人类研究员的分析笔记，而非信息堆砌型报告）：

1. 高信息密度：同一段内交代行业过去的变化、当前供需状态、导致变化的核心原因、未来可能发生什么、哪些条件会改变判断；不要逐条罗列所有资料。
2. 判断优先：先给判断（例如「行业名义产能过剩与头部有效产能偏紧并存」），再用必要数据解释判断；不要把数据罗列完才在末尾给一句宽泛总结。
3. 强调因果关系：解释规模如何影响单位制造成本、良率如何影响成本差异、技术迭代如何影响客户认证与产品溢价、海外布局如何改变竞争地位，而不是简单写「规模大、技术强、客户多」。
4. 有前瞻性但不编造：允许基于历史事实和研究材料做有条件判断（如「若新增产能按计划释放，2027 年行业供需紧张程度可能下降」），但不得把券商预测或公司计划写成已发生的事实。
5. 语言自然、简练：避免「在当前时代背景下」「充分彰显」「显著赋能」「构筑坚实护城河」「未来发展空间广阔」「机遇与挑战并存」等套话；不写长篇引言和总结套话。
6. 不盲目模仿结论：根据证据判断一项现象属于竞争优势、竞争劣势、经营风险还是暂时性现象，而不是照抄模板结论。"""

# ---- 统一写作约束（任务文档「七、统一写作要求」）----
UNIFIED_RULES = """统一写作约束（两组都必须遵守）：

1. 必须覆盖动力、储能两个业务的供给、需求及总结。
2. 必须分析公司优势、劣势和行业对比。
3. 不追求覆盖 Findings 的全部内容。
4. 优先选择对供需、价格、份额、盈利和未来竞争格局有解释力的内容。
5. 区分事实、推断、预测。
6. 保留必要引用：正文关键结论后用 [n] 编号标注来源，末尾附「来源」列表，只写券商研报或年报名称（不写页码、不写内部文件路径），不能伪造来源。
7. 不得混用出货量、销量、装机量、使用量等统计口径。
8. 数据有冲突时应说明，不能自行编造调和结果。
9. 缺少重要证据时可以写明不确定性，但不要堆砌长篇缺口清单。
10. 语言自然、紧凑，减少机械小标题和套话。

正文建议约 1,500–2,500 个汉字（非强制），如为保持重要判断和证据可略长。"""


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_task() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_task_context(task: dict, section: dict) -> str:
    company = task.get("company", "")
    period = task.get("report_period", "")
    brief = render_section_brief(section, company, period)
    return (
        f"任务：为「{company}」撰写 {period} 年「{section.get('name', '')}」研究分析报告"
        f"（目标风格：接近专业人类研究员、精简且有解释力的分析笔记，而非信息堆砌型报告）。\n\n{brief}"
    )


def _build_writing_spec() -> str:
    return "\n\n".join([TARGET_STRUCTURE, STYLE_GUIDE, UNIFIED_RULES])


def _render_findings(findings_dir: str, finding_files: list) -> str:
    parts = []
    for f in finding_files:
        text = _read_file(os.path.join(findings_dir, f))
        parts.append(f"===== findings/{f} =====\n{text}")
    return "\n\n".join(parts)


def _build_direct_prompt(task_context, writing_spec, plan_text, findings_text) -> str:
    return (
        f"{task_context}\n\n{writing_spec}\n\n"
        f"下面是本次研究的计划（plan）与 Findings（事实与判断来源）：\n\n"
        f"===== plan.md =====\n{plan_text}\n\n{findings_text}\n\n"
        f"请基于上述 plan 与 Findings 撰写最终行业竞争分析报告，直接输出 Markdown 正文。"
    )


def _build_synthesis_prompt(task_context, findings_text) -> str:
    return (
        f"{task_context}\n\n"
        f"下面是本次研究的 Findings（事实与判断来源）：\n\n{findings_text}\n\n"
        f"请从 Findings 中提炼 8–12 条真正值得写进报告的核心研究判断"
        f"（该数量是初始指导，不必强制凑满；有效观点更多或更少，请由你合理判断）。\n\n"
        f"每条判断至少包含：所属业务及主题、核心判断、支持判断的关键事实、因果或经营机制、"
        f"未来变化或判断条件、来源名称、重要性理由、不确定性及反面证据。\n\n"
        f"只输出一个 JSON 对象，键为 core_judgments，值为观点对象数组。字段示例：\n"
        f'{{"core_judgments": [{{"topic": "...", "thesis": "...", "supporting_facts": ["..."], '
        f'"mechanism": "...", "outlook": "...", "importance_reason": "...", '
        f'"source_refs": ["券商研报或年报名称"], "limitations": ["..."]}}]}}\n\n'
        f"来源名称只写券商研报或年报名称（如「宁德时代 2025 年年报」「东吴证券研报」），"
        f"不写页码、不写内部文件路径；不得伪造来源编号。"
    )


def _build_writer_b_prompt(task_context, writing_spec, findings_text, synthesis: dict) -> str:
    synthesis_json = json.dumps(synthesis, ensure_ascii=False, indent=2)
    return (
        f"{task_context}\n\n{writing_spec}\n\n"
        f"下面是第一步 Research Synthesis 提炼出的核心研究判断（观点提纲）：\n\n"
        f"{synthesis_json}\n\n"
        f"下面是原始 Findings（证据细节来源，可回查，不能只依赖上面的观点标题而丢失证据细节）：\n\n"
        f"{findings_text}\n\n"
        f"请基于上述 Research Synthesis 与 Findings 撰写最终行业竞争分析报告，直接输出 Markdown 正文。"
    )


def _extract_judgments(result) -> list:
    """从 chat_json 的结果中提取观点对象列表，兼容多种键名与数组形态。"""
    if isinstance(result, dict):
        for key in ("core_judgments", "judgments", "points", "items"):
            val = result.get(key)
            if isinstance(val, list):
                return [v for v in val if isinstance(v, dict)]
        val = result.get("result")
        if isinstance(val, list):
            return [v for v in val if isinstance(v, dict)]
    return []


def _synthesize(task_context, findings_text) -> dict:
    messages = [{"role": "user", "content": _build_synthesis_prompt(task_context, findings_text)}]
    for attempt in range(1 + MAX_SYNTHESIS_RETRIES):
        result = chat_json(messages, temperature=TEMPERATURE, max_tokens=SYNTHESIS_MAX_TOKENS)
        judgments = _extract_judgments(result)
        if judgments:
            return {"core_judgments": judgments, "count": len(judgments)}
        if attempt < MAX_SYNTHESIS_RETRIES:
            messages.append(
                {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)[:2000]}
            )
            messages.append(
                {
                    "role": "user",
                    "content": "上面的输出不是合法的 JSON 对象（或缺少 core_judgments 数组）。"
                    "请只输出一个 JSON 对象，键为 core_judgments，值为观点对象数组。",
                }
            )
    raise RuntimeError(
        f"Research Synthesis 连续 {1 + MAX_SYNTHESIS_RETRIES} 次未能返回合法 JSON 观点列表，"
        "请检查模型输出。"
    )


def _version_metrics(elapsed: float) -> dict:
    return {
        "elapsed_seconds": round(elapsed, 2),
        "usage": get_usage_summary(),
    }


def _run_direct(task_context, writing_spec, plan_text, findings_text, out_dir) -> dict:
    reset_usage()
    t0 = time.time()
    prompt = _build_direct_prompt(task_context, writing_spec, plan_text, findings_text)
    report = chat(
        [{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=WRITER_MAX_TOKENS,
    )
    elapsed = time.time() - t0
    _write_text(os.path.join(out_dir, "direct_report.md"), report)
    return _version_metrics(elapsed)


def _run_synthesis_writer(task_context, writing_spec, findings_text, out_dir) -> dict:
    reset_usage()
    t0 = time.time()
    synthesis = _synthesize(task_context, findings_text)
    _write_json(os.path.join(out_dir, "research_synthesis.json"), synthesis)

    prompt = _build_writer_b_prompt(task_context, writing_spec, findings_text, synthesis)
    report = chat(
        [{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=WRITER_MAX_TOKENS,
    )
    elapsed = time.time() - t0
    _write_text(os.path.join(out_dir, "synthesis_report.md"), report)
    return _version_metrics(elapsed)


def _build_manifest(run_dir, plan_path, plan_text, findings_dir, finding_files, task, cfg) -> dict:
    inputs = [
        {
            "type": "plan",
            "path": os.path.relpath(plan_path, run_dir),
            "sha256": _sha256(plan_path),
            "size_bytes": os.path.getsize(plan_path),
            "chars": len(plan_text),
        }
    ]
    total_chars = len(plan_text)
    for f in finding_files:
        p = os.path.join(findings_dir, f)
        text = _read_file(p)
        inputs.append(
            {
                "type": "finding",
                "path": os.path.join("findings", f),
                "sha256": _sha256(p),
                "size_bytes": os.path.getsize(p),
                "chars": len(text),
            }
        )
        total_chars += len(text)
    return {
        "run_dir": run_dir,
        "company": task.get("company"),
        "report_period": task.get("report_period"),
        "section_id": task.get("section_id"),
        "model_configured": cfg.deepseek_model,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "excluded_inputs": ["report.md（确定性兜底拼接结果，不作为事实来源）"],
        "inputs": inputs,
        "total_chars": total_chars,
        "truncation": "none（Findings 总量在模型上下文内，未截断）",
    }


def _analyze(text: str) -> dict:
    return {
        "chars": len(text.strip()),
        "citations": len(re.findall(r"\[\d+\]|第\s*\d", text)),
        "h1": len(re.findall(r"(?m)^#\s", text)),
        "h2": len(re.findall(r"(?m)^##\s", text)),
        "h3": len(re.findall(r"(?m)^###\s", text)),
        "table_lines": len(re.findall(r"(?m)^\s*\|.*\|\s*$", text)),
    }


def _write_comparison(out_dir):
    direct_path = os.path.join(out_dir, "direct_report.md")
    synth_path = os.path.join(out_dir, "synthesis_report.md")
    lines = ["# Writer 实验对比（供人工评审）", ""]

    if not (os.path.isfile(direct_path) and os.path.isfile(synth_path)):
        lines.append("（未生成全部报告，无法对比。请使用 --mode both 运行。）")
        _write_text(os.path.join(out_dir, "comparison.md"), "\n".join(lines))
        return

    d = _analyze(_read_file(direct_path))
    s = _analyze(_read_file(synth_path))
    lines.extend(
        [
            "> 说明：以下仅为本实验自动统计的结构性指标，**引用数量/字数多少不等同于质量高低**，"
            "仅供人工评审时参考。",
            "",
            "| 指标 | Version A (Direct) | Version B (Synthesis+Writer) |",
            "|---|---|---|",
            f"| 保存位置 | direct_report.md | synthesis_report.md |",
            f"| 正文字符数 | {d['chars']} | {s['chars']} |",
            f"| 引用/页码标记数 | {d['citations']} | {s['citations']} |",
            f"| 一级标题数 | {d['h1']} | {s['h1']} |",
            f"| 二级标题数 | {d['h2']} | {s['h2']} |",
            f"| 三级标题数 | {d['h3']} | {s['h3']} |",
            f"| 表格行数 | {d['table_lines']} | {s['table_lines']} |",
            "",
            "## 两版文本",
            "",
            "### Version A：direct_report.md",
            f"见 [direct_report.md](./direct_report.md)（{d['chars']} 字符）",
            "",
            "### Version B：synthesis_report.md",
            f"见 [synthesis_report.md](./synthesis_report.md)（{s['chars']} 字符）",
            "",
            "核心观点清单见 [research_synthesis.json](./research_synthesis.json)。",
        ]
    )
    _write_text(os.path.join(out_dir, "comparison.md"), "\n".join(lines))


def _print_usage(label: str, metrics: dict):
    usage = metrics["usage"]
    total = usage["total"]
    print(f"[{label}] 耗时 {metrics['elapsed_seconds']}s | 调用 {total['calls']} 次 | "
          f"输入 {total['prompt_tokens']:,} tokens（缓存命中 {total['cache_hit_tokens']:,} / "
          f"未命中 {total['cache_miss_tokens']:,}）| 输出 {total['completion_tokens']:,} tokens | "
          f"费用 ¥{total['cost_current']:.4f}（{usage['period_label']}时段）")


def run(run_dir_arg: str, mode: str):
    run_dir = os.path.abspath(run_dir_arg)
    plan_path = os.path.join(run_dir, "plan.md")
    findings_dir = os.path.join(run_dir, "findings")
    if not os.path.isdir(run_dir):
        raise SystemExit(f"运行目录不存在：{run_dir}")
    if not os.path.isfile(plan_path):
        raise SystemExit(f"未找到 plan.md：{plan_path}")
    if not os.path.isdir(findings_dir):
        raise SystemExit(f"未找到 findings 目录：{findings_dir}")

    cfg = get_config()
    if not cfg.deepseek_api_key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")

    out_dir = os.path.join(run_dir, "writer_experiment")
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录：{out_dir}")

    task = _load_task()
    section = load_section(task.get("section_id", "industry_competition"))
    task_context = _build_task_context(task, section)
    writing_spec = _build_writing_spec()

    finding_files = sorted(f for f in os.listdir(findings_dir) if f.endswith(".md"))
    plan_text = _read_file(plan_path)
    findings_text = _render_findings(findings_dir, finding_files)
    print(f"读取 plan.md + {len(finding_files)} 份 findings（合计 {len(plan_text) + len(findings_text):,} 字符），"
          f"未读取 report.md")

    manifest = _build_manifest(run_dir, plan_path, plan_text, findings_dir, finding_files, task, cfg)
    _write_json(os.path.join(out_dir, "input_manifest.json"), manifest)

    metrics = {
        "model_configured": cfg.deepseek_model,
        "temperature": TEMPERATURE,
        "writer_max_tokens": WRITER_MAX_TOKENS,
        "synthesis_max_tokens": SYNTHESIS_MAX_TOKENS,
        "versions": {},
    }

    if mode in ("direct", "both"):
        print("\n=== Version A（Direct Writer）===")
        metrics["versions"]["direct"] = _run_direct(
            task_context, writing_spec, plan_text, findings_text, out_dir
        )
        _print_usage("Version A", metrics["versions"]["direct"])

    if mode in ("synthesis", "both"):
        print("\n=== Version B（Synthesis + Writer）===")
        metrics["versions"]["synthesis"] = _run_synthesis_writer(
            task_context, writing_spec, findings_text, out_dir
        )
        _print_usage("Version B", metrics["versions"]["synthesis"])

    _write_json(os.path.join(out_dir, "run_metrics.json"), metrics)
    _write_comparison(out_dir)

    print(f"\n[完成] 输出目录：{out_dir}")
    for f in ("direct_report.md", "research_synthesis.json", "synthesis_report.md",
              "run_metrics.json", "comparison.md", "input_manifest.json"):
        if os.path.isfile(os.path.join(out_dir, f)):
            print(f"  - {f}")


def main():
    parser = argparse.ArgumentParser(description="Writer 独立实验（Experiment A）")
    parser.add_argument(
        "--run-dir", required=True, help="已有运行目录，如 outputs/run_52a6c22b"
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "synthesis", "both"],
        default="both",
        help="运行哪个版本（默认 both）",
    )
    args = parser.parse_args()
    run(args.run_dir, args.mode)


if __name__ == "__main__":
    main()