"""Deep Agents Harness：Chief Researcher + Researcher Subagent。"""
from __future__ import annotations

import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model

from my_deepagent.budget import BudgetManager, BudgetMiddleware
from my_deepagent.config import get_config
from my_deepagent.tools import build_research_tools


CHIEF_RESEARCHER_PROMPT = """你是一家上市公司研究团队的首席研究员。你的任务不是写通用投资报告，而是针对给定公司、报告期间和预先定义的 Section，完成业务研究并交付研究报告。

## 核心原则
固定交付目标，自主决定研究路径。Section Definition 规定最终需要回答什么，不规定你必须如何检索。不要擅自生成本 Section 不需要的估值、目标价或投资评级。

## 工作流程
1. 理解业务范围：先用 quick_search 了解公司主要业务，识别是否需要拆分多个行业或业务线。不要预设公司只有一个行业。quick_search 返回内容已为定位精简，仅用于发现线索，不要当作可引用的完整材料。
2. 制定研究计划：用 write_file 写入 plan.md。计划应说明业务范围、关键研究问题、问题依赖关系、优先级和预算安排。研究问题必须是可调查、可验证的问题，不要把交付标题直接改成搜索词。
3. 分派研究：使用 task 工具调用 researcher 子 Agent。每个 task 聚焦一个具体问题，并指定 findings/<两位序号>-<英文短名>.md。独立问题可并行；存在依赖的问题应先完成前置研究。
4. 阅读与反思：read_file 读取 Findings 时只会返回摘要索引（完整性标记、小节结构、来源标识、正文预览），不会返回全文——完整内容由独立 Writer 统一读取成稿。请依据摘要索引与 task 返回的摘要判断覆盖情况：检查证据是否支撑判断、是否存在数据冲突、是否缺少会改变结论的重要因素。只有存在明确且重要的证据缺口时才追加研究，不要为凑轮次重复检索，也不要反复 read_file 同一文件试图获取全文。
5. 综合成稿（可选）：如研究覆盖已充分，可用 write_file 写入 report.md。若你未写出完整报告，程序会在你结束后由独立 Writer 统一成稿，你无需强求在收尾强行拼凑报告。

## 研究与引用规则
- 只把 Findings 中有证据支撑的事实写成事实；区分事实、来源观点、测算和推断。
- Findings 中的 [SRC:xxxxxxxxxxxx] 是本次 Run 的稳定来源标识。最终报告必须原样保留这些标识，不得转换为 [1]、[2]，不得创建或猜测标识。
- 不要自行生成来源清单；程序会在报告完成后统一转换引用编号并生成来源列表。
- quick_search 结果仅用于定位线索，不得将其临时序号当作正式引用。
- 表格必须标注年份、单位和统计口径；口径不一致时明确说明。
- 公司优势要解释如何转化为经营优势；劣势不能因为没有检索到就写成“无”。
- 缺失信息应区分：已确认不适用、有数据且已完成研究、检索不足、当前无法核实。
- 不要把所有 Findings 都塞进正文。只保留对判断有价值的信息，避免重复论据和机械堆砌。

## 交付规则
- 报告交付由程序兜底：若你未写出 report.md，独立 Writer 会基于 Findings 自动成稿。
- 你的核心职责是完成研究规划与覆盖判断，并确保每个研究问题都有对应的 Findings 落盘。
- 如已写出 report.md，请保持其完整、可读；即使部分交付内容无法核实，也应标明缺口。
- 完成后简短说明 Findings 覆盖情况与 report.md 路径（如已生成）。

{budget_note}

{section_brief}
"""

RESEARCHER_PROMPT = """你是上市公司研究团队中的研究专员。主研究员会交给你一个具体研究问题和 Findings 文件路径。

## 任务
围绕研究问题自主检索，形成有证据、有判断的研究发现。不要机械复制检索结果。

## 工作方式
1. 先判断是否需要 quick_search 定位背景，再使用 deep_research 对关键问题做深入研究。不要为了调用工具而调用工具。
2. 可以在预算内对不同维度补充研究，但每次调用都必须解决一个明确的信息缺口。
3. 尽量用 write_file 写入指定 Findings 文件，建议结构：
   - ## 研究问题
   - ## 关键证据
   - ## 初步判断
   - ## 尚未解决的问题
   - ## 来源映射
4. deep_research 返回的 [SRC:xxxxxxxxxxxx] 必须原样保留在对应事实或判断旁边。不要改写为数字编号，不要自行创建来源标识。
5. 不得将 quick_search 的临时结果编号当作正式引用。
6. 只写资料能够支持的事实。未获取到数据时明确说明；自行推算的数值标注“测算”，并说明计算口径。
7. 如果子查询部分失败，明确记录哪些维度没有成功检索；不要把检索失败写成“没有相关信息”。

## 交付要求（重要）
- 你的最终回复必须包含完整 Findings 正文（含上述建议结构和所有 [SRC:...] 标识）。
- 程序会自动把你的最终回复持久化为 Findings 文件，并把精简摘要返回给首席研究员；即使 write_file 未成功，研究内容也不会丢失。
- 如果研究因预算或检索失败而质量不足，请在回复开头用一句话明确标注「研究不完整」，并说明缺口维度；不要伪装成完整研究结论。

研究预算耗尽时，基于已获取材料完成 Findings，并如实记录缺口。
"""


def build_agent(
    task: dict,
    run_dir: str,
    section_brief: str = "",
):
    model_name = (
        task.get("strategic_model")
        or os.environ.get("STRATEGIC_LLM")
        or task.get("model")
        or "deepseek:deepseek-chat"
    )
    model = init_chat_model(model_name, max_tokens=int(task.get("agent_max_tokens", 8192)))

    max_rounds = int(task.get("max_research_rounds", 4))
    max_deep = int(task.get("max_deep_research_calls", 12))
    max_quick = int(task.get("max_quick_search_calls", 30))
    max_subagent_tasks = int(task.get("max_subagent_tasks", 6))
    per_task_quick_quota = int(task.get("per_task_quick_quota", 4))
    per_task_deep_quota = int(task.get("per_task_deep_quota", 3))
    chief_quick_reserve = int(task.get("chief_quick_reserve", 8))
    max_context_chars = int(task.get("max_context_chars", 35000))
    run_max_tokens = task.get("max_tokens")
    run_max_cost = task.get("max_cost")

    budget_note = (
        "\n## 研究预算\n"
        f"- 计划研究轮次建议不超过 {max_rounds} 轮：这是规划层面的软约束，"
        "不等同于工具调用次数，不强制执行。\n"
        f"- 研究员子任务（task 派发）硬上限：{max_subagent_tasks} 个，达到后不得再调用 task。\n"
        f"- quick_search 全局硬上限：{max_quick} 次，其中 {chief_quick_reserve} 次"
        "为首席研究员预留，子任务不可占用。\n"
        f"- deep_research 全局硬上限：{max_deep} 次。\n"
        f"- 每个研究员子任务最多 quick_search {per_task_quick_quota} 次、"
        f"deep_research {per_task_deep_quota} 次。\n"
        "- 以上均为代码层硬约束，超限调用会直接失败并返回提示；失败调用同样计入次数。\n"
        "- 预算耗尽后停止新增研究，基于已有材料完成交付并如实标明证据缺口。\n"
    )
    if run_max_tokens is not None or run_max_cost is not None:
        budget_note += (
            "- 本次运行启用了 Token/费用硬预算，接近上限时会被强制停止，"
            "请尽早综合成稿。\n"
        )

    quick_search, deep_research = build_research_tools(task, run_dir=run_dir)

    budget = BudgetManager(
        max_subagent_tasks=max_subagent_tasks,
        max_quick_search_calls=max_quick,
        max_deep_research_calls=max_deep,
        per_task_quick_quota=per_task_quick_quota,
        per_task_deep_quota=per_task_deep_quota,
        chief_quick_reserve=chief_quick_reserve,
        max_context_chars=max_context_chars,
        max_tokens=run_max_tokens,
        max_cost=run_max_cost,
        pricing_tier=get_config().deepseek_pricing_tier,
    )
    middleware = BudgetMiddleware(budget, run_dir=run_dir)

    researcher_subagent = {
        "name": "researcher",
        "description": (
            "针对一个具体研究问题进行知识库调查，并把研究问题、关键证据、"
            "初步判断、待解决问题和稳定来源标识写入指定 Findings 文件。"
        ),
        "system_prompt": RESEARCHER_PROMPT,
        "tools": [quick_search, deep_research],
        "middleware": [middleware],
    }

    agent = create_deep_agent(
        model=model,
        tools=[quick_search],
        system_prompt=CHIEF_RESEARCHER_PROMPT.format(
            budget_note=budget_note,
            section_brief=section_brief,
        ),
        subagents=[researcher_subagent],
        middleware=[middleware],
        backend=FilesystemBackend(
            root_dir=str(Path(run_dir).resolve()),
            virtual_mode=True,
        ),
    )
    return agent, budget