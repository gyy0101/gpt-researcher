"""三层预算管理：全局硬上限 + 角色预算 + 单个研究任务预算。

通过 LangChain AgentMiddleware 的 ``awrap_tool_call`` / ``awrap_model_call`` 钩子，
在工具层拦截「task 派发」与「quick_search / deep_research 调用」，无需修改
Deep Agents 第三方库源码。

关键约束：

- 账本键为 ``(scope, tool_kind)``。``scope`` 来自运行时 config 的
  ``metadata.lc_subagent_tool_call_id``（子任务）或 ``"chief"``（首席研究员）。
- 区分工具调用 attempts / successes / failures 与实际 Token / 费用；失败调用同样
  占用预算（attempts 已计入，successes/failures 仅用于观测）。
- ``max_subagent_tasks`` 与 ``max_quick_search_calls`` / ``max_deep_research_calls``
  为硬约束，在工具层原子判定并短路返回文案，模型据此停止派发/检索。
- ``max_tokens`` / ``max_cost`` 为可选硬约束，在每次会产生 LLM 消耗的调用前
  按「输入估算 + 输出上界」做预留，调用结束后用单次请求返回的真实 usage 结算，
  预留阶段即可拦截并发的超额消耗。
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config as get_runnable_config
from langgraph.types import Command

from my_deepagent.config import get_config as get_app_config
from my_deepagent.cost import UsageTracker, estimate_cost, record_usage, usage_scope


SCOPE_CHIEF = "chief"
SCOPE_GLOBAL = "global"
KIND_TASK = "task"
KIND_QUICK = "quick_search"
KIND_DEEP = "deep_research"
KIND_REASONING = "reasoning"

# deep_research 内部两次 LLM 调用的输出 token 上界：子问题拆分 1200 + 合成 5000。
_DEEP_INTERNAL_OUTPUT_TOKENS = 1200 + 5000
# 无法从模型配置推断 max_tokens 时的输出上界兜底。
_DEFAULT_OUTPUT_TOKENS = 8192

# 从 task 描述中提取 Findings 文件路径（Chief 在派发时指定）。
_FINDINGS_PATH_RE = re.compile(r"findings/[A-Za-z0-9_\-./]+\.md")
# Subagent 返回内容过短时视为无效，不做持久化。
_MIN_PERSIST_CHARS = 40

# Chief 视角的返回精简：quick_search 只做业务 scoping，不需要全文；read_file
# 读 Findings 只返回摘要索引，完整内容由独立 Writer 统一读取，避免 Chief 上下文膨胀。
_CHIEF_QUICK_MAX_TOTAL = 2400
_CHIEF_FINDING_PREVIEW_CHARS = 500


@dataclass
class Account:
    """某个 ``(scope, tool_kind)`` 维度的预算账本。"""

    attempts: int = 0
    successes: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    def record_outcome(self, success: bool) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def add_usage(self, prompt_tokens: int, completion_tokens: int, cost: float) -> None:
        self.prompt_tokens += max(int(prompt_tokens), 0)
        self.completion_tokens += max(int(completion_tokens), 0)
        self.cost += cost


def scope_of(config: dict[str, Any] | None) -> str:
    """从运行时 config 解析调用方 scope：子任务 ID 或 chief。"""
    if not config:
        return SCOPE_CHIEF
    metadata = config.get("metadata") or {}
    call_id = metadata.get("lc_subagent_tool_call_id")
    if call_id:
        return f"task:{call_id}"
    return SCOPE_CHIEF


def _message_content(message: Any) -> str:
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part) for part in content)
    return str(content)


def _estimate_input_tokens(messages: list[Any]) -> int:
    """以「1 字符 ≈ 1 token」估算输入上界，对中文场景偏保守。"""
    total = 0
    for message in messages:
        total += len(_message_content(message))
    return total


def _extract_usage_metadata(response: Any) -> tuple[int, int] | None:
    result = getattr(response, "result", None)
    if not result:
        return None
    message = result[0] if result else None
    metadata = getattr(message, "usage_metadata", None)
    if not isinstance(metadata, dict):
        return None
    prompt = metadata.get("input_tokens")
    if prompt is None:
        prompt = metadata.get("prompt_tokens", 0)
    completion = metadata.get("output_tokens")
    if completion is None:
        completion = metadata.get("completion_tokens", 0)
    return int(prompt or 0), int(completion or 0)


def _is_success(result: Any) -> bool:
    return getattr(result, "status", None) != "error"


def _extract_task_result_content(result: Any) -> str:
    """从 task 工具返回结果中提取 Subagent 最终回复文本。

    兼容两种形态：``Command(update={"messages": [ToolMessage(...)]})`` 与
    直接返回的 ``ToolMessage``。
    """
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        for message in update.get("messages", []):
            content = _message_content(message)
            if content.strip():
                return content.strip()
        return ""

    if isinstance(result, ToolMessage):
        return _message_content(result).strip()

    return _message_content(result).strip()


def _replace_tool_content(result: Any, new_content: str) -> Any:
    """把工具返回的 ToolMessage 内容替换为精简摘要，避免完整内容进入 Chief 上下文。"""
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        new_messages = []
        for message in update.get("messages", []):
            if isinstance(message, ToolMessage):
                new_messages.append(
                    ToolMessage(
                        content=new_content,
                        tool_call_id=message.tool_call_id,
                        artifact=getattr(message, "artifact", None),
                    )
                )
            else:
                new_messages.append(message)
        return Command(update={**update, "messages": new_messages})

    if isinstance(result, ToolMessage):
        return ToolMessage(
            content=new_content,
            tool_call_id=result.tool_call_id,
            artifact=getattr(result, "artifact", None),
        )

    return result


def _trim_quick_result(content: str) -> str:
    """精简 Chief 的 quick_search 返回：只保留定位所需的片段前缀。"""
    if len(content) <= _CHIEF_QUICK_MAX_TOTAL:
        return content
    trimmed = content[:_CHIEF_QUICK_MAX_TOTAL]
    # 尽量在片段边界截断，避免把片段截断在句子中间。
    last_seg = trimmed.rfind("### 检索片段")
    if last_seg > 0:
        trimmed = trimmed[:last_seg]
    return (
        trimmed
        + "\n……[快速检索结果已为首席研究员精简；如需进一步定位，请针对具体方向发起新的 quick_search。]"
    )


def _is_within_dir(path: Path, base: Path) -> bool:
    """判断 ``path`` 是否位于 ``base`` 目录内（含目录本身）。

    使用 ``Path.relative_to`` 而非字符串前缀匹配，避免 ``/a/b`` 与
    ``/a/b2`` 这类前缀串导致越界路径被误判为安全。
    """
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _atomic_write_text(path: Path, content: str) -> None:
    """原子写入文本：先写临时文件再替换，避免进程中断留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def _locate_finding(run_dir: Path, file_path: str) -> Path | None:
    """把 read_file 传入的路径定位到 findings 目录下的真实文件；非 Findings 返回 None。"""
    run_dir = Path(run_dir).resolve()
    norm = file_path.replace("\\", "/")
    idx = norm.find("findings/")
    if idx < 0:
        return None
    rel = norm[idx:]
    if rel.rstrip("/").endswith("plan.md"):
        return None
    target = (run_dir / rel).resolve()
    if not _is_within_dir(target, run_dir):
        return None
    return target if target.is_file() else None


def _summarize_finding(path: Path) -> str:
    """把一份 Findings 压缩为给 Chief 的摘要索引，不调用 LLM。"""
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not content:
        return ""

    incomplete = "研究不完整" in content
    sections = re.findall(r"^#{1,3}\s+(.+)$", content, flags=re.MULTILINE)
    srcs = sorted(set(re.findall(r"\[SRC:[0-9a-f]{12}\]", content)))
    body_lines = [ln for ln in content.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    preview = " ".join(body_lines[:5])[:_CHIEF_FINDING_PREVIEW_CHARS]

    lines = [
        f"[摘要] {path.name}（{len(content)} 字符）",
        f"研究完整性：{'不完整（有明确缺口）' if incomplete else '完整'}",
    ]
    if sections:
        lines.append("小节：" + " / ".join(sections[:12]))
    if srcs:
        lines.append(f"来源标识（{len(srcs)} 个）：" + "、".join(srcs[:20]))
    if preview:
        suffix = "……" if len(preview) >= _CHIEF_FINDING_PREVIEW_CHARS else ""
        lines.append("正文预览：" + preview + suffix)
    lines.append("（完整内容已持久化，将由独立 Writer 统一读取成稿，你无需读取全文。）")
    return "\n".join(lines)


def _collector_totals(collector: UsageTracker) -> tuple[int, int]:
    prompt = 0
    completion = 0
    for usage in collector.snapshot().values():
        prompt += usage.prompt_tokens
        completion += usage.completion_tokens
    return prompt, completion


class BudgetManager:
    """统一负责全局预算、角色预算与任务预算的原子预留与结算。

    并发安全：单个 ``threading.RLock``，每次检查与自增在同一临界区内完成，
    避免 check-then-act 竞态。
    """

    def __init__(
        self,
        *,
        max_subagent_tasks: int,
        max_quick_search_calls: int,
        max_deep_research_calls: int,
        per_task_quick_quota: int,
        per_task_deep_quota: int,
        chief_quick_reserve: int,
        max_context_chars: int,
        max_tokens: int | None = None,
        max_cost: float | None = None,
        pricing_tier: str = "auto",
    ) -> None:
        self.max_subagent_tasks = max_subagent_tasks
        self.max_quick_search_calls = max_quick_search_calls
        self.max_deep_research_calls = max_deep_research_calls
        self.per_task_quick_quota = per_task_quick_quota
        self.per_task_deep_quota = per_task_deep_quota
        self.chief_quick_reserve = chief_quick_reserve
        self.max_context_chars = max_context_chars
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.pricing_tier = pricing_tier

        self._lock = threading.RLock()
        self._ledger: dict[tuple[str, str], Account] = {}
        self._task_dispatched = 0
        self._research_finished = False

        # token / 费用硬约束的预留与结算状态
        self._reserved_tokens = 0
        self._settled_tokens = 0
        self._reserved_cost = 0.0
        self._settled_cost = 0.0
        self._reservations: dict[int, tuple[int, float]] = {}
        self._reservation_counter = 0

    # ---- 账本 ----

    def _account(self, scope: str, kind: str) -> Account:
        key = (scope, kind)
        account = self._ledger.get(key)
        if account is None:
            account = Account()
            self._ledger[key] = account
        return account

    def _total_attempts(self, kind: str) -> int:
        return sum(
            acct.attempts for (_scope, k), acct in self._ledger.items() if k == kind
        )

    # ---- 硬约束判定 ----

    def try_dispatch_task(self) -> tuple[bool, str]:
        """task 工具层：是否允许再派发一个研究员子任务。"""
        with self._lock:
            if self._research_finished:
                return False, "研究阶段已结束，不再派发新的研究任务。"
            if self._task_dispatched >= self.max_subagent_tasks:
                return False, (
                    f"研究任务派发已达硬上限（{self.max_subagent_tasks} 个）。"
                    "请基于已有 Findings 完成报告，不要再调用 task。"
                )
            self._task_dispatched += 1
            self._account(SCOPE_GLOBAL, KIND_TASK).attempts += 1
            return True, ""

    def try_consume(self, scope: str, kind: str) -> tuple[bool, str]:
        """quick_search / deep_research 工具层：原子判定并占用一次额度。"""
        if kind == KIND_QUICK:
            global_cap = self.max_quick_search_calls
            per_task_cap = None if scope == SCOPE_CHIEF else self.per_task_quick_quota
            label = "快速检索"
        elif kind == KIND_DEEP:
            global_cap = self.max_deep_research_calls
            per_task_cap = None if scope == SCOPE_CHIEF else self.per_task_deep_quota
            label = "深度研究"
        else:
            raise ValueError(f"不支持的工具类型：{kind}")

        with self._lock:
            if self._total_attempts(kind) >= global_cap:
                return False, (
                    f"{label}全局硬上限（{global_cap} 次）已耗尽。"
                    "请基于已获得材料完成判断，不要继续重复检索。"
                )
            account = self._account(scope, kind)
            if per_task_cap is not None and account.attempts >= per_task_cap:
                return False, (
                    f"该研究任务的{label}额度（{per_task_cap} 次）已耗尽，"
                    "请基于已获得材料完成本任务。"
                )
            if kind == KIND_QUICK and scope != SCOPE_CHIEF:
                remaining_after = (
                    global_cap - self._total_attempts(KIND_QUICK) - 1
                )
                if remaining_after < self.chief_quick_reserve:
                    return False, (
                        f"quick_search 剩余额度需为首席研究员保留 "
                        f"{self.chief_quick_reserve} 次，当前子任务不再占用。"
                    )
            account.attempts += 1
            return True, ""

    def finish_research_phase(self) -> None:
        """研究阶段结束后调用，锁定任务派发。"""
        with self._lock:
            self._research_finished = True

    # ---- 结果与用量记账 ----

    def record_outcome(self, scope: str, kind: str, success: bool) -> None:
        with self._lock:
            self._account(scope, kind).record_outcome(success)

    def add_usage(
        self,
        scope: str,
        kind: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float = 0.0,
    ) -> None:
        with self._lock:
            self._account(scope, kind).add_usage(prompt_tokens, completion_tokens, cost)

    # ---- token / 费用硬约束的预留与结算 ----

    def reserve_usage(self, upper_tokens: int, model: str) -> tuple[bool, int | None]:
        """调用前按上界预留；返回 (是否放行, 预留 ID)。未启用时直接放行。"""
        if self.max_tokens is None and self.max_cost is None:
            return True, None
        upper_tokens = max(int(upper_tokens), 0)
        upper_cost = estimate_cost(0, upper_tokens, model, self.pricing_tier) or 0.0
        with self._lock:
            if self.max_tokens is not None and (
                self._settled_tokens + self._reserved_tokens + upper_tokens
                > self.max_tokens
            ):
                return False, None
            if self.max_cost is not None and (
                self._settled_cost + self._reserved_cost + upper_cost > self.max_cost
            ):
                return False, None
            self._reservation_counter += 1
            rid = self._reservation_counter
            self._reservations[rid] = (upper_tokens, upper_cost)
            self._reserved_tokens += upper_tokens
            self._reserved_cost += upper_cost
            return True, rid

    def settle_usage(
        self,
        reservation_id: int | None,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> None:
        """调用结束后用真实 usage 结算并释放预留。"""
        if reservation_id is None:
            return
        actual_cost = (
            estimate_cost(prompt_tokens, completion_tokens, model, self.pricing_tier)
            or 0.0
        )
        with self._lock:
            entry = self._reservations.pop(reservation_id, (0, 0.0))
            upper_tokens, upper_cost = entry
            self._reserved_tokens -= upper_tokens
            self._reserved_cost -= upper_cost
            self._settled_tokens += int(prompt_tokens) + int(completion_tokens)
            self._settled_cost += actual_cost

    def consume_reservation(self, reservation_id: int | None) -> None:
        """异常或无 usage 场景下释放预留，不追加结算用量。

        与 ``settle_usage`` 的区别：仅归还预留额度，不把任何用量计入已结算，
        用于模型调用抛异常或缺少 usage_metadata 时防止预留永久占用预算。
        """
        if reservation_id is None:
            return
        with self._lock:
            entry = self._reservations.pop(reservation_id, (0, 0.0))
            upper_tokens, upper_cost = entry
            self._reserved_tokens -= upper_tokens
            self._reserved_cost -= upper_cost

    # ---- 观测 ----

    def summary(self) -> dict:
        with self._lock:
            ledger: dict[str, dict] = {}
            for (scope, kind), acct in sorted(self._ledger.items()):
                ledger[f"{scope}:{kind}"] = {
                    "attempts": acct.attempts,
                    "successes": acct.successes,
                    "failures": acct.failures,
                    "prompt_tokens": acct.prompt_tokens,
                    "completion_tokens": acct.completion_tokens,
                    "cost": round(acct.cost, 6),
                }
            return {
                "limits": {
                    "max_subagent_tasks": self.max_subagent_tasks,
                    "max_quick_search_calls": self.max_quick_search_calls,
                    "max_deep_research_calls": self.max_deep_research_calls,
                    "per_task_quick_quota": self.per_task_quick_quota,
                    "per_task_deep_quota": self.per_task_deep_quota,
                    "chief_quick_reserve": self.chief_quick_reserve,
                    "max_tokens": self.max_tokens,
                    "max_cost": self.max_cost,
                },
                "task_dispatched": self._task_dispatched,
                "research_finished": self._research_finished,
                "settled_tokens": self._settled_tokens,
                "reserved_tokens": self._reserved_tokens,
                "settled_cost": round(self._settled_cost, 6),
                "reserved_cost": round(self._reserved_cost, 6),
                "ledger": ledger,
            }


class BudgetMiddleware(AgentMiddleware):
    """挂载到 Chief 与各 Subagent 的预算拦截中间件。

    拦截点在 ``awrap_tool_call`` / ``awrap_model_call``，其中工具层的额度判定
    为硬约束（短路返回文案），token/费用硬约束在模型调用层做预留。
    """

    def __init__(self, budget: BudgetManager, run_dir: str | Path | None = None) -> None:
        super().__init__()
        self.budget = budget
        self.run_dir = Path(run_dir).resolve() if run_dir else None

    @staticmethod
    def _tool_model() -> str:
        return get_app_config().deepseek_model

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_call = request.tool_call
        name = tool_call.get("name", "")
        runtime_config = getattr(request.runtime, "config", None)
        scope = scope_of(runtime_config)

        if name == KIND_TASK:
            ok, message = self.budget.try_dispatch_task()
            if not ok:
                return ToolMessage(content=message, tool_call_id=tool_call.get("id"))
            result = await handler(request)
            self.budget.record_outcome(SCOPE_GLOBAL, KIND_TASK, _is_success(result))
            return self._persist_task_result(request, result)

        if name == "read_file" and scope == SCOPE_CHIEF and self.run_dir is not None:
            args = tool_call.get("args") or {}
            file_path = str(args.get("file_path") or "")
            target = _locate_finding(self.run_dir, file_path)
            if target is not None:
                summary = _summarize_finding(target)
                if summary:
                    return ToolMessage(content=summary, tool_call_id=tool_call.get("id"))
            return await handler(request)

        if name in (KIND_QUICK, KIND_DEEP):
            ok, message = self.budget.try_consume(scope, name)
            if not ok:
                return ToolMessage(content=message, tool_call_id=tool_call.get("id"))

            reservation_id: int | None = None
            if name == KIND_DEEP:
                args = tool_call.get("args") or {}
                query = str(args.get("query") or "")
                parent_query = str(args.get("parent_query") or "")
                upper_tokens = (
                    len(query)
                    + len(parent_query)
                    + self.budget.max_context_chars
                    + _DEEP_INTERNAL_OUTPUT_TOKENS
                )
                ok, reservation_id = self.budget.reserve_usage(
                    upper_tokens, self._tool_model()
                )
                if not ok:
                    return ToolMessage(
                        content="Token/费用预算已耗尽，请基于已获得材料完成判断。",
                        tool_call_id=tool_call.get("id"),
                    )

            collector = UsageTracker()
            try:
                with usage_scope(collector):
                    result = await handler(request)
            except Exception:
                # 释放预留并记录失败，避免预留泄漏（attempts 已在 try_consume 计入）。
                prompt, completion = _collector_totals(collector)
                model = self._tool_model()
                if reservation_id is not None:
                    self.budget.settle_usage(reservation_id, prompt, completion, model)
                self.budget.record_outcome(scope, name, False)
                self._merge_back(collector)
                raise
            prompt, completion = _collector_totals(collector)
            model = self._tool_model()
            cost = estimate_cost(prompt, completion, model, self.budget.pricing_tier) or 0.0
            if reservation_id is not None:
                self.budget.settle_usage(reservation_id, prompt, completion, model)
            self.budget.add_usage(scope, name, prompt, completion, cost)
            self.budget.record_outcome(scope, name, _is_success(result))
            self._merge_back(collector)
            if name == KIND_QUICK and scope == SCOPE_CHIEF:
                content = _message_content(result)
                trimmed = _trim_quick_result(content)
                if trimmed != content:
                    result = _replace_tool_content(result, trimmed)
            return result

        return await handler(request)

    def _persist_task_result(self, request: Any, result: Any) -> Any:
        """统一持久化 Subagent 返回的研究内容。

        只要 Subagent 返回了足够长的研究内容，就落盘到 findings 目录，不再
        依赖 Subagent 主动调用 write_file。落盘后把返回给 Chief 的 ToolMessage
        内容替换为精简摘要，避免完整 Findings 反复堆积进 Chief 上下文。
        """
        content = _extract_task_result_content(result)
        if not content or len(content) < _MIN_PERSIST_CHARS:
            return result

        args = (request.tool_call or {}).get("args") or {}
        description = str(args.get("description") or "")
        rel_match = _FINDINGS_PATH_RE.search(description)
        call_id = str((request.tool_call or {}).get("id") or "")

        if self.run_dir is None:
            return result

        run_dir = self.run_dir.resolve()
        findings_dir = run_dir / "findings"
        findings_dir.mkdir(parents=True, exist_ok=True)

        # 从 task 描述中解析出的 Findings 路径优先；仅当路径确实落在 run 目录内
        # 才使用，否则退化为安全的固定文件名，避免越界写入。
        target: Path | None = None
        if rel_match:
            candidate = (run_dir / rel_match.group(0)).resolve()
            if _is_within_dir(candidate, run_dir):
                target = candidate

        if target is None:
            safe_id = re.sub(r"[^A-Za-z0-9_-]", "", call_id) or uuid.uuid4().hex[:8]
            target = findings_dir / f"task_{safe_id}.md"
            if not _is_within_dir(target, run_dir):
                target = findings_dir / f"task_{uuid.uuid4().hex[:8]}.md"

        try:
            already_saved = (
                target.is_file()
                and len(target.read_text(encoding="utf-8").strip()) > 0
            )
        except OSError:
            already_saved = False

        if not already_saved:
            header = (
                "> 程序自动持久化：该 Subagent 返回了研究内容，但未主动调用 "
                "write_file，以下为程序落盘的原始返回。\n\n"
            )
            try:
                _atomic_write_text(target, header + content + "\n")
            except OSError:
                pass

        summary = (
            f"研究任务已完成，结果已持久化到 {target.relative_to(run_dir)}"
            f"（{len(content)} 字符）。"
        )
        return _replace_tool_content(result, summary)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        try:
            scope = scope_of(get_runnable_config())
        except RuntimeError:
            scope = SCOPE_CHIEF
        model_name = (
            getattr(request.model, "model_name", None)
            or getattr(request.model, "model", None)
            or "unknown"
        )
        output_max = self._output_max_tokens(request)
        input_upper = _estimate_input_tokens(request.messages)
        if request.system_message is not None:
            input_upper += len(_message_content(request.system_message))
        upper_tokens = input_upper + output_max

        ok, reservation_id = self.budget.reserve_usage(upper_tokens, model_name)
        if not ok:
            return AIMessage(
                content="Token/费用预算已耗尽，请基于已有材料完成报告。"
            )

        try:
            response = await handler(request)
        except Exception:
            # 模型调用抛异常时释放预留，避免预留泄漏导致预算被永久占用。
            self.budget.consume_reservation(reservation_id)
            raise

        usage = _extract_usage_metadata(response)
        if usage is not None:
            prompt, completion = usage
            self.budget.settle_usage(reservation_id, prompt, completion, model_name)
            cost = estimate_cost(prompt, completion, model_name, self.budget.pricing_tier) or 0.0
            self.budget.add_usage(scope, KIND_REASONING, prompt, completion, cost)
        else:
            # 缺少 usage_metadata 时释放预留，避免预留泄漏。
            self.budget.consume_reservation(reservation_id)
        return response

    @staticmethod
    def _output_max_tokens(request: Any) -> int:
        settings = getattr(request, "model_settings", {}) or {}
        value = settings.get("max_tokens")
        if value is None:
            value = getattr(request.model, "max_tokens", None)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = None
        return value if value and value > 0 else _DEFAULT_OUTPUT_TOKENS

    @staticmethod
    def _merge_back(collector: UsageTracker) -> None:
        """把工具内部 LLM 用量合并回外层 Tracker，保证运行级报告不丢数据。"""
        for model, usage in collector.snapshot().items():
            if usage.prompt_tokens or usage.completion_tokens:
                record_usage(
                    model,
                    {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "prompt_cache_hit_tokens": usage.cache_hit_tokens,
                        "prompt_cache_miss_tokens": usage.cache_miss_tokens,
                    },
                    pricing_tier=usage.pricing_tier,
                )