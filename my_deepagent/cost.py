"""Token 统计与费用估算。

未知模型不会被错误归入旗舰模型费率；费用无法确认时明确标记为未配置。
峰谷判断暂按工作日时段计算，未接入中国法定节假日日历。
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Iterator

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # 允许不安装 LangChain 的纯工具单元测试导入本模块
    class BaseCallbackHandler:  # type: ignore[no-redef]
        pass

# 保留项目原有费率表；使用前请核对 DeepSeek 当前官方价格。
PRICING = {
    "deepseek-flash": {
        "peak": {"cache_hit": 0.04, "cache_miss": 2.0, "output": 8.0},
        "off_peak": {"cache_hit": 0.02, "cache_miss": 1.0, "output": 4.0},
    },
    "deepseek-v4-pro": {
        "peak": {"cache_hit": 0.30, "cache_miss": 9.0, "output": 27.0},
        "off_peak": {"cache_hit": 0.15, "cache_miss": 4.5, "output": 13.5},
    },
}
_BEIJING = timezone(timedelta(hours=8))


def _resolve_tier(model: str, configured: str | None = None) -> str | None:
    configured = (configured or "auto").lower()
    if configured in PRICING:
        return configured
    if configured not in {"", "auto"}:
        return None

    name = (model or "").lower()
    if "flash" in name:
        return "deepseek-flash"
    if "v4-pro" in name:
        return "deepseek-v4-pro"
    # 不再把 deepseek-chat 等未知模型默认按 V4 Pro 计费。
    return None


def is_peak_time(dt: datetime | None = None) -> bool:
    bj = (dt if dt is not None else datetime.now(timezone.utc)).astimezone(_BEIJING)
    if bj.weekday() >= 5:
        return False
    current = bj.time()
    return (time(9, 0) <= current < time(12, 0)) or (
        time(14, 0) <= current < time(18, 0)
    )


def period_label(dt: datetime | None = None) -> str:
    return "高峰" if is_peak_time(dt) else "空闲"


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            result = value.model_dump()
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
    return {}


def _extract(usage: dict) -> tuple[int, int, int, int]:
    usage = _as_dict(usage)
    if not usage:
        return 0, 0, 0, 0

    prompt = usage.get("prompt_tokens")
    if prompt is None:
        prompt = usage.get("input_tokens", 0)
    completion = usage.get("completion_tokens")
    if completion is None:
        completion = usage.get("output_tokens", 0)

    details = _as_dict(
        usage.get("prompt_tokens_details") or usage.get("input_token_details")
    )
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = details.get("cached_tokens")
    if hit is None:
        hit = details.get("cache_read", 0)

    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = max(int(prompt or 0) - int(hit or 0), 0)

    prompt = max(int(prompt or 0), 0)
    completion = max(int(completion or 0), 0)
    hit = min(max(int(hit or 0), 0), prompt)
    miss = max(int(miss or 0), 0)
    # 如果服务端给出的 hit/miss 之和异常，优先以 prompt-hit 兜底。
    if hit + miss != prompt:
        miss = max(prompt - hit, 0)
    return prompt, completion, hit, miss


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    calls: int = 0
    pricing_tier: str | None = None


class UsageTracker:
    """单次运行独立的线程安全用量统计器。"""

    def __init__(self):
        self._lock = threading.RLock()
        self._per_model: dict[str, Usage] = {}

    def record(self, model: str, usage: dict, pricing_tier: str | None = None) -> None:
        prompt, completion, hit, miss = _extract(usage)
        if prompt == 0 and completion == 0:
            return

        model_name = model or "unknown"
        tier = _resolve_tier(model_name, pricing_tier)
        with self._lock:
            current = self._per_model.setdefault(model_name, Usage(pricing_tier=tier))
            if current.pricing_tier != tier:
                current.pricing_tier = None
            current.prompt_tokens += prompt
            current.completion_tokens += completion
            current.cache_hit_tokens += hit
            current.cache_miss_tokens += miss
            current.calls += 1

    def reset(self) -> None:
        with self._lock:
            self._per_model.clear()

    def snapshot(self) -> dict[str, Usage]:
        with self._lock:
            return {
                name: Usage(
                    prompt_tokens=value.prompt_tokens,
                    completion_tokens=value.completion_tokens,
                    cache_hit_tokens=value.cache_hit_tokens,
                    cache_miss_tokens=value.cache_miss_tokens,
                    calls=value.calls,
                    pricing_tier=value.pricing_tier,
                )
                for name, value in self._per_model.items()
            }


_DEFAULT_TRACKER = UsageTracker()
_CURRENT_TRACKER: ContextVar[UsageTracker | None] = ContextVar(
    "deepagent_usage_tracker", default=None
)


@contextmanager
def usage_scope(tracker: UsageTracker) -> Iterator[UsageTracker]:
    """把直接 HTTP 调用与 LangChain 回调绑定到同一次运行的统计器。"""
    token = _CURRENT_TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _CURRENT_TRACKER.reset(token)


def _active_tracker() -> UsageTracker:
    return _CURRENT_TRACKER.get() or _DEFAULT_TRACKER


def record_usage(
    model: str, usage: dict, pricing_tier: str | None = None
) -> None:
    _active_tracker().record(model, usage, pricing_tier)


def reset_usage() -> None:
    _DEFAULT_TRACKER.reset()


def get_usage() -> dict[str, Usage]:
    return _active_tracker().snapshot()


def _cost(usage: Usage, period: str) -> float | None:
    if not usage.pricing_tier or usage.pricing_tier not in PRICING:
        return None
    rates = PRICING[usage.pricing_tier][period]
    return (
        usage.cache_hit_tokens / 1_000_000 * rates["cache_hit"]
        + usage.cache_miss_tokens / 1_000_000 * rates["cache_miss"]
        + usage.completion_tokens / 1_000_000 * rates["output"]
    )


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    pricing_tier: str | None = None,
) -> float | None:
    """按峰值费率保守估算单次调用的费用上界；未知模型返回 None。

    输入 token 统一按 cache_miss 峰值、输出 token 按 output 峰值计算，
    作为预算预留的费用上界。模型未匹配明确费率时返回 None，调用方不应
    据此触发费用硬约束。
    """
    tier = _resolve_tier(model, pricing_tier)
    if not tier or tier not in PRICING:
        return None
    rates = PRICING[tier]["peak"]
    return (
        max(int(prompt_tokens), 0) / 1_000_000 * rates["cache_miss"]
        + max(int(completion_tokens), 0) / 1_000_000 * rates["output"]
    )


def get_usage_summary(tracker: UsageTracker | None = None) -> dict:
    snapshot = (tracker or _active_tracker()).snapshot()
    now_bj = datetime.now(timezone.utc).astimezone(_BEIJING)
    current_period = "peak" if is_peak_time(now_bj) else "off_peak"

    models = {}
    for model, usage in snapshot.items():
        models[model] = {
            "tier": usage.pricing_tier,
            "calls": usage.calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cache_hit_tokens": usage.cache_hit_tokens,
            "cache_miss_tokens": usage.cache_miss_tokens,
            "cost_current": _cost(usage, current_period),
            "cost_peak": _cost(usage, "peak"),
            "cost_off_peak": _cost(usage, "off_peak"),
        }

    all_priced = all(item.pricing_tier for item in snapshot.values())
    current_partial = sum(
        _cost(item, current_period) or 0.0 for item in snapshot.values()
    )
    peak_partial = sum(_cost(item, "peak") or 0.0 for item in snapshot.values())
    off_peak_partial = sum(_cost(item, "off_peak") or 0.0 for item in snapshot.values())

    def total(field: str) -> int:
        return sum(getattr(item, field) for item in snapshot.values())

    return {
        "period": current_period,
        "period_label": period_label(now_bj),
        "models": models,
        "total": {
            "calls": total("calls"),
            "prompt_tokens": total("prompt_tokens"),
            "completion_tokens": total("completion_tokens"),
            "cache_hit_tokens": total("cache_hit_tokens"),
            "cache_miss_tokens": total("cache_miss_tokens"),
            "cost_complete": all_priced,
            "unpriced_models": [
                name for name, item in snapshot.items() if not item.pricing_tier
            ],
            "cost_current": current_partial if all_priced else None,
            "cost_peak": peak_partial if all_priced else None,
            "cost_off_peak": off_peak_partial if all_priced else None,
            "cost_current_partial": round(current_partial, 6),
            "cost_peak_partial": round(peak_partial, 6),
            "cost_off_peak_partial": round(off_peak_partial, 6),
        },
    }


class UsageCallback(BaseCallbackHandler):
    """记录 LangChain Agent 调用；与直接 HTTP 调用共享同一 Tracker。"""

    def __init__(
        self,
        tracker: UsageTracker | None = None,
        pricing_tier: str = "auto",
    ):
        self.tracker = tracker
        self.pricing_tier = pricing_tier

    def on_llm_end(self, response, **kwargs) -> None:
        llm_output = getattr(response, "llm_output", None) or {}
        llm_output = _as_dict(llm_output)
        usage = llm_output.get("token_usage")
        model = llm_output.get("model_name") or ""

        if not usage:
            try:
                message = response.generations[0][0].message
                metadata = _as_dict(getattr(message, "usage_metadata", None))
                if metadata:
                    details = _as_dict(metadata.get("input_token_details"))
                    usage = {
                        "input_tokens": metadata.get("input_tokens", 0),
                        "output_tokens": metadata.get("output_tokens", 0),
                        "input_token_details": details,
                    }
                response_metadata = _as_dict(
                    getattr(message, "response_metadata", None)
                )
                model = model or response_metadata.get("model_name", "")
            except Exception:
                usage = None

        if not usage:
            return

        tracker = self.tracker or _active_tracker()
        tracker.record(model, usage, self.pricing_tier)


def format_report(tracker: UsageTracker | None = None) -> str:
    summary = get_usage_summary(tracker)
    total = summary["total"]
    lines = [
        "==================== Token 用量与费用 ====================",
        f"统计时段（北京时间）: {summary['period_label']}",
        "",
    ]

    for model, item in summary["models"].items():
        lines.extend([
            f"[{model}]（费率档位: {item['tier'] or '未配置'}）",
            f"  调用次数      : {item['calls']}",
            (
                f"  输入 tokens   : {item['prompt_tokens']:,}"
                f"（缓存命中 {item['cache_hit_tokens']:,} / "
                f"未命中 {item['cache_miss_tokens']:,}）"
            ),
            f"  输出 tokens   : {item['completion_tokens']:,}",
        ])
        if item["cost_current"] is None:
            lines.append("  费用          : 未估算（当前模型未匹配明确费率）")
        else:
            lines.append(f"  当前时段费用  : ¥{item['cost_current']:.4f}")
            lines.append(
                f"  空闲/高峰参考 : ¥{item['cost_off_peak']:.4f} / "
                f"¥{item['cost_peak']:.4f}"
            )
        lines.append("")

    lines.append("-" * 52)
    if total["cost_complete"]:
        lines.append(f"当前时段费用合计: ¥{total['cost_current']:.4f}")
    else:
        lines.append(
            "费用合计: 未完整估算；"
            f"已知费率部分约 ¥{total['cost_current_partial']:.4f}"
        )
        lines.append(
            "未配置费率模型: " + ", ".join(total["unpriced_models"])
        )
    lines.append("=" * 52)
    return "\n".join(lines)


def print_report(tracker: UsageTracker | None = None) -> None:
    print(format_report(tracker))