"""DeepSeek token 用量统计与费用计算（元 / 百万 tokens 计费）。

费率表（来源：DeepSeek API 官方定价，峰谷定价于 2026-08-17 起生效）：
- 高峰时段：北京时间周一至周五（不含中国法定节假日）9:00-12:00、14:00-18:00。
- 空闲时段：其余时间，价格为高峰时段的一半。

本模块同时统计两条 DeepSeek 调用路径的 token：
1. `llm.py` 直接 HTTP 调用（工具内部 query 规划与合成），经 `record_usage` 记录；
2. LangChain `init_chat_model` 的 Agent 调用（Chief Researcher / researcher 子 agent），
   经 `UsageCallback`（LangChain 回调）记录。
"""
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, time, timezone

from langchain_core.callbacks import BaseCallbackHandler

# 每百万 tokens 价格（元）。空闲时段 = 高峰时段的一半。
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

# 北京时间时区
_BEIJING = timezone(timedelta(hours=8))


def _tier(model: str) -> str:
    """按模型名映射到费率档位。flash 系列走 flash 档，其余按旗舰档（v4-pro）计费。"""
    m = (model or "").lower()
    if "flash" in m:
        return "deepseek-flash"
    return "deepseek-v4-pro"


def is_peak_time(dt: datetime | None = None) -> bool:
    """判断是否为高峰时段（北京时间周一至周五 9:00-12:00、14:00-18:00）。

    未实现中国法定节假日判断（需要外部日历），默认按工作日处理。
    """
    bj = (dt if dt is not None else datetime.now(timezone.utc)).astimezone(_BEIJING)
    if bj.weekday() >= 5:  # 周六 / 周日
        return False
    t = bj.time()
    return (time(9, 0) <= t < time(12, 0)) or (time(14, 0) <= t < time(18, 0))


def period_label(dt: datetime | None = None) -> str:
    return "高峰" if is_peak_time(dt) else "空闲"


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    calls: int = 0


def _extract(usage: dict) -> tuple[int, int, int, int]:
    """从不同来源的 usage 结构中提取 (输入, 输出, 缓存命中, 缓存未命中)。"""
    if not usage:
        return 0, 0, 0, 0
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        details = (
            usage.get("prompt_tokens_details")
            or usage.get("input_token_details")
            or {}
        )
        hit = details.get("cached_tokens") or details.get("cache_read") or 0
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = max((prompt or 0) - (hit or 0), 0)
    return int(prompt or 0), int(completion or 0), int(hit or 0), int(miss or 0)


class _Tracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._per_model: dict[str, _Usage] = {}

    def record(self, model: str, usage: dict):
        prompt, completion, hit, miss = _extract(usage)
        if prompt == 0 and completion == 0:
            return
        with self._lock:
            u = self._per_model.setdefault(model or "unknown", _Usage())
            u.prompt_tokens += prompt
            u.completion_tokens += completion
            u.cache_hit_tokens += hit
            u.cache_miss_tokens += miss
            u.calls += 1

    def reset(self):
        with self._lock:
            self._per_model.clear()

    def snapshot(self) -> dict[str, _Usage]:
        with self._lock:
            return {k: _Usage(v.prompt_tokens, v.completion_tokens, v.cache_hit_tokens, v.cache_miss_tokens, v.calls) for k, v in self._per_model.items()}


_tracker = _Tracker()


def record_usage(model: str, usage: dict):
    """记录一次 DeepSeek 调用的 token 用量（供 llm.py 直接 HTTP 路径调用）。"""
    _tracker.record(model, usage)


def reset_usage():
    _tracker.reset()


def get_usage() -> dict[str, _Usage]:
    return _tracker.snapshot()


def get_usage_summary() -> dict:
    """返回可 JSON 序列化的用量与费用汇总（供外部脚本记录指标）。"""
    snapshot = _tracker.snapshot()
    now_bj = datetime.now(timezone.utc).astimezone(_BEIJING)
    cur_period = "peak" if is_peak_time(now_bj) else "off_peak"

    models = {}
    for model, u in snapshot.items():
        tier = _tier(model)
        models[model] = {
            "tier": tier,
            "calls": u.calls,
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "cache_hit_tokens": u.cache_hit_tokens,
            "cache_miss_tokens": u.cache_miss_tokens,
            "cost_current": round(_cost(u, tier, cur_period), 6),
            "cost_peak": round(_cost(u, tier, "peak"), 6),
            "cost_off_peak": round(_cost(u, tier, "off_peak"), 6),
        }

    def _sum(field):
        return sum(getattr(u, field) for u in snapshot.values())

    return {
        "period": cur_period,
        "period_label": period_label(now_bj),
        "models": models,
        "total": {
            "calls": _sum("calls"),
            "prompt_tokens": _sum("prompt_tokens"),
            "completion_tokens": _sum("completion_tokens"),
            "cache_hit_tokens": _sum("cache_hit_tokens"),
            "cache_miss_tokens": _sum("cache_miss_tokens"),
            "cost_current": round(
                sum(_cost(u, _tier(m), cur_period) for m, u in snapshot.items()), 6
            ),
            "cost_peak": round(
                sum(_cost(u, _tier(m), "peak") for m, u in snapshot.items()), 6
            ),
            "cost_off_peak": round(
                sum(_cost(u, _tier(m), "off_peak") for m, u in snapshot.items()), 6
            ),
        },
    }


class UsageCallback(BaseCallbackHandler):
    """LangChain 回调：捕获 Agent（Chief Researcher / researcher 子 agent）的 token 用量。"""

    def on_llm_end(self, response, **kwargs):
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage")
        model = llm_output.get("model_name") or ""
        if not usage:
            # 回退：从消息的 usage_metadata 提取（部分版本不放 llm_output["token_usage"]）
            try:
                msg = getattr(response.generations[0][0], "message", None)
                um = getattr(msg, "usage_metadata", None) or {}
                if um:
                    usage = {
                        "input_tokens": um.get("input_tokens", 0),
                        "output_tokens": um.get("output_tokens", 0),
                        "input_token_details": um.get("input_token_details", {}),
                    }
                    model = model or (getattr(msg, "response_metadata", None) or {}).get(
                        "model_name", ""
                    )
            except Exception:
                usage = None
        if not usage:
            return
        _tracker.record(model, usage)


def _cost(u: _Usage, tier: str, period: str) -> float:
    rates = PRICING[tier][period]
    return (
        u.cache_hit_tokens / 1e6 * rates["cache_hit"]
        + u.cache_miss_tokens / 1e6 * rates["cache_miss"]
        + u.completion_tokens / 1e6 * rates["output"]
    )


def format_report() -> str:
    """生成 token 用量与费用的可读汇总文本。"""
    snapshot = _tracker.snapshot()
    now_bj = datetime.now(timezone.utc).astimezone(_BEIJING)
    cur_period = "peak" if is_peak_time(now_bj) else "off_peak"

    lines = ["==================== Token 用量与费用 ===================="]
    lines.append(f"统计时刻（北京时间）: {now_bj:%Y-%m-%d %H:%M:%S}（{period_label(now_bj)}时段）")
    lines.append("")

    total_current = 0.0
    total_peak = 0.0
    for model, u in snapshot.items():
        tier = _tier(model)
        cur_cost = _cost(u, tier, cur_period)
        peak_cost = _cost(u, tier, "peak")
        off_cost = _cost(u, tier, "off_peak")
        total_current += cur_cost
        total_peak += peak_cost

        lines.append(f"[{model}]（费率档位: {tier.replace('deepseek-', '')}）")
        lines.append(f"  调用次数      : {u.calls}")
        lines.append(
            f"  输入 tokens   : {u.prompt_tokens:,}（缓存命中 {u.cache_hit_tokens:,} / 未命中 {u.cache_miss_tokens:,}）"
        )
        lines.append(f"  输出 tokens   : {u.completion_tokens:,}")
        lines.append(f"  费用（当前·{period_label(now_bj)}）: ¥{cur_cost:.4f}")
        lines.append(f"  费用（空闲参考）: ¥{off_cost:.4f}　（高峰参考）: ¥{peak_cost:.4f}")
        lines.append("")

    lines.append("-" * 52)
    lines.append(f"合计（当前·{period_label(now_bj)}时段）: ¥{total_current:.4f}")
    lines.append(f"合计（若全程高峰时段）        : ¥{total_peak:.4f}")
    lines.append("=" * 52)
    return "\n".join(lines)


def print_report():
    print(format_report())