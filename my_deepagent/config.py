"""集中管理 DeepSeek 与 RAG 配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass


def _parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数，当前值：{raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0，当前值：{value}")
    return value


@dataclass(slots=True)
class Config:
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout: int = 120
    # auto 表示只识别明确的模型名；未知模型不猜费率。
    deepseek_pricing_tier: str = "auto"

    rag_base_url: str = ""
    rag_kb_ids: list[str] = field(default_factory=list)
    rag_as_of_date: str = ""
    rag_timeout: int = 60

    @classmethod
    def from_env(cls) -> "Config":
        tier = os.getenv("DEEPSEEK_PRICING_TIER", "auto").strip().lower() or "auto"
        allowed = {"auto", "deepseek-flash", "deepseek-v4-pro"}
        if tier not in allowed:
            raise ValueError(
                "DEEPSEEK_PRICING_TIER 仅支持 auto、deepseek-flash、deepseek-v4-pro"
            )
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
            deepseek_timeout=_positive_int_env("DEEPSEEK_TIMEOUT", 120),
            deepseek_pricing_tier=tier,
            rag_base_url=os.getenv("RAG_BASE_URL", "").strip().rstrip("/"),
            rag_kb_ids=_parse_list(os.getenv("RAG_KB_IDS", "")),
            rag_as_of_date=os.getenv("RAG_AS_OF_DATE", "").strip(),
            rag_timeout=_positive_int_env("RAG_TIMEOUT", 60),
        )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """进程内缓存配置；测试或运行时修改环境变量后可调用 cache_clear()。"""
    return Config.from_env()
